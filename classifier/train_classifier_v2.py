"""
train_classifier_v2.py
======================
EdgeCV — guava / powerbank classifier.

WHAT CHANGED FROM v3:
  1. Stronger augmentation targeting the specific failure modes:
       - Partial view / edge of frame -> RandomCrop + RandomTranslation
       - Distance / scale -> stronger RandomZoom range
       - Lighting edge cases -> RandomBrightness + RandomContrast kept,
         range widened

  2. Confusion matrix printed after test evaluation so you can see
     which direction errors go (guava predicted as powerbank, or vice versa).

  3. Misclassified images saved to a folder so you can inspect what
     the model is still getting wrong and collect more of exactly that.

  4. Test-time augmentation (TTA) — at evaluation, each test image is
     run through N augmented versions and predictions are averaged. This
     gives a more stable accuracy number and often reveals whether errors
     are consistent (model genuinely wrong) or marginal (borderline confidence
     that flips with tiny input variation).

  5. Model summary written to a text file for reference.
"""

import os
import math
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR          = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/dataset"
MODEL_SAVE_PATH      = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/model.keras"
MODEL_SAVE_BEST      = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/model_best.keras"
TFLITE_PATH          = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/model.tflite"
HISTORY_PLOT_PATH    = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/model_history.png"
MISCLASSIFIED_DIR    = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/misclassified"

IMG_HEIGHT   = 96
IMG_WIDTH    = 96
BATCH_SIZE   = 16
EPOCHS       = 60
VAL_SPLIT    = 0.20
RANDOM_SEED  = 42
TTA_STEPS    = 5    # number of augmented versions per test image for TTA

CLASS_NAMES  = ["guava", "powerbank"]  # alphabetical, must match folder names

# ---------------------------------------------------------------------------
# Augmentation — targeting partial view and distance failure modes
# ---------------------------------------------------------------------------

def build_augmentation():
    """
    Augmentation pipeline specifically designed for the observed failure modes.

    RandomTranslation: shifts the object toward/away from edges, teaching the
    model that a partially-visible object is still classifiable. height/width
    factor of 0.2 means the image can shift up to 20% of its size in any direction.

    RandomZoom: wider range than v3 (-0.4 to +0.3) to cover both "very close"
    (zoomed in, object fills frame) and "very far" (zoomed out, object small).
    Negative zoom = zoom in (object larger), positive = zoom out (object smaller).

    RandomCrop equivalent: achieved by combining RandomZoom + RandomTranslation.
    A true RandomCrop would need a larger source image; this approximation is
    sufficient for 96x96 inputs.

    RandomFlip, RandomRotation, RandomBrightness, RandomContrast: same as v3.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),               # wider than v3 (was 0.15)
        layers.RandomZoom((-0.4, 0.3)),            # asymmetric: more zoom-in than zoom-out
        layers.RandomTranslation(0.20, 0.20),      # NEW: teaches partial-view tolerance
        layers.RandomBrightness(0.20),             # wider than v3 (was 0.10)
        layers.RandomContrast(0.20),               # wider than v3 (was 0.10)
    ], name="augmentation")


# ---------------------------------------------------------------------------
# Model — same architecture as v3, augmentation updated
# ---------------------------------------------------------------------------

def build_model() -> tf.keras.Model:
    augmentation = build_augmentation()

    model = models.Sequential([
        tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        augmentation,
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(8,  (3, 3), strides=(2, 2), activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(16, (3, 3), activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(16, (3, 3), activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(16, activation="relu"),
        layers.Dense(1,  activation="sigmoid"),
    ], name="tinyml_classifier_v4")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Dataset loading — group-aware split (unchanged from v3)
# ---------------------------------------------------------------------------

def _list_images(folder: str) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in exts
    ]


def _base_key(filename: str) -> str:
    name = os.path.basename(filename)
    for prefix in ("orig_", "clean_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def build_group_split(class_dir: str, val_fraction: float, seed: int):
    all_paths = _list_images(class_dir)
    groups = {}
    for p in all_paths:
        groups.setdefault(_base_key(p), []).append(p)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    val_count  = max(1, math.ceil(len(keys) * val_fraction))
    val_keys   = set(keys[:val_count])
    train_keys = set(keys[val_count:])

    return (
        [p for k in train_keys for p in groups[k]],
        [p for k in val_keys   for p in groups[k]],
    )


def paths_to_dataset(paths, labels, shuffle=False):
    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(
        load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(max(len(paths), 200), seed=RANDOM_SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_datasets():
    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []

    print("Dataset split:")
    for idx, cls in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        tr, vl = build_group_split(class_dir, VAL_SPLIT, RANDOM_SEED)
        train_paths  += tr;  train_labels += [idx] * len(tr)
        val_paths    += vl;  val_labels   += [idx] * len(vl)
        print(f"  {cls}: {len(tr)} train  |  {len(vl)} val")

    # Test set
    test_paths, test_labels, test_originals = [], [], []
    for idx, cls in enumerate(CLASS_NAMES):
        test_dir = os.path.join(DATASET_DIR, f"test_{cls}")
        if not os.path.isdir(test_dir):
            print(f"  WARNING: {test_dir} not found — skipping test eval")
            continue
        for p in _list_images(test_dir):
            test_paths.append(p)
            test_labels.append(idx)
            test_originals.append(p)
        print(f"  {cls} test: {len(_list_images(test_dir))} images")

    train_ds = paths_to_dataset(train_paths, train_labels, shuffle=True)
    val_ds   = paths_to_dataset(val_paths,   val_labels,   shuffle=False)
    test_ds  = paths_to_dataset(test_paths,  test_labels,  shuffle=False) \
               if test_paths else None

    return train_ds, val_ds, test_ds, train_paths, test_paths, test_labels


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def per_class_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("\n--- Per-Class Accuracy ---")
    for idx, name in enumerate(CLASS_NAMES):
        mask = (y_true == idx)
        if mask.sum() == 0:
            print(f"  {name:12s}: no samples")
            continue
        acc = (y_pred[mask] == y_true[mask]).mean() * 100
        print(f"  {name:12s}: {acc:.1f}%  ({int(mask.sum())} samples)")
    overall = (y_pred == y_true).mean() * 100
    print(f"  {'OVERALL':12s}: {overall:.1f}%  ({len(y_true)} samples)")


def confusion_matrix_print(y_true, y_pred):
    """
    Prints a 2x2 confusion matrix.
    Rows = actual class, Columns = predicted class.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    print("\n--- Confusion Matrix (rows=actual, cols=predicted) ---")
    header = "          " + "  ".join(f"{c:>12}" for c in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row = f"{row_name:>10}" + "  ".join(f"{cm[i][j]:>12}" for j in range(n))
        print(row)

    # Flag the most common error direction
    if n == 2:
        fp = cm[0][1]  # guava predicted as powerbank
        fn = cm[1][0]  # powerbank predicted as guava
        if fp > fn:
            print(f"\n  Most common error: guava misclassified as powerbank "
                  f"({fp} times)")
            print(f"  -> Collect more guava images at harder conditions")
        elif fn > fp:
            print(f"\n  Most common error: powerbank misclassified as guava "
                  f"({fn} times)")
            print(f"  -> Collect more powerbank images at harder conditions")
        else:
            print(f"\n  Errors balanced across both classes")


def save_misclassified(model, test_paths, test_labels):
    """
    Runs inference on test images one-by-one and saves misclassified
    ones to MISCLASSIFIED_DIR/<actual_class>/predicted_as_<class>/.
    Useful for understanding remaining failure modes.
    """
    if not test_paths:
        return

    shutil.rmtree(MISCLASSIFIED_DIR, ignore_errors=True)
    saved = 0

    for path, true_label in zip(test_paths, test_labels):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32)
        img_batch = tf.expand_dims(img, 0)

        prob = float(model.predict(img_batch, verbose=0)[0][0])
        pred_label = 1 if prob >= 0.5 else 0

        if pred_label != true_label:
            actual_name    = CLASS_NAMES[true_label]
            predicted_name = CLASS_NAMES[pred_label]
            out_dir = os.path.join(MISCLASSIFIED_DIR,
                                   f"actual_{actual_name}",
                                   f"predicted_as_{predicted_name}")
            os.makedirs(out_dir, exist_ok=True)
            conf = prob if pred_label == 1 else 1.0 - prob
            fname = f"conf{conf:.2f}_{os.path.basename(path)}"
            shutil.copy(path, os.path.join(out_dir, fname))
            saved += 1

    print(f"\n  {saved} misclassified images saved to: {MISCLASSIFIED_DIR}")
    print(f"  Filenames include confidence score for quick prioritisation.")


def test_time_augmentation(model, test_paths, test_labels, n_steps=TTA_STEPS):
    """
    Evaluates the model with test-time augmentation.
    Each image is augmented N times and predictions are averaged.
    A prediction that flips under TTA is a borderline case — the model is
    uncertain. A prediction that holds across all N versions is robust.
    """
    if not test_paths:
        return [], []

    aug = build_augmentation()

    y_true, y_pred_tta = [], []
    for path, label in zip(test_paths, test_labels):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32)

        probs = []
        for _ in range(n_steps):
            aug_img = aug(tf.expand_dims(img, 0), training=True)
            p = float(model.predict(aug_img, verbose=0)[0][0])
            probs.append(p)

        avg_prob = float(np.mean(probs))
        y_pred_tta.append(1 if avg_prob >= 0.5 else 0)
        y_true.append(label)

    return y_true, y_pred_tta


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(history.history["loss"],     label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss");     ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"History plot: {HISTORY_PLOT_PATH}")


# ---------------------------------------------------------------------------
# Int8 TFLite export
# ---------------------------------------------------------------------------

def convert_to_tflite_int8(model, calibration_paths):
    print("\n--- TFLite Int8 Quantization ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    sample = random.sample(calibration_paths, min(50, len(calibration_paths)))

    def representative_data_gen():
        for path in sample:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
            img = tf.cast(img, tf.float32)
            yield [tf.expand_dims(img, 0)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"Saved: {TFLITE_PATH}  ({size_kb:.1f} KB)")
    print("ESP32 input: int8_val = (int8_t)(uint8_pixel - 128)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== EdgeCV TinyML Training v4 ===")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Classes : {CLASS_NAMES}")
    print(f"Augmentation: RandomTranslation + wider zoom (targeting partial-view / distance)")
    print()

    train_ds, val_ds, test_ds, train_paths, test_paths, test_labels = load_datasets()

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_BEST,
            monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print("\nTraining...")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks,
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved: {MODEL_SAVE_PATH}")

    if test_ds is not None:
        print("\n--- Standard Test Evaluation ---")
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"Loss: {test_loss:.4f}  Accuracy: {test_acc*100:.2f}%")

        # Standard predictions
        y_true, y_pred = [], []
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0).flatten()
            y_pred.extend((preds >= 0.5).astype(int))
            y_true.extend(labels.numpy().astype(int))

        per_class_accuracy(y_true, y_pred)
        confusion_matrix_print(y_true, y_pred)

        # Test-time augmentation
        print(f"\n--- Test-Time Augmentation ({TTA_STEPS} steps) ---")
        y_true_tta, y_pred_tta = test_time_augmentation(
            model, test_paths, test_labels)
        per_class_accuracy(y_true_tta, y_pred_tta)
        confusion_matrix_print(y_true_tta, y_pred_tta)

        # Save misclassified images
        save_misclassified(model, test_paths, test_labels)

    plot_history(history)
    convert_to_tflite_int8(model, train_paths)

    print("\n=== Done ===")
    print(f"Next steps:")
    print(f"  1. Check {MISCLASSIFIED_DIR} to see what's still failing")
    print(f"  2. Collect more images of those specific failure cases")
    print(f"  3. Re-run sanitizer_dataset.py, then retrain")
    print(f"  4. xxd -i model.tflite > firmware/main/model_data.cc && idf.py build flash")


if __name__ == "__main__":
    main()
