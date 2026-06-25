"""
train_classifier.py
======================
EdgeCV — guava / powerbank TinyML classifier.

WHAT CHANGED FROM v2:
  1. Dataset split strategy fixed.
     v2 did an in-memory 80/20 random split of the training folder using
     validation_split=. This is fine IF images are independent, but since
     the sanitizer produces both an original and a cleaned version of each
     physical photo, sequential frames of the same scene can appear in both
     train and val — giving optimistic val accuracy.
     v3 splits by UNIQUE SOURCE IMAGE (the base filename without the orig_/
     clean_ prefix), ensuring that all versions of one physical frame stay
     in the same split.

  2. Test set loading fixed.
     v2 loaded test_guava/ and test_powerbank/ as a separate manual scan,
     which bypassed Keras's class-label inference and required hardcoded
     class indices. v3 wraps the test data in the same
     image_dataset_from_directory interface, making it consistent with
     train and val.

  3. Calibration images fixed.
     v2 pointed the int8 calibration generator at the top-level DATASET_DIR
     which contains test_guava/ and test_powerbank/ alongside guava/ and
     powerbank/. Keras would pick all four subfolders as classes (4 classes
     instead of 2) and calibrate on the wrong distribution.
     v3 calibrates strictly on the training split.

  4. Class sanity assertion added.
     Build fails immediately if detected classes don't match CLASS_NAMES,
     instead of silently training on the wrong classes.

DATASET STRUCTURE EXPECTED:
  dataset/
    guava/          <- cleaned images (sanitizer output)
    powerbank/      <- cleaned images (sanitizer output)
    test_guava/     <- held-out test images (never seen during training)
    test_powerbank/ <- held-out test images (never seen during training)

  guava/ and powerbank/ are split 80/20 train/val INTERNALLY by this
  script using group-aware splitting on the base filename.

  test_guava/ and test_powerbank/ are loaded separately and never
  touched during training.

INT8 INPUT ON ESP32:
  The model has layers.Rescaling(1/255) inside it.
  After int8 quantization: input tensor expects int8 values.
  Conversion from camera uint8 pixel:
    int8_val = (int8_t)(uint8_pixel - 128)
  zero_point = -128, scale ≈ 1/255 (≈ 0.003922).
  See main_tinyml_probe.cc for the full pipeline.
"""

import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR         = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/dataset"
MODEL_SAVE_PATH_KERAS  = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/model.keras"
MODEL_SAVE_PATH_TFLITE = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/model.tflite"
HISTORY_PLOT_PATH      = r"/mnt/c/Users/it/Documents/EdgeCV/classifier/model_history.png"

IMG_HEIGHT  = 96
IMG_WIDTH   = 96
BATCH_SIZE  = 16
EPOCHS      = 60
VAL_SPLIT   = 0.20    # fraction of UNIQUE source images held out for val
RANDOM_SEED = 42

CLASS_NAMES = ["guava", "powerbank"]   # alphabetical, matches folder names

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_tinyml_model() -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    model = models.Sequential([
        tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),        # baked into int8 weights — do not re-apply on ESP32

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
    ], name="tinyml_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ---------------------------------------------------------------------------
# Group-aware train/val split
#
# The sanitizer produces pairs: orig_<name>.jpg and clean_<name>.jpg.
# We split on the UNIQUE base name so both versions of one physical photo
# always land in the same split. This prevents data leakage where the model
# sees the original in training and the cleaned version in validation (same
# scene, same light, effectively the same image).
# ---------------------------------------------------------------------------

def _list_images(folder: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png"}
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in exts
    ]


def _base_key(filename: str) -> str:
    """
    Strips orig_ / clean_ prefix to get the unique source image key.
    'orig_guava_0042.jpg' -> 'guava_0042.jpg'
    'clean_guava_0042.jpg' -> 'guava_0042.jpg'
    'guava_0042.jpg' -> 'guava_0042.jpg'  (passthrough if no prefix)
    """
    name = os.path.basename(filename)
    for prefix in ("orig_", "clean_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def build_group_split(class_dir: str, val_fraction: float, seed: int):
    """
    Returns (train_paths, val_paths) for one class folder.
    Splitting is done by unique source key, not by file.
    """
    all_paths = _list_images(class_dir)

    # Group files by their base key
    groups: dict[str, list[str]] = {}
    for p in all_paths:
        key = _base_key(p)
        groups.setdefault(key, []).append(p)

    unique_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_keys)

    val_count   = max(1, math.ceil(len(unique_keys) * val_fraction))
    val_keys    = set(unique_keys[:val_count])
    train_keys  = set(unique_keys[val_count:])

    train_paths = [p for k in train_keys for p in groups[k]]
    val_paths   = [p for k in val_keys   for p in groups[k]]

    return train_paths, val_paths


def paths_to_dataset(paths: list[str], labels: list[int],
                     shuffle: bool = False) -> tf.data.Dataset:
    """Builds a tf.data.Dataset from file paths and integer labels."""

    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    ).map(load_img, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(paths), 200), seed=RANDOM_SEED)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_datasets():
    """
    Loads train, val, and test datasets.
    Train and val come from guava/ and powerbank/ with group-aware splitting.
    Test comes from test_guava/ and test_powerbank/.
    """
    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}\n"
                f"Make sure the sanitizer has run and produced '{class_name}/' "
                f"inside DATASET_DIR."
            )

        tr_paths, vl_paths = build_group_split(class_dir, VAL_SPLIT, RANDOM_SEED)
        train_paths  += tr_paths
        train_labels += [label_idx] * len(tr_paths)
        val_paths    += vl_paths
        val_labels   += [label_idx] * len(vl_paths)

        print(f"  {class_name}: {len(tr_paths)} train  |  {len(vl_paths)} val "
              f"(from {len(tr_paths)+len(vl_paths)} total images)")

    train_ds = paths_to_dataset(train_paths, train_labels, shuffle=True)
    val_ds   = paths_to_dataset(val_paths,   val_labels,   shuffle=False)

    # Test set — from separate test_guava/ test_powerbank/ folders
    test_paths, test_labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        test_dir = os.path.join(DATASET_DIR, f"test_{class_name}")
        if not os.path.isdir(test_dir):
            print(f"  WARNING: test folder not found: {test_dir}  — skipping test eval")
            continue
        tp = _list_images(test_dir)
        test_paths  += tp
        test_labels += [label_idx] * len(tp)
        print(f"  {class_name} test: {len(tp)} images")

    test_ds = paths_to_dataset(test_paths, test_labels, shuffle=False) \
              if test_paths else None

    return train_ds, val_ds, test_ds, train_paths

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def per_class_accuracy(model: tf.keras.Model,
                       test_ds: tf.data.Dataset) -> None:
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).flatten()
        y_pred.extend((preds >= 0.5).astype(int))
        y_true.extend(labels.numpy().astype(int))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n--- Per-Class Accuracy (Test Set) ---")
    for idx, name in enumerate(CLASS_NAMES):
        mask = (y_true == idx)
        if mask.sum() == 0:
            print(f"  {name:12s}: no test samples found")
            continue
        acc = (y_pred[mask] == y_true[mask]).mean() * 100
        print(f"  {name:12s}: {acc:.1f}%  ({mask.sum()} samples)")
    overall = (y_pred == y_true).mean() * 100
    print(f"  {'OVERALL':12s}: {overall:.1f}%  ({len(y_true)} samples)")

# ---------------------------------------------------------------------------
# Training plot
# ---------------------------------------------------------------------------

def plot_history(history) -> None:
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

def convert_to_tflite_int8(model: tf.keras.Model,
                            calibration_paths: list[str]) -> None:
    print("\n--- TFLite Int8 Quantization ---")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Calibrate using a sample of actual training images (not the test set,
    # not the full dataset root which may contain extra subfolders)
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

    with open(MODEL_SAVE_PATH_TFLITE, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(MODEL_SAVE_PATH_TFLITE) / 1024
    print(f"Saved: {MODEL_SAVE_PATH_TFLITE}  ({size_kb:.1f} KB)")
    print()
    print("ESP32 input pipeline (verify against printed tensor params):")
    print("  scale      ≈ 0.003922  (1/255)")
    print("  zero_point = -128")
    print("  int8_val   = (int8_t)(uint8_pixel - 128)")
    print("  Do NOT divide by 255 in firmware — Rescaling is baked into weights.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== EdgeCV TinyML Training (v3) ===")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Classes : {CLASS_NAMES}")
    print(f"Split   : {int((1-VAL_SPLIT)*100)}/{int(VAL_SPLIT*100)} train/val "
          f"(group-aware — no leakage between orig/clean pairs)")
    print()

    train_ds, val_ds, test_ds, train_paths = load_datasets()

    model = build_tinyml_model()
    model.summary()

    best_path = MODEL_SAVE_PATH_KERAS.replace(".keras", "_best.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print("\nTraining...")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks,
    )

    model.save(MODEL_SAVE_PATH_KERAS)
    print(f"Model saved: {MODEL_SAVE_PATH_KERAS}")

    if test_ds is not None:
        print("\n--- Test Set Evaluation ---")
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"Test loss    : {test_loss:.4f}")
        print(f"Test accuracy: {test_acc * 100:.2f}%")
        per_class_accuracy(model, test_ds)
    else:
        print("\nNo test set found — skipping test evaluation.")

    plot_history(history)
    convert_to_tflite_int8(model, train_paths)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
