"""
ESP32-CAM Image Classifier — Training Script
=============================================
Trains a small CNN on your image dataset and exports a .tflite file
that can be flashed to an ESP32-CAM via a C header array.

WHAT THIS SCRIPT DOES (step by step):
  1. Loads all images from class sub-folders, converts to RGB 96x96
  2. Runs sanity checks so you catch data problems before wasting training time
  3. Splits into train / validation sets (80% / 20% by default)
  4. Builds a CNN -- either scratch (default) or MobileNetV2 transfer (--transfer)
  5. Computes class weights so the model doesn't favour the bigger class
     (fixes the "predicts everything as one class" problem)
  6. Trains with augmentation, early stopping, and LR scheduling
     Transfer mode runs a two-phase strategy: frozen backbone first, then
     fine-tunes the top 30 guavaNetV2 layers at a very low LR.
  7. Shows a per-class accuracy breakdown after training
  8. Exports to .tflite (float32 or int8 quantized)

ARCHITECTURE:
  Input      -> (96, 96, 3) RGB, normalised to [0.0, 1.0]
  Conv2D(16) -> BatchNorm -> ReLU -> MaxPool   output: 48x48x16
  Conv2D(32) -> BatchNorm -> ReLU -> MaxPool   output: 24x24x32
  Conv2D(64) -> BatchNorm -> ReLU -> MaxPool   output: 12x12x64
  Flatten    -> 9216 units
  Dense(128) -> ReLU -> Dropout(0.5)
  Dense(N)   -> Softmax                        N = number of classes

WHY RGB INSTEAD OF GRAYSCALE:
  The ESP32-CAM OV3660 outputs colour frames. Using RGB gives the model
  more information to distinguish objects that look similar in grayscale
  (e.g. a silver phone vs a dark powerbank under similar lighting).

FOLDER STRUCTURE EXPECTED:
  dataset/
      guava/
          img001.jpg
          img002.jpg
          ...
      powerbank/
          img001.jpg
          ...
  Class index = alphabetical order of folder names.
  guava -> 0,  powerbank -> 1  (because m < p)

USAGE:
  Scratch CNN (default, faster, needs more data):
      python train_classifier.py --dataset .

  Transfer learning (recommended for small datasets like yours):
      python train_classifier.py --dataset . --transfer

  All options:
      python train_classifier.py \
          --dataset    .            \   <- folder containing class sub-folders
          --epochs     80           \   <- max epochs (early stopping may end sooner)
          --batch_size 16           \   <- images per gradient update step
          --lr         0.0003       \   <- learning rate for Adam optimiser
          --val_split  0.2          \   <- 20% of images held out for validation
          --output     model.tflite \   <- output file name
          --transfer                \   <- use MobileNetV2 pretrained on ImageNet
          --quantize                    <- shrink model ~4x using int8 quantization

AFTER TRAINING:
  Convert .tflite to a C header for Arduino:
      xxd -i model.tflite > model_data.h

DEPENDENCIES:
  pip install tensorflow pillow numpy scikit-learn matplotlib
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import os
import sys
import random
import numpy as np

# Check for tensorflow before going further -- gives a clean error message
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend, works without a display
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(
        f"\nMissing dependency: {e}"
        "\nInstall everything with:"
        "\n    pip install tensorflow pillow numpy scikit-learn matplotlib\n"
    )

# =============================================================================
# REPRODUCIBILITY
# Seeds make results consistent across runs so you can compare experiments.
# Set these AFTER importing tensorflow/numpy, not before.
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)   # covers tf, keras, and random ops inside tf.data

print(f"TensorFlow {tf.__version__}")

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(
    description="Train an ESP32-CAM image classifier",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset",    required=True,
                    help="Root folder containing one sub-folder per class")
parser.add_argument("--epochs",     type=int,   default=80,
                    help="Maximum number of training epochs")
parser.add_argument("--batch_size", type=int,   default=16,
                    help="Images per gradient update. Keep small for small datasets.")
parser.add_argument("--lr",         type=float, default=3e-4, dest="learning_rate",
                    help="Initial learning rate for Adam. 3e-4 is a safe starting point.")
parser.add_argument("--val_split",  type=float, default=0.2,
                    help="Fraction of data reserved for validation (not used in training)")
parser.add_argument("--output",     default="model.tflite",
                    help="Output .tflite file path")
parser.add_argument("--transfer",   action="store_true", help="Use MobileNetV2 pretrained weights. Much better on small datasets.")
parser.add_argument("--quantize",   action="store_true",
                    help="Apply post-training int8 quantization (~4x smaller model)")
args = parser.parse_args()

# =============================================================================
# CONSTANTS
# These match what the ESP32-CAM sends to the classifier.
# =============================================================================
IMG_W    = 96       # pixels -- must match EI_CLASSIFIER_INPUT_WIDTH
IMG_H    = 96       # pixels -- must match EI_CLASSIFIER_INPUT_HEIGHT
CHANNELS = 3        # 3 = RGB colour image


# =============================================================================
# STEP 1 -- LOAD DATASET
# =============================================================================
def load_dataset(root: str):
    """
    Walks the dataset root folder, loads every image, resizes to 96x96 RGB,
    normalises pixel values to [0.0, 1.0], and assigns integer labels.

    Label index = alphabetical position of the folder name.
    Example: folders ['guava', 'powerbank'] -> guava=0, powerbank=1
    """

    # Find all sub-folders (each sub-folder = one class)
    found = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    if len(found) < 2:
        sys.exit(
            f"\nERROR: Found only {len(found)} class folder(s) in '{root}'.\n"
            "Need at least 2. Check your --dataset path.\n"
        )

    print(f"\n{'='*50}")
    print(f"DATASET LOADING")
    print(f"{'='*50}")
    print(f"Root folder : {os.path.abspath(root)}")
    print(f"Classes found (alphabetical order): {found}")
    print()

    images, labels = [], []

    for idx, cls in enumerate(found):
        cls_dir = os.path.join(root, cls)

        # Find all image files in this class folder
        files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        print(f"  [{idx}] '{cls}' -- {len(files)} images")

        if len(files) < 10:
            print(f"      WARNING: Very few images ({len(files)})."
                  " Aim for 50+ per class for reliable results.")

        loaded = 0
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")          # force 3-channel colour
                img = img.resize((IMG_W, IMG_H), Image.LANCZOS) # resize to 96x96
                arr = np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]
                images.append(arr)      # shape: (96, 96, 3)
                labels.append(idx)      # integer class index
                loaded += 1
            except Exception as e:
                print(f"      Skipping '{fname}': {e}")

        print(f"      Loaded: {loaded}/{len(files)}")

    if not images:
        sys.exit("\nERROR: No images loaded at all. Check folder contents.\n")

    X = np.array(images)   # shape: (total_images, 96, 96, 3)
    y = np.array(labels)   # shape: (total_images,)

    return X, y, found


X, y, class_names = load_dataset(args.dataset)
num_classes = len(class_names)


# =============================================================================
# STEP 2 -- SANITY CHECKS
# Run these before touching the model. Catches problems early.
# If any of these fail, fix the data issue before retraining.
# =============================================================================
print(f"\n{'='*50}")
print("SANITY CHECKS")
print(f"{'='*50}")

# Check 1: Array shape
# Expected: (N, 96, 96, 3) -- N images, 96x96 pixels, 3 colour channels
print(f"\n[1] Array shape    : {X.shape}")
print(f"    Expected       : (N, {IMG_H}, {IMG_W}, {CHANNELS})")
shape_ok = X.shape[1:] == (IMG_H, IMG_W, CHANNELS)
print(f"    Status         : {'OK' if shape_ok else 'WRONG -- check CHANNELS constant'}")

# Check 2: Pixel value range
# Should be 0.0 to 1.0 after /255 normalisation
# If mean is very close to 0 or 1, something went wrong with normalisation
px_min  = X.min()
px_max  = X.max()
px_mean = X.mean()
px_std  = X.std()
print(f"\n[2] Pixel range    : {px_min:.3f} -- {px_max:.3f}")
print(f"    Mean / Std     : {px_mean:.3f} / {px_std:.3f}")
range_ok = px_min >= 0.0 and px_max <= 1.0
print(f"    Status         : {'OK' if range_ok else 'WRONG -- pixels not normalised to [0,1]'}")
if px_std < 0.05:
    print("    WARNING: Very low std -- images may all look nearly identical")

# Check 3: Label distribution
# Imbalanced classes cause the model to predict the bigger class for everything.
# We handle this with sample weights, but good to know the ratio upfront.
print(f"\n[3] Label distribution:")
counts = []
for i, name in enumerate(class_names):
    count = int(np.sum(y == i))
    counts.append(count)
    bar = "#" * (count // 5)
    print(f"    [{i}] {name:15s}: {count:4d}  {bar}")
ratio = max(counts) / min(counts)
print(f"    Imbalance ratio: {ratio:.2f}x  "
      f"{'(mild, class weights will fix it)' if ratio < 3 else '(severe -- collect more images for the smaller class)'}")

# Check 4: Sample label alignment
# Verifies the first 10 images are labelled with the right class.
# If labels look scrambled here, something is wrong with the load order.
print(f"\n[4] First 10 label assignments (spot-check):")
for i in range(min(10, len(y))):
    print(f"    X[{i}] -> label {y[i]} -> '{class_names[y[i]]}'")

# Check 5: Approximate duplicate detection
# Flags if many images have identical mean values -- sign of real duplicates.
# Not perfect, but catches obvious copy-paste errors.
means = X.reshape(len(X), -1).mean(axis=1)
unique_means = len(np.unique(means.round(4)))
dup_ratio = 1.0 - unique_means / len(X)
print(f"\n[5] Approximate duplicate check:")
print(f"    Unique image signatures : {unique_means} / {len(X)}")
print(f"    Estimated duplicate rate: {dup_ratio*100:.1f}%")
if dup_ratio > 0.3:
    print("    WARNING: Many near-duplicate images detected. Consider removing actual duplicates.")
else:
    print("    OK")

print(f"\n{'='*50}")


# =============================================================================
# STEP 3 -- TRAIN / VALIDATION SPLIT
#
# stratify=y means both splits keep the same class ratio as the full dataset.
# For example if overall 55% guava / 45% powerbank, both train and val
# will also be roughly 55/45 -- not accidentally all one class.
#
# random_state=SEED makes the split reproducible (same split every run).
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=args.val_split,
    stratify=y,             # preserve class balance in both splits
    random_state=SEED,
)

print(f"\nTRAIN / VAL SPLIT")
print(f"  Train : {len(X_train)} images")
print(f"  Val   : {len(X_val)} images")
print(f"  Val distribution:")
for i, name in enumerate(class_names):
    print(f"    [{i}] {name}: {int(np.sum(y_val==i))} images")


# =============================================================================
# STEP 4 -- CLASS WEIGHTS
#
# WHY WE NEED THIS:
#   If guava has 210 images and powerbank has 172, the model sees guava
#   more often and learns to favour it. Left uncorrected, the model starts
#   predicting the majority class for everything (class collapse).
#   Class weights compensate by making mistakes on the rarer class count
#   more in the loss function.
#
# WHY WE BAKE IT INTO THE DATASET (not class_weight= in model.fit):
#   Keras silently ignores class_weight= when using a tf.data pipeline
#   that has a .map() augmentation step. The weights just don't get applied.
#   The fix is to include sample weights as a third element in the dataset:
#   (image, label, weight) instead of (image, label).
# =============================================================================
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    # "balanced" formula: total_samples / (n_classes * samples_in_this_class)
    # So the rarer class gets a higher weight automatically.
    classes=np.unique(y_train),
    y=y_train,
)

# Map each training sample to its weight based on its class label
sample_weights = np.array(
    [class_weights_arr[label] for label in y_train],
    dtype=np.float32
)

print(f"\nCLASS WEIGHTS (computed from training set):")
for i, name in enumerate(class_names):
    print(f"  [{i}] {name}: weight = {class_weights_arr[i]:.4f}")
print("  (higher weight = model is penalised more for missing that class)")


# =============================================================================
# STEP 5 -- DATA AUGMENTATION
#
# Augmentation creates artificially varied versions of each image during
# training. This helps the model generalise instead of memorising exact images.
# It ONLY runs on the training set -- never on validation.
# During inference (on-device), no augmentation happens.
#
# For a small dataset (a few hundred images), heavier augmentation is better
# because it prevents the model from memorising the exact training images.
# =============================================================================
augment = keras.Sequential([
    # Randomly mirror images left-right AND top-bottom.
    # Objects should be recognisable from any orientation.
    layers.RandomFlip("horizontal_and_vertical"),

    # Rotate up to +-20% of a full turn (~72 degrees)
    layers.RandomRotation(0.2),

    # Zoom in or out by up to 20%
    layers.RandomZoom(0.2),

    # Randomly shift brightness +-30% -- simulates different lighting conditions
    layers.RandomBrightness(0.3),

    # Randomly change contrast +-30% -- helps generalise across shadows/reflections
    layers.RandomContrast(0.3),

], name="augmentation")


# =============================================================================
# STEP 6 -- BUILD tf.data PIPELINES
#
# tf.data is TensorFlow's optimised data loading system.
# It handles batching, shuffling, augmentation, and prefetching
# (loading the next batch in the background while current batch is training).
#
# Training pipeline:
#   images + labels + sample_weights -> shuffle -> batch -> augment -> prefetch
#
# Validation pipeline:
#   images + labels -> batch -> prefetch
#   (no augmentation on val -- we want to evaluate on clean real images)
# =============================================================================

def augment_with_weights(x, y, w):
    """
    Apply augmentation to a batch.
    Must accept and return all three elements (image, label, weight).
    The weight passes through unchanged -- it's just along for the ride.
    """
    return augment(x, training=True), y, w


# Training dataset: includes sample weights for class balancing
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights))
    .shuffle(len(X_train), reshuffle_each_iteration=True)  # re-shuffle every epoch
    .batch(args.batch_size)
    .map(augment_with_weights, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)   # load next batch while current one trains
)

# Validation dataset: no augmentation, no weights, no shuffle
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(args.batch_size)
    .prefetch(tf.data.AUTOTUNE)
)


# =============================================================================
# STEP 7 -- BUILD MODEL
#
# Two modes:
#   Default (no --transfer): small CNN trained from scratch.
#                            Faster, smaller model (~2 MB), but needs more data.
#   Transfer (--transfer):   MobileNetV2 backbone pretrained on ImageNet.
#                            The backbone already knows shapes/textures/edges.
#                            We freeze it and only train the final classifier.
#                            Much better accuracy on small datasets like ours.
#                            Produces a larger model (~8 MB, after quantization ~2 MB).
# =============================================================================
def build_scratch_model(num_classes: int) -> keras.Model:
    """
    Small CNN trained from scratch.
    Only use this if you have 500+ images per class OR need a tiny model.

    Sizes through the network (96x96 input):
      After Block 1 MaxPool: 48x48x16
      After Block 2 MaxPool: 24x24x32
      After Block 3 MaxPool: 12x12x64
      After Flatten:         9216  (12x12x64)
    """
    inp = keras.Input(shape=(IMG_H, IMG_W, CHANNELS), name="input_image")

    # Block 1: low-level features (edges, colour gradients)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv1")(inp)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)      # 96->48

    # Block 2: mid-level features (corners, shapes, textures)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)      # 48->24

    # Block 3: high-level features (object parts, distinctive shapes)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)      # 24->12

    x = layers.Flatten(name="flatten")(x)                 # 12x12x64 = 9216
    x = layers.Dense(128, activation="relu", name="fc_hidden")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.Model(inp, out, name="esp32_scratch_classifier")


def build_transfer_model(num_classes: int) -> keras.Model:
    """
    MobileNetV2 backbone pretrained on ImageNet + custom classifier head.

    WHY THIS WORKS BETTER ON SMALL DATASETS:
      MobileNetV2 was trained on 1.2 million images across 1000 categories.
      It has already learned to detect shapes, textures, edges, and object
      parts in a general way. We reuse those learned filters and only train
      the final classification layer on our small dataset.
      This is called 'transfer learning'.

    TRAINING STRATEGY (two phases):
      Phase 1 -- Feature extraction (frozen backbone):
        The MobileNetV2 weights are frozen (not updated).
        Only the new Dense layers at the top are trained.
        Run for a few epochs until the head stabilises.

      Phase 2 -- Fine-tuning (unfreeze top layers):
        The top layers of MobileNetV2 are unfrozen and trained with
        a very low learning rate. This adapts the high-level features
        to your specific objects while keeping low-level features intact.

    NOTE ON MODEL SIZE:
      MobileNetV2 is ~14 MB as float32, ~3.5 MB quantized.
      This will NOT fit as a C array in ESP32 flash.
      Use SPIFFS or SD card to store the model file.
    """
    # Load MobileNetV2 without the top classification layer
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_H, IMG_W, CHANNELS),
        include_top=False,          # remove the 1000-class ImageNet head
        weights="imagenet",         # use pretrained weights (~14 MB, downloaded once)
    )

    # Phase 1: freeze the entire backbone
    base.trainable = False

    inp = keras.Input(shape=(IMG_H, IMG_W, CHANNELS), name="input_image")

    # MobileNetV2 expects pixels in [-1, 1]; preprocess_input rescales from [0, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)

    # Pass through frozen backbone -- keeps BatchNorm in inference mode
    x = base(x, training=False)

    # Global average pooling: reduces (3, 3, 1280) -> (1280,)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classifier head
    x = layers.Dense(128, activation="relu", name="fc_hidden")(x)
    x = layers.Dropout(0.3, name="dropout")(x)   # lower dropout -- backbone handles regularisation
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="esp32_transfer_classifier")

    # Store base model reference so we can unfreeze it later for fine-tuning
    model._base_model = base

    return model


def unfreeze_top_layers(model: keras.Model, num_layers: int = 30, new_lr: float = 1e-5) -> keras.Model:
    """
    Phase 2 of transfer learning: unfreeze the top N layers of MobileNetV2
    and recompile with a very low learning rate for fine-tuning.

    WHY LOW LR: The pretrained weights are already good. A high LR would
    destroy them in a few steps. We want tiny adjustments only.

    WHY ONLY TOP LAYERS: Lower layers detect generic features (edges, colours)
    that are universal. Top layers detect more specific features that benefit
    from adaptation to your dataset.
    """
    base = model._base_model
    base.trainable = True

    # Freeze all layers except the last num_layers
    for layer in base.layers[:-num_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"  Unfroze top {num_layers} layers of MobileNetV2 "
          f"({trainable_count} trainable layers total in backbone)")

    # MUST recompile after changing trainable state
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=new_lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Build the model based on --transfer flag
if args.transfer:
    print("\nMode: TRANSFER LEARNING (MobileNetV2 pretrained on ImageNet)")
    print("Downloading pretrained weights if not cached (~14 MB, one-time)...")
    model = build_transfer_model(num_classes)
else:
    print("\nMode: SCRATCH CNN")
    model = build_scratch_model(num_classes)

model.summary()


# =============================================================================
# STEP 8 -- COMPILE
#
# optimizer: Adam -- adapts the learning rate per-weight automatically.
#            Good default for almost all vision tasks.
# loss:      sparse_categorical_crossentropy -- standard for multi-class
#            classification when labels are integers (not one-hot vectors).
# metrics:   accuracy -- fraction of correct predictions, shown each epoch.
#
# Transfer mode: the backbone is frozen in Phase 1, so we use the default LR
# for the classifier head only. After Phase 1 training, unfreeze_top_layers()
# recompiles with a much lower LR for fine-tuning.
# =============================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# =============================================================================
# STEP 9 -- CALLBACKS
# These run automatically at the end of each epoch.
# =============================================================================

class StopAtPerfectAccuracy(keras.callbacks.Callback):
    """
    Stops training immediately when BOTH train AND val accuracy hit 1.0.

    WHY BOTH: Val accuracy alone hitting 1.0 on a small val set can happen by
    luck before the model has genuinely converged. Requiring train accuracy to
    also be 1.0 means the model has actually saturated the data, not just
    got lucky on a handful of val images.

    WHY STOP: Once accuracy is perfect there is nothing left to learn.
    Continuing only wastes time and risks the weights drifting slightly worse.
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get("accuracy", 0.0)
        val_acc   = logs.get("val_accuracy", 0.0)
        if train_acc >= 1.0 and val_acc >= 1.0:
            print(f"\n  Both train and val accuracy = 1.0 at epoch {epoch + 1}. "
                  "Stopping early -- nothing left to learn.")
            self.model.stop_training = True


callbacks = [

    # EarlyStopping: stops training when val_accuracy stops improving.
    # patience=15 = wait 15 epochs without improvement before stopping.
    # restore_best_weights=True = reloads the best weights seen during training,
    # not the final weights (which may have gotten worse after the peak).
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),

    # ReduceLROnPlateau: halves learning rate when val_loss plateaus.
    # Helps squeeze more accuracy after the initial fast learning phase.
    # Example: lr starts at 3e-4, may reduce to 1.5e-4, then 7.5e-5, etc.
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,       # new_lr = current_lr * 0.5
        patience=7,       # wait 7 epochs before reducing
        min_lr=1e-6,      # never go below this
        verbose=1,
    ),

    # ModelCheckpoint: saves the best model weights to disk during training.
    # Useful if training crashes -- you can reload from best_model.keras.
    keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),

    # StopAtPerfectAccuracy: halts immediately when both train AND val hit 1.0.
    # No point running more epochs once the model is already perfect.
    StopAtPerfectAccuracy(),
]


# =============================================================================
# STEP 10 -- TRAIN
# =============================================================================
print(f"\n{'='*50}")
print(f"TRAINING")
print(f"{'='*50}")
print(f"  Max epochs   : {args.epochs}")
print(f"  Batch size   : {args.batch_size}")
print(f"  Learning rate: {args.learning_rate}")
print(f"  Classes      : {class_names}")
print()

history = model.fit(
    train_ds,                       # (image, label, sample_weight) batches
    validation_data=val_ds,         # (image, label) batches -- no weights for val
    epochs=args.epochs,
    callbacks=callbacks,
    # NOTE: class_weight= is intentionally NOT passed here.
    # Keras silently ignores it when the dataset pipeline has a .map() step.
    # Sample weights are baked into train_ds as a third tensor instead.
)


# =============================================================================
# STEP 10b -- FINE-TUNING (transfer learning only)
#
# Phase 2: unfreeze the top 30 layers of MobileNetV2 and train for a few more
# epochs with a very low LR. This adapts high-level ImageNet features to our
# specific objects (phone vs powerbank shapes/textures).
#
# WHY AFTER PHASE 1: If we unfroze from the start, the randomly-initialised
# Dense head would produce large gradients that destroy the pretrained weights.
# Stabilising the head first (Phase 1) means fine-tuning starts from a
# sensible position.
# =============================================================================
if args.transfer:
    print(f"\n{'='*50}")
    print("FINE-TUNING (Phase 2 of transfer learning)")
    print(f"{'='*50}")

    model = unfreeze_top_layers(model, num_layers=30, new_lr=1e-5)

    # Separate callbacks for fine-tuning -- reset patience counters
    ft_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        StopAtPerfectAccuracy(),
    ]

    fine_tune_epochs = 30   # fine-tuning needs far fewer epochs than Phase 1
    print(f"  Running up to {fine_tune_epochs} fine-tuning epochs (early stopping applies)...")

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=ft_callbacks,
    )

    # Merge histories so the training plot covers both phases
    for key in history.history:
        history.history[key].extend(history_ft.history.get(key, []))


# =============================================================================
# STEP 11 -- EVALUATE AND DIAGNOSE
# =============================================================================
val_loss, val_acc = model.evaluate(val_ds, verbose=0)

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"  Val accuracy : {val_acc*100:.1f}%")
print(f"  Val loss     : {val_loss:.4f}")

# Run predictions first so pred_classes is available for the warning check below
pred_probs   = model.predict(X_val, verbose=0)
pred_classes = np.argmax(pred_probs, axis=1)   # pick the class with highest score

# Loss ≈ 0.693 = -ln(0.5) = what you get when model outputs 50/50 for every image.
# Only warn if the model is ALSO predicting only one class -- otherwise loss can
# legitimately be near 0.693 while the model is still learning (72%+ accuracy case).
if val_loss > 0.68 and len(np.unique(pred_classes)) == 1:
    print("\n  WARNING: Loss is near 0.693 (= -ln(0.5)).")
    print("  This means the model is outputting ~50/50 for every image.")
    print("  The model has NOT learned to distinguish the classes.")
    print("  This is usually a data pipeline issue, not a hyperparameter issue.")
    print("  Check the sanity output above for clues.")

print(f"\n  Predicted distribution : {np.bincount(pred_classes, minlength=num_classes)}")
print(f"  Actual distribution    : {np.bincount(y_val,         minlength=num_classes)}")

print(f"\n  Per-class accuracy:")
for i, name in enumerate(class_names):
    correct  = int(np.sum((pred_classes == i) & (y_val == i)))
    total    = int(np.sum(y_val == i))
    pct      = correct / total * 100 if total > 0 else 0
    bar      = "#" * int(pct / 5)
    print(f"    [{i}] {name:15s}: {correct:3d}/{total:3d}  ({pct:5.1f}%)  {bar}")

# Detect class collapse: model predicts only one class for everything
unique_predicted = len(np.unique(pred_classes))
if unique_predicted == 1:
    collapsed_to = class_names[pred_classes[0]]
    print(f"\n  CLASS COLLAPSE DETECTED: Model predicts '{collapsed_to}' for everything.")
    print(  "  Possible causes:")
    print(  "    - Sample weights not being applied (check train_ds construction above)")
    print(  "    - Learning rate too high (try --lr 0.0001)")
    print(  "    - One class is much easier to detect visually")
    print(  "    - Not enough epochs (try --epochs 120)")
else:
    print(f"\n  OK: Model predicts {unique_predicted} out of {num_classes} classes")


# =============================================================================
# STEP 12 -- PLOT TRAINING HISTORY
# Saves training_history.png so you can see what happened during training.
#
# What to look for:
#   - Both train and val accuracy should increase together (good learning)
#   - If train accuracy is high but val is low: overfitting (model memorised training data)
#   - If both are stuck at ~50%: model never learned (data/pipeline problem)
#   - If val accuracy jumps around wildly: val set is too small
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"],     label="Train accuracy")
ax1.plot(history.history["val_accuracy"], label="Val accuracy")
ax1.set_title("Accuracy over epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

ax2.plot(history.history["loss"],     label="Train loss")
ax2.plot(history.history["val_loss"], label="Val loss")
ax2.set_title("Loss over epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=120)
print(f"\n  Training history plot saved -> training_history.png")
plt.close()


# =============================================================================
# STEP 13 -- EXPORT TO TFLITE
#
# TFLite is a compressed format designed for microcontrollers and phones.
#
# Float32 (default):
#   - Same precision as training
#   - Larger file size
#   - Straightforward to deploy
#
# Int8 quantized (--quantize flag):
#   - Weights stored as 8-bit integers instead of 32-bit floats -> ~4x smaller
#   - Needs a "representative dataset" to calibrate the int8 scale factors
#   - Slight accuracy drop (usually < 2%)
#   - Input/output kept as float32 so Arduino sketch doesn't need changes
# =============================================================================
print(f"\n{'='*50}")
print(f"TFLITE EXPORT")
print(f"{'='*50}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

if args.quantize:
    print("  Mode: int8 post-training quantization")
    print("  Calibrating int8 scale factors on training set...")

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        # Feed a sample of training images through the model to determine
        # the typical range of values at each layer.
        # This is used to set the int8 scale factors.
        # More samples = better calibration = less accuracy loss.
        for batch_x, _, _ in train_ds.take(100):   # 100 batches
            for img in batch_x:
                yield [img[tf.newaxis, ...]]        # one image at a time

    converter.representative_dataset    = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.float32   # keep float I/O for Arduino compat
    converter.inference_output_type     = tf.float32

else:
    print("  Mode: float32 (no quantization)")

tflite_model = converter.convert()

with open(args.output, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"  Saved: {args.output}  ({size_kb:.1f} KB)")

if size_kb > 3000:
    print("  WARNING: Model is over 3 MB.")
    print("  This may not fit in ESP32 flash as a C array.")
    print("  Options:")
    print("    1. Run with --quantize  to shrink ~4x")
    print("    2. Reduce conv filters in build_model() (16->8, 32->16, 64->32)")
    print("    3. Store model on SPIFFS instead of embedding as C array")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*50}")
print("CLASS LABELS (copy into Arduino sketch)")
print(f"{'='*50}")
print("  const char* labels[] = {")
for i, name in enumerate(class_names):
    print(f'      "{name}",   // index {i} -- result.classification[{i}].value')
print("  };")

print(f"""
{'='*50}
NEXT STEPS
{'='*50}
1. Open training_history.png and check:
   - Did accuracy improve steadily? (good)
   - Did it flatline at 50%? (pipeline issue -- check sanity output)
   - Is val accuracy much lower than train? (overfitting -- more data or more dropout)

2. Convert model to C header for Arduino:
       xxd -i {args.output} > model_data.h

3. Flash to ESP32-CAM and test.

4. If accuracy is still low, try:
   - --transfer  (MobileNetV2 pretrained backbone -- best option for small datasets)
   - More varied images (different backgrounds, lighting, distances)
   - --lr 0.0001  (slower, more careful learning)
   - --epochs 120  (more time to learn)
   - --quantize  (smaller model if size is the issue)
""")
