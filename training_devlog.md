# ESP32-CAM Classifier — Development Log
### Project: `custom cnn for esp-cam` | Started: `cup` vs `mobile` → Iterated to: `guava` vs `powerbank`

---

## Overview

This document tracks the full journey of building a custom image classifier
for an ESP32-CAM (OV3660 sensor) — from reverse-engineering an existing Edge
Impulse deployment, to writing a training script from scratch, to debugging
a series of increasingly subtle problems, to ultimately changing the object
classes entirely because the original pair was too visually similar.

The goal is a working .tflite model that fits on the ESP32-CAM and can
eventually drive a rover autonomously.

---

## Phase 1 — Reverse Engineering the Original Edge Impulse Model

### Starting point

The project began with an existing (but not understood) Edge Impulse deployment.
The files provided were:

- `esp32_camera.ino` — Arduino sketch running inference
- `tflite_learn_1018948_4_compiled.cpp/.h` — EON-compiled TFLite model (weights baked into C arrays)
- `trained_model_ops_define.h` — list of disabled TFLite op variants
- `model_metadata.h` — classifier configuration macros
- `model_variables.h` — full impulse definition including class labels

Edge Impulse's EON compiler converts a .tflite model into a C++ file where
all weights, tensor shapes, and operator registrations are hardcoded as arrays.
By reading those arrays we can reconstruct the original architecture.

### What the C++ files revealed

| Detail | How we found it | Value |
|---|---|---|
| Input shape | `tensor_dimension0 = {4, {1,96,96,1}}` | 96×96 grayscale, 1 channel |
| Conv block 1 | `tensor_data6[16*3*3*1]` — weight array size | 16 filters, 3×3 kernel |
| Conv block 2 | `tensor_data2[32*3*3*16]` — weight array size | 32 filters, 3×3 kernel |
| Flatten size | `tensor_data3 = {-1, 18432}` | 24×24×32 = 18432 (matches 2 pooling layers) |
| Output | `tensor_data5[2]` — 2 output weights | 2 classes |
| Quantization | All int8/u8 variants disabled in ops_define.h | Float32 model, not quantized |
| Class labels | `ei_classifier_inferencing_categories[]` in model_variables.h | `{ "cup", "mobile" }` |

### Initial architecture guess (mostly right, one mistake)

```
Input (96, 96, 1) grayscale
→ Conv2D(16, 3×3) → ReLU → MaxPool(2×2)
→ Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
→ Flatten (18432)
→ Dense(128) → ReLU        ← assumed a hidden layer existed
→ Dense(2) → Softmax
```

The mistake: we assumed there was a hidden Dense(128) layer because that's
common. But the compiled graph has exactly 7 node invocations with only one
`FULLY_CONNECTED` operator. There is no hidden layer — it goes straight from
Flatten to the output.

### Corrected architecture

```
Input (96, 96, 1) grayscale
→ Conv2D(16, 3×3) → ReLU → MaxPool(2×2)
→ Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
→ Flatten (18432)
→ Dense(2) → Softmax       ← single FC directly to output
```

---

## Phase 2 — Writing the Training Script

### Decisions made upfront

**Switch to RGB.** The OV3660 captures colour frames. The original EI model
used grayscale because that's what the DSP block was configured for, but there
is no hardware reason to throw away colour information. RGB gives the model
more signal to work with, especially if the two objects differ in colour.

**Add a 3rd conv block.** Going from 1 channel (grayscale) to 3 channels
(RGB) means 3× more input data. A deeper network can extract more useful
features from that extra information.

**Add Dense(128) hidden layer.** The original model had none, but we were
now training from scratch on new classes. A hidden layer gives the network
more capacity to combine spatial features before making a prediction.

**Add Dropout(0.5).** With only a few hundred images, the model can easily
memorise training samples. Dropout forces it to learn more robust features
by randomly disabling 50% of neurons during each training step.

### Resulting architecture

```
Input (96, 96, 3) RGB
→ Conv2D(16) → BatchNorm → ReLU → MaxPool    output: 48×48×16
→ Conv2D(32) → BatchNorm → ReLU → MaxPool    output: 24×24×32
→ Conv2D(64) → BatchNorm → ReLU → MaxPool    output: 12×12×64
→ Flatten (9216)
→ Dense(128) → ReLU → Dropout(0.5)
→ Dense(N) → Softmax
```

### Bugs caught immediately

| Bug | What went wrong | Fix |
|---|---|---|
| `convert("L")` but `CHANNELS=3` | Images loaded as grayscale (1 channel) but model expected RGB (3 channels) — silent shape mismatch | Changed to `convert("RGB")` |
| Seeds set before `import tensorflow` | `tf.keras.utils.set_random_seed()` called before tf existed — seeds did nothing | Moved all seed calls after imports |
| `batch_size=32` | With ~300 training images, batches of 32 meant only 9 gradient updates per epoch — too few to learn | Changed to 16 |
| `Dropout(0.1)` | Dropping only 10% of neurons is so mild it provides almost no regularisation | Changed to 0.4, later 0.5 |

---

## Phase 3 — Class Collapse: Model Predicts One Class for Everything

### What happened

```
Val accuracy : 54.5%
Val loss     : 0.6921
Predicted : [77  0]    ← every single image predicted as mobile
Actual    : [42 35]
```

54.5% accuracy sounds almost reasonable — until you notice it predicts mobile
for every image. Since 54.5% of the validation set is mobile, just saying
"mobile" for everything gives 54.5% for free. The model learned nothing.

### What loss = 0.693 means

Loss of exactly 0.693 is not a coincidence. It equals −ln(0.5), which is the
cross-entropy loss when the model outputs exactly 50% probability for each
class on every image. The model is outputting maximum uncertainty — a coin
flip — and it just happens to round to the majority class.

This is **class collapse**: the model abandons trying to learn the minority
class because it gets a lower average loss by always predicting the majority.

### Why it happened: dataset imbalance

Mobile had 210 images, powerbank had 172. A ~1.2× imbalance is mild but
enough to tip a model that isn't explicitly penalised for ignoring one class.

### Attempted fix: `class_weight=` in `model.fit()`

```python
model.fit(..., class_weight={0: 0.91, 1: 1.11})
```

This tells Keras to weight mistakes on powerbank more heavily in the loss.

**Result: model flipped to predicting powerbank for everything.**

```
Predicted : [0  77]    ← now every image predicted as powerbank
Actual    : [42 35]
```

The weights overshot and caused collapse in the opposite direction.

### Why `class_weight=` didn't work properly

Keras silently ignores `class_weight=` when the training dataset is a
`tf.data.Dataset` with a `.map()` augmentation step. The argument is accepted
without error, but the weights are never actually applied to the loss
computation. This is a known but underdocumented Keras/tf.data interaction.

### The real fix: sample weights baked into the dataset

Instead of relying on Keras to apply weights, include them as a third tensor
directly in the dataset:

```python
# Compute per-sample weights based on class
sample_weights = np.array([class_weights_arr[label] for label in y_train])

# Third element in the tuple — Keras reads this automatically
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train, sample_weights)
)

# Augmentation must pass the weight through unchanged
def augment_with_weights(x, y, w):
    return augment(x, training=True), y, w
```

When the dataset itself carries the weights, tf.data cannot drop them.

---

## Phase 4 — Gradient Explosion: Loss = 33 on Epoch 1

### What happened after the sample weight fix

```
Epoch 1: loss = 33.76    ← should be around 0.6-0.7
Epoch 2: loss = 0.6926   ← immediately collapsed
Epoch 3+: loss = 0.6930  ← stuck here forever
```

### What gradient explosion means

On epoch 1, the model computed a loss of 33 instead of the expected ~0.7.
That loss produced a gradient 50× larger than normal, which updated the
weights by a massive amount in one step. The weights were effectively destroyed
immediately, and the model could never recover — it sat at 0.693 for the rest
of training.

### Root cause: BatchNormalization + weighted inputs

BatchNorm tracks running statistics (mean and variance) of activations across
batches. It uses these to normalise each layer's output so training stays
stable.

The problem: on the very first batch, BatchNorm has no history and uses the
current batch to initialise its statistics. When that batch comes from a
`(image, label, weight)` dataset rather than a plain `(image, label)` dataset,
the weighted inputs on the first pass produce statistics that are far from
normal. This creates abnormally large activation values, which produce an
enormous loss, which destroys the weights in one update.

### How we confirmed it

We ran a minimal test with completely random data (no real signal):

```python
X = np.random.rand(100, 96, 96, 3).astype('float32')
y = np.array([0]*50 + [1]*50)
# Result: loss = 33.76 on epoch 1, then 0.693 forever
# Same exact pattern = same root cause
```

If it were a data issue, random data would not reproduce it.

### Fix 1: Remove BatchNormalization

Plain `Conv2D` with `activation="relu"` does not have this interaction:

```python
# Before (exploding):
x = layers.Conv2D(16, (3,3), use_bias=False)(inp)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# After (stable):
x = layers.Conv2D(16, (3,3), padding="same", activation="relu")(inp)
```

Note: `use_bias=False` is only appropriate when BatchNorm follows (BatchNorm
has its own bias term). With BatchNorm removed, `use_bias` goes back to the
default `True`.

### Fix 2: Gradient clipping as a safety net

```python
optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
```

`clipnorm=1.0` rescales the gradient so its magnitude never exceeds 1.0
before the weight update. Even if something produces a large gradient, it
cannot destroy the weights in a single step. With BatchNorm removed this
shouldn't be needed, but it costs nothing to keep.

---

## Phase 5 — Partial Progress: 72.7% but Mobile Still Weak

### Results after gradient explosion fix

```
Val accuracy : 72.7%
Per-class accuracy:
  [0] mobile    : 22/42  (52.4%)
  [1] powerbank : 27/35  (77.1%)
```

The model is genuinely learning both classes now (no collapse). But mobile
accuracy at 52% is barely above random. The model finds powerbank easy
(distinctive strap/connector visible from most angles) but struggles with
mobile (flat rectangular slab, similar shape from most angles).

### False alarm in the script

The warning `"Loss is near 0.693"` was firing at 72.7% accuracy because the
threshold `val_loss > 0.68` was too aggressive. A model can have val_loss of
0.69 while still learning — the loss being near 0.693 only indicates a problem
when the model is ALSO predicting only one class.

Fix:
```python
# Before: fires on loss alone
if val_loss > 0.68:

# After: only fires on genuine collapse
if val_loss > 0.68 and len(np.unique(pred_classes)) == 1:
```

### Also fixed: NameError crash

`pred_classes` was referenced in the warning check before it was defined a
few lines later. Fixed by moving `model.predict()` to before the warning.

---

## Phase 6 — The Real Problem: Objects Are Visually Identical

### Learning rate experiments showed the fundamental issue

| LR | Mobile accuracy | Powerbank accuracy | Overall |
|---|---|---|---|
| 3e-4 | 52% | 77% | 72.7% |
| 1e-4 | 81% | 31% | 58.4% |
| 5e-5 | 45% | 74% | 58.4% |

The model keeps trading one class for the other. Improving mobile hurts
powerbank by exactly the same amount and vice versa. This is not a learning
rate problem — the model is finding one shortcut feature and using it as a
binary switch.

### Colour statistics confirmed the problem

```python
mobile:    R=0.507  G=0.522  B=0.472   mean=0.500  std=0.196
powerbank: R=0.506  G=0.517  B=0.474   mean=0.499  std=0.172
```

The two classes are colour-identical — less than 0.01 difference per channel.
Both are grey/silver objects on similar backgrounds. RGB provides zero extra
signal over grayscale. The model must distinguish them by shape and texture
alone, at 96×96 resolution.

### Real-world testing confirmed the background bias

Testing with `test_laptop.py` revealed the model classifying faces and random
office scenes as "mobile" with 0.82 confidence. The training images were taken
against a white wall with the object close-up. The model learned to recognise
the white background + rectangular object combination, not the object itself.

When tested in a real environment with a busy background, the model had
no idea what to do — because it had never seen a busy background in training.

### Root cause summary

Three compounding problems:
1. **Too visually similar**: a phone and a powerbank are both dark rectangles
2. **Colour provides no signal**: identical RGB stats between classes
3. **Background bias**: training images used plain white backgrounds;
   real-world backgrounds are complex

No amount of learning rate tuning, architecture tweaking, or more epochs
will fix these — they are data and class selection problems.

---

## Phase 7 — Transfer Learning (MobileNetV2)

### Why transfer learning helps

A scratch CNN with 380 images must learn everything — edges, shapes, textures,
object parts — from scratch. MobileNetV2 was trained on 1.2 million images
across 1000 categories. It already knows how to detect shapes, textures, and
object parts in a general way. We reuse those learned filters and only train
the final classification layers on our data.

This is called **transfer learning** — transferring visual knowledge from a
large dataset to a small one.

### Architecture

```
Input (96, 96, 3)
→ preprocess_input       (rescale pixels from [0,1] to [-1,1] for MobileNetV2)
→ MobileNetV2 backbone   (frozen, 154 layers, pretrained on ImageNet)
→ GlobalAveragePooling2D (reduces (3,3,1280) → (1280,) — more efficient than Flatten)
→ Dense(128, relu)
→ Dropout(0.3)
→ Dense(N, softmax)
```

### Two-phase training strategy

Training in two phases avoids accidentally destroying the pretrained weights.

**Phase 1 — Head only (backbone frozen):**
The MobileNetV2 weights are locked. Only the Dense layers we added are
trained. This is necessary because the new head starts with random weights
and produces large gradients initially — if the backbone were unfrozen at
this point, those gradients would corrupt it immediately.

**Phase 2 — Fine-tuning (top 30 layers unfrozen):**
After the head stabilises, the top 30 layers of MobileNetV2 are unlocked
and trained at a very low learning rate (1e-5). The lower layers (which
detect generic edges and colours) remain frozen. Only the higher-level feature
detectors are adapted to distinguish phones from powerbanks.

Very low LR is critical in Phase 2 — the pretrained weights are already good.
We want tiny adjustments, not large updates.

### Model size

MobileNetV2 quantized comes to ~3.5 MB. This does NOT fit in ESP32 flash as
a C array. It must be stored on SPIFFS or SD card and loaded at runtime.

The scratch CNN quantized is ~1.2 MB and fits in SPIFFS with the default
partition scheme.

---

## Phase 8 — Class Change: Mobile/Powerbank → Guava/Powerbank

### Why we changed classes

After all the above debugging, the conclusion was clear: mobile and powerbank
are too visually similar for a small scratch CNN to reliably distinguish.

Both are:
- Dark rectangular slabs
- Similar size
- Held in a hand against varying backgrounds
- Colour-identical (RGB means differ by less than 0.01)

The shape difference (camera bump vs cable port) is subtle at 96×96 pixels.
Even MobileNetV2 was only marginally better because the core problem is the
objects themselves, not the model.

### The new pair: guava vs powerbank

A round, green/yellow fruit vs a flat dark rectangle. These are about as
visually different as two objects can be:

- **Shape**: completely different (round vs rectangular)
- **Colour**: completely different (green/yellow vs dark grey)
- **Texture**: different (rough/bumpy vs smooth)
- **Silhouette**: obvious at 96×96 resolution

A scratch CNN of this size should reach 90%+ accuracy on this pair without
any of the tricks needed for mobile/powerbank.

### What changes in the script

- Folder names: `guava/` and `powerbank/`
- Class order (alphabetical): `[0] guava`, `[1] powerbank`
- Update `LABELS` in `test_laptop.py` to `["guava", "powerbank"]`
- Everything else stays the same

### Training recommendations for the new dataset

- 100-150 images per class minimum
- Varied backgrounds (not just white wall — this was the key lesson)
- Varied distances (close, arm's length, far)
- Varied orientations (upright, sideways, tilted)
- With and without hand in frame
- Different lighting conditions

---

## Summary of All Fixes

| # | Problem | Symptom | Fix |
|---|---|---|---|
| 1 | `convert("L")` with `CHANNELS=3` | Images loaded as grayscale, model expected RGB | Changed to `convert("RGB")` |
| 2 | Seeds set before `import tensorflow` | Non-reproducible runs | Moved seeds after all imports |
| 3 | `class_weight=` silently dropped by tf.data | Model collapsed to majority class | Baked sample weights into dataset as 3rd tensor |
| 4 | BatchNorm + weighted dataset inputs | loss=33 on epoch 1, then stuck at 0.693 | Removed BatchNorm, added `clipnorm=1.0` |
| 5 | False collapse warning threshold | Warning fired at 72.7% accuracy | Added `len(np.unique(pred_classes))==1` condition |
| 6 | `pred_classes` used before defined | `NameError` crash after training | Moved `model.predict()` before the warning check |
| 7 | Mobile/powerbank colour-identical | Per-class accuracy flip-flops with any LR | Changed classes to guava/powerbank |

---

## Key Lessons

**On tf.data and Keras:**
`class_weight=` in `model.fit()` is silently ignored when the dataset has a
`.map()` step. No error, no warning — it just doesn't apply. Always bake
class weights into the dataset as a third `(image, label, weight)` tensor
if you are using augmentation via `.map()`.

**On BatchNormalization with weighted datasets:**
BatchNorm initialises its running statistics on the first batch. If that
batch comes from a weighted dataset, the statistics are skewed and activations
can explode (loss = 30+). The model then collapses to 0.693 and never
recovers. Fix: remove BatchNorm, or use Layer Normalization instead (which
normalises per-sample rather than per-batch).

**On loss = 0.693:**
This is −ln(0.5) — the loss value when a model outputs exactly 50%
probability for every class. It means the model has learned nothing and is
guessing. However, loss near 0.693 alone is NOT sufficient to diagnose
collapse — a model can have this loss briefly at the start of training while
it's still learning. Genuine collapse requires BOTH loss≈0.693 AND the
predicted distribution being all one class.

**On class collapse:**
A model collapses to predicting one class when the majority class provides
a lower average loss than trying to learn both. Symptoms: overall accuracy
≈ majority class fraction, predicted distribution = [N, 0] or [0, N].
Fix: sample weights. But watch for the tf.data silent-drop bug above.

**On per-class accuracy as the real metric:**
Overall accuracy is misleading when classes are imbalanced or when collapse
is happening. Always look at per-class accuracy. A 55% overall accuracy with
[77, 0] predicted distribution is very different from 55% with [42, 35].

**On background bias in training data:**
If training images all use the same background (plain white wall), the model
may learn to classify the background rather than the object. Always collect
training data in varied environments that match real usage conditions.

**On choosing object classes:**
The most impactful decision in a small-dataset classification project is
which classes to use. Choose objects that are visually distinct in:
- Shape (round vs rectangular)
- Colour (green vs grey)
- Texture (rough vs smooth)
Classes that differ only in subtle details (camera bump vs cable port on two
dark rectangles) require much more data and a more powerful model.

**On model size for ESP32:**
- Scratch CNN (~1.2 MB quantized): fits in SPIFFS with default partition
- MobileNetV2 (~3.5 MB quantized): requires custom SPIFFS partition or SD card
- Rule of thumb: if the objects are visually distinct, use the scratch CNN.
  Only reach for transfer learning when the objects are similar and you have
  ruled out data/class selection issues first.

---

## File Reference

| File | Purpose |
|---|---|
| `train_classifier.py` | Main training script |
| `test_laptop.py` | Webcam test script — test model before flashing to ESP32 |
| `best_model.keras` | Best weights saved during training (Keras format) |
| `model.tflite` | Exported model ready for ESP32 deployment |
| `model_data.h` | C header generated by `xxd -i model.tflite` |
| `training_history.png` | Accuracy and loss curves saved after each run |

---

## Current Status

- Classes: `guava` vs `powerbank`
- Dataset: being collected (target: 100-150 images per class)
- Model: scratch CNN, quantized, target size ~1.2 MB
- Deployment plan: SPIFFS on ESP32-CAM (default partition fits 1.2 MB)
- Next milestone: 90%+ val accuracy on guava/powerbank before flashing

## Next Steps

1. Collect 100-150 images per class with varied backgrounds and lighting
2. Train: `python train_classifier.py --dataset . --quantize`
3. Check size: should be ~1.2 MB
4. Test on laptop: `python test_laptop.py`
5. If accuracy is good: `xxd -i model.tflite > model_data.h`
6. Flash to ESP32-CAM via SPIFFS uploader
7. Wire up to rover motors once classification is reliable
