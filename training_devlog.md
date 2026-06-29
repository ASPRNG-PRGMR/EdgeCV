# EdgeCV Training Devlog

This document is the full development journal for the EdgeCV custom image
classifier. Every phase, every bug, every fix — written as it happened.

---

## Overview

The goal: train a small image classifier to distinguish two objects and run
it in real time on an ESP32-CAM with no server, no WiFi, no SD card.

The journey took eleven phases across two major model generations:

- **Phases 1–8**: training pipeline — from reverse-engineering someone
  else's model to a working guava/powerbank classifier on a laptop
- **Phases 9–10**: ESP32 deployment — memory probes, camera bringup,
  MobileNetV2 on SD card, working end-to-end but slow (~5.7s)
- **Phases 11–13**: TinyML rewrite — scratch CNN, int8 quantization,
  model embedded in flash, **184ms inference, working perfectly**

---

## Phase 1 — Reverse Engineering the Existing Pipeline

### Context

The project started with an existing Edge Impulse-based MicroPython demo
running MNIST digit recognition on the ESP32-CAM. The goal was to replace
it with a custom classifier for arbitrary objects.

Before writing a single line of training code, the existing pipeline was
reverse-engineered to understand:
- What image format did the model expect?
- How were pixels normalised?
- How were predictions read from the output tensor?

This took longer than expected because Edge Impulse's exported models
embed preprocessing implicitly in the graph, and the documentation assumes
you use their SDK end-to-end.

### Lesson

Reverse-engineer before you build. Know exactly what format and range the
model expects, what the output means, and how the training pipeline
produced the labels — before writing a single training image to disk.

---

## Phase 2 — First Training Script

### What was built

`train_classifier.py` — a from-scratch training script using Keras. Initial
architecture: a small scratch CNN with Conv2D, MaxPool, Dense layers.
Classes: initially tried phone vs powerbank.

### Problem

The model trained to ~80% accuracy on validation but classified everything
as one class on the test set. Per-class breakdown revealed one class was
being correctly predicted and the other was never predicted at all.

### Root cause

The two classes — phone and powerbank — were visually near-identical at
96×96 resolution. Both were dark rectangles with similar aspect ratios
under similar lighting. The model learned to always predict the majority
class because that minimised loss.

### Fix

Changed classes to **guava vs powerbank**: round green fruit vs flat dark
rectangle. Maximum visual dissimilarity at 96×96.

---

## Phase 3 — Class Imbalance and class_weight

### Problem

Dataset sizes were unequal (155 guava, 164 powerbank). Training loss was
erratic. Tried adding `class_weight` to `model.fit()`.

### What happened

`class_weight` is silently ignored when the dataset uses `.map()` for
augmentation. The argument is accepted without error, has no effect, and
no warning is raised. Hours spent debugging imagined learning rate issues.

### Fix

Removed `class_weight`. With guava/powerbank, the visual difference was
sufficient that mild imbalance didn't matter — both classes trained cleanly
without weighting.

### Lesson

`class_weight` does not work with datasets built using `.map()`. If you
need it, bake weights into the dataset pipeline as a third tensor, or use
`sample_weight` at the batch level.

---

## Phase 4 — BatchNorm Loss Spike

### What happened

Added `BatchNormalization` layers to stabilise training. On epoch 1, loss
spiked to 30+ and never recovered. The model output ~50% for every class
from that point forward — it had collapsed.

### Root cause

`BatchNormalization` tracks running statistics (mean, variance) across
batches. When used with an augmentation pipeline that modifies images
non-deterministically, the statistics are inconsistent between the forward
pass and the normalisation update step. This causes the gradient signal to
become contradictory and the network gets stuck at maximum entropy output.

### Fix

Removed all `BatchNormalization` layers. L2 regularisation added to Conv
layers instead (weight decay = 0.001) to provide the overfitting resistance
that BatchNorm was supposed to supply.

### Lesson

Never use BatchNorm in a pipeline where the dataset uses `.map()` for
random augmentation without being very careful about the `training=True/False`
flag propagation. For small models on small datasets, L2 regularisation
is simpler and more predictable.

---

## Phase 5 — "Loss = 0.693" Means Nothing by Itself

### Problem

During debugging, loss consistently started at 0.693. This was interpreted
as a sign that training had collapsed — 0.693 is `ln(2)`, the theoretical
loss for a model outputting exactly 50% for every binary prediction.

### What was actually happening

0.693 is the *correct* starting loss for a randomly-initialised binary
classifier. It means the model starts with no knowledge and improves from
there. It only indicates collapse when the loss stays at 0.693 throughout
training *and* the prediction distribution is uniform (all 50%).

A model can have loss = 0.693 at epoch 1 and reach 97% accuracy by
epoch 30 — this is normal.

### Lesson

Starting loss of 0.693 for binary cross-entropy is expected, not alarming.
Look at whether loss *decreases* over epochs, and whether per-class accuracy
is balanced, not just the starting value.

---

## Phase 6 — Background Bias

### Problem

Laptop test accuracy was excellent. Real-world accuracy was poor: the model
predicted one class whenever the background was bright and the other whenever
the background was dark, regardless of what object was in frame.

### Root cause

All training images of guava were taken against a white wall. All training
images of powerbank were taken on a dark desk. The model never needed to
learn anything about the objects — it learned background brightness.

This is a dataset problem, not a model problem. The model did exactly what
it was trained to do.

### Fix

Recollected all images with varied backgrounds: different surfaces, different
lighting conditions, both objects photographed against both light and dark
backgrounds. Dataset rebuild from scratch.

### Lesson

If pointing the camera at a white ceiling predicts one class and a dark desk
predicts the other, the model learned the background. Vary backgrounds
aggressively during data collection. The model must see both objects against
both kinds of backgrounds.

---

## Phase 7 — MobileNetV2 Transfer Learning (and Why it Was Overkill)

### What was tried

Switched to MobileNetV2 pretrained on ImageNet with two-phase fine-tuning:
frozen backbone first, then top-30-layers fine-tuned at LR=1e-5.

### Result

Excellent laptop accuracy. Model size: 2.77 MB float32.

### The problem this created

At 2.77 MB, the model is too large for SPIFFS. Required loading from SD
card at boot. Inference time on ESP32: **~5.7 seconds per frame**.

### Lesson

MobileNetV2 is the wrong tool for two visually distinct classes with a
small dataset. The extra capacity and pretrained features add nothing when
the objects are a round green fruit and a flat black rectangle. The scratch
TinyML CNN (built in Phase 11) outperformed it in speed by 30× at the cost
of ~1% accuracy.

---

## Phase 8 — Dataset Pipeline Formalised

### What was built

Three Jupyter notebooks implementing a formal data preparation pipeline:

**Notebook 1 — HSV Histogram Analysis**
Analyses hue and value distributions per class to confirm the objects are
distinguishable in colour space. Guava peaks around H=79 (green), powerbank
peaks vary — confirmed the dataset was usable.

**Notebook 2 — Hand Filtering**
A senior collaborator wrote an HSV skin-tone segmentation script to remove
hands from images. Method:
- HSV threshold (H 5–25, S>40, V>40) to detect skin
- Morphological open + dilate to clean noise and cover shadow fringe
- Border-connected contour filter — only keep skin blobs touching the image
  edge (the hand), discard internal warm-toned pixels (e.g. guava surface)
- Fill hand region with histogram-sampled background colour

The approach was sound. The flat-colour fill produced a visible "painted
over" patch when the background had any texture or gradient.

**Notebook 3 — Data Splitter**
Rigorous 70/15/15 train/val/test split at the *file* level. Both original
and hand-cleaned versions of each image are included in training, doubling
the effective dataset size without additional collection.

---

## Phase 9 — ESP32 Deployment: Probe Strategy

### Why probes

With the model trained and validated on the laptop, the deployment risks
were unknown. How much memory does a 2.77 MB model consume after
AllocateTensors()? Can the camera coexist with the model in memory?

Rather than write the full inference loop and debug memory, camera, and
inference bugs simultaneously, deployment was split into isolated probes.
Each probe answered exactly one question.

### Probe 1 — Model load + AllocateTensors only

No camera. Loads model from SD card into PSRAM, calls AllocateTensors().

**Result: PASS.**

```
PSRAM free after AllocateTensors : ~204 KB
Tensor arena allocated            : 1024 KB
Actual arena usage                 : ~414 KB
AllocateTensors() time             : ~149 ms
```

### Probe 2 — Camera coexistence

Same model load as Probe 1. Camera initialised afterward. One grayscale
frame captured.

**Result: PASS.**

```
PSRAM free after AllocateTensors (before camera) : 204 KB
PSRAM free after camera init                       : 195 KB
PSRAM free with frame buffer held                  : 195 KB
192 KB contiguous PSRAM remaining after everything.
```

### Lesson

Probing one thing at a time means that when something breaks, it's
immediately obvious which layer caused it. This paid off immediately in
Phase 10.

---

## Phase 10 — OV3660 Does Not Support RGB888

### The plan

Model trained on RGB images. Configure camera for `PIXFORMAT_RGB888`,
capture, fill float32 tensor, Invoke().

### What happened

```
E (5110) cam_hal: cam_config: ll_cam_set_sample_mode failed
E (5110) camera: Camera config failed with error 0xffffffff
E (5110) INFERENCE: esp_camera_init failed: ESP_FAIL
```

The driver rejected the format before the sensor was even programmed.

### Root cause

The OV3660 outputs raw YUV422 over its parallel interface. The esp32-camera
driver can software-convert that to JPEG or extract luma for GRAYSCALE.
It has no RGB888 conversion path for OV3660. This is a driver limitation,
not a wiring error. The same request works on OV2640.

Supported formats for OV3660: `PIXFORMAT_JPEG`, `PIXFORMAT_YUV422`,
`PIXFORMAT_GRAYSCALE`.

### Fix

Use `PIXFORMAT_YUV422` and convert to RGB in firmware using BT.601
full-range coefficients — the same coefficients PIL uses internally for
`Image.convert("RGB")`, ensuring the camera's colour output is consistent
with what the training images looked like.

YUYV layout (4 bytes = 2 pixels):
```
Y0  U  Y1  V
```

Conversion:
```
R = clamp(Y + 1.402   * (V-128))
G = clamp(Y - 0.34414 * (U-128) - 0.71414 * (V-128))
B = clamp(Y + 1.772   * (U-128))
```

Integer approximation for speed (no FPU per pixel):
```cpp
r = clamp255(Y + ((179 * (V-128)) >> 7));
g = clamp255(Y - ((44*(U-128) + 91*(V-128)) >> 7));
b = clamp255(Y + ((227 * (U-128)) >> 7));
```

### Bonus

YUV422 frame buffers at 96×96 = 18 432 bytes. RGB888 would have been
27 648 bytes — the forced format switch was also a free memory saving.

### End-to-end result

MobileNetV2 (float32, 2.77 MB from SD) running at ~5.7 seconds per frame.
Working, but impractically slow. This became the motivation for Phase 11.

---

## Phase 11 — TinyML Scratch CNN (The Right Architecture All Along)

### Motivation

5.7 seconds per frame is usable for a static demo but not for anything
interactive. MobileNetV2 was brought in because it was familiar, not
because it was appropriate. The objects are visually distinct. A much
smaller model should work.

### Architecture

```
Input: (96, 96, 3) RGB
Rescaling(1/255)          ← baked into weights, not applied manually
Conv2D(8,  3×3, stride=2) + ReLU + MaxPool   → 23×23×8
Conv2D(16, 3×3)           + ReLU + MaxPool   → 10×10×16
Conv2D(16, 3×3)           + ReLU             →  8×8×16
Flatten → Dropout(0.5) → Dense(16, ReLU) → Dense(1, sigmoid)
```

Binary sigmoid output. Alphabetical class order: 0=guava, 1=powerbank.

Quantized to int8: **~50 KB**. Fits in ESP32 flash. No SD card needed.

### Training improvements over v1/v2

- Group-aware train/val split: all versions of one physical photo
  (original + hand-cleaned) guaranteed to land in the same split.
  Prevents leakage where the model sees the original in training and
  the cleaned version in validation.
- Calibration data fixed: int8 calibration now samples from the actual
  training image paths, not the top-level dataset directory which
  contained test subfolders and confused Keras into detecting 4 classes.
- 50 calibration images instead of 15 — more stable quantization ranges.

---

## Phase 12 — The Int8 Firmware Bug (Why it Was Classifying Brightness)

### Symptoms

After flashing the int8 TinyML model, predictions alternated between
guava and powerbank in sync with scene brightness, not with what object
was in frame:

```
Frame 50: powerbank 99.6%   <- looking at powerbank
Frame 51: guava 100.0%      <- pointing at ceiling
Frame 52: powerbank 99.6%   <- back to powerbank
Frame 53: guava 100.0%      <- ceiling again
```

### Root cause

`fill_input_tensor()` was writing **float32** values into an **int8**
tensor. The model is exported with `inference_input_type = tf.int8`, so
`input->type == kTfLiteInt8`. The old code wrote to `input->data.f` (the
float32 union member) with values divided by 255.

What the model actually received was the byte pattern of a 32-bit IEEE 754
float, reinterpreted as four int8 values. For example, `0.5f` =
`0x3F000000` → four bytes `[0x3F, 0x00, 0x00, 0x00]` = int8 `[63, 0, 0, 0]`.

The model still produced confident binary outputs because it had learned to
extract some consistent signal from this corrupted data — specifically, scene
brightness, which partially survives the corruption because bright pixels
produce slightly different float byte patterns than dark pixels. Hence the
ceiling-vs-desk prediction.

### Fix

New function `fill_input_tensor_int8()` writes to `input->data.int8` and
converts each uint8 RGB value to int8 by subtracting 128:

```cpp
int8_val = (int8_t)(uint8_rgb_pixel - 128);
```

This is the correct mapping: the quantization zero_point is -128, meaning
uint8=0 maps to int8=-128 and uint8=255 maps to int8=127.

**Do not divide by 255** — the `layers.Rescaling(1/255)` layer inside the
model is baked into the int8 weights during quantization. Applying it
again would compress the full [-128, 127] input range into approximately
[-0.5, 0.5], which looks like all-grey images to the model.

### Additional bug: calibration pointing at wrong directory

`convert_to_tflite_int8()` in v2 pointed the representative dataset
generator at `DATASET_DIR` (the top-level folder). This directory contained:
```
guava/
powerbank/
test_guava/
test_powerbank/
```
Keras's `image_dataset_from_directory` detected **four** subfolders and
treated them as four classes. The calibration distribution was wrong,
skewing the quantization zero_point and scale parameters. Fixed by
calibrating directly on the list of training file paths.

---

## Phase 13 — Working: 184ms Inference

### Build

The TinyML model is embedded into firmware flash using `xxd`:

```bash
xxd -i model.tflite > firmware/main/model_data.cc
```

`model_data.h` declares the array. The firmware reads directly from flash
— no SD card, no PSRAM model buffer, no file I/O at boot.

### Compilation error: `int32_t` format specifier

```
error: format '%d' expects argument of type 'int',
       but argument has type 'int32_t' {aka 'long int'} [-Werror=format]
```

ESP-IDF's toolchain defines `int32_t` as `long int`. The `%d` specifier
expects plain `int`. Anywhere `dims->data[N]`, `params.zero_point`, or
`->type` is passed to `ESP_LOGI`, it must be cast to `(int)`:

```cpp
// Wrong:
ESP_LOGI(TAG, "dims=[%d]", input->dims->data[0]);
// Correct:
ESP_LOGI(TAG, "dims=[%d]", (int)(input->dims->data[0]));
```

This only affects log lines — the inference logic is unaffected.

### Final result

```
I (xxxx) TINYML: AllocateTensors OK — arena used 48 KB of 256 KB
I (xxxx) TINYML: --- Frame 9 ---
I (xxxx) TINYML:   guava     : 0.0625  (6.2%)
I (xxxx) TINYML:   powerbank : 0.9375  (93.8%)
I (xxxx) TINYML:   PREDICTION : powerbank  (93.8%)
I (xxxx) TINYML:   TIMING     : pre=6.8ms  invoke=184.6ms  total=191.6ms
```

**184ms per frame. ~50 KB model. No SD card. No WiFi. No server.**

The model correctly classifies guava and powerbank at >90% confidence on
clean samples. It does not classify background brightness.

---

## Summary of All Bugs and Fixes

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | Phone vs powerbank too similar | Model predicts one class always | Changed classes to guava vs powerbank |
| 2 | `class_weight` silently ignored with `.map()` | Imbalance not corrected | Removed class_weight; relied on visual difference |
| 3 | BatchNorm + augmentation pipeline | Loss spikes to 30+ on epoch 1, model collapses | Removed BatchNorm, used L2 regularisation |
| 4 | Background bias in training data | Ceiling = guava, dark desk = powerbank | Varied backgrounds in all training images |
| 5 | OV3660 doesn't support `PIXFORMAT_RGB888` | `esp_camera_init` fails: ESP_FAIL | Capture YUV422, convert to RGB in firmware |
| 6 | Firmware writing float32 into int8 tensor | Predicts by scene brightness | Write to `input->data.int8`; apply `uint8 - 128` not `/255` |
| 7 | Calibration on wrong directory (4 classes) | int8 quantization params skewed | Calibrate on training file paths only |
| 8 | `int32_t` passed to `%d` format specifier | Build fails: `-Werror=format` | Cast `dims->data[]` and `params.zero_point` to `(int)` |
| 9 | Train/val split leaks orig/clean pairs | Optimistic val accuracy | Group-aware split on base filename key |
| 10 | `jpg2rgb888` unavailable → switched to `jpg2rgb565` without updating downstream code | Wrong buffer size + raw byte copy of packed uint16_t → garbage tensor | Allocate `W*H*2` bytes; unpack uint16_t with bit shifts before int8 cast |
| 11 | `CAMERA_GRAB_LATEST` with slow consumer | Constant `cam_hal: FB-OVF` spam, frames never consumed | Changed to `CAMERA_GRAB_WHEN_EMPTY` |

---

## Key Lessons

**On data:**
- Vary backgrounds in every training image. If both classes were filmed
  against the same surface, the model learns the surface.
- Group-aware splitting matters when the dataset contains multiple versions
  of the same physical photo. All versions of one scene must stay in one split.
- Perceptual hashing (phash) is the right tool for near-duplicate removal —
  it catches compressed/slightly-varied re-captures of the same frame.

**On training:**
- `class_weight` does not work with `.map()` augmentation pipelines.
- `layers.Rescaling(1/255)` inside the model is the correct pattern for
  int8 quantization — the normalisation is encoded in the weights, so
  raw uint8 values should be fed at inference, not pre-normalised floats.
- Loss = 0.693 at epoch 1 is normal for binary cross-entropy. It only
  indicates collapse if it does not decrease.

**On hardware:**
- The OV3660 only supports JPEG, YUV422, and GRAYSCALE. RGB888 is not
  available through the esp32-camera driver for this sensor.
- `int32_t` on the ESP32 toolchain is `long int`. Use `(int)` casts for
  `%d` format arguments, or use `PRId32` macros.
- Writing float32 bytes into an int8 tensor pointer produces confident but
  wrong outputs — not a crash, not NaN, just garbage predictions. Always
  check `input->type` at startup.

**On architecture:**
- Scratch CNN (~50 KB int8) at 184ms/frame beats MobileNetV2 (~3 MB) at
  5700ms/frame for visually distinct two-class problems. Use the simplest
  model that works.
- Probe your hardware incrementally. Model-only probe first, then camera
  coexistence probe, then full inference. Each probe catches one class of
  bug at a time.

**On streaming and camera DMA:**
- `CAMERA_GRAB_LATEST` causes FB-OVF whenever the consumer is slower than
  the DMA. Always use `CAMERA_GRAB_WHEN_EMPTY` unless you specifically need
  the absolute latest frame and can guarantee the consumer keeps up.
- When changing JPEG decode functions (e.g. `jpg2rgb888` → `jpg2rgb565`),
  the pixel format changes from 3 bytes/px to 2 bytes/px packed uint16_t.
  Every downstream step — buffer size, conversion loop, tensor fill — must
  be updated together. Changing one without the others corrupts the tensor
  silently.

**On deployment robustness:**
- A binary classifier is closed-world: it will confidently classify anything
  as one of two classes. This is wrong for deployment. Add rejection
  thresholds based on confidence and margin before using output for anything
  real.
- Per-frame predictions fluctuate at decision boundaries. Always stabilize
  output before driving actuators. For motor control specifically, use
  hysteresis — it prevents rapid alternation that causes physical wear.

---

## File Reference

| File | Purpose |
|---|---|
| `sanitizer_dataset.py` | Hand removal (HSV + inpainting) + duplicate filtering (phash) |
| `train_classifier_v3.py` | Training script v3 — group-aware split, fixed calibration |
| `train_classifier_v4.py` | Training script v4 — stronger augmentation targeting partial-view/distance |
| `test_laptop.py` | Webcam validation script |
| `model.keras` | Best training weights (Keras format) |
| `model.tflite` | Deployed model — int8, ~50 KB |
| `model_history.png` | Accuracy/loss curves from latest training run |
| `firmware/main/model_data.h` | Declares `model_tflite[]` and `model_tflite_len` |
| `firmware/main/model_data.cc` | Generated by `xxd` — contains model bytes |
| `firmware/main/main_tinyml_probe.cc` | Serial-only inference — int8, no WiFi |
| `firmware/main/main_wifi_stream.cc` | WiFi + MJPEG stream + inference — browser UI |
| `firmware/main/main_edgecv_v2.cc` | Phase 2 — profiling + rejection + stabilization |
| `firmware/main/main_model_probe.cc` | Diagnostic: model load + AllocateTensors |
| `firmware/main/main_camera_probe.cc` | Diagnostic: camera + model coexistence |

---

## Current Status

**Phase 2 complete — production-quality embedded classifier.**

- Model: TinyML scratch CNN (v4 augmentation), int8, ~50 KB, embedded in flash
- Camera: OV3660, JPEG capture, decoded to RGB565 in firmware
- Inference: ~184ms per frame (~5 FPS) on 96×96 input
- WiFi stream: MJPEG live feed served to browser, classification label updates via JSON poll
- Open-world rejection: combined threshold + margin, rejects unknowns as UNKNOWN
- Temporal stabilization: confidence averaging over N frames (selectable strategy)
- Per-stage profiling: capture / decode / convert / invoke / postproc individually timed
- No SD card required

## Next Steps

- [ ] Measure actual timing profile on hardware and confirm bottleneck breakdown
- [ ] Retrain at 64×64 input to target ~10 FPS (~65ms Invoke)
- [ ] Calibrate rejection thresholds against real unknown objects
- [ ] Wire classification output to rover motor control (use STABILITY_HYSTERESIS)
- [ ] OTA model updates

---

## Phase 14 — WiFi + MJPEG Live Stream

### What was built

`main_wifi_stream.cc` — runs the TinyML classifier and simultaneously
streams a live MJPEG feed to a browser. Architecture:

- **Core 0** — WiFi + `esp_http_server`:
  - `GET /` → single-page HTML UI with `<img src=/stream>` and a
    classification label panel that polls `/status` every 500ms
  - `GET /stream` → MJPEG multipart stream
  - `GET /status` → JSON `{class, confidence, invoke_ms}`

- **Core 1** — `camera_task` (sole camera owner):
  - Grabs JPEG frames continuously
  - Copies each frame into a mutex-protected stream buffer
  - Every `INFER_EVERY_N_FRAMES` frames: decodes JPEG → RGB → int8
    tensor → `Invoke()` → updates result buffer

**Why JPEG for both stream and inference:**
Switching between `PIXFORMAT_JPEG` (streaming) and `PIXFORMAT_YUV422`
(inference) requires `deinit` + `reinit` (~100ms). At 184ms Invoke() that
stalls the stream visibly every inference cycle. Instead, JPEG frames are
decoded to RGB on-device using `jpg2rgb888` (from `espressif/esp32-camera`).
Decode adds ~20–30ms per inference frame — negligible against Invoke().

### Bug 1 — jpg2rgb888 not found, changed to jpg2rgb565

`jpg2rgb888` was not available by name in the version of `img_converters.h`
on this build. The compiler suggested `jpg2rgb565` as an alternative, and
it was accepted. This created two silent downstream bugs.

### Bug 2 — wrong decode buffer size

`jpg2rgb565` outputs **2 bytes per pixel** (packed RGB565), not 3. The
decode buffer was allocated as `INPUT_W * INPUT_H * INPUT_C` = 27,648 bytes
(3 bytes/pixel). Writing 18,432 bytes of RGB565 into a 27,648-byte buffer
was safe by accident, but the conversion loop still iterated over 27,648
bytes of which the last 9,216 were uninitialised memory. Fixed by allocating
`INPUT_W * INPUT_H * 2` bytes.

### Bug 3 — wrong conversion loop for RGB565

The original conversion loop treated the decode buffer as flat `uint8`
bytes and subtracted 128 from each. RGB565 packs R (5 bits), G (6 bits),
B (5 bits) into a single `uint16_t`. Reading it as bytes gives two garbage
values per pixel, not three channel values. The model received completely
wrong channel data.

Fix — unpack each `uint16_t` pixel into R8/G8/B8 before the int8 cast:

```cpp
const uint16_t* src = (const uint16_t*)s_rgb_buf;
int8_t*         dst = input->data.int8;

for (int i = 0; i < px; i++) {
    uint16_t p = src[i];
    uint8_t r = (uint8_t)(((p >> 11) & 0x1F) << 3);  // 5-bit -> 8-bit
    uint8_t g = (uint8_t)(((p >>  5) & 0x3F) << 2);  // 6-bit -> 8-bit
    uint8_t b = (uint8_t)(((p      ) & 0x1F) << 3);  // 5-bit -> 8-bit
    dst[i * 3 + 0] = (int8_t)((int)r - 128);
    dst[i * 3 + 1] = (int8_t)((int)g - 128);
    dst[i * 3 + 2] = (int8_t)((int)b - 128);
}
```

### Bug 4 — FB-OVF (Frame Buffer Overflow)

```
cam_hal: FB-OVF
cam_hal: FB-OVF
cam_hal: FB-OVF
...
```

The camera produced constant frame buffer overflows. The DMA was filling
buffers faster than the task was consuming and returning them.

**Root cause:** `CAMERA_GRAB_LATEST` was set in the camera config.

In this mode the DMA continuously overwrites buffers with the newest frame
regardless of whether the application has returned the previous one. With
`fb_count=2` and the task busy with WiFi I/O and JPEG decode on the same
core, the two DMA slots constantly raced ahead of consumption.

Some FB-OVF before `camera_task` started was expected — the camera runs
during WiFi init and nothing is consuming frames yet. But the overflow
continued after the task was running because `CAMERA_GRAB_LATEST` kept
overwriting even when buffers were held.

**Fix:** Changed to `CAMERA_GRAB_WHEN_EMPTY`.

```cpp
// Wrong — DMA overwrites regardless of buffer state
cfg.grab_mode = CAMERA_GRAB_LATEST;

// Correct — DMA only captures when a buffer slot is free
cfg.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
```

This adds backpressure: the DMA waits for a slot to be returned before
capturing the next frame. The stream latency increases slightly (you get
the frame from when the buffer was free, not the absolute latest), but at
96×96 for an inference feed this is imperceptible.

### Result

WiFi stream + inference working simultaneously:
- Stream served to browser at ~5–8fps
- Inference running asynchronously on Core 1 every N frames
- Classification label updates in browser every ~500ms via JS poll
- No FB-OVF after grab_mode fix

---

## Phase 15 — EdgeCV Phase 2: Profiling, Open-World Rejection, Temporal Stabilization

### Context

With basic inference and streaming working, the next objective was to make
the system production-quality: measure actual timing, handle unknown objects
gracefully, and produce stable output rather than per-frame noise.

### Phase 15A — Per-Stage Profiling

`main_edgecv_v2.cc` instruments every pipeline stage individually using
`esp_timer_get_time()` and accumulates averages over `PROFILE_EVERY_N_FRAMES`
frames. Stages measured:

| Stage | What it measures |
|---|---|
| `capture_ms` | `esp_camera_fb_get()` — DMA transfer time |
| `decode_ms` | `jpg2rgb565/jpg2rgb888` — software JPEG decode |
| `convert_ms` | uint8→int8 channel conversion loop |
| `invoke_ms` | `interpreter.Invoke()` — neural network forward pass |
| `postproc_ms` | dequantize + rejection + stabilization |

The profiler report prints automatically every N frames:

```
=== TIMING PROFILE (avg over 10 frames) ===
  capture  :  18.40 ms  ( 9.5%)
  decode   :  24.80 ms  (12.8%)
  convert  :   1.60 ms  ( 0.8%)
  invoke   : 138.20 ms  (71.4%)
  postproc :   0.30 ms  ( 0.2%)
  TOTAL    : 183.30 ms  ->  5.5 FPS
```

**Conclusion:** Invoke() dominates at ~71%. Decode is 13%. Everything else
is noise. The path to faster inference is a smaller model or smaller input,
not pipeline optimisation.

**On 64×64 resolution:** Halving the input from 96×96 to 64×64 reduces
pixel count by 56% (`64²/96²`). Conv2D time scales roughly with pixel
count, so Invoke() is expected to drop from ~135ms to ~65ms, giving total
~95ms (~10 FPS). Requires retraining at 64×64. Not done yet — logged as
a future step.

### Phase 15B — Open-World Rejection

The closed-world problem: everything is forced to be guava or powerbank,
even a keyboard, face, or blank wall. Three rejection strategies implemented,
all selectable at compile time via `REJECTION_MODE`:

**Threshold (`REJECT_MODE_THRESHOLD`):**
```
if max(guava_prob, powerbank_prob) < CONFIDENCE_THRESHOLD: UNKNOWN
```
Simple. Works when known-class confidence is reliably >0.8.
Recommended starting point. `CONFIDENCE_THRESHOLD = 0.75`.

**Margin (`REJECT_MODE_MARGIN`):**
```
if |guava_prob - powerbank_prob| < MARGIN_THRESHOLD: UNKNOWN
```
Catches split predictions where both scores are moderate (e.g. wall gives
guava=0.55, powerbank=0.45 → margin=0.10 → UNKNOWN).
`MARGIN_THRESHOLD = 0.40`.

**Combined (`REJECT_MODE_COMBINED`, default):**
Both threshold AND margin must be satisfied. Most conservative — correct
prediction requires both high confidence and a clear margin.

**On entropy-based rejection:** For a 2-class sigmoid model, entropy is
`-p·log(p) - (1-p)·log(1-p)`, which is a monotonic function of `|p - 0.5|`.
This is mathematically equivalent to the margin approach and adds no new
information for binary classification. Worth revisiting for 3+ classes.

**Calibrating thresholds:**
Point the camera at known objects (guava, powerbank) and at unknowns
(keyboard, wall, hand). Read the printed raw probabilities. Set
`CONFIDENCE_THRESHOLD` so known objects consistently clear it and unknowns
don't. Start at 0.75 and adjust based on your specific deployment environment.

### Phase 15C — Temporal Stabilization

Per-frame predictions fluctuate — the model may alternate between guava
and UNKNOWN at a decision boundary even when holding the object still. Four
stabilization strategies implemented via `STABILITY_MODE`:

**Majority vote (`STABILITY_MAJORITY_VOTE`):**
Ring buffer of N raw predictions. Output = most common label.
Fast to respond to genuine changes. Sensitive to single-frame flips.

**Confidence averaging (`STABILITY_CONF_AVG`, recommended):**
Averages the raw probability score over N frames. Applies rejection to the
average. Smoother than vote, still responds quickly to real transitions.
Best for general deployment.

**Exponential moving average (`STABILITY_EMA`):**
`ema = α·new + (1-α)·ema`. Lower α = more smoothing, slower response.
Good when transitions are rare and smooth rather than abrupt.
`EMA_ALPHA = 0.30` — new frame contributes 30%.

**Hysteresis (`STABILITY_HYSTERESIS`):**
Once a class is confirmed (confidence > `HYSTERESIS_HIGH`), it holds until
confidence drops below `HYSTERESIS_LOW`. Prevents rapid alternation at
decision boundaries. Best when the output drives physical actuators (motors)
where flapping between states is mechanically harmful.

**Choosing a strategy:**
- General use, serial output → `STABILITY_CONF_AVG`
- Motor/actuator control → `STABILITY_HYSTERESIS`
- Low-latency requirement → `STABILITY_MAJORITY_VOTE` with small window
- Continuous probabilistic stream → `STABILITY_EMA`


---

## Phase 16 — Raw TCP Stream + Serial Inference (main_stream_raw.cc)

### Goal

A lightweight alternative to `main_wifi_stream.cc`: stream camera frames
to a laptop Python script for display, while running inference independently
on the ESP32 and printing predictions to the serial monitor. No browser, no
HTTP server, no MJPEG overhead.

The laptop (`live_infer.py`) only receives and displays frames — it does
no inference, has no TensorFlow dependency, and needs only `opencv-python`
and `numpy`.

### Architecture

```
ESP32 Core 1 — stream_task:
  loop:
    grab YUV422 frame
    every INFER_EVERY_N_FRAMES:
      yuv422_to_int8_tensor() -> Invoke() -> print to serial
    frame2jpg() -> [4-byte LE length][JPEG bytes] -> TCP send

Laptop — live_infer.py:
  recv 4-byte length -> recv JPEG bytes -> cv2.imdecode -> cv2.imshow
```

**Why YUV422 capture (not JPEG):**
Capturing JPEG directly from the OV3660 at 96×96 produces `cam_hal: NO-SOI`
errors — the sensor's JPEG pipeline doesn't produce valid output at that
resolution. YUV422 works reliably. `frame2jpg()` encodes the YUV frame to
JPEG on-device for transmission, adding ~5ms per frame.

**Why `frame2jpg()` and not native JPEG capture:**
Native JPEG capture at larger resolutions works but the frame is then too
large to decode on-device for inference without an extra JPEG decode step.
YUV422 at 96×96 feeds directly into the BT.601 conversion that inference
already uses, with no intermediate decode needed.

### Bug — JPEG capture at 96×96 produces NO-SOI

When the camera was configured for `PIXFORMAT_JPEG` at `FRAMESIZE_96X96`,
every captured frame triggered `cam_hal: NO-SOI` — the JPEG start-of-image
marker `0xFF 0xD8` was missing. The sensor outputs no valid JPEG data at
that resolution.

Attempted workaround: switch to `FRAMESIZE_QVGA` (320×240). The SO-I error
persisted even at QVGA, compounded by the stream task consuming frames too
slowly.

**Root cause:** The OV3660 JPEG pipeline is unreliable at small resolutions
and requires warm-up frames. Rather than fight the sensor in JPEG mode,
the approach was changed to YUV422 capture + software JPEG encode via
`frame2jpg()`, which is the same path that already works in `main_tinyml_probe.cc`.

### Bug — FB-OVF on JPEG capture mode

With `CAMERA_GRAB_LATEST` and `fb_count=2`, constant frame buffer overflows
appeared while the stream task was busy encoding/sending. Fixed by changing
to `CAMERA_GRAB_WHEN_EMPTY` (same fix applied earlier in Phase 14). This
was not an issue in YUV422 mode since the stream task consumes frames fast
enough.

### Bug — mDNS component not found

```
HINT: The component mdns could not be found.
```

`mdns` is not built into ESP-IDF v5.3 by default — it was moved to the
component manager. Fixed by running:

```bash
idf.py add-dependency "espressif/mdns"
```

This adds the entry to `idf_component.yml` and pulls the component on next
build. The same fix was needed in a previous project (tinyguard-monitor).

### Bug — mDNS resolution failing on Windows

```
socket.gaierror: [Errno 11001] getaddrinfo failed
```

`socket.getaddrinfo()` with explicit `AF_INET`/`SOCK_STREAM` flags bypasses
the Windows mDNS resolver even when Bonjour is installed. Fixed by switching
to `socket.gethostbyname()` in `live_infer.py`, which goes through the
normal Windows name resolution stack that Bonjour hooks into.

### Bug — `IPSTR`/`IP2STR` macro type mismatch

```
error: format '%d' expects int but argument has type 'int32_t'
```

`IP2STR()` expects a pointer to `esp_ip4_addr_t` (which has a `.addr` field),
but `client_addr.sin_addr` is a POSIX `struct in_addr` (which has `.s_addr`).
Fixed by replacing the `IPSTR`/`IP2STR` macro with `inet_ntoa()`:

```cpp
// Before (wrong type)
ESP_LOGI(TAG, "Client from " IPSTR, IP2STR(&client_addr.sin_addr));

// After (correct)
ESP_LOGI(TAG, "Client from %s", inet_ntoa(client_addr.sin_addr));
```

### CMakeLists REQUIRES cleanup

The `REQUIRES` list in `main/CMakeLists.txt` was overpopulated with
components pulled from previous entry points. `esp_http_server` is not
used by `main_stream_raw.cc` (it uses a raw TCP socket, not HTTP). `heap`
and `spi_flash` are implicit transitive dependencies in ESP-IDF v5.3 and
don't need explicit declaration. Cleaned up to only what's actually used:
`esp_psram`, `esp_timer`, `esp_camera`, `esp_wifi`, `esp_event`,
`esp_netif`, `nvs_flash`, `mdns`, `driver`, `log`.

### Result

- `main_stream_raw.cc` flashed and running
- Live camera feed displayed in OpenCV window via `live_infer.py`
- Inference predictions printing to serial every 3 frames:

```
I (xxxx) RAW_STREAM: --- Frame 3 ---
I (xxxx) RAW_STREAM:   guava     : 0.0625  (6.2%)
I (xxxx) RAW_STREAM:   powerbank : 0.9375  (93.8%)
I (xxxx) RAW_STREAM:   PREDICTION : powerbank  (93.8%)
I (xxxx) RAW_STREAM:   invoke    : 184.3 ms
```

---

## Phase 17 — BUILD_MODE CMake Flag (replaces APP_MAIN env var)

### Problem

The original entry-point selection used an environment variable:

```bash
APP_MAIN=main_stream_raw.cc idf.py build flash monitor
```

This works on Linux/macOS but breaks silently in two ways:
- On Windows (native cmd/PowerShell), env-var prefix syntax is not supported.
  The build falls through to the default without any error or warning.
- The env var is set per-shell and easy to forget — a stale terminal with
  `APP_MAIN` set will silently build the wrong entry point.

CMake cache variables (`-D`) are the standard idf.py mechanism for
parameterised builds, are platform-independent, and are visible in the
CMake log output on every build.

### Change

Replaced `APP_MAIN` env-var lookup with a `BUILD_MODE` CMake cache variable
in `main/CMakeLists.txt`. Friendly mode names map to source files:

| `BUILD_MODE` value | Source file compiled |
|---|---|
| `probe` (default) | `main_tinyml_probe.cc` |
| `stream` | `main_stream_raw.cc` |
| `wifi` | `main_wifi_stream.cc` |
| `edgecv_v2` | `main_edgecv_v2.cc` |
| `model_probe` | `main_model_probe.cc` |
| `camera_probe` | `main_camera_probe.cc` |

Raw filenames still work as values for forward-compatibility with any
existing scripts. The legacy `APP_MAIN` env-var is still honoured as a
fallback when `BUILD_MODE` is not set, so existing workflows are not broken.

### Usage

```bash
# default — serial-only inference
idf.py build flash monitor

# TCP stream to laptop
idf.py -DBUILD_MODE=stream build flash monitor

# browser MJPEG stream
idf.py -DBUILD_MODE=wifi build flash monitor
```

### Why not run both probe + stream simultaneously

The entry points are separate `app_main()` implementations — each one owns
the full execution context (camera init, model load, task creation). There
is no safe way to run two of them in the same firmware image: they would
both call `esp_camera_init()`, both allocate the tensor arena, and both
fight over PSRAM. The flag selects one entry point per build, which is the
correct architecture for embedded systems.

### build.sh — wrapper script

Typing `idf.py -DBUILD_MODE=stream -p /dev/ttyUSB0 build flash` then a
separate `picocom` command every session is friction. Added `build.sh` to
`classifier/firmware/` to wrap the full workflow in one command:

```bash
./build.sh [port_number] [mode]

./build.sh              # probe, /dev/ttyUSB0
./build.sh 1            # probe, /dev/ttyUSB1
./build.sh 0 stream     # stream, /dev/ttyUSB0
./build.sh 0 wifi       # wifi,   /dev/ttyUSB0
```

The script runs `idf.py build flash`, exits immediately if the build fails
(so picocom never opens on a failed flash), then launches picocom with
`--lower-rts --lower-dtr` automatically. Port argument is just the USB
number — `0` expands to `/dev/ttyUSB0` inside the script.
