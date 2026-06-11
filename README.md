# EdgeCV

Embedded edge vision framework — real-time computer vision inference on an ESP32 camera node.

EdgeCV explores moving lightweight CV workloads off centralised servers and onto low-cost embedded hardware. The current prototype runs inference entirely on an ESP32-CAM with an OV3660 camera — no server involved, no cloud dependency.

Two demos are included:

- **MNIST digit recogniser** — the original prototype; classifies handwritten digits from the camera feed using an INT8 CNN and the emlearn-micropython runtime
- **Custom object classifier** — the active project; a full training pipeline for any two-class object recogniser, currently trained to distinguish guava vs powerbank. Includes a laptop webcam tester for rapid iteration before flashing

---

## Repository Structure

```
EdgeCV/
├── README.md
├── training_devlog.md              # Full development log: MNIST bring-up + custom classifier
│
├── mnistDigits/                    # MNIST digit recogniser — self-contained
│   ├── src/
│   │   ├── cam_sanity.py           # Camera initialisation sanity check
│   │   ├── downscale.py            # Average-pool helper (96×96 → 32×32)
│   │   └── mnist_cnn_camera.py     # Main inference loop (runs on ESP32)
│   ├── models/
│   │   └── mnist_cnn_int8.tmdl     # Trained INT8 CNN weights (emlearn format)
│   └── firmware/
│       ├── esp-dev.bin             # MicroPython firmware for ESP32 Dev Module
│       └── esp-cam.bin             # MicroPython firmware for ESP32-CAM
│
└── classifier/                     # Custom two-class object classifier
    ├── train_classifier.py         # Training script — scratch CNN or MobileNetV2 transfer
    ├── test_laptop.py              # Webcam tester — validate model before flashing
    ├── training_history.png        # Accuracy/loss curves from latest training run
    └── model.tflite                # Custom classifier — TFLite float32 or int8
```

---

## Hardware

| Component             | Notes                           |
|-----------------------|---------------------------------|
| ESP32-CAM             | Main MCU + camera board         |
| OV3660 Camera Module  | Attached via parallel interface |
| USB–UART adapter      | Flashing and serial monitor     |
| Host PC (Linux / WSL) | Tooling, `esptool`, `picocom`   |

---

## Firmware Binaries

Tested MicroPython firmware binaries are kept under `mnistDigits/firmware/` for convenience and reproducibility.

Original firmware source: https://micropython.org/download/

| Board            | Firmware file                   |
|------------------|---------------------------------|
| ESP32 Dev Module | `mnistDigits/firmware/esp-dev.bin` |
| ESP32-CAM        | `mnistDigits/firmware/esp-cam.bin` |

---

## Software Dependencies

### On the host (training + testing)

```bash
pip install tensorflow pillow numpy scikit-learn matplotlib
pip install opencv-python    # for test_laptop.py only
pip install esptool mpremote
sudo apt install picocom
```

### On the ESP32 (MicroPython modules — MNIST demo only)

- [`emlearn_cnn_int8`](https://github.com/emlearn/emlearn-micropython) — compiled `.mpy` from emlearn-micropython
- `camera` — MicroPython camera driver for ESP32

The custom classifier runs TFLite directly and does not require emlearn.

---

## Demo 1 — MNIST Digit Recogniser

All source files for this demo live under `mnistDigits/`.

### Inference pipeline

```
OV3660 capture (96×96 greyscale)
        │
        ▼
Average-pool → 32×32            downscale.py
        │
        ▼
Centre-crop → 28×28             crop32to28()
        │
        ▼
Binary threshold                white digit on black background
        │
        ▼
INT8 CNN inference              emlearn_cnn_int8
        │
        ▼
Predicted digit + latency       printed to serial
```

### Running

```bash
# Step 1 — camera sanity check
import cam_sanity
# Expected: OK: captured 19200 bytes

# Step 2 — main inference loop
import mnist_cnn_camera
# Expected:
# camera: ok
# model: ready
# digit: 3  time: 214 ms
```

Hold a piece of paper with a handwritten digit in front of the camera.

### Flashing the MNIST demo

> All commands assume `/dev/ttyUSB0`. Run `ls /dev/ttyUSB*` to find your port.

#### 1 — Erase flash

```bash
esptool.py --port /dev/ttyUSB0 --baud 460800 erase_flash
```

#### 2 — Flash MicroPython firmware

```bash
esptool.py \
    --chip esp32 \
    --port /dev/ttyUSB0 \
    --baud 460800 \
    write_flash \
    -z 0x1000 \
    mnistDigits/firmware/esp-cam.bin
```

#### 3 — Upload project files

```bash
pip install mpremote

mpremote connect /dev/ttyUSB0 cp mnistDigits/src/cam_sanity.py       :cam_sanity.py
mpremote connect /dev/ttyUSB0 cp mnistDigits/src/downscale.py         :downscale.py
mpremote connect /dev/ttyUSB0 cp mnistDigits/src/mnist_cnn_camera.py  :mnist_cnn_camera.py
mpremote connect /dev/ttyUSB0 cp mnistDigits/models/mnist_cnn_int8.tmdl :mnist_cnn_int8.tmdl
mpremote connect /dev/ttyUSB0 cp emlearn_cnn_int8.mpy                 :emlearn_cnn_int8.mpy
```

---

## Demo 2 — Custom Object Classifier

All source files for this demo live under `classifier/`.

### Overview

A from-scratch training pipeline for a two-class image classifier targeting the ESP32-CAM. The current trained model distinguishes **guava** from **powerbank** — chosen because they are visually distinct (round green fruit vs flat dark rectangle), which is the right property to optimise for at 96×96 resolution with a small dataset.

The pipeline went through eight development phases to get here — see `training_devlog.md` for the full story including every bug hit along the way.

### Inference pipeline

```
OV3660 capture (96×96 RGB)
        │
        ▼
Normalise to [0.0, 1.0]
        │
        ▼
TFLite inference (float32 or int8)
        │
        ▼
Softmax → argmax → class label + confidence
        │
        ▼
Printed to serial
```

### Architecture — scratch CNN (default)

```
Input (96, 96, 3) RGB
→ Conv2D(16, 3×3)  → ReLU → MaxPool(2×2)    output: 48×48×16
→ Conv2D(32, 3×3)  → ReLU → MaxPool(2×2)    output: 24×24×32
→ Conv2D(64, 3×3)  → ReLU → MaxPool(2×2)    output: 12×12×64
→ Flatten (9216)
→ Dense(128) → ReLU → Dropout(0.5)
→ Dense(N) → Softmax
```

Quantized to int8, this produces a model of approximately 1.2 MB — fits in SPIFFS with the default ESP32 partition scheme.

### Architecture — transfer learning (--transfer flag)

```
Input (96, 96, 3)
→ preprocess_input              rescale [0,1] → [-1,1] for MobileNetV2
→ MobileNetV2 backbone          frozen (pretrained ImageNet, 154 layers)
→ GlobalAveragePooling2D        (3,3,1280) → (1280,)
→ Dense(128, relu)
→ Dropout(0.3)
→ Dense(N, softmax)
```

Training runs in two phases: head-only first (backbone frozen), then fine-tuning of the top 30 backbone layers at LR=1e-5. Quantized MobileNetV2 comes to approximately 3.5 MB — requires a custom SPIFFS partition or SD card. Use the scratch CNN unless the object classes are genuinely visually similar.

### Training your own classifier

#### 1. Collect images

Organise your dataset into class sub-folders. Folder names determine class labels (alphabetical order = class index):

```
dataset/
    guava/
        img001.jpg
        img002.jpg
        ...
    powerbank/
        img001.jpg
        ...
```

Recommended minimum: 100–150 images per class. Vary backgrounds, distances, orientations, lighting, and whether a hand is in frame. The biggest mistake is using a plain white wall for everything — the model will learn the background, not the object.

#### 2. Train

```bash
# Scratch CNN — fast, good for visually distinct objects
python classifier/train_classifier.py --dataset ./dataset --quantize

# Transfer learning — slower, better for visually similar objects
python classifier/train_classifier.py --dataset ./dataset --transfer --quantize

# All options
python classifier/train_classifier.py \
    --dataset    ./dataset  \   # folder containing class sub-folders
    --epochs     80         \   # max epochs (early stopping may end sooner)
    --batch_size 16         \   # images per gradient update
    --lr         0.0003     \   # learning rate for Adam
    --val_split  0.2        \   # fraction held out for validation
    --output     classifier/models/model.tflite \
    --transfer               \  # use MobileNetV2 pretrained on ImageNet
    --quantize                  # int8 quantization (~4× size reduction)
```

The script outputs per-class accuracy after training (not just overall accuracy), saves the best weights to `best_model.keras`, and writes training curves to `classifier/training_history.png`.

#### 3. Test on your laptop

Before flashing anything, validate the model using your webcam:

```bash
python classifier/test_laptop.py
```

Edit the top of `test_laptop.py` to match your trained classes:

```python
LABELS = ["guava", "powerbank"]   # alphabetical order, matches training folders
```

Controls: `ESC` to quit, `S` to save a debug screenshot, `+`/`-` to adjust the confidence threshold live. The display shows a confidence bar per class, the raw probability scores, and a 96×96 preview of exactly what the model sees.

#### 4. Prepare for ESP32 deployment

Once laptop testing looks good, convert the model to a C header:

```bash
xxd -i classifier/models/model.tflite > model_data.h
```

Flash `model_data.h` to the ESP32 via SPIFFS uploader. The MicroPython inference loop reads the model from flash at startup and runs classification in a loop.

---

## Serial Monitor (picocom)

The ESP32-CAM's USB-UART chip often holds RTS/DTR high, which keeps the ESP32 in reset. Pass `--lower-rts --lower-dtr` to release those lines:

```bash
picocom \
    --baud 115200 \
    --lower-rts \
    --lower-dtr \
    /dev/ttyUSB0
```

Exit with **Ctrl-A then Ctrl-X**.

> **WSL note:** WSL 2 does not expose USB devices by default. Use [usbipd-win](https://github.com/dorssel/usbipd-win) to attach the adapter to WSL:
> ```powershell
> usbipd list
> usbipd attach --wsl --busid <BUSID>
> ```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `capture returned None` | Bad pin map or no power | Check wiring; confirm `xclk_freq` |
| Board stuck in reset | RTS/DTR held high | Use `--lower-rts --lower-dtr` in picocom |
| `ImportError: emlearn_cnn_int8` | Module not uploaded | Build and copy `.mpy` from emlearn-micropython |
| Very low MNIST accuracy | Lighting or background | Dark background, good contrast, even lighting |
| `MemoryError` on ESP32 | Heap fragmentation | Call `gc.collect()` more frequently; reduce `fb_count` |
| Model outputs ~50% for both classes | Class collapse or early training | Check `training_history.png`; see `training_devlog.md` Phase 3 |
| Laptop test misclassifies everything | Background bias in training data | Recollect images with varied backgrounds |
| `model.tflite` too large for SPIFFS | Using MobileNetV2 (~3.5 MB) | Use `--quantize` flag, or use scratch CNN (~1.2 MB) |

---

## Current Status

**Active prototype.** Two demos working:
- MNIST digit recognition: working on hardware - Custom object classifier (guava / powerbank): training pipeline complete, laptop testing passing, ESP32 deployment in progress

---

## Roadmap

- [x] MNIST digit recognition on ESP32-CAM hardware
- [x] Custom training pipeline with class weight handling, augmentation, early stopping
- [x] Transfer learning support (MobileNetV2, two-phase training)
- [x] Laptop webcam tester with live confidence display
- [x] Custom object classifier (guava vs powerbank)
- [ ] Flash custom classifier to ESP32-CAM and verify inference loop
- [ ] Improved preprocessing (adaptive threshold, contrast normalisation)
- [ ] Serial / WebSocket output visualiser
- [ ] Real-time object detection driving rover motors
- [ ] Multi-node distributed camera system
- [ ] OTA model updates
- [ ] Latency benchmarks vs cloud inference

---

## Key Lessons (Short Version)

The full development journey — including the MNIST bring-up and all eight phases of the custom classifier — is documented in `training_devlog.md`. The short version:

- `class_weight=` in `model.fit()` is silently ignored when the dataset uses `.map()` for augmentation. Always bake class weights into the dataset as a third tensor.
- `BatchNormalization` + a weighted `(image, label, weight)` dataset causes loss to spike to 30+ on epoch 1, permanently collapsing the model. Remove BatchNorm or use Layer Normalisation instead.
- Loss = 0.693 means the model is outputting 50% for every class — it's learned nothing. This only indicates collapse when the prediction distribution is also all one class.
- If training images all use the same background, the model will learn the background. Always vary backgrounds to match real usage conditions.
- Choose object classes that differ in shape, colour, and texture. Two dark rectangles of similar size (phone vs powerbank) are a bad choice at 96×96 with a small dataset.
- On the MNIST bring-up: the horizontal flip (`hflip_inplace`) is necessary because the OV3660 outputs a mirrored image. Without it, accuracy on physical digits is noticeably worse.

---

## References

- [emlearn-micropython](https://github.com/emlearn/emlearn-micropython) — TinyML inference for MicroPython
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [MicroPython ESP32](https://micropython.org/download/ESP32_GENERIC/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
