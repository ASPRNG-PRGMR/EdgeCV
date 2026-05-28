# EdgeCV

Embedded edge vision framework — real-time computer vision inference on an ESP32 camera node.

EdgeCV explores moving lightweight CV workloads off centralised servers and onto low-cost embedded hardware. The current prototype runs an MNIST handwritten-digit classifier entirely on an ESP32 with an OV3660 camera, with no server involved.

---

## Repository Structure

```
EdgeCV/
├── README.md
├── src/
|   ├── cam_sanity.py
|   ├── mnist_cnn_camera.py
├── models/
|    └── mnist_cnn_int8.tmdl  # Trained INT8 CNN weights (emlearn format)
└── firmware
     ├── esp-dev.bin # micropython firmware for esp32 dev module
     └── esp-cam.bin # micropython firmware for esp32 cam module 
```

---

## Hardware

| Component             | Notes                           |
|-----------------------|---------------------------------|
| ESP32 Dev Board       | Main MCU                        |
| OV3660 Camera Module  | Attached via parallel interface |
| USB–UART adapter      | Flashing and serial monitor     |
| Host PC (Linux / WSL) | Tooling, `esptool`, `picocom`   |

---

## Firmware Binaries

This repository includes tested MicroPython firmware binaries for convenience and reproducibility.

Original firmware source:

* https://micropython.org/download/

Included binaries:

| Board            | Firmware                             |
| ---------------- | ------------------------------------ |
| ESP32 Dev Module | `ESP32_GENERIC-20250415-v1.25.0.bin` |
| ESP32-CAM        | `firmware.bin`                       |

These binaries are included to preserve a known-working environment for the current prototype.


---

## Software Dependencies

### On the host

```bash
pip install esptool
sudo apt install picocom        # or: pip install pyserial
```

### MicroPython firmware

Download the latest ESP32 MicroPython firmware from https://micropython.org/download/ESP32_GENERIC/

### On the ESP32 (MicroPython modules)

- [`emlearn_cnn_int8`](https://github.com/emlearn/emlearn-micropython) — compiled `.mpy` from emlearn-micropython (follow their build / install instructions)
- `camera` — MicroPython camera driver for ESP32 (e.g. `esp32-camera` port)

---

## Inference Pipeline

```
OV3660 capture (96×96 greyscale)
        │
        ▼
Average-pool → 32×32        downscale.py
        │
        ▼
Centre-crop → 28×28         crop32to28()
        │
        ▼
Binary threshold             white digit on black background
        │
        ▼
INT8 CNN inference           emlearn_cnn_int8
        │
        ▼
Predicted digit + latency   printed to serial
```

---

## Flashing Guide (Linux / WSL)

> All commands below assume `/dev/ttyUSB0`.  
> Run `ls /dev/ttyUSB*` or `ls /dev/ttyACM*` to find your actual port.

### 1 — Erase flash

```bash
esptool.py --port /dev/ttyUSB0 --baud 460800 erase_flash
```

### 2 — Flash MicroPython firmware

```bash
esptool.py \
    --chip esp32 \
    --port /dev/ttyUSB0 \
    --baud 460800 \
    write_flash \
    -z 0x1000 \
    <firmware-name>.bin
```

Replace `ESP32_GENERIC-<version>.bin` with the actual filename you downloaded.

### 3 — Upload project files

Use `mpremote` (comes with MicroPython tools) or `ampy`:

```bash
# Install mpremote if needed
pip install mpremote

# Upload files one by one
mpremote connect /dev/ttyUSB0 cp cam_sanity.py :cam_sanity.py
mpremote connect /dev/ttyUSB0 cp downscale.py :downscale.py
mpremote connect /dev/ttyUSB0 cp mnist_cnn_camera.py :mnist_cnn_camera.py
mpremote connect /dev/ttyUSB0 cp models/mnist_cnn_int8.tmdl :mnist_cnn_int8.tmdl

# Also upload the emlearn_cnn_int8.mpy you built from emlearn-micropython
mpremote connect /dev/ttyUSB0 cp emlearn_cnn_int8.mpy :emlearn_cnn_int8.mpy
```

---

## Serial Monitor (picocom)

The ESP32-CAM's USB-UART chip often keeps RTS/DTR asserted, which holds the ESP32 in reset.  
Pass `--lower-rts --lower-dtr` to release those lines before opening the port:

```bash
picocom \
    --baud 115200 \
    --lower-rts \
    --lower-dtr \
    /dev/ttyUSB0
```

Exit picocom with **Ctrl-A then Ctrl-X**.

> **WSL note:** WSL 2 does not expose USB devices by default.  
> Use [usbipd-win](https://github.com/dorssel/usbipd-win) to attach the USB-UART adapter to WSL:
> ```powershell
> # In a Windows PowerShell (admin)
> usbipd list
> usbipd attach --wsl --busid <BUSID>
> ```
> Then the device will appear as `/dev/ttyUSB0` inside WSL.

---

## Running

### Step 1 — Camera sanity check

```bash
# In picocom or mpremote REPL
import cam_sanity
```

Expected output:
```
OK: captured 19200 bytes
```

If you see `ERROR: capture returned None`, double-check pin mapping and camera power.

### Step 2 — Main inference loop

```bash
# In picocom REPL
import mnist_cnn_camera
```

Hold a piece of paper with a handwritten digit in front of the camera.  
Expected output:
```
camera: ok
model: ready
digit: 3  time: 214 ms
digit: 3  time: 211 ms
...
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `capture returned None` | Bad pin map or no power | Check wiring; confirm `xclk_freq` |
| Board stuck in reset | RTS/DTR held high | Use `--lower-rts --lower-dtr` in picocom |
| `ImportError: emlearn_cnn_int8` | Module not uploaded | Build and copy `.mpy` from emlearn-micropython |
| Very low accuracy | Lighting / background | Ensure dark background, good contrast, even lighting |
| `MemoryError` | Heap fragmentation | Call `gc.collect()` more frequently; reduce `fb_count` |

---

## Motivation

Most embedded camera deployments stream raw frames to a server for processing — high bandwidth, high latency, poor scalability.  
EdgeCV investigates the opposite: run inference *on the sensor*, transmit only the result.

Current status: **active prototype** — MNIST digit recognition working on hardware.

---

## Roadmap

- [ ] Improved preprocessing (adaptive threshold, contrast normalisation)
- [ ] Serial / WebSocket output visualiser
- [ ] TinyML model optimisation (pruning, quantisation)
- [ ] Real-time object detection (beyond MNIST)
- [ ] Multi-node distributed camera system
- [ ] OTA model updates
- [ ] Latency benchmarks vs cloud inference

---

## References

- [emlearn-micropython](https://github.com/emlearn/emlearn-micropython) — TinyML inference for MicroPython
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [MicroPython ESP32](https://micropython.org/download/ESP32_GENERIC/)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)

---

## License

MIT
