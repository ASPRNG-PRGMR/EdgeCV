"""
mnist_cnn_camera.py
-------------------
Real-time MNIST handwritten-digit classifier running on an ESP32
with an OV3660 camera module.

Based on the emlearn-micropython CNN example:
https://github.com/emlearn/emlearn-micropython

Pipeline
--------
    96×96 greyscale capture
        → average-pool to 32×32   (downscale module)
        → centre-crop to 28×28    (crop32to28)
        → horizontal flip         (mirror each row in-place)
        → binary threshold        (white digit on black background)
        → INT8 CNN inference      (emlearn_cnn_int8)
        → predicted digit + time  (printed to serial)

Dependencies (must be present on the ESP32 filesystem)
-------------------------------------------------------
    emlearn_cnn_int8  – compiled .mpy from emlearn-micropython
    downscale.py      – average-pool helper (this repo)
    mnist_cnn_int8.tmdl – trained model weights
"""

import array
import gc
import time

from camera import Camera, FrameSize, GrabMode, PixelFormat

import downscale
import emlearn_cnn_int8

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH     = "mnist_cnn_int8.tmdl"
CAPTURE_SIZE   = 96
POOL_SIZE      = 32
CROP_SIZE      = 28
CROP_OFFSET    = (POOL_SIZE - CROP_SIZE) // 2   # = 2
THRESHOLD      = 120
LOOP_DELAY_MS  = 50
ERROR_DELAY_MS = 200
NUM_CLASSES    = 10

DATA_PINS = [5, 18, 19, 21, 36, 39, 34, 35]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def argmax(arr) -> int:
    idx_max = 0
    val_max = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > val_max:
            val_max = arr[i]
            idx_max = i
    return idx_max


def crop32to28(src, dst) -> None:
    for r in range(CROP_SIZE):
        for c in range(CROP_SIZE):
            dst[r * CROP_SIZE + c] = src[(r + CROP_OFFSET) * POOL_SIZE + (c + CROP_OFFSET)]


def hflip_inplace(buf, width) -> None:
    """Mirror each row of a flat square buffer horizontally, in-place."""
    for r in range(width):
        row_start = r * width
        row_end   = row_start + width - 1
        for c in range(width // 2):
            l = row_start + c
            r2 = row_end - c
            buf[l], buf[r2] = buf[r2], buf[l]


def threshold_inplace(buf, cutoff: int) -> None:
    for i in range(len(buf)):
        buf[i] = 255 if buf[i] > cutoff else 0


# ---------------------------------------------------------------------------
# Camera init
# ---------------------------------------------------------------------------

cam = Camera(
    data_pins=DATA_PINS,
    vsync_pin=25,
    href_pin=23,
    sda_pin=26,
    scl_pin=27,
    pclk_pin=22,
    xclk_pin=0,
    xclk_freq=20_000_000,
    powerdown_pin=32,
    reset_pin=-1,
    pixel_format=PixelFormat.GRAYSCALE,
    frame_size=FrameSize.R96X96,
    fb_count=1,
    grab_mode=GrabMode.LATEST,
)

print("camera: ok")

# ---------------------------------------------------------------------------
# Model + pre-allocated buffers
# ---------------------------------------------------------------------------

with open(MODEL_PATH, "rb") as f:
    model_data = array.array("B", f.read())

model    = emlearn_cnn_int8.new(model_data)
probs    = array.array("f", (0.0 for _ in range(NUM_CLASSES)))
scaled32 = array.array("B", (0 for _ in range(POOL_SIZE * POOL_SIZE)))
scaled28 = array.array("B", (0 for _ in range(CROP_SIZE * CROP_SIZE)))

print("model: ready")

# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

while True:
    try:
        t0 = time.ticks_ms()

        img = cam.capture()
        if img is None:
            print("capture: failed — skipping frame")
            continue

        buf = bytes(img)
        cam.free_buffer()

        downscale.downscale(buf, scaled32, CAPTURE_SIZE, POOL_SIZE)
        crop32to28(scaled32, scaled28)
        hflip_inplace(scaled28, CROP_SIZE)   # fix mirrored camera output
        threshold_inplace(scaled28, THRESHOLD)

        model.run(scaled28, probs)
        digit   = argmax(probs)
        elapsed = time.ticks_diff(time.ticks_ms(), t0)

        print(f"digit: {digit}  time: {elapsed} ms")

        gc.collect()
        time.sleep_ms(LOOP_DELAY_MS)

    except Exception as e:
        print(f"error: {e}")
        gc.collect()
        time.sleep_ms(ERROR_DELAY_MS)