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
CAPTURE_SIZE   = 96    # OV3660 capture resolution (square, pixels per side)
POOL_SIZE      = 32    # intermediate size after average pooling
CROP_SIZE      = 28    # final size fed to the model (MNIST native resolution)
CROP_OFFSET    = (POOL_SIZE - CROP_SIZE) // 2   # = 2 pixels from each edge
THRESHOLD      = 120   # pixels above this → 255 (digit), else → 0 (background)
LOOP_DELAY_MS  = 50    # inter-frame sleep; helps camera stability
ERROR_DELAY_MS = 200   # sleep after an exception before retrying

NUM_CLASSES    = 10    # digits 0-9

# Camera pin mapping for a typical ESP32-CAM board
DATA_PINS = [5, 18, 19, 21, 36, 39, 34, 35]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def argmax(arr) -> int:
    """Return the index of the maximum value in arr."""
    idx_max = 0
    val_max = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > val_max:
            val_max = arr[i]
            idx_max = i
    return idx_max


def crop32to28(src, dst) -> None:
    """
    Centre-crop a flat 32×32 buffer into a flat 28×28 buffer.
    Removes CROP_OFFSET (2) pixels from each edge.
    """
    for r in range(CROP_SIZE):
        for c in range(CROP_SIZE):
            dst[r * CROP_SIZE + c] = src[(r + CROP_OFFSET) * POOL_SIZE + (c + CROP_OFFSET)]


def threshold_inplace(buf, cutoff: int) -> None:
    """Binarise buf in-place: values > cutoff → 255, else → 0."""
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

model       = emlearn_cnn_int8.new(model_data)
probs       = array.array("f", (0.0 for _ in range(NUM_CLASSES)))
scaled32    = array.array("B", (0 for _ in range(POOL_SIZE  * POOL_SIZE)))
scaled28    = array.array("B", (0 for _ in range(CROP_SIZE  * CROP_SIZE)))

print("model: ready")

# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

while True:
    try:
        t0 = time.ticks_ms()

        # --- Capture --------------------------------------------------------
        img = cam.capture()
        if img is None:
            print("capture: failed — skipping frame")
            continue

        buf = bytes(img)
        cam.free_buffer()

        # --- Preprocess -----------------------------------------------------
        downscale.downscale(buf, scaled32, CAPTURE_SIZE, POOL_SIZE)
        crop32to28(scaled32, scaled28)
        threshold_inplace(scaled28, THRESHOLD)

        # --- Inference ------------------------------------------------------
        model.run(scaled28, probs)
        digit    = argmax(probs)
        elapsed  = time.ticks_diff(time.ticks_ms(), t0)

        print(f"digit: {digit}  time: {elapsed} ms")

        gc.collect()
        time.sleep_ms(LOOP_DELAY_MS)

    except Exception as e:
        print(f"error: {e}")
        gc.collect()
        time.sleep_ms(ERROR_DELAY_MS)
