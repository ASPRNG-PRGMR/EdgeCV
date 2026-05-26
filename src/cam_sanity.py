"""
cam_sanity.py
-------------
Quick sanity check for the OV3660 camera on ESP32.
Initialises the camera, captures a single frame, and prints
the number of bytes received.  Run this first to confirm the
camera wiring and driver are working before running the main
inference script.
"""

from camera import Camera, PixelFormat, FrameSize

cam = Camera(
    data_pins=[5, 18, 19, 21, 36, 39, 34, 35],
    vsync_pin=25,
    href_pin=23,
    sda_pin=26,
    scl_pin=27,
    pclk_pin=22,
    xclk_pin=0,
    xclk_freq=10_000_000,
    powerdown_pin=32,
    reset_pin=-1,
    pixel_format=PixelFormat.GRAYSCALE,
    frame_size=FrameSize.QQVGA,   # 160 x 120
)

img = cam.capture()

if img is None:
    print("ERROR: capture returned None — check wiring / pin mapping")
else:
    print(f"OK: captured {len(img)} bytes")
