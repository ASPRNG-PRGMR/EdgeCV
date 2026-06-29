#!/usr/bin/env python3
"""
live_view.py — receive raw JPEG frames from ESP32-CAM and display live feed.

Protocol: [4-byte LE length][JPEG bytes] repeated.
Connects to classifier-espcam.local — no hardcoded IP needed.
Press Q in the display window to quit.
"""

import socket
import struct
import numpy as np
import cv2

ESP32_HOST = "classifier-espcam.local"
ESP32_PORT = 5555
DISPLAY_W  = 480
DISPLAY_H  = 480


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Block until exactly n bytes are received (TCP can short-read on WiFi)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("ESP32 disconnected")
        buf += chunk
    return bytes(buf)


def resolve_esp32() -> str:
    try:
        return socket.gethostbyname(ESP32_HOST)
    except socket.gaierror:
        raise RuntimeError(
            f"Could not resolve {ESP32_HOST}.\n"
            "Ensure the ESP32-CAM is powered on, connected to WiFi,\n"
            "and mDNS is working (Bonjour installed on Windows)."
        )


def main() -> None:
    print(f"Resolving {ESP32_HOST} ...")
    ip = resolve_esp32()
    print(f"Connecting to {ip}:{ESP32_PORT} ...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip, ESP32_PORT))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Connected. Press Q in the window to quit.")

        while True:
            # 1. Read 4-byte little-endian length prefix
            hdr       = recv_exact(s, 4)
            frame_len = struct.unpack('<I', hdr)[0]

            # 2. Read exactly that many JPEG bytes
            jpeg_bytes = recv_exact(s, frame_len)

            # 3. Decode JPEG → BGR numpy array
            arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[warn] Bad JPEG ({frame_len} B), skipping")
                continue

            # 4. Scale up for display (96px native is tiny on screen)
            disp = cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                              interpolation=cv2.INTER_NEAREST)

            # 5. Invert
            disp = cv2.bitwise_not(disp)

            cv2.imshow("EdgeCV — Live Feed", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()