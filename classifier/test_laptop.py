"""
ESP32-CAM Classifier — Laptop Webcam Tester
============================================
Tests your trained model.tflite using your laptop webcam before flashing
to the ESP32-CAM. Lets you catch problems early without touching the board.

USAGE:
    python test_laptop.py

CONTROLS:
    ESC       — quit
    S         — save a screenshot of the current frame to debug_frame.jpg
    +/-       — raise/lower the confidence threshold on the fly

WHAT THE DISPLAY SHOWS:
    Top line    — predicted class and confidence, or "uncertain (class?)"
                  if below threshold. The (class?) tells you which way
                  it's leaning even when uncertain — useful for debugging.
    Second line — raw probability scores for each class (should sum to 1.0)
    Bar chart   — visual confidence bars for each class
    Bottom      — current confidence threshold

REQUIREMENTS:
    pip install opencv-python tensorflow numpy
"""

import cv2
import numpy as np
import tensorflow as tf
import platform
# =============================================================================
# CONFIGURATION — edit these if needed
# =============================================================================

MODEL_PATH     = "model.tflite"   # path to your exported model
LABELS         = ["guava", "powerbank"]  # must match training folder order (alphabetical)
CONF_THRESHOLD = 0.75             # minimum confidence to show a class name
                                  # below this = "uncertain". Range: 0.0 - 1.0
CAMERA_INDEX   = 0                # 0 = default webcam, try 1 or 2 if wrong camera opens
IMG_SIZE       = 96               # must match what the model was trained on

# Colour scheme (BGR format for OpenCV)
COL_GREEN  = (0,   220, 0)
COL_RED    = (0,   60,  220)
COL_YELLOW = (0,   200, 220)
COL_WHITE  = (255, 255, 255)
COL_GRAY   = (160, 160, 160)
COL_BLACK  = (0,   0,   0)

# =============================================================================
# LOAD MODEL
# =============================================================================
print(f"Loading model: {MODEL_PATH}")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    raise SystemExit(f"Failed to load model: {e}\nMake sure {MODEL_PATH} exists.")

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Model input  shape : {input_details[0]['shape']}")
print(f"Model output shape : {output_details[0]['shape']}")
print(f"Classes: {LABELS}")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print("\nControls: ESC=quit  S=screenshot  +/-=adjust threshold\n")

# =============================================================================
# OPEN WEBCAM (Windows + Linux)
# =============================================================================

os_name = platform.system()

if os_name == "Windows":
    print("Detected Windows")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

elif os_name == "Linux":
    print("Detected Linux")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

else:
    print(f"Unknown OS ({os_name}), using default backend")
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise SystemExit(
        f"Could not open camera index {CAMERA_INDEX} on {os_name}.\n"
        f"Try CAMERA_INDEX=1 or CAMERA_INDEX=2."
    )

# =============================================================================
# HELPER: draw a filled rectangle as background for text (readability)
# =============================================================================
def draw_label(img, text, pos, font_scale, colour, thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4),
                      COL_BLACK, -1)
    cv2.putText(img, text, pos, font, font_scale, colour, thickness, cv2.LINE_AA)

# =============================================================================
# HELPER: draw a confidence bar for one class
# =============================================================================
def draw_bar(img, label, score, x, y, bar_width=200, highlight=False):
    filled = int(bar_width * score)
    colour = COL_GREEN if highlight else COL_GRAY
    # Background bar
    cv2.rectangle(img, (x, y), (x + bar_width, y + 18), (40, 40, 40), -1)
    # Filled portion
    cv2.rectangle(img, (x, y), (x + filled, y + 18), colour, -1)
    # Label + score text
    draw_label(img, f"{label}: {score:.2f}", (x + bar_width + 8, y + 14),
               0.5, COL_WHITE, 1, bg=False)

# =============================================================================
# MAIN LOOP
# =============================================================================
threshold = CONF_THRESHOLD   # local copy so +/- can adjust it

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    h, w    = display.shape[:2]

    # ── Preprocess ────────────────────────────────────────────────────────────
    # Match exactly what the training script did:
    #   convert BGR -> RGB, resize to 96x96, normalise to [0, 1]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Center crop — take the middle square of the frame
    
    # This reduces background and forces the object to fill more of the 96x96 input
    h_f, w_f = img.shape[:2]
    crop_size = min(h_f, w_f)
    y0 = (h_f - crop_size) // 2
    x0 = (w_f - crop_size) // 2
    img = img[y0:y0+crop_size, x0:x0+crop_size]

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)          # add batch dimension: (1, 96, 96, 3)

    # ── Run inference ─────────────────────────────────────────────────────────
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # ── Parse output ──────────────────────────────────────────────────────────
    scores     = [float(output[i]) for i in range(len(LABELS))]
    top_idx    = int(np.argmax(scores))
    confidence = scores[top_idx]

    # Show which direction it's leaning even when uncertain.
    # "uncertain (guava?)" is more useful than just "undefined"
    # because it tells you what the model is almost-thinking.
    if confidence < threshold:
        pred_text  = f"uncertain ({LABELS[top_idx]}?)"
        pred_color = COL_YELLOW
    else:
        pred_text  = LABELS[top_idx]
        pred_color = COL_GREEN

    # ── Draw UI ───────────────────────────────────────────────────────────────

    # Main prediction — large text at top
    draw_label(display, f"{pred_text}  {confidence:.2f}",
               (20, 45), 1.1, pred_color, 2)

    # Confidence bars for each class
    for i, (label, score) in enumerate(zip(LABELS, scores)):
        draw_bar(display, label, score,
                 x=20, y=70 + i * 28,
                 highlight=(i == top_idx and confidence >= threshold))

    # Raw scores line — useful for spotting if model is outputting 0.50/0.50
    raw_str = "  ".join([f"{l}:{s:.3f}" for l, s in zip(LABELS, scores)])
    draw_label(display, raw_str, (20, 70 + len(LABELS) * 28 + 20),
               0.5, COL_GRAY, 1)

    # Threshold indicator at bottom
    draw_label(display, f"threshold: {threshold:.2f}  (+/- to adjust)",
               (20, h - 15), 0.45, COL_GRAY, 1)

    # Show a small 96x96 preview of what the model actually sees
    # (the resized and cropped input — useful to check framing)
    preview = cv2.resize(frame, (IMG_SIZE * 2, IMG_SIZE * 2),
                         interpolation=cv2.INTER_NEAREST)
    display[10 : 10 + IMG_SIZE*2, w - IMG_SIZE*2 - 10 : w - 10] = preview
    draw_label(display, "model sees ->", (w - IMG_SIZE*2 - 10, 8),
               0.4, COL_GRAY, 1, bg=False)

    cv2.imshow("Classifier Test  |  ESC=quit  S=save  +/-=threshold", display)

    # ── Key handling ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == 27:           # ESC — quit
        break

    elif key == ord('s'):   # S — save screenshot
        fname = "debug_frame.jpg"
        cv2.imwrite(fname, display)
        print(f"Saved screenshot: {fname}")
        print(f"  Scores: { {l: f'{s:.3f}' for l, s in zip(LABELS, scores)} }")

    elif key == ord('+') or key == ord('='):   # + — raise threshold
        threshold = min(0.99, round(threshold + 0.05, 2))
        print(f"Threshold -> {threshold:.2f}")

    elif key == ord('-'):   # - — lower threshold
        threshold = max(0.50, round(threshold - 0.05, 2))
        print(f"Threshold -> {threshold:.2f}")

# =============================================================================
# CLEANUP
# =============================================================================
cap.release()
cv2.destroyAllWindows()
print("Done.")
