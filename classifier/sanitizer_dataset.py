import os
import cv2
import shutil
import imagehash
import numpy as np

from PIL import Image
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ROOT = Path("dataset")

INPUTS = {
    "guava": ROOT / "test_guava",
    "powerbank": ROOT / "test_powerbank"
}

OUTPUTS = {
    "guava": ROOT / "guava",
    "powerbank": ROOT / "powerbank"
}

PHASH_THRESHOLD = 5

# ============================================================
# HAND REMOVAL
# ============================================================

def get_histogram_background_peak_mean(img, hand_mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    bright_hist = hist[180:256]

    if np.sum(bright_hist) == 0:
        return (240, 240, 240)

    peak_intensity = int(np.argmax(bright_hist) + 180)

    lower_bound = max(0, peak_intensity - 5)
    upper_bound = min(255, peak_intensity + 5)

    peak_mask = cv2.inRange(gray, lower_bound, upper_bound)

    hand_mask_inv = cv2.bitwise_not(hand_mask)

    valid_bg_mask = cv2.bitwise_and(
        peak_mask,
        hand_mask_inv
    )

    mean_color = cv2.mean(img, mask=valid_bg_mask)

    if np.sum(valid_bg_mask) == 0:
        return (240, 240, 240)

    return (
        int(mean_color[0]),
        int(mean_color[1]),
        int(mean_color[2])
    )


def remove_hand(img):
    h, w = img.shape[:2]

    hsv_img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower_skin = np.array([5, 40, 40])
    upper_skin = np.array([25, 255, 255])

    skin_mask = cv2.inRange(
        hsv_img,
        lower_skin,
        upper_skin
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_DILATE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        skin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(skin_mask)

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        touches_edge = (
            x <= 5 or
            y <= 5 or
            x + bw >= w - 5 or
            y + bh >= h - 5
        )

        if touches_edge:
            cv2.drawContours(
                final_mask,
                [cnt],
                -1,
                255,
                thickness=cv2.FILLED
            )

    bg_color = get_histogram_background_peak_mean(
        img,
        final_mask
    )

    result = img.copy()

    result[final_mask == 255] = bg_color

    return result

# ============================================================
# DUPLICATE DETECTION
# ============================================================

def sharpness(path):
    img = cv2.imread(str(path))

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def remove_duplicates(folder):
    files = sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.png"))
    )

    hashes = {}

    removed = 0

    for f in files:
        try:
            h = imagehash.phash(
                Image.open(f)
            )
        except:
            continue

        duplicate_found = False

        for existing_file, existing_hash in hashes.items():

            if (h - existing_hash) <= PHASH_THRESHOLD:

                duplicate_found = True

                s_new = sharpness(f)
                s_old = sharpness(existing_file)

                if s_new > s_old:
                    existing_file.unlink()
                    hashes.pop(existing_file)

                    hashes[f] = h

                else:
                    f.unlink()

                removed += 1
                break

        if not duplicate_found:
            hashes[f] = h

    return removed

# ============================================================
# MAIN
# ============================================================

def build_clean_dataset(class_name):

    input_dir = INPUTS[class_name]
    output_dir = OUTPUTS[class_name]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    image_files = sorted(
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.png"))
    )

    for idx, path in enumerate(image_files, start=1):

        img = cv2.imread(str(path))

        if img is None:
            continue

        cleaned = remove_hand(img)

        ext = path.suffix.lower()

        out_path = output_dir / (
            f"{class_name}_{idx:04d}{ext}"
        )

        cv2.imwrite(
            str(out_path),
            cleaned
        )

        count += 1

    removed = remove_duplicates(output_dir)

    print(
        f"{class_name}: "
        f"{count} cleaned images created, "
        f"{removed} duplicates removed."
    )

# ============================================================

if __name__ == "__main__":

    print("===================================")
    print("EdgeCV Dataset Sanitizer")
    print("===================================")

    build_clean_dataset("guava")
    build_clean_dataset("powerbank")

    print()
    print("Done.")
