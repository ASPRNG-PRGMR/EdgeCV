"""
eval_model.py — measure what the deployed int8 model actually does.

WHY THIS SCRIPT EXISTS
======================
The repo previously reported model quality as ">90% confidence on clean
samples". Confidence is not accuracy. A model that is wrong can be confidently
wrong — and in this project it demonstrably is: classifier/misclassified/
contains a powerbank frame classified as guava at 0.97 confidence. Confidence
tells you how far the sigmoid output is from 0.5; it says nothing about
whether the answer is right.

This script evaluates the INT8 .tflite — the artefact that actually ships to
flash — rather than the Keras float model, because int8 quantisation is a real
source of accuracy loss and the float number does not predict the device.

WHAT IT REPORTS
===============
  1. Contamination check. For every eval image, whether that image (or its
     sanitised twin) appears in the training manifest. If the model trained on
     your test set, every number below is meaningless, and the script says so
     instead of printing a reassuring 99%.
  2. Confusion matrix + per-class precision / recall / F1, and balanced
     accuracy — not just the single accuracy figure, which hides which class
     is failing.
  3. Rejection threshold sweep. For each candidate confidence threshold, the
     fraction of frames accepted and the accuracy among accepted frames. This
     is how you pick CONFIDENCE_THRESHOLD in main_edgecv_v2.cc from data
     rather than from a guess. Note that for a single-sigmoid binary model the
     file's own three REJECT_MODE_* options collapse to this one knob —
     max(p, 1-p) < t is exactly |p - (1-p)| < 2t-1 — so one sweep tunes all
     three modes.
  4. Misclassified frames dumped to disk, named with their confidence, so the
     failures can be looked at rather than inferred.

PREPROCESSING FIDELITY
======================
Device and host do not see identical pixels, and the difference is worth
knowing when host numbers disagree with serial output:

  device : OV3660 captures natively at FRAMESIZE_96X96 (YUV422), converted to
           RGB in firmware with BT.601 coefficients, then int8 = uint8 - 128.
  host   : this script decodes the stored 240x240 JPEG and bilinearly
           downscales it to 96x96, matching train_classifier.py's
           tf.image.resize(img, [96, 96]) (bilinear, antialias=False, which is
           cv2.INTER_LINEAR), then int8 = uint8 - 128.

So the host evaluates a downscaled 240x240 capture while the device evaluates
a native 96x96 sensor frame. Same nominal resolution, different optics and
different resampling. That is itself a train/serve skew: the model was trained
on downscaled 240x240 images and deployed on native 96x96 ones.

USAGE
=====
    python eval_model.py                       # evaluate dataset/test
    python eval_model.py --split raw           # evaluate every raw frame
    python eval_model.py --split train         # sanity check: should be high
    python eval_model.py --model model.tflite --dump-misses

Requires an int8 TFLite interpreter. Any one of these works:
    pip install ai-edge-litert        # smallest, current LiteRT runtime
    pip install tensorflow            # also fine, much larger
    pip install tflite-runtime        # older standalone runtime
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"

IMG_H = IMG_W = 96

# Alphabetical, matching Keras image_dataset_from_directory and the firmware's
# CLASS_LABELS[] = { "guava", "powerbank" }. Index 1 is the sigmoid's positive
# class: output >= 0.5 means powerbank.
CLASS_NAMES = ["guava", "powerbank"]


# ---------------------------------------------------------------------------
# Interpreter loading — try the runtimes in order of preference
# ---------------------------------------------------------------------------

def load_interpreter(model_path: Path):
    errors = []
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=str(model_path)), "ai-edge-litert"
    except Exception as e:
        errors.append(f"ai_edge_litert: {e}")
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=str(model_path)), "tflite-runtime"
    except Exception as e:
        errors.append(f"tflite_runtime: {e}")
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(model_path)), "tensorflow.lite"
    except Exception as e:
        errors.append(f"tensorflow: {e}")

    print("ERROR: no int8 TFLite interpreter available. Tried:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    print("\nInstall one of:\n"
          "  pip install ai-edge-litert     (smallest)\n"
          "  pip install tflite-runtime\n"
          "  pip install tensorflow         (largest)", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Preprocessing — mirrors train_classifier.py and the firmware
# ---------------------------------------------------------------------------

def preprocess(path: Path, in_dtype) -> np.ndarray | None:
    """JPEG -> RGB -> 96x96 bilinear -> int8 (uint8 - 128) or uint8/float32."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # cv2.INTER_LINEAR == tf.image.resize bilinear with antialias=False,
    # which is what train_classifier.py used.
    rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    if in_dtype == np.int8:
        # Rescaling(1/255) is baked into the quantised weights and
        # zero_point is -128, so the device does exactly this.
        return (rgb.astype(np.int16) - 128).astype(np.int8)[None, ...]
    if in_dtype == np.uint8:
        return rgb.astype(np.uint8)[None, ...]
    return (rgb.astype(np.float32) / 255.0)[None, ...]


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------

def base_key(name: str) -> str:
    """Strip directory and extension; capture filenames are unique by timestamp."""
    return Path(name).stem


def load_train_keys() -> set[str] | None:
    mpath = DATASET / "manifest.json"
    if not mpath.is_file():
        return None
    man = json.loads(mpath.read_text())
    keys = set()
    for cls, info in man.get("classes", {}).items():
        for row in info.get("train_files", []):
            keys.add(base_key(row["file"]))
            keys.add(base_key(row.get("source", "")))
    keys.discard("")
    return keys


def model_provenance(model_path: Path) -> dict | None:
    """Read <model>.provenance.json, written by the training script.

    Without it there is no way to know which images a model was trained on,
    and a contamination check against the *current* manifest proves nothing:
    the manifest describes today's split, not the split the model was fitted
    under. This distinction is not academic here — the model.tflite committed
    to this repo was trained by the old pipeline on all 319 raw frames, so it
    has seen every image in every split, including ones a freshly generated
    manifest lists as held out.
    """
    ppath = model_path.with_suffix(model_path.suffix + ".provenance.json")
    if ppath.is_file():
        try:
            return json.loads(ppath.read_text())
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def report_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    n_cls = len(CLASS_NAMES)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    width = max(len(c) for c in CLASS_NAMES) + 2
    print("\n  Confusion matrix (rows = actual, cols = predicted)")
    print("    " + " " * width + "".join(f"{c:>{width}}" for c in CLASS_NAMES) + "   total")
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:>{width}}" + "".join(f"{cm[i, j]:>{width}}" for j in range(n_cls))
              + f"{cm[i].sum():>8}")

    acc = cm.trace() / cm.sum() if cm.sum() else 0.0
    print(f"\n  Overall accuracy : {acc * 100:6.2f}%   ({cm.trace()}/{cm.sum()})")

    recalls = []
    print(f"\n  {'class':>{width}} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>8}")
    for i, c in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recalls.append(rec)
        print(f"  {c:>{width}} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f} {cm[i].sum():>8}")
    print(f"\n  Balanced accuracy (mean recall): {np.mean(recalls) * 100:6.2f}%")
    if abs(recalls[0] - recalls[1]) > 0.10:
        worse = CLASS_NAMES[int(np.argmin(recalls))]
        print(f"  NOTE: per-class recall differs by "
              f"{abs(recalls[0] - recalls[1]) * 100:.0f} points — "
              f"'{worse}' is the failing class, which the overall accuracy hides.")


def report_threshold_sweep(y_true: np.ndarray, probs: np.ndarray) -> None:
    """Accept-rate vs accuracy-on-accepted, for picking CONFIDENCE_THRESHOLD."""
    conf = np.maximum(probs, 1.0 - probs)        # max(p, 1-p)
    pred = (probs >= 0.5).astype(int)
    correct = (pred == y_true)

    print("\n  Rejection threshold sweep")
    print("  (conf = max(p, 1-p). 'accepted' = conf >= threshold.")
    print("   equivalent MARGIN_THRESHOLD = 2*threshold - 1)")
    print(f"\n  {'thresh':>7} {'margin':>7} {'accepted':>10} {'acc|accepted':>14} {'missed':>8}")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        keep = conf >= t
        n_keep = int(keep.sum())
        if n_keep == 0:
            print(f"  {t:>7.2f} {2 * t - 1:>7.2f} {0:>9d}  {'—':>14} {len(y_true):>8}")
            continue
        acc_keep = correct[keep].mean()
        print(f"  {t:>7.2f} {2 * t - 1:>7.2f} "
              f"{n_keep:>4d}/{len(y_true):<5d} {acc_keep * 100:>13.2f}% "
              f"{int((~keep).sum()):>8}")

    # Wrong-and-confident is the failure mode that matters for actuation.
    for t in (0.75, 0.90):
        bad = (~correct) & (conf >= t)
        print(f"\n  Wrong AND confident (conf >= {t:.2f}): {int(bad.sum())} frames"
              f"  — these are what rejection cannot catch.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect(split_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for idx, cls in enumerate(CLASS_NAMES):
        d = split_dir / cls
        if not d.is_dir():
            sys.exit(f"ERROR: {d} not found. Run prepare_dataset.py first.")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((p, idx))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.tflite")
    ap.add_argument("--split", default="test", choices=["test", "train", "raw"],
                    help="which dataset/ subfolder to evaluate (default: test)")
    ap.add_argument("--dump-misses", action="store_true",
                    help="write misclassified frames to misclassified_<split>/")
    args = ap.parse_args()

    model_path = (HERE / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not model_path.is_file():
        sys.exit(f"ERROR: model not found: {model_path}")

    interp, backend = load_interpreter(model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    print("=" * 72)
    print("EdgeCV model evaluation")
    print("=" * 72)
    print(f"  model    : {model_path.name}  ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  backend  : {backend}")
    print(f"  input    : shape={list(inp['shape'])} dtype={np.dtype(inp['dtype']).name} "
          f"scale={inp['quantization'][0]:.6f} zero_point={inp['quantization'][1]}")
    print(f"  output   : shape={list(out['shape'])} dtype={np.dtype(out['dtype']).name} "
          f"scale={out['quantization'][0]:.6f} zero_point={out['quantization'][1]}")

    if np.dtype(inp["dtype"]) == np.int8 and inp["quantization"][1] != -128:
        print(f"  WARNING: zero_point is {inp['quantization'][1]}, not -128. The firmware "
              f"hardcodes (uint8 - 128); it will feed this model wrong values.")

    items = collect(DATASET / args.split)
    print(f"\n  split    : dataset/{args.split}  ({len(items)} images)")

    # --- contamination ----------------------------------------------------
    prov = model_provenance(model_path)
    train_keys = load_train_keys()
    contaminated = True   # assume the worst until provenance proves otherwise

    if prov is None:
        print("\n  CONTAMINATION: UNKNOWN — cannot be established.")
        print("  " + "!" * 66)
        print(f"  No {model_path.name}.provenance.json, so there is no record of which")
        print("  images this model was fitted on. The manifest in dataset/ describes")
        print("  the CURRENT split, not the one this model was trained under, so")
        print("  checking the eval set against it would prove nothing.")
        print()
        print("  For the model.tflite committed to this repo the answer is known from")
        print("  the pipeline's history: it was trained by the old script on all 319")
        print("  raw frames, so it has already seen every image in every split. Treat")
        print("  every number below as an upper bound, not a measurement.")
        print()
        print("  Retrain via prepare_dataset.py + train_classifier.py to get a model")
        print("  with recorded provenance and a genuinely unseen test set.")
        print("  " + "!" * 66)
    elif train_keys is None:
        print("\n  CONTAMINATION: dataset/manifest.json not found — run prepare_dataset.py.")
    else:
        prov_keys = set(prov.get("train_keys", []))
        overlap = sum(1 for p, _ in items if base_key(p.name) in prov_keys)
        pct = 100 * overlap / len(items) if items else 0
        contaminated = overlap > 0
        print(f"\n  CONTAMINATION: {overlap}/{len(items)} ({pct:.0f}%) of these images "
              f"are in this model's recorded training set.")
        if contaminated:
            print("  " + "!" * 66)
            print("  Every metric below is inflated. A model evaluated on its own")
            print("  training data is not being measured.")
            print("  " + "!" * 66)
        else:
            print("  Clean: none of these images were used to fit this model.")

    # --- inference --------------------------------------------------------
    in_dtype = np.dtype(inp["dtype"]).type
    o_scale, o_zp = out["quantization"]

    y_true, probs, paths = [], [], []
    for p, label in items:
        x = preprocess(p, in_dtype)
        if x is None:
            print(f"  unreadable, skipped: {p.name}")
            continue
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0][0]
        prob = (float(raw) - o_zp) * o_scale if np.dtype(out["dtype"]) == np.int8 else float(raw)
        probs.append(min(max(prob, 0.0), 1.0))
        y_true.append(label)
        paths.append(p)

    y_true = np.array(y_true)
    probs = np.array(probs)
    y_pred = (probs >= 0.5).astype(int)

    report_metrics(y_true, y_pred)
    report_threshold_sweep(y_true, probs)

    # --- dump misses ------------------------------------------------------
    wrong = np.flatnonzero(y_pred != y_true)
    if len(wrong):
        print(f"\n  Misclassified frames ({len(wrong)}):")
        order = wrong[np.argsort(-np.maximum(probs[wrong], 1 - probs[wrong]))]
        for i in order[:15]:
            conf = max(probs[i], 1 - probs[i])
            print(f"    conf {conf:.2f}  {CLASS_NAMES[y_true[i]]:>10} "
                  f"-> {CLASS_NAMES[y_pred[i]]:<10} {paths[i].name}")
        if len(order) > 15:
            print(f"    ... and {len(order) - 15} more")

        if args.dump_misses:
            root = HERE / f"misclassified_{args.split}"
            if root.exists():
                shutil.rmtree(root)
            for i in order:
                conf = max(probs[i], 1 - probs[i])
                d = root / f"actual_{CLASS_NAMES[y_true[i]]}" / f"predicted_as_{CLASS_NAMES[y_pred[i]]}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(paths[i], d / f"conf{conf:.2f}_{paths[i].name}")
            print(f"\n  Dumped to {root.relative_to(HERE)}/")
    else:
        print("\n  No misclassified frames.")
        if contaminated:
            print("  A perfect score here is the expected result of evaluating a model")
            print("  on data it may have trained on — see the contamination notice above.")
            print("  It is not evidence that the model generalises.")


if __name__ == "__main__":
    main()
