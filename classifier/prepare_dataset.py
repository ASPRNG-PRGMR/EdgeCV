"""
prepare_dataset.py — build a leakage-free train/test split from the raw captures.

WHY THIS SCRIPT EXISTS
======================
The previous pipeline leaked the test set into training, in two separate ways.

Leak 1 — the test folder WAS the training source.
    sanitizer_dataset.py reads   dataset/test_guava/, dataset/test_powerbank/
    and writes hand-removed copies to
                                 dataset/guava/,      dataset/powerbank/
    train_classifier.py then trains on guava/ + powerbank/ and evaluates on
    test_guava/ + test_powerbank/, describing them as "held-out test images
    (never seen during training)". They are not held out: every single test
    image is the raw original of a photo whose inpainted twin is in the
    training set. Same object, same pose, same background, same frame — only
    the hand pixels differ. The reported test accuracy measured "can the model
    recognise an image it trained on, minus inpainting".

Leak 2 — the group-aware split is a no-op, and the data is video.
    build_group_split() groups files by base filename to keep two versions of
    one photo in the same split. But the sanitizer renames its output to a
    fresh sequential index (guava_0001.jpg), so every group has exactly one
    member and the group split degenerates into a plain random split.
    That would be survivable if the frames were independent. They are not:

        guava     : 155 frames, one unbroken 38.5 s capture at 4 fps
        powerbank : 164 frames, one unbroken 40.8 s capture at 4 fps
        largest gap anywhere in either timeline: 265 ms

    It is two continuous video clips, one object per class, one background,
    one lighting setup. Frames 250 ms apart are near-identical (median
    adjacent dHash distance 8; 382 guava pairs within dHash 10 sit a mean
    2.2 s apart). A random split therefore puts near-duplicate frames on both
    sides of the boundary no matter what, and the phash dedup at threshold 5
    only removes 40/155 and 20/164 of them.

WHAT THIS SCRIPT DOES INSTEAD
=============================
Splits on the TIMELINE, not on files, and splits BEFORE sanitising:

    raw clip  ├──────────── train ────────────┤ guard ├──── test ────┤
                                               (dropped)

  * The tail TEST_TAIL_FRACTION of each clip becomes the test set.
  * A GUARD_SECONDS band either side of the boundary is discarded, so the
    last training frame and the first test frame are never adjacent frames
    of the same motion.
  * Hand removal runs on the TRAIN portion only. The test portion stays raw,
    because raw is what the camera actually sees at deployment — testing on
    inpainted images would hide the train/serve skew that inpainting creates.
  * Every output file is recorded in dataset/manifest.json with its sha256,
    so a trained model can be traced back to exact inputs. dataset/ is
    gitignored; the manifest is small and should be committed.

READ THIS BEFORE TRUSTING ANY NUMBER THIS ENABLES
=================================================
A temporal split makes the metric honest, it does not make the dataset good.
With one object instance, one background and one lighting condition per class,
even a clean held-out tail only answers "can it recognise the back half of
this one clip". It cannot answer "can it recognise a guava". For that the
dataset needs multiple instances, backgrounds, distances and lighting
conditions per class, captured in separate sessions — and the split should
then hold out whole SESSIONS, which this script already supports via the
gap-detection in group_sessions().

USAGE
=====
    python prepare_dataset.py                  # ingest zips, split, sanitise
    python prepare_dataset.py --test-frac 0.3 --guard 2.0
    python prepare_dataset.py --no-sanitize    # skip hand removal (faster)

Output layout:
    dataset/
      raw/<class>/        every raw frame, untouched, as ingested
      train/<class>/      sanitised training frames
      test/<class>/       raw held-out tail frames
      manifest.json       sha256 + split + timestamp for every file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"

# Raw capture archives shipped with the repo, mapped to class name.
ZIPS = {
    "guava": HERE / "images_of_guava.zip",
    "powerbank": HERE / "images_of_powerbank.zip",
}

CLASS_NAMES = sorted(ZIPS)  # alphabetical — must match the firmware's label order

# Fraction of each clip's timeline held out for test, taken from the END.
TEST_TAIL_FRACTION = 0.30

# Frames within this many seconds of the train/test boundary are dropped, so
# the split boundary is not two consecutive frames of one continuous motion.
GUARD_SECONDS = 2.0

# A gap larger than this in a class's timeline means a separate capture
# session. When the dataset eventually has several sessions per class, the
# split should hold out whole sessions rather than a tail.
SESSION_GAP_SECONDS = 2.0

TS_RE = re.compile(r"_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_zips(force: bool = False) -> None:
    """Extract the raw archives into dataset/raw/<class>/.

    Entries are extracted by basename only. Anything with a path separator,
    a drive letter or a .. component is refused rather than trusted, and
    non-image entries are skipped — the archives are data, so they get
    checked rather than assumed flat.
    """
    for cls, zpath in ZIPS.items():
        out = DATASET / "raw" / cls
        if out.is_dir() and any(out.iterdir()) and not force:
            print(f"  {cls:<10} raw/ already populated ({len(list(out.iterdir()))} files) — skipping")
            continue
        if not zpath.is_file():
            sys.exit(f"ERROR: archive not found: {zpath}")

        out.mkdir(parents=True, exist_ok=True)
        n, skipped = 0, 0
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                name = info.filename
                if info.is_dir():
                    continue
                if "/" in name or "\\" in name or ".." in name or ":" in name:
                    print(f"    refusing entry with a path component: {name!r}")
                    skipped += 1
                    continue
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    skipped += 1
                    continue
                with zf.open(info) as src, open(out / Path(name).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
        print(f"  {cls:<10} extracted {n} images"
              + (f" ({skipped} entries skipped)" if skipped else ""))


# ---------------------------------------------------------------------------
# Timeline analysis
# ---------------------------------------------------------------------------

def frame_timeline(class_dir: Path) -> list[tuple[int, Path]]:
    """Return [(timestamp_ms, path)] sorted by timestamp.

    Capture filenames look like guava_1781092996904.jpg — the suffix is a
    millisecond epoch. Files without one cannot be ordered in time, so they
    are reported and dropped rather than silently sorted by name.
    """
    out, undated = [], []
    for p in sorted(class_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = TS_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
        else:
            undated.append(p.name)
    if undated:
        print(f"    WARNING: {len(undated)} file(s) have no _<timestamp> suffix "
              f"and cannot be placed on the timeline — excluded. "
              f"First few: {undated[:3]}")
    out.sort(key=lambda t: t[0])
    return out


def group_sessions(timeline: list[tuple[int, Path]],
                   gap_s: float = SESSION_GAP_SECONDS) -> list[list[tuple[int, Path]]]:
    """Split a timeline into capture sessions wherever there is a long gap."""
    if not timeline:
        return []
    sessions, cur = [], [timeline[0]]
    for prev, nxt in zip(timeline, timeline[1:]):
        if (nxt[0] - prev[0]) / 1000.0 > gap_s:
            sessions.append(cur)
            cur = []
        cur.append(nxt)
    sessions.append(cur)
    return sessions


def temporal_split(timeline: list[tuple[int, Path]],
                   test_frac: float,
                   guard_s: float):
    """Hold out the tail of the timeline, dropping a guard band at the seam.

    Returns (train, test, dropped).
    """
    if not timeline:
        return [], [], []
    t0, t1 = timeline[0][0], timeline[-1][0]
    span = t1 - t0
    if span <= 0:
        return list(timeline), [], []

    boundary = t1 - span * test_frac
    guard_ms = guard_s * 1000.0

    train = [f for f in timeline if f[0] <= boundary - guard_ms / 2]
    test = [f for f in timeline if f[0] >= boundary + guard_ms / 2]
    dropped = [f for f in timeline if f not in set(train) | set(test)]
    return train, test, dropped


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(frames, out_dir: Path, sanitize: bool, remove_hand) -> list[dict]:
    """Copy (or sanitise then write) frames into out_dir. Returns manifest rows."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rows = []
    for ts, src in frames:
        dst = out_dir / src.name
        if sanitize:
            img = cv2.imread(str(src))
            if img is None:
                print(f"    unreadable, skipped: {src.name}")
                continue
            cv2.imwrite(str(dst), remove_hand(img))
        else:
            shutil.copy2(src, dst)
        rows.append({
            "file": str(dst.relative_to(DATASET)),
            "source": str(src.relative_to(DATASET)),
            "timestamp_ms": ts,
            "sanitized": bool(sanitize),
            "sha256": sha256(dst),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--test-frac", type=float, default=TEST_TAIL_FRACTION,
                    help=f"fraction of each clip's timeline held out (default {TEST_TAIL_FRACTION})")
    ap.add_argument("--guard", type=float, default=GUARD_SECONDS,
                    help=f"seconds dropped at the split seam (default {GUARD_SECONDS})")
    ap.add_argument("--no-sanitize", action="store_true",
                    help="skip hand removal on the training split")
    ap.add_argument("--force-ingest", action="store_true",
                    help="re-extract the zips even if raw/ is populated")
    args = ap.parse_args()

    remove_hand = None
    if not args.no_sanitize:
        try:
            from sanitizer_dataset import remove_hand  # noqa: F401
        except Exception as exc:
            sys.exit(f"ERROR: cannot import remove_hand from sanitizer_dataset.py ({exc}).\n"
                     f"       Run with --no-sanitize to skip hand removal.")

    print("=" * 70)
    print("EdgeCV dataset preparation — temporal split, sanitise train only")
    print("=" * 70)

    print("\n[1/3] Ingesting raw archives")
    DATASET.mkdir(exist_ok=True)
    ingest_zips(force=args.force_ingest)

    print("\n[2/3] Timeline analysis and split")
    manifest = {
        "class_names": CLASS_NAMES,
        "test_tail_fraction": args.test_frac,
        "guard_seconds": args.guard,
        "sanitized_train": not args.no_sanitize,
        "classes": {},
    }

    for cls in CLASS_NAMES:
        raw_dir = DATASET / "raw" / cls
        if not raw_dir.is_dir():
            sys.exit(f"ERROR: {raw_dir} missing — ingest failed?")

        timeline = frame_timeline(raw_dir)
        sessions = group_sessions(timeline)
        span = (timeline[-1][0] - timeline[0][0]) / 1000.0 if timeline else 0.0

        print(f"\n  {cls}")
        print(f"    frames   : {len(timeline)}")
        print(f"    span     : {span:.1f} s")
        print(f"    sessions : {len(sessions)} "
              f"(gap > {SESSION_GAP_SECONDS}s = new session)")
        if len(sessions) == 1:
            print(f"    NOTE: one continuous clip. A temporal tail split is the best")
            print(f"          available here, but it still cannot measure whether the")
            print(f"          model generalises past this single recording.")

        train, test, dropped = temporal_split(timeline, args.test_frac, args.guard)
        print(f"    split    : {len(train)} train / {len(test)} test "
              f"({len(dropped)} dropped in the {args.guard}s guard band)")
        if train and test:
            seam = (test[0][0] - train[-1][0]) / 1000.0
            print(f"    seam gap : {seam:.1f} s between last train and first test frame")

        manifest["classes"][cls] = {
            "n_raw": len(timeline),
            "n_sessions": len(sessions),
            "span_seconds": round(span, 1),
            "n_train": len(train),
            "n_test": len(test),
            "n_dropped_guard": len(dropped),
        }

        print(f"    writing train/ ({'sanitised' if not args.no_sanitize else 'raw copy'}) ...", end="", flush=True)
        tr_rows = write_split(train, DATASET / "train" / cls,
                              sanitize=not args.no_sanitize, remove_hand=remove_hand)
        print(f" {len(tr_rows)} files")

        print(f"    writing test/  (raw, never sanitised) ...", end="", flush=True)
        te_rows = write_split(test, DATASET / "test" / cls,
                              sanitize=False, remove_hand=None)
        print(f" {len(te_rows)} files")

        manifest["classes"][cls]["train_files"] = tr_rows
        manifest["classes"][cls]["test_files"] = te_rows

    print("\n[3/3] Writing manifest")
    mpath = DATASET / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"  {mpath}  ({mpath.stat().st_size / 1024:.0f} KB)")
    print("\n  dataset/ is gitignored. COMMIT dataset/manifest.json — it is what")
    print("  makes a trained model traceable to its exact inputs.")

    print("\nDone. Train with the train/ and test/ folders this created,")
    print("then measure with:  python eval_model.py")


if __name__ == "__main__":
    main()
