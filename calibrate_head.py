"""
Fit the zero-shot decision threshold on labeled webcam frames.

SigLIP's "not touching" prompt outscores the "touching" prompt on nearly
every frame, but the gap between the two still separates the classes. This
runs the deployed model on calibration_frames/ and stores the threshold
midway between the classes' mean gap in models/face_touch_head.npz.

If frames are missing (or with --recapture), the camera records 50 frames
of you not touching your face, then 50 of you touching it in varied ways.

Re-run after export_model.py, which rewrites the head.

Usage:
  python calibrate_head.py [--recapture]
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from inference import HEAD_PATH, ZeroShotHead, create_session, open_camera, preprocess

CALIB_DIR = Path("calibration_frames")
FRAMES_PER_PHASE = 50
CAPTURE_INTERVAL = 0.3  # seconds between saved frames, for pose variety
COUNTDOWN = 3           # seconds to get into position before each phase

# (filename prefix, on-screen instruction); the prefix doubles as the label
PHASES = [
    ("not_touching", "Do NOT touch your face"),
    ("touching", "TOUCH your face (vary hand & spot)"),
]


def capture_frames() -> None:
    CALIB_DIR.mkdir(exist_ok=True)
    cap = open_camera()
    try:
        for prefix, instruction in PHASES:
            start, last_save, saved = time.monotonic(), 0.0, 0
            while saved < FRAMES_PER_PHASE:
                ret, frame = cap.read()
                if not ret:
                    continue
                now = time.monotonic()
                if now - start < COUNTDOWN:
                    status, color = f"starting in {COUNTDOWN - (now - start):.0f}s", (0, 255, 255)
                else:
                    if now - last_save >= CAPTURE_INTERVAL:
                        cv2.imwrite(str(CALIB_DIR / f"{prefix}_{saved:03d}.jpg"), frame)
                        saved, last_save = saved + 1, now
                    status, color = f"{saved}/{FRAMES_PER_PHASE}", (0, 255, 0)
                for i, text in enumerate((instruction, status)):
                    cv2.putText(frame, text, (10, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow("Calibration capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    sys.exit("Capture aborted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the zero-shot decision threshold.")
    parser.add_argument("--recapture", action="store_true", help="discard saved frames and record new ones")
    args = parser.parse_args()

    if args.recapture:
        shutil.rmtree(CALIB_DIR, ignore_errors=True)
    if len(list(CALIB_DIR.glob("*.jpg"))) < FRAMES_PER_PHASE * len(PHASES):
        print("Capturing calibration frames — follow the on-screen prompts ('q' aborts).")
        capture_frames()

    session, head = create_session(), ZeroShotHead()
    margins, labels = [], []
    for path in sorted(CALIB_DIR.glob("*.jpg")):
        embeds = session.run(None, {"pixel_values": preprocess(cv2.imread(str(path)))})[0][0]
        margins.append(head.margin(embeds))
        labels.append(path.name.startswith("touching"))
    margins, labels = np.array(margins), np.array(labels)

    mean_not, mean_touch = margins[~labels].mean(), margins[labels].mean()
    if mean_touch <= mean_not:
        sys.exit(f"Classes not separable (mean margin: not={mean_not:+.4f}, touch={mean_touch:+.4f}). "
                 "Try other prompts in export_model.py or --recapture with clearer touches.")
    threshold = (mean_not + mean_touch) / 2

    print(f"Mean margin: not touching={mean_not:+.4f}  touching={mean_touch:+.4f}  → threshold={threshold:+.4f}")
    print(f"Accuracy on calibration frames: uncalibrated={((margins > 0) == labels).mean():.2f}  "
          f"calibrated={((margins > threshold) == labels).mean():.2f}")

    with np.load(HEAD_PATH) as saved:
        arrays = dict(saved)
    arrays["threshold"] = np.float32(threshold)
    np.savez(HEAD_PATH, **arrays)
    print(f"Saved threshold to {HEAD_PATH}")


if __name__ == "__main__":
    main()
