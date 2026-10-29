"""
Live face-touching detection on the Ryzen AI NPU.

Each frame is classified as "no face detected", "not touching face" or
"touching face". YuNet (CPU) checks for a face first; only then does the
SigLIP encoder run on the NPU — the VitisAI EP compiles the FP32 model to
BF16 (~5 min on first launch, then cached) — followed by the zero-shot head
in FP32 numpy.

Usage:
  python inference.py      # press 'q' in the preview window to quit
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ONNX_PATH = Path("models/face_touch_siglip_fp32.onnx")
HEAD_PATH = Path("models/face_touch_head.npz")
NPU_CACHE_DIR = Path("models/npu_cache")
FACE_MODEL_PATH = Path("models/face_detection_yunet.onnx")

# YuNet scores your face ≥0.84 even with a hand over it, and leftover
# background ≤0.18 with the face masked out.
FACE_SCORE_THRESHOLD = 0.6

NO_FACE, NOT_TOUCHING, TOUCHING = "no face detected", "not touching face", "touching face"
COLORS = {NO_FACE: (160, 160, 160), NOT_TOUCHING: (0, 255, 0), TOUCHING: (0, 0, 255)}   # BGR

IMAGE_SIZE = 224
TARGET_FPS = 1      # inference rate; the preview runs at camera rate
CAMERA_INDEX = 0


def create_session() -> ort.InferenceSession:
    """Encoder session on the NPU, falling back to CPU without the Ryzen AI SDK."""
    if not ONNX_PATH.exists():
        sys.exit(f"Model not found: {ONNX_PATH} — run export_model.py first.")

    providers: list = ["CPUExecutionProvider"]
    if "VitisAIExecutionProvider" in ort.get_available_providers():
        # Cache key tracks the model file, so a re-export forces a recompile.
        stat = ONNX_PATH.stat()
        providers.insert(0, ("VitisAIExecutionProvider", {
            "cache_dir": str(NPU_CACHE_DIR.resolve()),
            "cache_key": f"{ONNX_PATH.stem}_{stat.st_size}_{int(stat.st_mtime)}",
        }))
        print("Using NPU (first launch compiles the model, ~5 min).")
    else:
        print("WARNING: VitisAIExecutionProvider unavailable — running on CPU. "
              "Activate the Ryzen AI SDK env for NPU inference.")
    return ort.InferenceSession(str(ONNX_PATH), providers=providers)


class ZeroShotHead:
    """SigLIP zero-shot head with a calibrated decision threshold."""

    def __init__(self, path: Path = HEAD_PATH):
        with np.load(path) as head:
            self.text_embeds = head["text_embeds"]      # [2, 768]: not touching, touching
            self.logit_scale = float(head["logit_scale"])
            self.calibrated = "threshold" in head.files
            self.threshold = float(head["threshold"]) if self.calibrated else 0.0

    def margin(self, image_embeds: np.ndarray) -> float:
        """Cosine similarity to the "touching" prompt minus the "not touching" one."""
        cos = self.text_embeds @ (image_embeds / np.linalg.norm(image_embeds))
        return float(cos[1] - cos[0])

    def __call__(self, image_embeds: np.ndarray) -> float:
        """Probability of touching; 0.5 exactly at the threshold."""
        return float(1 / (1 + np.exp(-self.logit_scale * (self.margin(image_embeds) - self.threshold))))


class FaceDetector:
    """YuNet face detector (OpenCV, CPU, ~17 ms per 640x480 frame)."""

    def __init__(self, path: Path = FACE_MODEL_PATH):
        if not path.exists():
            sys.exit(f"Face detector not found: {path} — run export_model.py first.")
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (0, 0), FACE_SCORE_THRESHOLD)

    def __call__(self, frame: np.ndarray) -> bool:
        """Whether the frame contains a face."""
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        return self.detector.detect(frame)[1] is not None


def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR uint8 frame → [1, 3, 224, 224] float32 in [-1, 1] (SigLIP normalization)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])


def open_camera() -> cv2.VideoCapture:
    # DirectShow avoids Media Foundation's slow startup on some Ryzen laptops
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {CAMERA_INDEX}. "
                 "Check Settings > Privacy > Camera allows desktop apps.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main() -> None:
    session = create_session()
    head = ZeroShotHead()
    if not head.calibrated:
        print("WARNING: head is not calibrated — run calibrate_head.py.")
    has_face = FaceDetector()
    cap = open_camera()

    status, text, last_infer = NO_FACE, NO_FACE, 0.0
    print("Running — press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        now = time.monotonic()
        if now - last_infer >= 1 / TARGET_FPS:
            last_infer = now
            if has_face(frame):
                embeds = session.run(None, {"pixel_values": preprocess(frame)})[0][0]
                p_touch = head(embeds)
                status = TOUCHING if p_touch > 0.5 else NOT_TOUCHING
                text = f"{status} ({max(p_touch, 1 - p_touch):.2f})"
            else:
                status = text = NO_FACE
            print(text)

        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[status], 2)
        cv2.imshow("Face Touching Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
