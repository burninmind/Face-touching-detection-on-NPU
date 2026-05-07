"""
Face-touching detection — Windows-native inference on Ryzen AI NPU.

Prerequisites (Windows machine):
  1. Install Ryzen AI SDK 1.7.1 from AMD (provides VitisAI EP + xclbin + vaip_config.json)
  2. pip install numpy opencv-python
     Do NOT pip install onnxruntime — the SDK already installed it with the VitisAI EP.
  3. Copy models/face_touch_siglip_bf16.onnx here (produced by export_model.py).

Usage:
  python inference.py

Press 'q' in the preview window to quit.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONNX_PATH = Path("models/face_touch_siglip_bf16.onnx")

# Default vaip_config.json location after Ryzen AI SDK 1.7.1 installation on Windows.
# Adjust if your SDK is installed elsewhere.
VAIP_CONFIG = Path(r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json")

CLASSES = ["not touching face", "touching face"]

# SigLIP base-patch16-224 normalization: (pixel/255 - 0.5) / 0.5  →  [-1, 1]
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

IMAGE_SIZE = 224
TARGET_FPS = 1          # inference rate (preview window stays at camera FPS)
CAMERA_INDEX = 0        # change if the laptop has multiple cameras


# ---------------------------------------------------------------------------
# ONNX Runtime session
# ---------------------------------------------------------------------------

def create_session() -> ort.InferenceSession:
    if not ONNX_PATH.exists():
        sys.exit(
            f"Model not found: {ONNX_PATH}\n"
            "Run export_model.py first, then copy the .onnx file here."
        )

    providers: list = []
    if VAIP_CONFIG.exists():
        # VitisAI EP config path must be a plain str — Path objects are not accepted
        # by some SDK versions.
        providers.append(("VitisAIExecutionProvider", {"config_file": str(VAIP_CONFIG)}))
        print(f"VitisAI EP enabled — config: {VAIP_CONFIG}")
        print("First run triggers NPU graph compilation (30–90 s). Subsequent runs use cache.")
    else:
        print(
            f"WARNING: vaip_config.json not found at {VAIP_CONFIG}\n"
            "Falling back to CPU. Install Ryzen AI SDK 1.7.1 to enable NPU inference."
        )
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=providers)
    print(f"Active providers: {session.get_providers()}")
    return session


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 OpenCV frame to a [1, 3, 224, 224] float32 tensor."""
    # OpenCV uses BGR; SigLIP expects RGB — conversion is mandatory
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    img = (resized.astype(np.float32) / 255.0 - MEAN) / STD   # → [-1, 1]
    img = img.transpose(2, 0, 1)                                # HWC → CHW
    return np.ascontiguousarray(img[np.newaxis])                # [1, 3, 224, 224]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    session = create_session()

    # cv2.CAP_DSHOW uses Windows DirectShow backend, which avoids the ~800 ms
    # Media Foundation initialization delay present on some Ryzen laptops.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(
            f"Cannot open camera at index {CAMERA_INDEX}.\n"
            "Check Settings > Privacy > Camera on Windows 11 to allow desktop app access."
        )

    # Camera runs at its native rate for a smooth preview window;
    # inference is throttled separately by time.monotonic().
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    label = CLASSES[0]
    conf  = 0.0
    last_infer = 0.0
    infer_interval = 1.0 / TARGET_FPS

    print("Capturing — press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            # Transient grab failure; retry without crashing
            time.sleep(0.05)
            continue

        now = time.monotonic()
        if now - last_infer >= infer_interval:
            pixel_values = preprocess(frame)
            probs = session.run(
                output_names=["probabilities"],
                input_feed={"pixel_values": pixel_values},
            )[0][0]                                             # [2] float32
            idx   = int(np.argmax(probs))
            label = CLASSES[idx]
            conf  = float(probs[idx])
            last_infer = now
            print(f"[{label}]  confidence={conf:.3f}")

        # Overlay result on live preview (color-coded: red = touching, green = safe)
        color = (0, 0, 255) if "touching" in label else (0, 255, 0)
        cv2.putText(
            frame,
            f"{label} ({conf:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )
        cv2.imshow("Face Touching Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
