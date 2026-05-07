# Face Touching Detection on NPU

Real-time face-touching detection using [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) zero-shot image classification, accelerated by the NPU on AMD Ryzen AI 7 350 (Strix Point) Copilot+ PCs.

## How it works

The pipeline has two phases:

**Export (one-time)** — `export_model.py` downloads `google/siglip-base-patch16-224`, precomputes text embeddings for the two classes, and exports a self-contained ONNX model at BF16 precision. The text pipeline is baked in as constants; only the vision encoder runs at inference time.

**Inference** — `inference.py` opens the laptop camera, runs the ONNX model on the NPU via the VitisAI Execution Provider at 1 FPS, and overlays the result on a live preview window.

```
Input: pixel_values [1, 3, 224, 224]  FP32
  → Cast to BF16
  → SigLIP ViT-B/16 vision encoder    (all weights BF16)
  → visual_projection
  → L2-normalize
  → dot product with baked text embeddings ["not touching face", "touching face"]
  → sigmoid
  → Cast to FP32
Output: probabilities [1, 2]           FP32
```

## Requirements

- AMD Ryzen AI 7 350 (or any Ryzen AI 300-series / Strix Point)
- [Ryzen AI SDK 1.7.1](https://ryzenai.docs.amd.com) installed on Windows
- Python 3.10+

## Setup

### Step 1 — Export the model

Run this anywhere that has PyTorch (WSL2, Linux, or Windows):

```bash
pip install -r requirements.txt
python export_model.py
```

This downloads the model (~400 MB from HuggingFace) and produces:

```
models/face_touch_siglip_bf16.onnx   (~350 MB)
```

### Step 2 — Copy to the Windows machine

If you exported in WSL2 or on another machine, copy `models/face_touch_siglip_bf16.onnx` to the same directory on the Windows machine.

### Step 3 — Install inference dependencies on Windows

> **Important:** Run `pip install -r requirements.txt` **before** installing the Ryzen AI SDK, or at minimum before running it again afterward. The SDK overwrites the PyPI `onnxruntime` with its own build (which includes the VitisAI Execution Provider). Running `pip install -r requirements.txt` *after* the SDK re-installs the vanilla PyPI version and silently removes NPU support.

```bash
pip install -r requirements.txt
# then install the Ryzen AI SDK 1.7.1
```

### Step 4 — Run

```bash
python inference.py
```

Press `q` to quit.

## Notes

- **First launch is slow** — the VitisAI EP compiles the ONNX graph for the NPU on first run (30–90 s). Subsequent launches load from cache and start in ~2 s.
- **CPU fallback** — if the Ryzen AI SDK is not installed or `vaip_config.json` is not found, inference falls back to CPU automatically.
- **Camera index** — if `inference.py` cannot open the camera, check `Settings > Privacy > Camera` on Windows 11 to allow desktop app access, or change `CAMERA_INDEX` in `inference.py`.
- **Prompt tuning** — if accuracy is poor, edit the `CLASSES` list in `export_model.py` with longer descriptions and re-run the export. No retraining is needed.

## File overview

| File | Purpose |
|------|---------|
| `export_model.py` | Export SigLIP to ONNX BF16 (run once) |
| `inference.py` | Live camera inference on NPU |
| `requirements.txt` | Python dependencies |
