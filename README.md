# Face Touching Detection on NPU

Real-time face-touching detection from a laptop webcam, using [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) zero-shot image classification on the NPU of AMD Ryzen AI 300-series laptops (developed on a Ryzen AI 7 350). Each frame is labeled **no face detected**, **not touching face** or **touching face**.

## How it works

```
CPU           webcam frame → YuNet face detector ──no face──→ "no face detected"   ~17 ms
                                   │ face
NPU (BF16)                         └→ SigLIP ViT-B/16 vision encoder → image embedding [768]   ~40 ms
CPU (FP32)    cosine similarity to two text prompts → margin vs. calibrated threshold → touching / not touching
```

- **Face gate:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) runs first, so the classifier never has to judge a frame with no one in it. It still finds the face with a hand over it: it detected 100% of the calibration frames where the face was being touched, while OpenCV's Haar cascade found only 68%. When no face is found, the NPU step is skipped.
- **Zero-shot:** no training. The two classes are described by text prompts ("a photo of a person (not) touching their face"), and their embeddings are precomputed at export time.
- **NPU without quantization:** the FP32 ONNX model is deployed as-is. The VitisAI Execution Provider compiles float models to BF16 with its VAIML compiler and runs the whole encoder on the NPU as one fused node. That's about 4× faster than the CPU (~155 ms), and the embeddings match FP32 at 0.999 cosine similarity.
- **Calibrated threshold:** SigLIP's "not touching" prompt scores higher on almost every frame. The *gap* between the two prompt scores still separates the classes, but it's offset. `calibrate_head.py` measures that offset on ~100 frames of you, which raises accuracy from ~53% to ~97%.
- **Head in FP32 on the CPU:** the decision depends on a ~0.01 difference between two cosine similarities, which is too fine for reduced precision. It's only two 768-length dot products, so it costs nothing.

## Requirements

- AMD Ryzen AI 300-series laptop (Strix or Krackan Point NPU)
- Windows 11 with the [Ryzen AI SDK 1.7.1](https://ryzenai.docs.amd.com) installed; it provides the `ryzen-ai-1.7.1` conda env
- A webcam

## Setup

All steps run in the SDK's conda env:

```bash
conda activate ryzen-ai-1.7.1
pip install transformers opencv-python   # if not already present
```

> **Don't `pip install onnxruntime`** into this env. The SDK ships its own build with the VitisAI Execution Provider, and the PyPI package would silently replace it, which means no NPU.

### 1. Export the model

```bash
python export_model.py
```

This downloads SigLIP (~800 MB from Hugging Face) and YuNet, and writes:

```
models/face_touch_siglip_fp32.onnx   vision encoder (~370 MB)
models/face_touch_head.npz           text embeddings + logit scale
models/face_detection_yunet.onnx     face detector (~230 KB)
```

The export also works in any other env with `torch`, `transformers`, `onnx` and `onnxruntime` (see `requirements.txt`). If you export on another machine, copy `models/` over.

### 2. Calibrate

```bash
python calibrate_head.py
```

A preview window opens. Sit naturally **without** touching your face, then, when prompted, touch your face in varied ways (chin, cheek, eyes, nose). The frames are saved to `calibration_frames/` and reused on later runs. Use `--recapture` to record a fresh set, for example in different lighting.

The first run also compiles the model for the NPU, which takes about 5 minutes.

### 3. Run

```bash
python inference.py
```

A frame is classified once per second, with the result overlaid on the live preview: gray means no face detected, green means not touching, red means touching. Press `q` to quit.

## Notes

- **NPU compile cache:** the compiled model is cached in `models/npu_cache/`, keyed to the model file. Later launches take seconds, and re-exporting triggers one recompile.
- **CPU fallback:** if the VitisAI Execution Provider isn't available (wrong env), inference runs on the CPU and prints a warning.
- **Checking NPU offload:** `session.get_providers()` lists VitisAI even when it runs nothing, so look at latency instead: ~40 ms per frame on the NPU vs ~155 ms on the CPU.
- **Why not INT8?** Quark-quantized models were tried. The EP's default INT8 flow only matches CNN patterns, so for this ViT every MatMul and LayerNorm stayed on the CPU. The `XINT8` config also wrecked the embeddings (0.28 cosine similarity to FP32).
- **Face threshold:** `FACE_SCORE_THRESHOLD = 0.6` in `inference.py`. Real faces scored at least 0.84, even while being touched, and the background scored at most 0.18 with the face masked out. Lower it if faces at an angle or far from the camera get missed.
- **Prompt tuning:** to change the class descriptions, edit `PROMPTS` in `export_model.py`, then re-export and re-calibrate.
- **Camera:** if the camera won't open, allow desktop apps under *Settings > Privacy > Camera*, or change `CAMERA_INDEX` in `inference.py`.

## Files

| File | Purpose |
|------|---------|
| `export_model.py` | Export the SigLIP vision encoder to ONNX, save the zero-shot head, download YuNet |
| `calibrate_head.py` | Capture labeled webcam frames and fit the decision threshold |
| `inference.py` | Live webcam detection on the NPU |
| `requirements.txt` | Dependencies for exporting outside the Ryzen AI env |
