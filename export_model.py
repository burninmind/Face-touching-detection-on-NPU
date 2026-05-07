"""
One-time script: exports google/siglip-base-patch16-224 to ONNX at BF16 precision.

The exported model:
  - Input:  pixel_values  [batch, 3, 224, 224]  float32
  - Output: probabilities [batch, 2]             float32  (sigmoid scores)
  - Classes (index 0 / 1): "not touching face" / "touching face"

Text embeddings for both classes are baked as constants in the graph,
so no text pipeline is needed at inference time.

Run once, then copy models/face_touch_siglip_bf16.onnx to the Windows machine.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from transformers import SiglipModel, SiglipProcessor

MODEL_ID = "google/siglip-base-patch16-224"
CLASSES = ["not touching face", "touching face"]
OUTPUT_DIR = Path("models")
ONNX_PATH = OUTPUT_DIR / "face_touch_siglip_bf16.onnx"
OPSET = 17


class SigLIPFaceTouchClassifier(nn.Module):
    """
    Vision-only wrapper around SigLIP for two-class zero-shot classification.

    Accepts FP32 pixel values, casts internally to BF16 for NPU-friendly
    computation, then returns FP32 sigmoid probabilities.  Text embeddings
    and the logit scale/bias are baked as buffers (ONNX initializer constants).
    """

    def __init__(
        self,
        vision_model: nn.Module,
        visual_projection: nn.Module,
        text_embeds_bf16: torch.Tensor,
        logit_scale: float,
        logit_bias: float,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.register_buffer("text_embeds", text_embeds_bf16)                        # [2, 768]
        self.register_buffer("logit_scale", torch.tensor(logit_scale, dtype=torch.bfloat16))
        self.register_buffer("logit_bias",  torch.tensor(logit_bias,  dtype=torch.bfloat16))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Cast FP32 input to BF16 — the Cast node appears in the ONNX graph,
        # keeping the external interface FP32 (numpy-compatible).
        x = pixel_values.to(torch.bfloat16)

        vision_out = self.vision_model(pixel_values=x)
        img_feat = self.visual_projection(vision_out.pooler_output)       # [B, 768] BF16
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)         # L2 normalize

        # SigLIP zero-shot: cosine-sim scaled by logit_scale + logit_bias, then sigmoid
        logits = img_feat @ self.text_embeds.T                             # [B, 2] BF16
        logits = logits * self.logit_scale + self.logit_bias

        return torch.sigmoid(logits).to(torch.float32)                    # [B, 2] FP32


def main() -> None:
    print(f"Loading {MODEL_ID} …")
    model = SiglipModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    processor = SiglipProcessor.from_pretrained(MODEL_ID)
    model.eval()

    # --- Precompute text embeddings in FP32 then bake as BF16 ---
    print("Computing text embeddings …")
    text_inputs = processor(
        text=CLASSES,
        return_tensors="pt",
        padding="max_length",   # SigLIP requires fixed-length input sequences
    )
    with torch.no_grad():
        # get_text_features() does NOT apply L2 normalization — must do it here
        text_features = model.get_text_features(**text_inputs)            # [2, 768] FP32
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features_bf16 = text_features.to(torch.bfloat16)
    print(f"  text_embeds shape: {text_features_bf16.shape}")

    # logit_scale is stored as its log; exp() gives the actual scale multiplier
    logit_scale = model.logit_scale.exp().item()
    logit_bias  = model.logit_bias.item()
    print(f"  logit_scale={logit_scale:.4f}, logit_bias={logit_bias:.4f}")

    # --- Build wrapper with BF16 vision backbone ---
    wrapper = SigLIPFaceTouchClassifier(
        vision_model=model.vision_model.to(torch.bfloat16).eval(),
        visual_projection=model.visual_projection.to(torch.bfloat16).eval(),
        text_embeds_bf16=text_features_bf16,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
    )

    # --- Dry-run to catch shape or dtype issues before export ---
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        pytorch_out = wrapper(dummy)
    assert pytorch_out.shape == (1, 2), f"Expected (1,2), got {pytorch_out.shape}"
    assert pytorch_out.dtype == torch.float32, f"Expected float32, got {pytorch_out.dtype}"
    assert torch.all((pytorch_out >= 0) & (pytorch_out <= 1)), "sigmoid outputs must be in [0,1]"
    print(f"Dry-run OK — class probs: {pytorch_out.squeeze().tolist()}")

    # --- ONNX export ---
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Exporting to {ONNX_PATH} (opset {OPSET}) …")
    torch.onnx.export(
        wrapper,
        dummy,
        str(ONNX_PATH),
        opset_version=OPSET,        # 17 required for first-class LayerNormalization
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamic_axes={
            "pixel_values":  {0: "batch"},
            "probabilities": {0: "batch"},
        },
        do_constant_folding=True,   # folds logit_scale/bias and text_embeds constants
    )
    print("Export done.")

    # --- ONNX graph validation ---
    print("Validating ONNX graph …")
    model_onnx = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model_onnx)
    print("  ONNX checker: OK")

    # Numerical sanity-check against PyTorch reference (CPU EP)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["probabilities"], {"pixel_values": dummy.numpy()})[0]
    np.testing.assert_allclose(ort_out, pytorch_out.numpy(), atol=1e-2)
    # atol=1e-2 is appropriate: BF16 has 7-bit mantissa (~0.5% relative error)
    print("  Numerical match within BF16 tolerance: OK")

    print(f"\nAll done. Model saved to: {ONNX_PATH.resolve()}")
    print("Copy this file to the Windows Ryzen AI machine, then run inference.py.")


if __name__ == "__main__":
    main()
