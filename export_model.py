"""
Export the SigLIP vision encoder to FP32 ONNX and save the zero-shot head.

  models/face_touch_siglip_fp32.onnx  pixel_values [1, 3, 224, 224] → image_embeds [1, 768]
  models/face_touch_head.npz          L2-normalized text embeddings + logit_scale

The head (normalize + dot product with the text embeddings) stays out of the
graph: the decision hinges on a ~0.01 gap between two cosine similarities,
too fine for reduced precision. inference.py evaluates it in FP32 numpy.

Usage:
  python export_model.py
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import SiglipModel, SiglipTokenizer

MODEL_ID = "google/siglip-base-patch16-224"
PROMPTS = [
    "a photo of a person not touching their face",
    "a photo of a person touching their face with their hand",
]
ONNX_PATH = Path("models/face_touch_siglip_fp32.onnx")
HEAD_PATH = Path("models/face_touch_head.npz")


class VisionEncoder(torch.nn.Module):
    """Returns SigLIP's image embedding (the pooler output) as a single tensor."""

    def __init__(self, vision_model: torch.nn.Module):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values=pixel_values).pooler_output


def main() -> None:
    print(f"Loading {MODEL_ID} …")
    # Eager attention exports as plain MatMul/Softmax; the SDPA path crashes the
    # legacy TorchScript exporter in older torch builds (e.g. the ryzen-ai env).
    model = SiglipModel.from_pretrained(MODEL_ID, attn_implementation="eager").eval()
    tokenizer = SiglipTokenizer.from_pretrained(MODEL_ID)

    # SigLIP requires fixed-length (max_length) text inputs
    text_inputs = tokenizer(PROMPTS, return_tensors="pt", padding="max_length")
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        if not isinstance(text_embeds, torch.Tensor):   # newer transformers return an output object
            text_embeds = text_embeds.pooler_output
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    ONNX_PATH.parent.mkdir(exist_ok=True)
    np.savez(
        HEAD_PATH,
        text_embeds=text_embeds.numpy().astype(np.float32),
        logit_scale=np.float32(model.logit_scale.exp().item()),   # stored as log; ~117
    )
    print(f"Saved head to {HEAD_PATH}")

    # Static batch of 1: the NPU compiler does not support dynamic shapes.
    encoder = VisionEncoder(model.vision_model).eval()
    dummy = torch.randn(1, 3, 224, 224)
    print(f"Exporting {ONNX_PATH} …")
    torch.onnx.export(
        encoder,
        dummy,
        str(ONNX_PATH),
        opset_version=17,   # first-class LayerNormalization
        input_names=["pixel_values"],
        output_names=["image_embeds"],
    )

    onnx.checker.check_model(str(ONNX_PATH))
    with torch.no_grad():
        expected = encoder(dummy).numpy()
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    actual = sess.run(None, {"pixel_values": dummy.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
    print("ONNX graph valid and matches PyTorch. Next: python calibrate_head.py")


if __name__ == "__main__":
    main()
