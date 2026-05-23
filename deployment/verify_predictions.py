"""
Single-image sanity check: feed one JPEG through both PyTorch and TFLite,
verify they produce the same malignancy probability within tolerance.

Usage:
    python deployment/verify_predictions.py path/to/image.jpg
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import EfficientNetB4Binary  # noqa: E402

CHECKPOINT_PATH = ROOT / "model_checkpoint.pt"
TFLITE_PATH = ROOT / "deployment" / "skin_lesion_b4.tflite"


def preprocess(image_path: Path) -> np.ndarray:
    """Mimic the prepare.py preprocessing: resize to 128×128, float32 in [0,1],
    return NHWC (matches the TFLite model's input layout)."""
    img = Image.open(image_path).convert("RGB").resize((128, 128), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0       # (128, 128, 3) NHWC
    return arr[None, ...]                                  # (1, 128, 128, 3) NHWC


def pytorch_prob(x_nhwc: np.ndarray) -> float:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    module = EfficientNetB4Binary()
    module.load_state_dict(ckpt["state_dict"])
    module.eval()
    # PyTorch backbone wants NCHW — transpose from the NHWC preprocess.
    x_nchw = torch.from_numpy(x_nhwc).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        logits = module(x_nchw).squeeze(-1)
        p = torch.sigmoid(logits).item()
    return p


def tflite_prob(x_nhwc: np.ndarray) -> float:
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH.as_posix())
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    interp.set_tensor(in_d["index"], x_nhwc.astype(in_d["dtype"]))
    interp.invoke()
    return float(interp.get_tensor(out_d["index"]).flatten()[0])


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python deployment/verify_predictions.py <image.jpg>")
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise SystemExit(f"No file at {image_path}")

    x = preprocess(image_path)
    print(f"Image: {image_path.name}  shape={x.shape}  range=[{x.min():.3f}, {x.max():.3f}]")

    pt_p = pytorch_prob(x)
    tfl_p = tflite_prob(x)
    diff = abs(pt_p - tfl_p)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    threshold = float(ckpt["threshold"])

    print(f"\nPyTorch P(malignant)  = {pt_p:.4f}")
    print(f"TFLite  P(malignant)  = {tfl_p:.4f}")
    print(f"Difference            = {diff:.6f}  " + ("✓ PASS" if diff < 1e-2 else "✗ FAIL"))
    print(f"\nDeployment decision (threshold = {threshold:.4f}):")
    label = "MALIGNANT — recommend dermatologist consult" if tfl_p >= threshold else "benign"
    print(f"  → {label}")


if __name__ == "__main__":
    main()
