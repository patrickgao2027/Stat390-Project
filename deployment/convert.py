"""
End-to-end PyTorch → ONNX → TFLite conversion.

Loads model_checkpoint.pt (saved by SkinLesionModel during training), wraps the
EfficientNet-B4 module with sigmoid (so TFLite outputs probability not logits),
exports ONNX, converts to TFLite with fp16 quantization, and verifies that the
TFLite outputs agree with PyTorch on a dummy input.

Outputs:
    deployment/skin_lesion_b4.onnx
    deployment/skin_lesion_b4.tflite

Run:
    python deployment/convert.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Project root on sys.path so we can import the model definition.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import EfficientNetB4Binary  # noqa: E402

CHECKPOINT_PATH = ROOT / "model_checkpoint.pt"
ONNX_PATH = ROOT / "deployment" / "skin_lesion_b4.onnx"
TFLITE_PATH = ROOT / "deployment" / "skin_lesion_b4.tflite"


class InferenceWrapper(nn.Module):
    """Wrap EfficientNetB4Binary so the exported graph:
       1. accepts NHWC float32 input in [0, 1] (matches what the app produces
          from a Bitmap), shape (N, 128, 128, 3);
       2. internally permutes NHWC → NCHW and calls the backbone (which itself
          upsamples 128→224 and applies ImageNet normalization);
       3. applies sigmoid so the output is probability of malignant, shape (N,).
    """
    def __init__(self, backbone_module: EfficientNetB4Binary):
        super().__init__()
        self.backbone_module = backbone_module

    def forward(self, x_nhwc):
        # NHWC → NCHW
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        logits = self.backbone_module(x).squeeze(-1)
        return torch.sigmoid(logits)


def load_checkpoint_and_build():
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(
            f"No checkpoint at {CHECKPOINT_PATH}. "
            f"Train with SAVE_CHECKPOINT=True first."
        )
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    threshold = float(ckpt["threshold"])
    raw_threshold = float(ckpt.get("raw_threshold", float("nan")))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    print(f"  threshold      = {threshold:.4f}")
    print(f"  raw_threshold  = {raw_threshold:.4f}")
    print(f"  safety_margin  = {ckpt.get('safety_margin', float('nan')):.4f}")
    print(f"  use_tta        = {ckpt.get('use_tta', 'n/a')}")

    backbone = EfficientNetB4Binary()
    backbone.load_state_dict(ckpt["state_dict"])
    backbone.eval()
    wrapper = InferenceWrapper(backbone).eval()
    return wrapper, threshold


def export_onnx(wrapper):
    print(f"\nExporting ONNX → {ONNX_PATH}")
    dummy = torch.zeros(1, 128, 128, 3, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH.as_posix(),
        input_names=["image_nhwc"],
        output_names=["malignancy_prob"],
        opset_version=17,
        dynamic_axes={"image_nhwc": {0: "batch"}, "malignancy_prob": {0: "batch"}},
    )
    print(f"  ONNX size: {ONNX_PATH.stat().st_size / 1e6:.1f} MB")


def convert_onnx_to_tflite():
    """ONNX → TFLite using onnx2tf (more reliable than onnx-tf for modern ops).

    onnx2tf converts ONNX directly to a TF SavedModel that uses NHWC natively,
    then runs the TFLite converter with fp16 quantization.
    """
    try:
        import onnx2tf
    except ImportError:
        raise SystemExit("pip install onnx2tf  (required for ONNX→TFLite)")

    print(f"\nConverting ONNX → TFLite via onnx2tf (fp16 quantization)")
    out_dir = ROOT / "deployment" / "tflite_build"
    out_dir.mkdir(exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH.as_posix(),
        output_folder_path=out_dir.as_posix(),
        output_signaturedefs=True,
        output_h5=False,
        copy_onnx_input_output_names_to_tflite=True,
        # fp16 quantization halves model size with negligible accuracy loss
        output_dynamic_range_quantized_tflite=False,
        output_float16_quantized_tflite=True,
        not_use_onnxsim=False,
        non_verbose=False,
    )
    # onnx2tf emits multiple .tflite variants; pick the fp16 one.
    candidates = sorted(out_dir.glob("*float16*.tflite"))
    if not candidates:
        candidates = sorted(out_dir.glob("*.tflite"))
    if not candidates:
        raise SystemExit(f"No .tflite produced in {out_dir}")
    chosen = candidates[0]
    chosen.replace(TFLITE_PATH)
    print(f"  TFLite size: {TFLITE_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"  TFLite path: {TFLITE_PATH}")


def sanity_check(wrapper, threshold):
    """Compare PyTorch vs TFLite on a random NHWC input."""
    try:
        import tensorflow as tf
    except ImportError:
        print("\n(skipping sanity check: tensorflow not installed)")
        return

    rng = np.random.default_rng(0)
    x_np = rng.random((1, 128, 128, 3)).astype(np.float32)

    with torch.no_grad():
        pt_out = wrapper(torch.from_numpy(x_np)).numpy()

    interp = tf.lite.Interpreter(model_path=TFLITE_PATH.as_posix())
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    interp.set_tensor(in_d["index"], x_np.astype(in_d["dtype"]))
    interp.invoke()
    tfl_out = interp.get_tensor(out_d["index"])

    diff = float(np.max(np.abs(pt_out.flatten() - tfl_out.flatten())))
    print(f"\nSanity check:")
    print(f"  PyTorch prob = {pt_out.flatten()[0]:.4f}")
    print(f"  TFLite prob  = {tfl_out.flatten()[0]:.4f}")
    print(f"  max abs diff = {diff:.6f}  " + ("✓ PASS" if diff < 1e-2 else "✗ FAIL (diff > 1e-2)"))
    print(f"\nDeployment threshold to bake into Android app: {threshold:.4f}")
    print(f"  (open SkinLesionClassifier.kt, set: private const val THRESHOLD = {threshold:.4f}f)")


def main():
    wrapper, threshold = load_checkpoint_and_build()
    export_onnx(wrapper)
    convert_onnx_to_tflite()
    sanity_check(wrapper, threshold)
    print("\nDone. Copy the .tflite into the Android assets folder:")
    print(f"  cp {TFLITE_PATH} deployment/android/app/src/main/assets/")


if __name__ == "__main__":
    main()
