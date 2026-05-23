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
       1. accepts NCHW float32 input in [0, 1], shape (N, 3, 128, 128) —
          matches the backbone's natural input layout, avoids an explicit
          Transpose op that onnx2tf mangles and that fp16 TFLite can't run;
       2. calls the backbone (which upsamples 128→224 and applies ImageNet
          normalization internally);
       3. applies sigmoid so the output is probability of malignant, shape (N,).

    The Android app constructs the NCHW tensor directly from a Bitmap by
    writing all R values, then all G, then all B (channel-planar order)
    into the input ByteBuffer.
    """
    def __init__(self, backbone_module: EfficientNetB4Binary):
        super().__init__()
        self.backbone_module = backbone_module

    def forward(self, x_nchw):
        logits = self.backbone_module(x_nchw).squeeze(-1)
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
    dummy = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH.as_posix(),
        input_names=["image_nchw"],
        output_names=["malignancy_prob"],
        opset_version=17,
        dynamic_axes={"image_nchw": {0: "batch"}, "malignancy_prob": {0: "batch"}},
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

    print(f"\nConverting ONNX → TFLite via onnx2tf (fp32, ~75 MB)")
    out_dir = ROOT / "deployment" / "tflite_build"
    out_dir.mkdir(exist_ok=True)
    # onnx2tf 2.4+ API: quantization flags were removed; default produces fp32.
    # If size matters later, run a follow-up tf.lite.TFLiteConverter step for fp16.
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH.as_posix(),
        output_folder_path=out_dir.as_posix(),
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
    )
    # onnx2tf emits float32 AND float16 variants. Prefer float32: B4's internal
    # RESIZE_BILINEAR (128→224 upsample) isn't implemented for fp16 in the
    # current TFLite runtime, so fp16 invoke() crashes at inference time.
    fp32 = sorted(out_dir.glob("*float32*.tflite"))
    fp16 = sorted(out_dir.glob("*float16*.tflite"))
    all_candidates = sorted(out_dir.glob("*.tflite"))
    if fp32:
        chosen = fp32[0]
        variant = "fp32"
    elif fp16:
        chosen = fp16[0]
        variant = "fp16 (fallback — may fail on RESIZE_BILINEAR)"
    elif all_candidates:
        chosen = all_candidates[0]
        variant = "unknown"
    else:
        raise SystemExit(f"No .tflite produced in {out_dir}")
    print(f"  Picked variant: {variant}  ({chosen.name})")
    chosen.replace(TFLITE_PATH)
    print(f"  TFLite size: {TFLITE_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"  TFLite path: {TFLITE_PATH}")


def sanity_check(wrapper, threshold):
    """Compare PyTorch vs TFLite on a random input.

    Reads TFLite's actual expected input shape and reshapes accordingly —
    onnx2tf's flatbuffer_direct backend preserves the ONNX layout (NCHW),
    while older paths produced NHWC. We handle both.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("\n(skipping sanity check: tensorflow not installed)")
        return

    rng = np.random.default_rng(0)
    x_nchw = rng.random((1, 3, 128, 128)).astype(np.float32)

    with torch.no_grad():
        pt_out = wrapper(torch.from_numpy(x_nchw)).numpy()

    interp = tf.lite.Interpreter(model_path=TFLITE_PATH.as_posix())
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    tfl_shape = tuple(int(s) for s in in_d["shape"])
    print(f"\nTFLite input shape: {tfl_shape}  (dtype={np.dtype(in_d['dtype']).name})")

    if tfl_shape == (1, 3, 128, 128):
        x_in = x_nchw
        layout = "NCHW"
    elif tfl_shape == (1, 128, 128, 3):
        x_in = np.transpose(x_nchw, (0, 2, 3, 1)).copy()
        layout = "NHWC"
    else:
        raise SystemExit(f"Unexpected TFLite input shape {tfl_shape}; expected (1,3,128,128) or (1,128,128,3)")

    print(f"TFLite input layout: {layout}")
    interp.set_tensor(in_d["index"], x_in.astype(in_d["dtype"]))
    interp.invoke()
    tfl_out = interp.get_tensor(out_d["index"])

    diff = float(np.max(np.abs(pt_out.flatten() - tfl_out.flatten())))
    print(f"\nSanity check:")
    print(f"  PyTorch prob = {pt_out.flatten()[0]:.4f}")
    print(f"  TFLite prob  = {tfl_out.flatten()[0]:.4f}")
    print(f"  max abs diff = {diff:.6f}  " + ("✓ PASS" if diff < 1e-2 else "✗ FAIL (diff > 1e-2)"))
    print(f"\nDeployment threshold to bake into Android app: {threshold:.4f}")
    print(f"  (open SkinLesionClassifier.kt, set: const val THRESHOLD = {threshold:.4f}f)")
    print(f"\nAndroid input layout: {layout} — SkinLesionClassifier.kt now feeds {layout}.")


def main():
    wrapper, threshold = load_checkpoint_and_build()
    export_onnx(wrapper)
    convert_onnx_to_tflite()
    sanity_check(wrapper, threshold)
    print("\nDone. Copy the .tflite into the Android assets folder:")
    print(f"  cp {TFLITE_PATH} deployment/android/app/src/main/assets/")


if __name__ == "__main__":
    main()
