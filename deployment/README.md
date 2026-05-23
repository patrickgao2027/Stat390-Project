# Deployment — PyTorch → TFLite → Android (S22+)

End-to-end pipeline that takes the saved PyTorch checkpoint from training and produces a TFLite model bundled into a runnable Android app for the Samsung S22+.

## Pipeline overview

```
model_checkpoint.pt   (PyTorch, ~75 MB)
    ↓  convert.py  (torch.onnx.export → onnx2tf → TFLite fp16)
skin_lesion_b4.tflite (~40 MB after fp16 quantization)
    ↓  copy into android/app/src/main/assets/
Android app          (Kotlin + CameraX + TFLite + NNAPI delegate)
```

## Prerequisites

```powershell
# Python conversion side (anaconda base)
pip install onnx onnx2tf tensorflow tflite-runtime
# Android side
# Install Android Studio (https://developer.android.com/studio)
# Open the android/ folder as a project; SDK auto-syncs.
```

## Step 1 — Convert PyTorch checkpoint to TFLite

From project root:

```powershell
cd C:\Users\Owner\Documents\Stat390-Project
python deployment/convert.py
```

This:
1. Loads `model_checkpoint.pt` (best-of-N saved during training)
2. Wraps the module with sigmoid (so TFLite outputs probability, not logits)
3. Exports ONNX at `deployment/skin_lesion_b4.onnx`
4. Converts ONNX → TFLite (fp16 quantization) at `deployment/skin_lesion_b4.tflite`
5. Runs a sanity check: feeds the same dummy input through PyTorch and TFLite, verifies outputs agree within 1e-3
6. Prints the **deployment threshold** to bake into the Android app

Expected output:
```
Loaded checkpoint: threshold=0.0500, raw_threshold=0.1247
Exported ONNX → deployment/skin_lesion_b4.onnx
Converted to TFLite → deployment/skin_lesion_b4.tflite (size: 41.2 MB)
Sanity check: PyTorch prob=0.7345, TFLite prob=0.7341, diff=0.0004 ✓
Use threshold = 0.0500 in Android app (SkinLesionClassifier.kt: THRESHOLD constant)
```

## Step 2 — Bundle TFLite into Android app

```powershell
cp deployment/skin_lesion_b4.tflite deployment/android/app/src/main/assets/
```

Then open `deployment/android/` in Android Studio. Verify `SkinLesionClassifier.kt` has `THRESHOLD = 0.05f` (from convert.py output).

## Step 3 — Run on S22+

1. Enable Developer Mode on the S22+ (Settings → About → tap Build Number 7×)
2. Enable USB Debugging in Developer Options
3. Plug into your PC; Android Studio detects the device
4. Press Run → app installs and launches

App flow:
- Opens camera preview
- "Capture & Analyze" button takes a photo, runs inference (~2-5 sec with TTA, ~0.5-1 sec without)
- Result displays: "Likely benign" or "**Recommend dermatologist consultation**" + probability
- Disclaimer: "Screening aid only — not a diagnostic tool"

## TTA toggle

In `SkinLesionClassifier.kt`, set `USE_TTA = true` for higher accuracy (~3-5 sec) or `false` for speed (~0.5-1 sec, ~0.02 recall loss).

## Inference performance notes (S22+)

| Mode | Backend | Per-frame latency |
|---|---|---|
| Single view, NNAPI | NPU | ~0.4-0.7 sec |
| Single view, GPU delegate | Adreno | ~0.6-1.0 sec |
| Single view, CPU | CPU | ~1.5-3 sec |
| 4-view TTA, NNAPI | NPU | ~1.5-3 sec |
| 4-view TTA, GPU | Adreno | ~2.5-4 sec |

The app prefers NNAPI → GPU → CPU automatically.

## Sanity check after deployment

Run `python deployment/verify_predictions.py path/to/test_image.jpg` to confirm a single inference matches what the PyTorch model produces. Should give the same probability ± 1e-3.

## Files in this directory

| File | Purpose |
|---|---|
| `convert.py` | PyTorch → ONNX → TFLite + sanity check |
| `verify_predictions.py` | Single-image sanity check (PyTorch vs TFLite) |
| `android/` | Kotlin + CameraX + TFLite app source tree |

## Known limits

- **Pipeline ceiling**: model trained on 30%-subsampled, 128×128-source images. Native S22+ camera resolution is downscaled to 128×128 inside the app before inference; you don't gain accuracy from higher-resolution captures.
- **No medical certification**: this is a screening aid. The on-app disclaimer text makes this explicit.
- **TTA is approximate on TFLite**: the converted model handles a single forward pass. TTA = the app runs inference 4× with flipped Bitmaps and averages, not a single fused TFLite op.
