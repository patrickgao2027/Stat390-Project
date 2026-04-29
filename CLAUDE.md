# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Northwestern University STAT 390 medical imaging project: a binary skin lesion classifier (benign vs. malignant) using dermoscopy images from the ISIC 2019 and 2020 datasets (~57,586 images combined, ~5.3:1 class imbalance).

## Running Experiments

The project uses the Anaconda base environment. Always invoke Python via its full path:

```bash
# Run an experiment (after editing model.py)
C:\Users\Owner\anaconda3\python.exe run.py "description of change"            # status=keep
C:\Users\Owner\anaconda3\python.exe run.py "description of change" --baseline # status=baseline
C:\Users\Owner\anaconda3\python.exe run.py "description of change" --discard  # status=discard
```

Results are logged to `results.tsv` (columns: experiment, roc_auc, recall, status, description, git commit hash).

## Critical Constraints

- **Only `model.py` may be modified** — `prepare.py` and `run.py` are frozen.
- `model.py` must define a `build_model()` function that returns a PyTorch or sklearn-compatible estimator.
- The returned model must implement `predict()` and `predict_proba()`.
- No external datasets beyond ISIC 2019/2020.

## Architecture

**Main workflow:**
- `prepare.py` — Frozen. Loads ISIC CSV metadata, merges datasets, filters indeterminate diagnoses, stratified 80/20 split, subsamples training to 30%, computes class weights (~5.3 for malignant), returns `tf.data.Dataset` pipelines at 128×128.
- `model.py` — Editable stub. Implement `build_model()` here.
- `run.py` — Frozen experiment runner. Calls `load_data()` → `build_model()` → train → `evaluate()` → log to TSV → plot ROC/confusion matrix.

**Reference implementations** (in `skin-lesion-autoresearch/src/`, do not import directly from run.py):
- `src/baseline_model.py` — Simple 3-block CNN from scratch.
- `src/cnn_model.py` — EfficientNet-B0 with two-phase training (frozen backbone, then full fine-tune).
- `src/utils.py` — `SkinLesionDataset`, transforms, metrics, plotting helpers.

**Data pipeline:** TensorFlow `tf.data.Dataset` streams images on-the-fly (never fully loaded into memory). Training augmentation: random flip, rotation, brightness/contrast/saturation. Images normalized to [0, 1].

## Success Criteria

- ROC-AUC ≥ 0.85 on the test set (primary metric — preferred over accuracy due to class imbalance)
- Recall ≥ 0.95 (minimize false negatives for malignant lesions)
- Plain-English summary required for non-technical stakeholders

## Class Imbalance Handling

Pass `pos_weight` to `BCEWithLogitsLoss`:
```python
# class_weight is returned by load_data() as the malignant weight (~5.3)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight]))
```

## Deployment Target (Stretch)

PyTorch → ONNX → TensorFlow → TFLite. See `skin-lesion-autoresearch/deployment/instructions.md`.
