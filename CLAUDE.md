# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Northwestern STAT 390 medical imaging capstone: a binary skin lesion classifier (benign vs. malignant) on the combined ISIC 2019 + 2020 datasets (~57,586 images, ~5.3:1 class imbalance favoring benign). Iteration is auto-research style — edit one file, run one command, log one row to `results.tsv`, decide keep / discard / revert.

## Critical Constraints

- **Only `model.py` may be modified.** `prepare.py` and `run.py` are frozen — never edit them.
- `model.py` must define `build_model()` returning an object with `fit(train_ds, epochs, steps_per_epoch, class_weight)`, `predict(x_ds) -> int labels`, and `predict_proba(x_ds) -> [n, 2]` (col 1 = P(malignant)).
- No external datasets beyond ISIC 2019/2020.
- **Operate only in the main project root** (`C:\Users\Owner\Documents\Stat390-Project`). Git worktrees under `.claude/worktrees/` cannot run training because the data CSVs and ISIC image folders are gitignored — they exist only in the main root. Edits to a worktree's `model.py` never reach the trainer.

## Running Experiments

The project uses the Anaconda base environment. Always invoke Python via its full path, and use `-u` for unbuffered stdout (otherwise long training runs look like they've stalled because per-epoch lines stay block-buffered until the buffer fills):

```bash
C:\Users\Owner\anaconda3\python.exe -u run.py "description Nth iteration"             # status=keep (default)
C:\Users\Owner\anaconda3\python.exe -u run.py "description" --baseline                # status=baseline
C:\Users\Owner\anaconda3\python.exe -u run.py "description" --discard                 # status=discard
```

Training time on the RTX 4050: ~50–60 min for EfficientNet-B2 at 10 epochs, ~22–45 min for B0/AlexNet. Plan budgets accordingly.

`results.tsv` columns: `experiment` (short git hash at run time), `roc_auc`, `accuracy`, `recall`, `precision`, `train_time_s`, `status`, `description`. Note: the commit hash logged is the *parent* commit at run time — uncommitted `model.py` changes still log against the parent's hash, so "keep" rows can share a hash with "baseline" rows.

## Architecture

### Two-file model contract

```
prepare.py (frozen)  ──────►  load_data()
                                  │
                                  ▼
model.py (editable)  ──────►  build_model()  ──┐
                                                ▼
torch_adapter.py     ──────►  BaseTorchModel (predict, predict_proba)
                                                │
                                                ▼
run.py (frozen)      ──────►  fit, evaluate, log to results.tsv
```

- **`prepare.py`** — loads ISIC CSV metadata, merges 2019+2020, filters indeterminate diagnoses, **stratified 80/20 split** seeded by `RANDOM_STATE=67`, subsamples training to 30%, returns `tf.data.Dataset` pipelines yielding NHWC float32 in `[0, 1]` at **128×128**, plus `class_weight` dict (~5.31 for malignant). Note: the seed in `prepare.py` only seeds the data pipeline — PyTorch model init and cuDNN ops are *not* seeded, which is why repeated runs of identical code produce different fits.
- **`torch_adapter.py`** — `BaseTorchModel` adapter that wraps a `nn.Module` producing logits. Handles the TF→Torch conversion (`NHWC → NCHW`), the BCEWithLogitsLoss training loop with `pos_weight`, and exposes `predict()` / `predict_proba()`. **`predict()` uses a hardcoded `threshold = 0.5`** unless the subclass overrides it. Most architectural choices and any threshold tuning logic live in `model.py` subclasses, not here.
- **`model.py`** — must subclass `BaseTorchModel`, override `_build_module()` to return the `nn.Module`, and may override `fit()` for two-phase training, threshold calibration, etc. The `nn.Module` typically upscales the 128×128 input to **224×224** with `F.interpolate` and applies ImageNet normalization before passing to a pretrained torchvision backbone.
- **`run.py`** — calls `load_data()` → `build_model()` → `model.fit(...)` → `evaluate(model, test_ds, y_test)` → log row → save ROC + confusion-matrix plots.

### Class imbalance handling

`load_data()` returns `class_weight = {0: 1.0, 1: ~5.31}` automatically. The adapter's `fit()` reads `class_weight[1]` into `BCEWithLogitsLoss(pos_weight=...)`. Any `model.py` subclass that overrides `fit()` should preserve this — pass a custom `class_weight` dict (e.g. `{0: 1.0, 1: 10.0}`) to push recall further by amplifying the malignant penalty in the loss.

### Per-iteration workflow

1. Edit `model.py` (one variable at a time — confounded multi-change runs are uninterpretable).
2. Run `run.py` with a description that includes the iteration number ("Nth iteration").
3. Check the new row in `results.tsv`.
4. If improved: `git add model.py results.tsv && git commit -m "iter N: <change>" && git push`
5. If worse: `git checkout model.py` to revert.

## Success Criteria

- ROC-AUC ≥ 0.85 on the test set (primary — preferred over accuracy due to class imbalance)
- Recall ≥ 0.95 (minimize false negatives for malignant lesions)
- Plain-English summary required for non-technical stakeholders

## Other directories

- **`week4/`** — Week 4 deliverable artifacts: controlled-experiment writeup, results matrix, metric-over-time plot, error taxonomy, failure analysis memo, and `make_plot.py` to regenerate the plots from `results.tsv`.
- **`skin-lesion-autoresearch/`** — earlier exploratory scaffold (separate `src/`, `models/`, `results/` trees). **Not imported by the live loop** in `run.py` — only kept as historical reference. Don't import from it; copy code in if useful.

## Deployment Target (Stretch)

PyTorch → ONNX → TensorFlow → TFLite for on-device smartphone inference. See `skin-lesion-autoresearch/deployment/instructions.md`.
