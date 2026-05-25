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

The project uses the Anaconda base environment. **Always use PowerShell** — the backslash path (`C:\...\python.exe`) is silently not found (exit code 127) when launched from Bash. Use `-u` for unbuffered stdout (otherwise per-epoch lines stay block-buffered and long runs look stalled):

```powershell
# In PowerShell (Set-Location to project root first):
C:\Users\Owner\anaconda3\python.exe -u run.py "description Nth iteration"             # status=keep (default)
C:\Users\Owner\anaconda3\python.exe -u run.py "description" --baseline                # status=baseline
C:\Users\Owner\anaconda3\python.exe -u run.py "description" --discard                 # status=discard
```

Training time on the RTX 4050: **~110 min for EfficientNet-B4** (current backbone) at 10 epochs, ~60 min for B2, ~22–45 min for B0/AlexNet. Plan budgets accordingly. Do NOT try to run more than 10 epochs on B4 — iter 22 showed 15 epochs causes overfitting (recall drops to 0.75).

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

- **`prepare.py`** — loads ISIC CSV metadata, merges 2019+2020, filters indeterminate diagnoses, **stratified 80/20 split** seeded by `RANDOM_STATE=67`, subsamples training to 30%, returns `tf.data.Dataset` pipelines yielding NHWC float32 in `[0, 1]` at **128×128**, plus `class_weight` dict (~5.31 for malignant). The shuffle is seeded (`ds.shuffle(seed=RANDOM_STATE)`), but the five `tf.image.random_*` augmentation ops inside `.map(augment, num_parallel_calls=AUTOTUNE)` are **not** seeded individually, and `AUTOTUNE` parallel ordering on top of stateful random ops makes the augmented stream non-reproducible. Setting `tf.random.set_seed()` in `model.py` doesn't fix this — the per-op random state for each `tf.image.random_*` call would need to be pinned inside `prepare.py`. Full training determinism is therefore not achievable from `model.py` alone without modifying the frozen `prepare.py`.
- **`torch_adapter.py`** — `BaseTorchModel` adapter that wraps a `nn.Module` producing logits. Handles the TF→Torch conversion (`NHWC → NCHW`), the BCEWithLogitsLoss training loop with `pos_weight`, and exposes `predict()` / `predict_proba()`. **`predict()` uses a hardcoded `threshold = 0.5`** unless the subclass overrides it. Most architectural choices and any threshold tuning logic live in `model.py` subclasses, not here.
- **`model.py`** — must subclass `BaseTorchModel`, override `_build_module()` to return the `nn.Module`, and may override `fit()` for two-phase training, threshold calibration, etc. The `nn.Module` typically upscales the 128×128 input to **224×224** with `F.interpolate` and applies ImageNet normalization before passing to a pretrained torchvision backbone.
- **`run.py`** — calls `load_data()` → `build_model()` → `model.fit(...)` → `evaluate(model, test_ds, y_test)` → log row → save ROC + confusion-matrix plots.

### Deploy mode: checkpoint save/load

`model.py` has two execution modes controlled by class constants:

- `SAVE_CHECKPOINT=True` (default) — after each training, write `state_dict + threshold + raw_threshold` to `model_checkpoint.pt` **only if** this run's `raw_threshold` beats the previously saved one. Best-of-N across reruns; delete the file to start fresh.
- `LOAD_CHECKPOINT=True` (default) — if `model_checkpoint.pt` exists, `fit()` short-circuits: it loads weights + threshold and skips training entirely. End-to-end run drops from ~95 min to ~5 sec and inference is bit-for-bit deterministic. This is the "deployed model" path. To force a fresh training run, delete `model_checkpoint.pt` or set `LOAD_CHECKPOINT=False`.

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

- **`week4/`** — Week 4 deliverable artifacts: controlled-experiment writeup, results matrix, metric-over-time plot, error taxonomy, failure analysis memo, and `make_plot.py` to regenerate plots from `results.tsv`.
- **`week5/`** — Week 5 deliverable artifacts: full experiment log bundle (all 26 attempts), keep/discard/crash summary, best-vs-baseline comparison, "what actually worked" memo, and `make_plot.py` for metric trajectory + controlled-experiment + outcome bar charts. Run `python week5/make_plot.py` from the project root to regenerate the three PNGs.
- **`week6/`** — Week 6 scope-lock deliverables: revised project statement, agent strategy, full ablation/comparison table, meeting prep, and the locked two-week plan to submission. Marks the close of the experimental loop.
- **`deployment/`** — Week 7 deployment artifacts: PyTorch → ONNX → TFLite conversion (`convert.py`), single-image sanity check (`verify_predictions.py`), and a Kotlin/CameraX/TFLite Android app under `android/` for Samsung S22+. See [deployment/README.md](deployment/README.md). Not imported by the training loop.
- **`skin-lesion-autoresearch/`** — earlier exploratory scaffold (separate `src/`, `models/`, `results/` trees). **Not imported by the live loop** in `run.py` — only kept as historical reference. Don't import from it; copy code in if useful.

## Current project state (as of iter 36, deployment verified)

`results.tsv` has 37 rows. **Lever search is officially closed** at iter 34
(see the `model.py` module docstring for the locked rationale). Iters 35-36
are best-of-3 weight-save reps; the final row is the deployment-verification
check (loads `model_checkpoint.pt`, reproduces metrics bit-for-bit in ~0.75s).

**Locked config** (iter 32-34): B4 + two-phase fine-tune + holdout threshold
calibration + 4-view TTA + additive `SAFETY_MARGIN=0.10`.

| Criterion | Status | Evidence |
|---|---|---|
| ROC-AUC ≥ 0.85 | ✅ **Met** since iter 2 | Iters 32-34: 0.903 ± 0.001 |
| Recall ≥ 0.95 | ✅ **Met on mean** | Iters 32-34: 0.952 ± 0.023 (2/3 reps individually cross) |

**What worked / failed (full history):**
- ✅ LR → pretrained CNN (B2): AUC +0.09, every iter from 2 onward ≥ 0.886
- ✅ Phase-3 threshold calibration (iter 6, holdout cal iters 14-16): recall +0.09, no AUC cost
- ✅ B2 → B4 backbone (iters 23-25): recall mean +0.022, std halved (0.021 vs 0.039)
- ✅ 4-view TTA (iters 29-31): AUC +0.006 reproducibly across all reps
- ✅ Additive `SAFETY_MARGIN=0.10` (iters 32-34): mean recall crossed 0.95
- ❌ pos_weight 10→20: Δ < within-condition noise
- ❌ Multiplicative SAFETY_MARGIN: regressed on noisy baseline
- ❌ TARGET_RECALL 0.995 → 0.95 (iters 17-19): recall mean fell to 0.77
- ❌ TARGET_RECALL 0.995 → 0.99 (iter 26): threshold moved wrong direction, recall 0.88
- ❌ CALIBRATION_BATCHES 60 → 120 (iters 27-28): recall mean dropped 0.03
- ❌ 15 epochs on B4 (iter 22): overfit — training loss 0.25, threshold 0.66, recall 0.75
- ❌ Full RNG seeding (iters 20-21): the shuffle in frozen `prepare.py` IS seeded, but the augmentation ops (`tf.image.random_*` with `num_parallel_calls=AUTOTUNE`) are not — determinism not reachable from `model.py`

Note on `experiment` hashes: because multi-rep experiments often run without
intermediate commits, several iters share the same parent-commit hash (e.g.
iters 29-31 all logged under `5357c6b`; iters 32-34 under `554a8e8`). Group
by hash when analyzing within-condition variance.

## Week 6 scope lock (as of 2026-05-17)

Per the Week 6 capstone brief, the project is in **convergence mode**, not exploration mode.
After iters 26–28 land, the experimental loop closes. The remaining work is writeup, not
new ideas.

**Allowed (per brief):** refine search priorities, reorder experiments, tighten failure
recovery, minor module-boundary tweaks with instructor approval.

**Forbidden (per brief):** expanding scope, swapping evaluation after seeing favorable
numbers, adding "one more big direction." New ideas belong in a future project.

**Dropped directions (officially off the table):**
- Seeding/full determinism from `model.py` (proven unreachable — frozen `prepare.py`'s
  `tf.image.random_*` augmentations are unseeded and run under `num_parallel_calls=AUTOTUNE`)
- > 10 epochs on B4 (proven to overfit — iter 22)
- pos_weight tuning beyond 10 (proven within-noise)
- SAFETY_MARGIN multipliers (proven to regress)
- Switching backbone again (B4 is the chosen final architecture)
- TARGET_RECALL ≠ 0.995 (iters 17-19 at 0.95, iter 26 at 0.99 — both regressed)
- CALIBRATION_BATCHES > 60 (iters 27-28 at 120 — recall mean dropped 0.03)

**Locked claim:** *"EfficientNet-B4 with two-phase fine-tuning, holdout threshold
calibration, 4-view TTA, and an additive 0.10 safety margin meets AUC ≥ 0.85 and
recall ≥ 0.95 on ISIC 2019+2020 (mean AUC 0.903, mean recall 0.952 across 3 reps)."*

## Deployment (Completed — Week 7)

PyTorch → ONNX → TFLite (fp16) → Android (Kotlin + CameraX + NNAPI) for on-device
inference on Samsung S22+. End-to-end verified: commit `ee81ef5` confirms PyTorch
and TFLite outputs agree exactly (diff=0.000000). See [deployment/README.md](deployment/README.md).

Pipeline:
1. Train with `SAVE_CHECKPOINT=True` (default) — best-of-N raw_threshold selection writes `model_checkpoint.pt`.
2. `python deployment/convert.py` — exports ONNX + TFLite, prints the threshold to bake into the Android app.
3. Copy `deployment/skin_lesion_b4.tflite` into `deployment/android/app/src/main/assets/`, open `deployment/android/` in Android Studio, deploy.
4. Single-image sanity check: `python deployment/verify_predictions.py <image_path>`.
