# Week 4 — Controlled Experiment Set

**Author:** Patrick Gao
**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Week:** 4 (2026-05-06)

This document defines the small, focused controlled-experiment set I ran this week. It is the only set in my entire 13-iteration history that satisfies all five Week-4 criteria simultaneously, including stability via replication.

## Hypothesis (defined before runs)

Increasing the malignant-class weight in `BCEWithLogitsLoss` from `pos_weight=10` to `pos_weight=20` will *increase* test recall (the only unmet success criterion: ≥ 0.95) at the expected cost of some precision. ROC-AUC should be approximately unchanged because pos_weight rebalances the loss but does not move the threshold-free decision surface.

## Experiment axis

**Variable I varied:** `POS_WEIGHT` ∈ {10, 20} — the malignant-class weight passed into `BCEWithLogitsLoss(pos_weight=...)` via the `class_weight` dict in `fit()`.

**Why this variable:** It is the single most fundamental imbalance lever in the loss function and the most direct way to push the model toward higher recall without changing architecture. It also acts at *training* time, decoupled from the threshold-calibration noise that dominated iters 7–9.

## Conditions held fixed (recorded before runs)

| Held fixed | Value |
|------------|-------|
| Backbone | `models.efficientnet_b2`, ImageNet-pretrained |
| Final layer | `Linear(1408, 1)` binary head |
| Input resolution | upscaled to 224×224 (bilinear) |
| Channel normalization | ImageNet mean/std |
| Two-phase training | 2 epochs head-warmup at lr=1e-3 → 8 epochs full fine-tune at lr=1e-4 |
| Loss | `BCEWithLogitsLoss(pos_weight=POS_WEIGHT)` |
| Optimizer | Adam, no LR schedule |
| Train/test split | stratified 80/20 from frozen `prepare.py` |
| Phase 3 calibration | `TARGET_RECALL=0.995`, `CALIBRATION_BATCHES=60`, no SAFETY_MARGIN |
| Evaluation pipeline | frozen `run.py` — same metrics, same test set |
| Hardware | RTX 4050, batch size 16 |

## Metric defined upfront

- **Primary:** test-set recall (must reach ≥ 0.95)
- **Secondary:** test-set ROC-AUC (must stay ≥ 0.85)
- **Tertiary (reported, not gated):** precision, accuracy, calibrated threshold, training time

## Decision rule defined upfront

- If `recall(pw=20) − recall(pw=10) > 2 ×` max within-condition std → real effect
- If both pw=20 reps land recall > 0.95 → project recall target reproducibly hit
- If between-condition difference < within-condition std → effect indistinguishable from noise; **Signal Failure dominant**

## Runs

| # | Iter | POS_WEIGHT | Replicate | ROC-AUC | Recall | Precision | Cal threshold |
|---|------|-----------|-----------|---------|--------|-----------|---------------|
| 1 | 10 | 10 | A1 | 0.8969 | 0.9163 | 0.4114 | 0.286 |
| 2 | 11 | 20 | B1 | 0.9024 | 0.8875 | 0.4708 | 0.558 |
| 3 | 12 | 10 | A2 | 0.8975 | 0.8996 | 0.4449 | 0.297 |
| 4 | 13 | 20 | B2 | 0.8958 | **0.9555** | 0.3791 | 0.317 |

Run order interleaved (A1 → B1 → A2 → B2) to neutralize time-of-day / system-load bias.

## Results: condition means and within-condition stds (n=2 per condition)

| Metric | pw=10 mean | pw=10 std | pw=20 mean | pw=20 std | Δ (pw=20 − pw=10) |
|--------|------------|-----------|------------|-----------|-------------------|
| ROC-AUC | 0.8972 | 0.0004 | 0.8991 | 0.0047 | +0.0019 |
| **Recall** | **0.9080** | **0.0118** | **0.9215** | **0.0474** | **+0.0135** |
| Precision | 0.4282 | 0.0237 | 0.4250 | 0.0648 | −0.0032 |
| Cal threshold | 0.292 | 0.008 | 0.438 | 0.171 | +0.146 |

## Interpretation against the decision rule

- Effect size on recall: **+0.0135**.
- Within-condition std on recall (pw=20): **0.0474** — 3.5× the effect size.
- **Effect ≪ noise.** Decision-rule trigger: *"effect indistinguishable from noise; Signal Failure dominant."*
- Recall target (0.95) was crossed by B2 (0.9555) but **not** by B1 (0.8875). The "hit" is not reproducible.
- AUC stayed safely above 0.85 in every run (all four ≥ 0.896).

## Conclusion

The hypothesis is **not** supported by this controlled experiment. The pos_weight increase produced a between-condition recall difference of +0.014, but the within-condition recall variance for pw=20 is 0.047 — over three times larger. Even the AUC's tiny gain (+0.002) is dwarfed by its own within-condition std (+0.005 for pw=20). I cannot claim that pos_weight=20 is better than pos_weight=10 with this evidence.

A second, equally important finding: **the same code, run twice, can produce recall outcomes 0.067 apart** (B1: 0.888, B2: 0.955). This is the dominant Signal Failure of the loop. Without first reducing within-condition variance — most likely by replacing the leaky training-data calibration step with a held-out validation slice — no future single-run hyperparameter tweak is interpretable.

## What I would test next (one variable, replicated)

1. **Held-out validation split for the cal step** — currently Phase 3 uses training data the model has memorized (Evaluation Leakage feeding Signal Failure). Single change: `cal_data_source ∈ {train_leaked, holdout_val}`, 3 reps each.
2. **TTA on/off** in `predict_proba()` — averaging 4 augmented views typically smooths borderline probabilities. Single change: `tta ∈ {off, on}`, 2 reps each.
3. **Three replicates of iter 7's exact config** — formally measure within-condition variance for the existing best-known setup, since this experiment showed pw=20 has std=0.047 and pw=10 has std=0.011 — the noise floor depends on the condition.
