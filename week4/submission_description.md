# Week 4 Submission — Short Description

**Author:** Patrick Gao | **Project:** ISIC binary skin lesion classifier (benign vs. malignant) | **2026-05-06**

## How I controlled the experiment

ROC-AUC has cleared its 0.85 success target since iter 2 (best 0.901), so the only unmet criterion is **recall ≥ 0.95**. I therefore picked a single experimental axis for week 4: **`pos_weight`** — the malignant-class weight in `BCEWithLogitsLoss` — the most fundamental class-imbalance lever in the loss function and the most direct way to push the model toward higher recall without changing architecture.

The controlled experiment has 2 conditions (`pos_weight ∈ {10, 20}`) and 2 replicates per condition for a total of 4 fresh runs (iters 10–13 in `results.tsv`). Run order was interleaved (A1 → B1 → A2 → B2) to neutralize time-of-day and system-load bias.

**Held fixed across all four runs and explicitly recorded:** EfficientNet-B2 backbone (ImageNet-pretrained), `Linear(1408, 1)` binary head, 224×224 input upscale with ImageNet mean/std normalization, two-phase training (2 epochs head warmup at lr=1e-3 then 8 epochs full fine-tune at lr=1e-4), Adam optimizer with no schedule, Phase 3 threshold calibration with `TARGET_RECALL=0.995` and `CALIBRATION_BATCHES=60` and no SAFETY_MARGIN, the same stratified 80/20 train/test split from frozen `prepare.py`, and the same evaluation pipeline in frozen `run.py`. Same hardware (RTX 4050) and batch size 16 throughout.

The metric was defined *before* the runs: primary = test recall (gated at 0.95), secondary = test ROC-AUC (must stay ≥ 0.85). Comparability is preserved because `run.py` is frozen and the test set never changes. The decision rule was also defined upfront: "if between-condition recall difference < within-condition std, the effect is indistinguishable from noise — Signal Failure dominant."

This is the only experiment in my whole 13-iteration history that satisfies all five Week-4 criteria simultaneously: one variable changed (criterion 1), all others held fixed (2), metric defined upfront (3), comparable evaluation (4), and **2 replicates per condition giving real within-condition std for the stability claim (5)**.

## What the experiment showed

| | pw=10 mean ± std | pw=20 mean ± std | Δ (pw=20 − pw=10) |
|---|---|---|---|
| ROC-AUC | 0.8972 ± 0.0004 | 0.8991 ± 0.0047 | +0.0019 |
| **Recall** | **0.9080 ± 0.0118** | **0.9215 ± 0.0474** | **+0.0135** |
| Precision | 0.4282 ± 0.0237 | 0.4250 ± 0.0648 | −0.0032 |

**The decision rule fired the "Signal Failure dominant" branch.** Between-condition recall difference (+0.014) is *smaller than* within-condition recall std at pw=20 (0.047). Two replicate runs of the same exact code at pw=20 produced recalls of **0.888 and 0.955** — a 0.067 spread.

B2 (iter 13) is the only run in my whole loop to clear the 0.95 recall target, but its sibling B1 produced 0.888. **I cannot claim the project's recall target is reached** without first reducing within-condition variance.

## Error taxonomy (4-category framework, ranked by severity in *my* loop)

| Rank | Category | How it manifests in this loop |
|------|----------|-------------------------------|
| **1** | **Signal Failure** *(dominant)* | Two replicates of identical code at pw=20 produced recalls 0.888 and 0.955 — within-condition std (0.047) is 3.5× the between-condition effect (0.014). Calibrated threshold also varies wildly across runs (std=0.171 at pw=20). |
| 2 | **Evaluation Leakage** *(feeds #1)* | Phase 3 calibration runs inference on training batches the model has already memorized, so cal-set recall is trivially 1.000 and the resulting threshold is systematically too high for unseen test data. The 0.06–0.09 train→test recall gap is the direct symptom. |
| 3 | **Agent Misbehavior** | Iters 4, 5, 7, 9 each changed two variables at once (confounded). Iters 8/9 logged with `status=keep` despite regressing — `run.py`'s default. The matrix corrects the record. |
| 4 | **Code Instability** *(resolved)* | Iter 7 first launch crashed because training was launched from a git worktree where data CSVs were gitignored. A 47-min apparent stall turned out to be Python `print()` block-buffering when stdout is redirected — `python -u` in subsequent runs. None recurred during iters 10–13. |

**Highest-leverage week-5 fix is replacing the cal subsystem's leaky training-data source with a held-out 10 % validation slice** (priority 1 in `failure_analysis.md`). Until that is in place, no further hyperparameter sweep is interpretable.

## Files in this submission

| Deliverable | File |
|-------------|------|
| Controlled Experiment Set | `week4/controlled_experiment_set.md` |
| Experiment-Result Matrix | `week4/experiment_matrix.md` |
| Metric-Over-Time Plot | `week4/metric_over_time.png` (full 13-iter timeline) and `week4/controlled_experiment.png` (per-condition mean ± std for iters 10–13) |
| Error Taxonomy | `week4/error_taxonomy.md` |
| Failure Analysis Memo | `week4/failure_analysis.md` |
