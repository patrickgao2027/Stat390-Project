# Week 4 — Failure Analysis Memo

**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Author:** Patrick Gao
**Week:** 4 (2026-05-06)

## Headline

**Dominant failure mode: Signal Failure caused by Evaluation Leakage in the threshold-calibration subsystem.** Two replicated runs of identical code at `pos_weight=20` produced test recalls **0.888 and 0.955**, a within-condition spread of 0.067 — *4.8 ×* larger than the +0.014 between-condition difference I was attempting to measure. The candidate "pos_weight=20 raises recall" effect is statistically indistinguishable from noise at this sample size.

A consequence: B2's recall = 0.9555 is the only run in the whole 13-iteration history to clear the project's 0.95 recall target — but its sibling B1, identical except for the random seed, produced 0.888. **Treating B2 as a project win would be a numbered failure mode in the week-4 framework** ("the score changed, but I'm not sure why").

## What I varied this week

Architecture has cleared the AUC ≥ 0.85 success target since iter 2 (best 0.901), so the only unmet criterion is recall ≥ 0.95. I therefore picked a single experimental axis: **`pos_weight`** — the malignant-class weight in `BCEWithLogitsLoss` — and ran a 4-run controlled set with 2 replicates per condition (pw=10 vs pw=20). This is the only experiment in my whole loop that satisfies all five Week-4 criteria, including stability via replication.

## Numbers (the actual evidence)

| | pw=10 mean ± std | pw=20 mean ± std | Δ (pw=20 − pw=10) | Effect / pw=20 std |
|---|---|---|---|---|
| ROC-AUC | 0.8972 ± 0.0004 | 0.8991 ± 0.0047 | +0.0019 | 0.40 |
| **Recall** | **0.9080 ± 0.0118** | **0.9215 ± 0.0474** | **+0.0135** | **0.30** |
| Precision | 0.4282 ± 0.0237 | 0.4250 ± 0.0648 | −0.0032 | — |
| Cal threshold | 0.292 ± 0.008 | 0.438 ± 0.171 | +0.146 | — |

## Why it likely happened

Each iteration does a fresh fine-tune from ImageNet weights with a randomly initialized binary head. Different head inits, different shuffle orders, and dropout/batch-norm noise during training cause each run to converge to a slightly different local optimum. Crucially, **the calibration threshold is tightly coupled to the model's specific probability output distribution**, not to anything stable like the held-out test set. So:

- A model that lands more confident on its hardest training-set positives produces a higher cal threshold (B1: 0.558).
- A model that lands less confident on the same positives produces a lower one (B2: 0.317).
- The threshold dispersion (std=0.171 in probability space at pw=20) directly causes the recall dispersion at test time, because predict() applies the threshold to a smoother test distribution.

Layered on top of this is the deeper problem: **Phase 3 calibration uses training data the model has already memorized.** Cal-set recall is therefore trivially 1.000 in every run, and the threshold is systematically too high for unseen test data — the 0.06–0.09 train→test recall gap I have observed since iter 6. This is Evaluation Leakage feeding Signal Failure.

The pos_weight increase did the textbook thing in the loss function — it just couldn't punch through the noise produced by leaky cal at this experimental sample size.

## What I'm most confident in

**The 4-run controlled set as a unit, not any single iteration.** Each iteration alone is a noisy point. The four together let me say with measured numbers: "within-condition recall variance is large enough at pw=20 to drown the between-condition effect." That is a research finding. I am explicitly *not* confident in B2's 0.9555 as a reproducible result.

I am also confident in the AUC trajectory across iters 1–13: it crossed 0.85 at iter 2 and has stayed there, with within-condition std at the controlled-experiment runs of 0.0004 (pw=10) and 0.0047 (pw=20). AUC is genuinely stable; recall is not.

## What I distrust most

**Iter 13 (B2) recall = 0.9555.** Single tail of a wide noise distribution. If I had stopped after iter 13 and said "we hit the recall target," I would be doing exactly what the week-4 PDF warns against.

## Proposed next steps (one variable per future run, with replicates)

| Priority | Step | Expected effect | Cost |
|----------|------|-----------------|------|
| 1 | **Replace the cal data source** with a held-out 10 % validation slice of the training data | Removes the Evaluation Leakage that mechanistically causes the train→test recall gap and the threshold dispersion. Single change: cal source ∈ {train_leaked, holdout_val}, **3 reps each.** | ~5.5 hours |
| 2 | **Add Test-Time Augmentation** in `predict_proba()` (4 views: original + hflip + vflip + 180° rot, averaged) | Smooths borderline probabilities, typically lifts AUC by 0.005–0.01 and stabilizes per-image probabilities at the threshold boundary. Single change: `tta ∈ {off, on}`, 2 reps each. | ~3.7 hours |
| 3 | **3 replicates of iter 7's exact config** to formally measure the noise floor of the existing best-known setup | Gives a baseline std for future hypothesis testing. The within-condition std measured here (0.012 at pw=10, 0.047 at pw=20) suggests noise depends on condition. | ~2.75 hours |

Week-5 first move is priority 1: fix the leaky cal subsystem. Without that fix, every subsequent hyperparameter sweep is uninterpretable for the same reason iters 8 and 9 are uninterpretable today.

## Lightning-round answers (for the meeting)

- **Experiment axis varied:** `pos_weight` ∈ {10, 20}, 2 reps per condition. Held fixed: EfficientNet-B2, two-phase training, 224 upscale, lr schedule, cal hyperparameters, frozen splits and eval pipeline.
- **Most important result:** the within-condition recall std at pw=20 is 0.047 — larger than the between-condition effect of 0.014. *That* is the result, not B2's 0.9555.
- **Dominant error type:** Signal Failure, mechanistically caused by Evaluation Leakage in the threshold calibration step.
- **Open uncertainty:** is the train→test recall gap closeable by switching to a held-out validation slice for cal, or is the variance intrinsic to the model + dataset combo? Priority-1 next-step answers this.
