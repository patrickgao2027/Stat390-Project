# Week 4 — Experiment-Result Matrix

**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Targets:** ROC-AUC ≥ 0.85 (met since iter 2) **and** recall ≥ 0.95 (B2/iter 13 crossed it once at 0.9555 but the result is *not* reproducible — see headline finding below).

## Headline finding (this is the actual evidence I trust most)

The week-4 controlled experiment set (iters 10–13) is the only set in my whole loop that satisfies all five Week-4 criteria, including stability via 2 replicates per condition. **Within-condition recall std is 0.011 at pw=10 and 0.047 at pw=20.** The between-condition recall difference is only 0.014 — *3.5× smaller* than the within-condition std at pw=20. The effect of `pos_weight` cannot be cleanly attributed at this sample size; **Signal Failure dominates.**

## Full iteration history (all 13 runs)

| Iter | Commit | Variable changed vs prior? | ROC-AUC | Recall | Status |
|------|--------|----------------------------|---------|--------|--------|
| 1 (baseline) | bb13e84 | n/a | 0.793 | 0.898 | baseline |
| 2 | fc3e951 | model class (LR → AlexNet) | 0.887 | 0.827 | keep |
| 3 | 99ff7d6 | ✅ architecture (AlexNet → EfficientNet-B0) | 0.884 | 0.765 | keep |
| 4 | 817e6fa | ❌ **two** (224 upscale **+** pos_weight=10) | 0.897 | 0.831 | keep |
| 5 | 82ff045 | ❌ **two** (B0 → B2 **+** two-phase training) | 0.892 | 0.791 | keep |
| 6 | fc6c6f0 | ✅ added threshold calibration (target=0.97, 30 batches) | 0.895 | 0.884 | keep |
| 7 | 55ed2e7 | ❌ **two** (cal target 0.97→0.995 **+** batches 30→60) | 0.901 | 0.937 | keep |
| 8 | 7a57f2c | ✅ added SAFETY_MARGIN=0.85 multiplier | 0.895 | 0.848 | regression — code reverted |
| 9 | 7a57f2c | ❌ **two** (target 0.995→1.0 **+** batches 60→200) | 0.896 | 0.891 | regression — code reverted |
| **10** | 7a57f2c | ✅ controlled-set replicate of pw=10 (A1) | 0.897 | 0.916 | keep |
| **11** | 7a57f2c | ✅ pw=10 → 20 (B1) | 0.902 | 0.888 | keep |
| **12** | 7a57f2c | ✅ pw=10 (A2 — replicate of A1) | 0.898 | 0.900 | keep |
| **13** | 7a57f2c | ✅ pw=20 (B2 — replicate of B1) | 0.896 | **0.955** | keep — but not reproducible |

✅ = single-variable change. ❌ = confounded (multi-variable). Of 13 iterations, **only iters 3, 6, 8, 10, 11, 12, 13 are strict single-variable comparisons.** Of those, only iters 10–13 also have replicates (criterion 5: stability) — making them the only set that satisfies all five Week-4 criteria.

## Controlled experiment sub-table (the only stability-verified runs)

This is the actual evidence base for week 4. See `controlled_experiment_set.md` for the full design and decision rule.

### Per-run

| Iter | Replicate | POS_WEIGHT | ROC-AUC | Recall | Precision | Cal threshold |
|------|-----------|-----------|---------|--------|-----------|---------------|
| 10 | A1 | 10 | 0.8969 | 0.9163 | 0.4114 | 0.286 |
| 11 | B1 | 20 | 0.9024 | 0.8875 | 0.4708 | 0.558 |
| 12 | A2 | 10 | 0.8975 | 0.8996 | 0.4449 | 0.297 |
| 13 | B2 | 20 | 0.8958 | 0.9555 | 0.3791 | 0.317 |

### Per-condition (mean ± std, n=2)

| POS_WEIGHT | ROC-AUC | Recall | Precision | Cal threshold |
|------------|---------|--------|-----------|---------------|
| 10 | 0.8972 ± 0.0004 | **0.9080 ± 0.0118** | 0.4282 ± 0.0237 | 0.292 ± 0.008 |
| 20 | 0.8991 ± 0.0047 | **0.9215 ± 0.0474** | 0.4250 ± 0.0648 | 0.438 ± 0.171 |
| **Δ (20−10)** | +0.0019 | **+0.0135** | −0.0032 | +0.146 |

## Most-trusted result (Week-4 question 3)

**The 4-run controlled set as a unit, *not* any single iteration.** Each iteration alone is a noisy single point — but the four together let me say with measured confidence: "within-condition recall variance is large enough at pw=20 to drown the between-condition effect." That is a real research finding, not a number with no explanation.

## Most-distrusted result (Week-4 question 4)

**Iter 13 (B2)'s recall = 0.9555.** This is the only run in the entire 13-iteration history to clear the project's 0.95 recall target. But its sibling B1 (same code, same data, different random seed) produced recall = 0.888. The "hit" is one tail of a wide noise distribution, not a reproducible result. Treating it as a project win would be exactly the failure mode the PDF warns against: *"the score changed, but I'm not sure why."*

## Comparability (Week-4 criterion 4)

`run.py` and `prepare.py` are frozen across the entire history; the train/test split is deterministic; the same metrics computation runs after every fit. All 13 iterations are thus directly comparable on AUC and recall. The only subsystem with comparability concerns is the threshold calibration — see Evaluation Leakage in `error_taxonomy.md`.
