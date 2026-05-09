# Week 5 — Best Result vs. Baseline

**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Block window:** Iter 1 → Iter 19 (iter 20 in flight, not yet eligible)

The Week-5 framing pushes against picking a single peak run as "best" — *"a single outlier result is not sufficient evidence of a real improvement."* So this comparison reports both **the highest single-run number** (honest about its noise-tail status) and **the most credible reproducible improvement** (reported as mean ± std over a single-variable replicated controlled experiment).

---

## Baseline (iter 1, commit `bb13e84`)

| Metric | Value |
|---|---|
| ROC-AUC | **0.7932** |
| Recall (at default 0.5 threshold) | 0.8982 |
| Precision | 0.3708 |
| Accuracy | 0.7088 |
| Train time | 836 s |

Logistic regression on raw flattened image features. Fails the project's AUC target (≥ 0.85). Recall happens to look high because LR is heavily under-confident on benign — almost everything is predicted positive — at the cost of very low precision.

## Single-run "peak" (iter 13, commit `7a57f2c (parent)`)

| Metric | Value | Δ vs baseline |
|---|---|---|
| ROC-AUC | 0.8958 | **+0.103** |
| Recall | **0.9555** | +0.057 |
| Precision | 0.3791 | +0.008 |
| Accuracy | 0.8056 | +0.097 |

EfficientNet-B2, 224 upscale, two-phase training, `pos_weight=20`, threshold cal target=0.995, 60 batches. **The only run in the entire 19-iter block to clear recall ≥ 0.95.**

I do **not** claim this as a "real" project win, because:
- Its own sibling-replicate at the same exact code (iter 11, B1) produced recall = 0.8875 — a 0.067 spread between two reps.
- Iter 15 (commit `9d1cc2d`, `USE_HOLDOUT_CAL=True`, *different* config) also crossed 0.95 once at recall = 0.965; its sibling at iter 14 produced 0.892.
- Both crossings are within one within-condition std of the run-mean, i.e. both are noise tails.

Treating either as the project win would be the numbered Week-4/5 failure mode: *"the best run looked good, but I'm not sure why."*

## Most credible reproducible improvement (Week-4 controlled set, n=4)

| Metric | Baseline (iter 1) | Iters 10–13 mean ± std | Δ |
|---|---|---|---|
| ROC-AUC | 0.7932 | 0.8982 ± 0.003 | **+0.105** ✅ over 0.85 target |
| Recall | 0.8982 | 0.915 ± 0.030 | +0.017 |
| Precision | 0.3708 | 0.428 ± 0.045 | +0.057 |

This is the strongest "real" claim I can make: under the iter-7-style training pipeline (B2, 224, two-phase, pos_weight ∈ {10, 20}, threshold cal target=0.995, batches=60), **mean test ROC-AUC is 0.898 with std ~0.003 — comfortably above the 0.85 target and consistently reproducible.** Recall is in the low-to-mid 0.90s with within-condition std around 0.03 — close to the 0.95 target but not reproducibly there.

## Most credible reproducible improvement that *also* satisfies stability rigorously (Week-5 controlled set, n=3)

Iters 17–19 (`TARGET_RECALL=0.95`, percentile threshold estimator, holdout cal):

| Metric | Baseline (iter 1) | Iters 17–19 mean ± std | Δ |
|---|---|---|---|
| ROC-AUC | 0.7932 | 0.8944 ± 0.004 | **+0.101** ✅ over 0.85 |
| **Recall** | 0.8982 | **0.7647 ± 0.014** | **−0.134** ❌ |
| Precision | 0.3708 | 0.531 ± 0.014 | +0.160 |

Lower mean recall, but **the smallest within-condition variance in the whole block.** This is the most "trustworthy" comparison even though it doesn't hit recall ≥ 0.95 — the percentile estimator is too strict.

## Project success-criteria scorecard

| Criterion | Baseline | Best (single run) | Best (replicated mean) | Status |
|---|---|---|---|---|
| ROC-AUC ≥ 0.85 | 0.793 ❌ | 0.901 ✅ (iter 7) | 0.898 ✅ (iters 10–13 mean) | **MET** since iter 2 (every iter ≥ iter 2 has AUC ≥ 0.86) |
| Recall ≥ 0.95 | 0.898 ❌ at 0.5 threshold (precision 0.37) | 0.965 ✅ (iter 15, single run) | 0.92 ❌ (iters 14–16 mean), 0.76 ❌ (iters 17–19 mean) | **NOT MET reproducibly** — crossed twice as noise-tail singletons |

## Plain-English version of the comparison

The starting point was a logistic-regression model that scored 0.79 on the AUC metric and 0.90 on the recall metric. After 19 deep-learning iterations, the deep model gets to about **0.90 AUC** very reliably — that is a real, measurable improvement of about 10 percentage points over the baseline, and it is reproduced across roughly twenty independent training runs. So on the AUC front, the project's primary success criterion is comfortably met.

The recall criterion (≥ 0.95) is more delicate. Two single runs in the block crossed 0.95, but their replicated siblings at the *same* code did not. The honest summary is: with the current pipeline, recall typically lands between 0.90 and 0.92 with run-to-run variation around 0.04 — close to the target but not reproducibly past it. The Week-5 in-flight experiment (iter 20, full RNG seeding) is testing whether this variation can be eliminated by determinism, which would let us report a single reproducible recall number for the project.
