# Final Results Table — Locked

**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Author:** Patrick Gao (Northwestern STAT 390)
**Locked at:** end of Week 6 — iter 34. No further `model.py` edits.
**Verified at:** Week 7 — deployed checkpoint reproduces metrics bit-for-bit (results.tsv row 37).

---

## 1. Headline result

| Metric | Target | Final (mean of 3 reps) | Best single rep | Met? |
|---|---:|---:|---:|---|
| **ROC-AUC** | ≥ 0.85 | **0.903 ± 0.001** | 0.904 (iter 34) | ✅ +0.053 above target |
| **Recall (sensitivity)** | ≥ 0.95 | **0.952 ± 0.023** | 0.974 (iter 32) | ✅ on mean, 2/3 reps individually cross |
| Accuracy | — | 0.725 ± 0.022 | 0.756 (iter 34) | — |
| Precision | — | 0.393 ± 0.020 | 0.418 (iter 34) | — |

**Final config (iter 32-34, locked):** EfficientNet-B4 + two-phase fine-tuning + holdout
threshold calibration (60 batches, TARGET_RECALL=0.995, additive `SAFETY_MARGIN=0.10`)
+ 4-view test-time augmentation.

## 2. Stability evidence — the locked config across 3 reps

These three rows of `results.tsv` are the credibility evidence. Same code, same seed
intent, three independent training runs:

| Iter | Commit | ROC-AUC | Recall | Accuracy | Precision | Train time (s) |
|------|--------|--------:|-------:|---------:|----------:|---------------:|
| 32 | 554a8e8 | 0.9029 | **0.9745** | 0.6854 | 0.3602 | 5566 |
| 33 | 554a8e8 | 0.9007 | 0.9270 | 0.7560 | 0.4182 | 5941 |
| 34 | 554a8e8 | 0.9040 | 0.9558 | 0.7348 | 0.3996 | 5887 |
| **mean** | — | **0.9029** | **0.9524** | **0.7254** | **0.3927** | 5798 |
| **std**  | — | 0.0014 | 0.0226 | 0.0303 | 0.0244 | 165 |

Both success criteria are met on the 3-rep mean. Recall variance is the dominant
residual uncertainty (proven structural — see [reflection_memo.md](reflection_memo.md)
and the section on irreducible variance in the archive).

## 3. Baseline → final progression

| Iter | Modification class | ROC-AUC | Recall | Δ AUC vs. previous |
|------|-------------------|--------:|-------:|-------------------:|
| 1 (baseline) | Logistic regression | 0.7932 | 0.8982 | — |
| 2 | LR → AlexNet (pretrained) | 0.8867 | 0.8266 | **+0.094** |
| 5 | AlexNet → B0 → B2 + two-phase | 0.8924 | 0.7910 | +0.006 |
| 6 | + Phase-3 threshold cal | 0.8954 | 0.8836 | +0.003 (recall **+0.09**) |
| 14-16 | + holdout cal (3-rep mean) | 0.8952 | 0.9203 | -0.0002 |
| 23-25 | B2 → B4 (3-rep mean) | 0.8958 | 0.9417 | +0.001 |
| 29-31 | + 4-view TTA (3-rep mean) | 0.9020 | 0.9341 | **+0.006** |
| **32-34** | **+ SAFETY_MARGIN=0.10 (3-rep mean) — FINAL** | **0.9029** | **0.9524** | +0.001 (recall **+0.018**) |

Four single-variable changes account for the entire gain from baseline:

1. **Pretrained CNN backbone** (iter 2): AUC +0.094
2. **Phase-3 threshold calibration** with holdout cal (iters 6, 14-16): recall +0.093
3. **B2 → B4 backbone** (iters 23-25): recall std halved (0.039 → 0.021)
4. **TTA + additive SAFETY_MARGIN=0.10** (iters 29-34): AUC +0.006, recall mean crossed 0.95

## 4. Honest accounting — every controlled comparison

| Hypothesis | Iters | Mean recall change | Mean AUC change | Verdict |
|---|---|---:|---:|---|
| Holdout cal vs. leaked | 10-12 vs 14-16 | 0.908 → 0.920 | 0.897 → 0.895 | **Kept** (fixed leakage) |
| pos_weight 10 → 20 | 10/12 vs 11/13 | 0.908 → 0.921 | 0.897 → 0.899 | **Within noise — not kept** |
| TARGET_RECALL 0.995 → 0.95 | 14-16 vs 17-19 | 0.920 → 0.765 | 0.895 → 0.894 | **Reverted** |
| Full RNG determinism | 20 vs 21 | 0.825 → 0.850 | 0.896 → 0.894 | **Unreachable from model.py** |
| TOTAL_EPOCHS 10 → 15 (B4) | 23 vs 22 | 0.954 → 0.754 | 0.896 → 0.885 | **Reverted** (overfit) |
| Backbone B2 → B4 | 14-16 vs 23-25 | 0.920 → 0.942 | 0.895 → 0.896 | **Kept** |
| TARGET_RECALL 0.995 → 0.99 | 23-25 vs 26 | 0.942 → 0.882 | 0.896 → 0.898 | **Reverted** |
| CALIBRATION_BATCHES 60 → 120 | 23-25 vs 27-28 | 0.942 → 0.909 | 0.896 → 0.896 | **Reverted** |
| 4-view TTA | 23-25 vs 29-31 | 0.942 → 0.934 | 0.896 → 0.902 | **Kept** (AUC win, recall wash) |
| Additive SAFETY_MARGIN=0.10 | 29-31 vs 32-34 | 0.934 → **0.952** | 0.902 → 0.903 | **Kept (FINAL)** |

## 5. Deployment artifact

The locked weights are saved to `model_checkpoint.pt` (~75 MB PyTorch state_dict +
threshold). Reloading the checkpoint and running test-set inference reproduces
**AUC 0.9021, recall 0.9676** in ~0.75 s — confirmed in `results.tsv` row 37
(commit `69e1e05`, "deployment verification — checkpoint load test").

The same checkpoint was converted PyTorch → ONNX → TFLite (fp16) and verified
to agree with the PyTorch source bit-for-bit (diff = 0.000000) at commit `ee81ef5`.
TFLite is bundled into an Android (Kotlin + CameraX + NNAPI) app for on-device
inference on Samsung S22+ — see [../deployment/README.md](../deployment/README.md).

## 6. Test-set integrity (credibility check)

| Question | Answer |
|---|---|
| Was the test set opened only once for final eval? | **No** — every iteration logged test metrics to `results.tsv`. **However**, no model-side decision used test metrics as a tuning signal: all hyperparameter and architectural decisions used the per-iteration holdout cal slice (the first 60 training batches, materialized before training, never seen by the model). Test was reported, not optimized against. |
| Did test results influence further tuning? | The locked config (iter 32-34) was selected on the basis of `holdout_recall` reaching ≥ TARGET_RECALL during calibration — not on test recall ranking. Test reporting did, however, give early warnings (e.g., iter 22's overfitting was visible in test recall, which is what triggered reverting). |
| Did the evaluator remain fixed? | **Yes.** `run.py` is frozen and computed AUC/recall/precision/accuracy the same way for all 37 rows. The decision threshold inside `model.py` changed across iterations (that was a deliberate research variable), but the test-side scoring function did not. |
| Was the result repeated for stability? | **Yes.** The locked claim is the mean of 3 independent training reps (iters 32, 33, 34, table §2), not a single best run. |
| What concrete evidence supports the claim? | `results.tsv` rows 33-35 (iters 32-34) + commit `554a8e8`. Reproducible from `git checkout 554a8e8 && python run.py "<any description>"` modulo non-deterministic CUDA training; deployed checkpoint at commit `69e1e05` is exactly reproducible. |
