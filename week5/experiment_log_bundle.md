# Week 5 — Complete Experiment Log Bundle

**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Author:** Patrick Gao
**Block window:** Iter 1 (baseline) → Iter 21 (determinism verification, completed)
**Iterations logged:** 21 successfully completed + 1 crash (recovered) = 22 attempts

This file is the single source of truth for every run in the autonomous block. Every row in `results.tsv` is reflected here, plus the one run that crashed before logging and the operational events that are not visible from `results.tsv` alone.

---

## Master log (all runs)

| Iter | Commit | Status | What changed (ONE variable per row, when controlled) | ROC-AUC | Recall | Precision | Train s | Decision/notes |
|------|--------|--------|---------------------------------------------------|--------:|-------:|----------:|--------:|----------------|
| 1 | bb13e84 | **baseline** | logistic regression on raw image features | 0.7932 | 0.8982 | 0.3708 | 836 | Pre-block reference. Fails AUC target (0.85) but recall happens to be high because LR is heavily under-confident on benign. |
| 2 | fc3e951 | keep | model class: LR → AlexNet (pretrained, full fine-tune, lr=1e-4) | 0.8867 | 0.8266 | 0.4822 | 2465 | Architecture leap. AUC target met (+0.09). |
| 3 | 99ff7d6 | keep | architecture: AlexNet → EfficientNet-B0 (pretrained, full fine-tune, lr=1e-4) | 0.8841 | 0.7645 | 0.5004 | 1176 | Single-variable swap. AUC ~unchanged; recall fell 0.06. |
| 4 | 817e6fa | keep (**confounded**) | TWO changes: 224×224 upscale + `pos_weight=10` | 0.8975 | 0.8314 | 0.4942 | 2691 | AUC up, recall recovered. Cannot attribute to either change individually — flagged as confounded. |
| 5 | 82ff045 | keep (**confounded**) | TWO changes: B0 → B2 + two-phase training | 0.8924 | 0.7910 | 0.5005 | 3531 | Slight AUC drop, recall fell. Cannot attribute. |
| 6 | fc6c6f0 | keep | added Phase-3 threshold calibration (target=0.97, batches=30) | 0.8954 | 0.8836 | 0.4564 | 1324 | Recall +0.09 vs iter 5 from threshold tuning. AUC unchanged. |
| 7 (1st attempt) | — | **CRASH** | same code as iter 7 successful — launched from worktree | — | — | — | — | `FileNotFoundError: 'challenge-2019-training_metadata...'`. Cause: data CSVs gitignored, only exist in main project root. Worktree run never reached training. Documented in error taxonomy as Code Instability. |
| 7 | 55ed2e7 | keep (**confounded**) | TWO changes: cal target 0.97→0.995 + cal batches 30→60 | 0.9005 | 0.9374 | 0.4154 | 3508 | Best AUC and best recall to date. Confounded — can't attribute to either change alone. |
| 8 | 7a57f2c (parent) | discard (logged as keep) | added `SAFETY_MARGIN=0.85` multiplier on threshold | 0.8951 | 0.8480 | 0.4815 | 3127 | Single variable; regressed recall by 0.09. **Reverted via `git checkout model.py` after the run.** Logged status was `keep` (run.py default); reality was discard. |
| 9 | 7a57f2c (parent) | discard (logged as keep) | TWO changes: target 0.995→1.0 + batches 60→200 | 0.8962 | 0.8910 | 0.4356 | 3389 | Confounded; regressed. Reverted. Same `keep`/`discard` mismatch as iter 8. |
| 10 | 7a57f2c (parent) | keep | Week-4 controlled set A1: `pos_weight=10` rep 1/2 | 0.8969 | 0.9163 | 0.4114 | 3592 | Part of the single-variable controlled experiment over `pos_weight ∈ {10, 20}`. |
| 11 | 7a57f2c (parent) | keep | Week-4 controlled set B1: `pos_weight=20` rep 1/2 | 0.9024 | 0.8875 | 0.4708 | 3249 | |
| 12 | 7a57f2c (parent) | keep | Week-4 controlled set A2: `pos_weight=10` rep 2/2 | 0.8975 | 0.8996 | 0.4449 | 3571 | Within-condition recall std at pw=10: **0.012** |
| 13 | 7a57f2c (parent) | keep | Week-4 controlled set B2: `pos_weight=20` rep 2/2 | 0.8958 | **0.9555** | 0.3791 | 3637 | Only single run in the whole block to clear recall ≥ 0.95. **Treated as a noise tail per Week-4 failure memo** — sibling B1 produced 0.888. Within-condition recall std at pw=20: **0.047**. |
| 14 | 9d1cc2d | keep | Week-5 priority-1 controlled set: `USE_HOLDOUT_CAL=True` rep 1/3 | 0.8928 | 0.8916 | 0.4228 | 3501 | Hypothesis: holdout cal removes Evaluation Leakage and tightens variance. Single variable changed vs iter 12. |
| 15 | 9d1cc2d | keep | rep 2/3 | 0.8952 | **0.9650** | 0.3623 | 3677 | Threshold collapsed to 0.062. Second time recall ≥ 0.95 — also a noise tail. |
| 16 | 9d1cc2d | keep | rep 3/3 | 0.8976 | 0.9044 | 0.4361 | 3554 | Holdout-cal recall std came in at **0.039** vs leaked-cal std 0.019 — holdout *widened* variance. **Hypothesis rejected.** |
| 17 | 9d1cc2d | keep | corrected priority-1 controlled set: `TARGET_RECALL` 0.995→0.95 (min → 5th-percentile) rep 1/3 | 0.8984 | 0.7743 | 0.5458 | 3472 | Different single variable. Cal threshold variance dropped 4.5× (0.135 → 0.030). |
| 18 | 9d1cc2d | keep | rep 2/3 | 0.8938 | 0.7705 | 0.5180 | 3807 | |
| 19 | 9d1cc2d | keep | rep 3/3 | 0.8909 | 0.7491 | 0.5290 | 3681 | Recall std 0.014 vs 0.039 with min estimator — **2.8× tighter.** Mean recall fell 0.92→0.77 (tradeoff quantified). |
| 20 | a73de59 | keep | full PyTorch+CUDA+TF determinism (seed=67, cudnn deterministic) + revert to TARGET_RECALL=0.995 | 0.8962 | 0.8245 | 0.4917 | 3552 | Threshold=0.560 (min of 157 holdout positives). Seed 67 produced a model where all positives score ≥ 0.56 → recall lower than unseeded min estimator mean (0.92). |
| 21 | 7965f2b | keep | determinism verification — identical code, no changes | 0.8936 | 0.8498 | 0.4719 | 3673 | Threshold=0.603. **Determinism NOT achieved**: losses differed from epoch 1 (1.0396 vs 1.0288). Root cause: `prepare.py` calls `tf.data.shuffle()` before `model.py` is imported, so `tf.random.set_seed()` in model.py runs too late. Full determinism requires modifying the frozen `prepare.py`. |

`(parent)` in the commit column means `model.py` had uncommitted changes when the run logged — `run.py` records the parent commit hash, so several rows share `7a57f2c`. The "what changed" column is the source of truth for what code was actually used.

## Run-status accounting

| Status | Count | Iters |
|---|---:|---|
| Baseline | 1 | 1 |
| Keep (committed and pushed) | 18 | 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 |
| Discard (logged as `keep` by `run.py` default, but `model.py` reverted via `git checkout`) | 2 | 8, 9 |
| Crash (no row in `results.tsv`) | 1 | iter 7 first attempt — `FileNotFoundError` from worktree |
| **Total attempts** | **22** | |

The status mismatch on iters 8 and 9 (logged as `keep` but reality is `discard`) is itself a documented bit of trace integrity — the experiment-result matrix corrects the record.

## Git / version history (block commits, in order)

```
7965f2b → f9fa804   iters 20-21 results: determinism test — NOT achieved
a73de59 → 7965f2b   iter 20: full RNG seeding (seed=67, cudnn deterministic) + revert TARGET_RECALL=0.995
155d0d3 → a73de59   Week 5 deliverables (iters 1-19 trace)
9d1cc2d → 155d0d3   Week 5 iters 14-19: holdout cal + percentile estimator experiments
7a57f2c → 9d1cc2d   Week 4: pos_weight controlled experiment + updated deliverables
69d87b4 → 7a57f2c   iter 7: tighter calibration target=0.995, 60 batches — recall 0.937, AUC 0.9005
fc6c6f0 → 69d87b4   ignore .claude/ worktrees
82ff045 → fc6c6f0   EfficientNet-B2 + 224 + two-phase + threshold calibration for recall>=0.97 6th iteration
817e6fa → 82ff045   EfficientNet-B2 + 224 upscale + two-phase + pos_weight=10 5th iteration
99ff7d6 → 817e6fa   EfficientNet-B0 + 224x224 upscale + pos_weight=10 4th iteration
fc3e951 → 99ff7d6   EfficientNet-B0 pretrained full fine-tune lr=1e-4 3rd iteration
bb13e84 → fc3e951   AlexNet pretrained full fine-tune lr=1e-4 2nd iteration
                    (pre-block: logistic regression baseline)
```

Every commit is on `main` and pushed to `github.com:patrickgao2027/Stat390-Project`.

## Resource usage

- Hardware: RTX 4050 (laptop), 6 GiB VRAM, batch size 16
- Training time per iter (deep learning runs): **~22–60 min** (avg ~55 min for B2)
- GPU thermals during a 3-hour block: **63 °C, 39 W (~⅓ TDP), 98 % util** — well within healthy range
- Total wall-clock spent on this block (iters 6–21, the autonomous-block portion): roughly **~16 hours of training** + ~1 hour of agent overhead (file edits, plotting, deliverable writing, git)
- Disk: `results.tsv` (~2 KB), training has no checkpoint persistence between runs (each run trains from ImageNet weights)
- Network: ~0 (no external data downloads)

## Operational events not visible in `results.tsv`

| Event | When | What happened | How resolved |
|---|---|---|---|
| Worktree crash | iter 7 first attempt | training launched from `.claude/worktrees/elastic-shirley-5607b5/` and crashed because data CSVs are gitignored, only exist in main project root | switched to running from main project root; preference saved to memory so future sessions don't repeat the mistake |
| Apparent stall | iter 7 successful run | training log file stayed at 6 lines for ~47 min — looked like the process had hung | diagnosed as Python `print()` block-buffering when stdout is redirected. All subsequent runs use `python -u` to force unbuffered output. |
| `keep`/`discard` mismatch | iters 8, 9 | `run.py` defaults to `--keep`. Both runs regressed and were reverted via `git checkout model.py` after, but the `results.tsv` row still says `keep` | documented in this file and the experiment-result matrix |

## How to reproduce any single run

```bash
git checkout <commit-hash-from-master-log>
C:\Users\Owner\anaconda3\python.exe -u run.py "any description"
```

None of the runs are bit-exactly reproducible. Pre-iter-20 runs lack PyTorch/CUDA seeding. Iters 20-21 added full PyTorch+TF+CUDA seeding but still produced different results (AUC 0.896 vs 0.894, recall 0.825 vs 0.850) because `prepare.py`'s `tf.data.shuffle()` runs before `model.py` is imported — the data pipeline cannot be seeded from `model.py` alone.
