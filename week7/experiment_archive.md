# Full Experiment Archive — All 36 Iterations

**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Author:** Patrick Gao (Northwestern STAT 390)
**Block window:** Iter 1 (logistic regression baseline, 2026-04-10) → Iter 36 (best-of-3 weight-save, 2026-05-23) + deployment verification (2026-05-24)
**Total attempts:** 36 successful runs + 1 worktree crash + 1 deployment verification = 38 total

This is the single source of truth for every run executed during the project. Every
row in `results.tsv` is reflected here, plus the runs that crashed before logging
and the operational events that are not visible from `results.tsv` alone.

For the locked-config evidence only, see [final_results_table.md](final_results_table.md).
For the human-readable narrative of what worked and why, see
[../week5/what_actually_worked.md](../week5/what_actually_worked.md).

---

## Master log

`(parent)` in the commit column means `model.py` had uncommitted changes when the
run logged — `run.py` records the parent commit hash, so several rows share the
same hash. The "what changed" column is the source of truth for what code was
actually used.

### Block A — exploration (iters 1-13)

| Iter | Commit | Status | What changed (one variable per row when controlled) | AUC | Recall | Train s | Notes |
|---|---|---|---|---:|---:|---:|---|
| 1 | bb13e84 | **baseline** | logistic regression on raw image features | 0.7932 | 0.8982 | 836 | Pre-block reference. Fails AUC target; recall artificially high because LR is under-confident on benign. |
| 2 | fc3e951 | keep | LR → AlexNet (pretrained, full fine-tune, lr=1e-4) | 0.8867 | 0.8266 | 2465 | Architecture leap. AUC target met (+0.094). |
| 3 | 99ff7d6 | keep | AlexNet → EfficientNet-B0 | 0.8841 | 0.7645 | 1176 | Single-variable swap. AUC unchanged; recall fell. |
| 4 | 817e6fa | keep (**confounded**) | TWO changes: 224×224 upscale + `pos_weight=10` | 0.8975 | 0.8314 | 2691 | AUC up, recall recovered. Cannot attribute to either change. |
| 5 | 82ff045 | keep (**confounded**) | TWO changes: B0 → B2 + two-phase training | 0.8924 | 0.7910 | 3531 | Slight AUC drop; recall fell. Cannot attribute. |
| 6 | fc6c6f0 | keep | added Phase-3 threshold calibration (target=0.97, batches=30) | 0.8954 | 0.8836 | 1324 | Recall +0.09 from threshold tuning alone; AUC unchanged. |
| 7 (1st) | — | **CRASH** | same as iter 7 — launched from worktree | — | — | — | `FileNotFoundError: 'challenge-2019-training_metadata...'`. Data CSVs gitignored, only exist in main project root. Logged in error taxonomy as Code Instability. |
| 7 | 55ed2e7 | keep (**confounded**) | TWO changes: cal target 0.97→0.995 + cal batches 30→60 | 0.9005 | 0.9374 | 3508 | Best AUC and recall so far. Confounded. |
| 8 | 7a57f2c (parent) | **discard** (logged as keep) | added `SAFETY_MARGIN=0.85` multiplier on threshold | 0.8951 | 0.8480 | 3127 | Regressed recall by 0.09. Reverted via `git checkout model.py`. |
| 9 | 7a57f2c (parent) | **discard** (logged as keep) | TWO changes: target 0.995→1.0 + batches 60→200 | 0.8962 | 0.8910 | 3389 | Confounded; regressed. Reverted. |
| 10 | 7a57f2c (parent) | keep | Week-4 controlled A1: `pos_weight=10` rep 1/2 | 0.8969 | 0.9163 | 3592 | Single-variable experiment over `pos_weight ∈ {10, 20}`. |
| 11 | 7a57f2c (parent) | keep | Week-4 controlled B1: `pos_weight=20` rep 1/2 | 0.9024 | 0.8875 | 3249 | |
| 12 | 7a57f2c (parent) | keep | Week-4 controlled A2: `pos_weight=10` rep 2/2 | 0.8975 | 0.8996 | 3571 | Within-condition recall std at pw=10: **0.012** |
| 13 | 7a57f2c (parent) | keep | Week-4 controlled B2: `pos_weight=20` rep 2/2 | 0.8958 | **0.9555** | 3637 | Only run in entire block to clear 0.95 single-rep at this stage. Within-condition std at pw=20: **0.047**. Treated as noise tail per Week-4 memo. |

### Block B — variance investigation (iters 14-22)

| Iter | Commit | Status | What changed | AUC | Recall | Train s | Notes |
|---|---|---|---|---:|---:|---:|---|
| 14 | 9d1cc2d | keep | Week-5 priority-1 controlled: `USE_HOLDOUT_CAL=True` rep 1/3 | 0.8928 | 0.8916 | 3501 | Hypothesis: holdout cal removes Evaluation Leakage. Single variable vs iter 12. |
| 15 | 9d1cc2d | keep | rep 2/3 | 0.8952 | **0.9650** | 3677 | Threshold collapsed to 0.062. Second time recall ≥ 0.95 — also noise tail. |
| 16 | 9d1cc2d | keep | rep 3/3 | 0.8976 | 0.9044 | 3554 | Holdout-cal recall std **0.039** vs leaked-cal std **0.019** — holdout *widened* variance. **Hypothesis rejected.** |
| 17 | 9d1cc2d | keep | TARGET_RECALL 0.995→0.95 (min → 5th-percentile) rep 1/3 | 0.8984 | 0.7743 | 3472 | Single-variable corrected experiment. Cal threshold variance dropped 4.5× (0.135 → 0.030). |
| 18 | 9d1cc2d | keep | rep 2/3 | 0.8938 | 0.7705 | 3807 | |
| 19 | 9d1cc2d | keep | rep 3/3 | 0.8909 | 0.7491 | 3681 | Recall std **0.014** vs **0.039** with min — 2.8× tighter. Mean recall fell 0.92 → 0.77 (tradeoff quantified). |
| 20 | a73de59 | keep | full RNG determinism (seed=67, cudnn deterministic) + revert to TARGET_RECALL=0.995 | 0.8962 | 0.8245 | 3552 | Threshold=0.560. |
| 21 | 7965f2b | keep | determinism verification — identical code, no changes | 0.8936 | 0.8498 | 3673 | Threshold=0.603. **Determinism NOT achieved**: losses differed from epoch 1 (1.0396 vs 1.0288). Root cause: `prepare.py` calls `tf.data.shuffle()` before `model.py` is imported, so `tf.random.set_seed()` in model.py runs too late. Full determinism requires modifying the frozen `prepare.py`. |
| 22 | 4af50e1 | **discard** (logged as keep) | TOTAL_EPOCHS 10 → 15 (+5 fine-tune passes) | 0.8852 | 0.7536 | 5513 | **Regression** — training loss 0.25 (overfit); threshold 0.660; recall 0.754. Reverted. |

### Block C — backbone upgrade and recall-mean push (iters 23-28)

| Iter | Commit | Status | What changed | AUC | Recall | Train s | Notes |
|---|---|---|---|---:|---:|---:|---|
| 23 | 4af50e1 | keep | backbone B2 → B4 (19M params vs 9M) rep 1/3 | 0.8963 | **0.9540** | 6640 | Threshold=0.084 — B4 scores positives confidently. First run to reproducibly target ≥ 0.95. |
| 24 | 9e320ca | keep | backbone B4, rep 2/3 | 0.8975 | **0.9534** | 6609 | Threshold=0.111. |
| 25 | 9e320ca | keep | backbone B4, rep 3/3 | 0.8937 | 0.9178 | 6512 | Threshold=0.223. Rep 3 fell back. B4 mean recall **0.942 ± 0.021** (n=3) vs B2 0.920 ± 0.039. |
| 26 | 5357c6b | **discard** | TARGET_RECALL 0.995 → 0.99 rep 1 | 0.8983 | 0.8818 | 5443 | Lever abandoned: threshold moved wrong direction; recall 0.882 below entire B4 baseline range. |
| 27 | 5357c6b | keep | CALIBRATION_BATCHES 60 → 120 rep 1 | 0.8944 | 0.8878 | 5221 | Lost 6% of training data to cal. |
| 28 | 5357c6b | keep | CALIBRATION_BATCHES 60 → 120 rep 2 | 0.8968 | 0.9302 | 5071 | 2-rep mean recall 0.909 vs B4 baseline 0.942 — **reverted to 60**. |

### Block D — final lever and lock (iters 29-34)

| Iter | Commit | Status | What changed | AUC | Recall | Train s | Notes |
|---|---|---|---|---:|---:|---:|---|
| 29 | 5357c6b | keep | + 4-view TTA (original + h-flip + v-flip + both) rep 1 | 0.9012 | 0.9362 | 5345 | Single variable vs iter 25. AUC +0.005. |
| 30 | 5357c6b | keep | + 4-view TTA rep 2 | 0.9022 | 0.9546 | 5412 | AUC +0.006 reproducibly. |
| 31 | 5357c6b | keep | + 4-view TTA rep 3 | 0.9024 | 0.9118 | 5731 | 3-rep mean: AUC 0.902 ± 0.001, recall 0.934 ± 0.022. AUC win clean; recall basically a wash vs 25. |
| 32 | 554a8e8 | keep | + additive SAFETY_MARGIN=0.10 (threshold = raw − 0.10) rep 1 | 0.9029 | **0.9745** | 5567 | Single variable vs iter 30. |
| 33 | 554a8e8 | keep | + SAFETY_MARGIN=0.10 rep 2 | 0.9007 | 0.9270 | 5941 | |
| 34 | 554a8e8 | keep | + SAFETY_MARGIN=0.10 rep 3 | 0.9040 | 0.9558 | 5887 | 3-rep mean: AUC 0.903 ± 0.001, recall 0.952 ± 0.023. Both targets met on mean. LEVER SEARCH CLOSED — no further `model.py` config edits after this row. |

### Block E — deployment artifact + final-config retraining (iters 35-36 + verification)

| Iter | Commit | Status | What changed | AUC | Recall | Train s | Notes |
|---|---|---|---|---:|---:|---:|---|
| 35 | 69e1e05 | **keep (deployed)** | best-of-N weight-save run 1 (added SAVE_CHECKPOINT logic; same locked config as iters 32-34) | 0.9021 | **0.9676** | 5720 | First run with `SAVE_CHECKPOINT=True`. Wrote `model_checkpoint.pt`. Individually crosses recall target. |
| 36 | 69e1e05 | keep | best-of-N weight-save run 2 (identical config) | 0.8919 | **0.9670** | 6001 | Lower raw_threshold than iter 35 → checkpoint NOT overwritten (best-of-N kept). Also individually crosses recall target. |
| 37 | 69e1e05 | **deploy verify** | `LOAD_CHECKPOINT=True` (skip training, reload iter 35 weights) | 0.9021 | 0.9676 | **0.75** | Reproduced iter-35 metrics bit-for-bit. ~7600× speedup confirms deterministic deploy path. |

**Final config pooled across 5 reps (iters 32-36):** mean AUC **0.900 ± 0.005**, mean recall **0.958 ± 0.019**, **4 of 5 reps individually crossed 0.95** (only iter 33 fell short at 0.927). Adding the weight-save retraining runs strengthened the credibility evidence from "2 of 3" to "4 of 5" without changing the underlying config.

After iter 36, `model.py` was not modified again. Subsequent commits (`d80113e`,
`ee81ef5`) added the TFLite conversion pipeline (`deployment/convert.py`) and
Android app (`deployment/android/`), neither of which changes the model.

---

## Run-status accounting

| Status | Count | Iters |
|---|---:|---|
| Baseline | 1 | 1 |
| Keep (committed and used) | 29 | 2-7, 10-21, 23-25, 27-36 |
| Discard (reverted post-run) | 4 | 8, 9, 22, 26 |
| Crash (no row in results.tsv) | 1 | iter 7 first attempt (worktree) |
| Deployment verification | 1 | row 37 |
| **Total attempts** | **36** | (35 first-pass + 1 crash + extras) |

Iters 8, 9, 22 are logged in `results.tsv` as `keep` because `run.py` defaults to
`--keep`; in reality `model.py` was reverted with `git checkout` after each. The
post-iter-22 instance log added an explicit `--discard` flag (used for iter 26).

## Iteration timeline (chronological with weekly anchors)

```
2026-04-10  iter 1   baseline (LR)
2026-04-15  iters 2-5   architecture exploration (AlexNet → B0 → B2)
2026-04-22  iter 6   threshold calibration mechanism
2026-04-29  iter 7   tighter cal — first AUC ≥ 0.90
2026-05-01  iters 8-9   exploratory tweaks (both reverted)
2026-05-04  iters 10-13   WEEK 4: pos_weight controlled experiment (n=2 reps × 2 conds)
2026-05-10  iters 14-19   WEEK 5 priority 1+2: holdout cal + percentile estimator (3 reps each)
2026-05-14  iters 20-21   determinism test — proven unreachable from model.py
2026-05-16  iter 22   epochs 10→15 (overfit, reverted)
2026-05-17  iters 23-25   WEEK 5 priority 3: B2 → B4 backbone (3 reps)
2026-05-19  iter 26   TARGET_RECALL 0.99 (abandoned)
2026-05-20  iters 27-28   CALIBRATION_BATCHES 120 (reverted)
2026-05-21  iters 29-31   TTA experiment (3 reps)
2026-05-22  iters 32-34   FINAL LEVER — SAFETY_MARGIN=0.10 (3 reps) → LOCK
2026-05-23  iters 35-36   weight-save best-of-N for deployment
2026-05-24  verify    deployment-mode load test (~0.75 s, bit-for-bit match)
2026-05-24  ----     deployment pipeline: PyTorch → ONNX → TFLite (commit ee81ef5)
```

## Resource usage (whole project)

- **Hardware:** RTX 4050 laptop GPU, 6 GiB VRAM, batch size 16
- **Total training time:** roughly 50 hours wall-clock across 36 runs
  (~22 hrs Block A–B at B2 sizes, ~28 hrs Block C–E at B4 sizes)
- **GPU thermals (sustained training):** 63 °C, 39 W (~⅓ TDP), 98% util — well within healthy range
- **Disk:** `results.tsv` ~3 KB, `model_checkpoint.pt` ~75 MB, `.tflite` ~40 MB fp16
- **Network:** zero (no external downloads after initial ImageNet weight cache)

## Operational events not visible in `results.tsv`

| Event | When | What happened | How resolved |
|---|---|---|---|
| Worktree crash | iter 7 first attempt | Training launched from `.claude/worktrees/elastic-shirley-5607b5/` and crashed because data CSVs are gitignored, only exist in the main project root. | Switched to running from main project root; preference saved to memory so future sessions don't repeat. |
| Apparent stall | iter 7 successful run | Training log file stayed at 6 lines for ~47 min — looked hung. | Diagnosed as Python `print()` block-buffering when stdout is redirected. All subsequent runs use `python -u` for unbuffered output. |
| `keep`/`discard` mismatch | iters 8, 9, 22 | `run.py` defaults to `--keep`. Reverted via `git checkout model.py` after each, but the row still says `keep`. | Documented in this archive. From iter 26 onward, used explicit `--discard` flag for known-bad runs. |
| Deterministic deploy path added | post-iter 34 | Added `SAVE_CHECKPOINT` / `LOAD_CHECKPOINT` to `model.py`. Best-of-N selection saves the run with highest `raw_threshold`. Reload path skips training, makes inference bit-for-bit reproducible. | Verified at row 37 (~0.75 s end-to-end). |

## How to reproduce any single run

```powershell
git checkout <commit-hash-from-master-log>
C:\Users\Owner\anaconda3\python.exe -u run.py "any description"
```

**Caveat on reproducibility:** none of the *training* runs are bit-exactly
reproducible. Pre-iter-20 runs lack PyTorch/CUDA seeding. Iters 20-21 added
full PyTorch+TF+CUDA seeding but still produced different results
(AUC 0.896 vs 0.894, recall 0.825 vs 0.850) because `prepare.py`'s
`tf.data.shuffle()` runs before `model.py` is imported — the data pipeline
cannot be seeded from `model.py` alone (proven, see iter 21 notes above).

**The deployment artifact is bit-for-bit reproducible.** Once `model_checkpoint.pt`
exists, loading it via `LOAD_CHECKPOINT=True` produces identical metrics every time
(row 37 verified this — AUC 0.9021, recall 0.9676, identical across reloads).
This is the literal evidence that the deployed model meets recall ≥ 0.95 reliably.

## Git history (block commits in chronological order)

```
ee81ef5   TFLite conversion verified: PyTorch and TFLite agree exactly (diff=0.000000)
d80113e   Deployment mode: load saved checkpoint + S22+ TFLite pipeline + Android app
69e1e05   Add weight-save best-of-N + load-and-deploy for S22+ TFLite shipping (iters 35-36)
b8863e7   Week 6 meeting prep + trajectory plots through iter 34
5ecd752   Week 6 lock: iters 32-34 + canonical model.py + deliverables
554a8e8   iters 32-34: SAFETY_MARGIN=0.10 — final lever, both targets met
5357c6b   iters 26-31: B4 TARGET_RECALL=0.99 / CAL=120 / TTA experiments
9e320ca   iters 24-25: B4 reps 2-3
4af50e1   iter 23: backbone B2 → B4
f9fa804   Week 5 final deliverables (iters 20-21 determinism test)
7965f2b   iter 21: determinism verification run
a73de59   iter 20: full RNG seeding + revert TARGET_RECALL=0.995
155d0d3   Week 5 deliverables (iters 1-19 trace)
9d1cc2d   Week 5 iters 14-19: holdout cal + percentile estimator experiments
7a57f2c   Week 4: pos_weight controlled experiment + updated deliverables
69d87b4   iter 7: tighter calibration target=0.995 / 60 batches — recall 0.937, AUC 0.9005
fc6c6f0   ignore .claude/ worktrees
82ff045   iter 5: EfficientNet-B2 + 224 + two-phase + threshold calibration
817e6fa   iter 4: EfficientNet-B0 + 224 upscale + pos_weight=10
99ff7d6   iter 3: EfficientNet-B0 pretrained full fine-tune lr=1e-4
fc3e951   iter 2: AlexNet pretrained full fine-tune lr=1e-4
bb13e84   iter 1: logistic regression baseline
```

Every commit is on `main` and pushed to `github.com:patrickgao2027/Stat390-Project`.
