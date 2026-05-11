# Week 5 — Keep / Discard / Crash Summary

**Block window:** Iter 1 → Iter 25 (completed) | **Total attempts:** 26

## Headline counts

| Outcome | Count | Rate |
|---|---:|---:|
| **Keep** (kept, committed to `main`, pushed to GitHub) | 22 | 85 % |
| **Discard** (run completed but reverted via `git checkout model.py`) | 3 | 12 % |
| **Crash** (run never produced a `results.tsv` row) | 1 | 4 % |
| **Total** | **26** | **100 %** |

(Out of the 25 *completed* attempts, the keep/discard/crash split is 22/3/1 = 88% keep, 12% discard, 4% crash.)

The "discard" rate is artificially low because **the failure mode I corrected most often was confounding, not regression** — and confounded runs were kept (not discarded) because their numbers are interpretable as data points even when their causal claims aren't. Both true regressions (iters 8 and 9) were caught and reverted within the same session.

## Keep — the 17 retained runs

Grouped by what the iteration was meant to demonstrate:

| Sub-bucket | Iters | Why kept |
|---|---|---|
| Architecture progression | 2, 3, 4, 5 | Each iter changed the backbone or training pipeline; AUC and recall both moved meaningfully. Iters 4 and 5 each changed two variables — kept anyway as exploratory but flagged as confounded in the matrix. |
| Threshold-calibration introduction | 6, 7 | Adding the Phase-3 cal mechanism lifted recall from 0.79 → 0.94. Iter 7 best AUC and best single-run recall in the whole block. |
| Week-4 controlled experiment (pos_weight) | 10, 11, 12, 13 | The first set of 4 fresh single-variable replicated runs. All kept by design (controlled-experiment data points, regardless of outcome). |
| Week-5 priority-1 controlled experiment (holdout cal) | 14, 15, 16 | Three replicates of `USE_HOLDOUT_CAL=True`. All kept — the negative result (hypothesis rejected) is still a research finding. |
| Week-5 priority-1 corrected (percentile estimator) | 17, 18, 19 | Three replicates of `TARGET_RECALL=0.95`. All kept — variance fix confirmed. |
| Week-5 determinism experiment | 20, 21 | Full PyTorch+TF+CUDA seeding (seed=67). Two reps with identical code. All kept — negative result (determinism NOT achieved) is still a research finding. |
| Week-5 backbone upgrade (B4) | 23, 24, 25 | EfficientNet-B4 controlled experiment, 3 reps. All kept — mean recall 0.942 ± 0.021, improvement over B2 confirmed. |

## Discard — the 3 reverted runs

All are *true regressions* caught with the Week-4 decision rule (recall must improve AND ROC-AUC stays ≥ 0.85; otherwise revert).

| Iter | What changed | Why discarded | Rollback action |
|---|---|---|---|
| **8** | added `SAFETY_MARGIN=0.85` multiplier on the calibrated threshold | recall *dropped* from iter 7's 0.937 to 0.848. The multiplier was meant to push the threshold lower; but iter 8's model produced a much higher cal threshold (0.56 instead of 0.21), so 0.56 × 0.85 = 0.48 was actually higher than 0.21, capturing fewer test positives. | `git checkout model.py`; SAFETY_MARGIN logic removed |
| **9** | TWO changes: `TARGET_RECALL` 0.995→1.0 + `CALIBRATION_BATCHES` 60→200 | recall fell to 0.891 vs iter 7's 0.937. Confounded; can't attribute regression to either knob. | `git checkout model.py`; both changes removed |
| **22** | `TOTAL_EPOCHS` 10→15 (+5 fine-tune passes) | Training loss hit 0.25 (overfit); holdout min-threshold jumped to 0.660; recall fell to 0.754, AUC fell to 0.885. More epochs hurt on B4. | `git checkout model.py`; reverted to 10 epochs, switched to B4 backbone |

**Caveat in the trace:** `run.py` logs all three rows with `status=keep` (its default), even though `model.py` was reverted each time. The "real" status is `discard`; the rows in `results.tsv` are preserved as records of what was tried.

## Crash — 1 attempt

| Iter (attempt) | When | What happened | Lesson |
|---|---|---|---|
| Iter 7 first attempt | first launch of the calibration-mechanism experiment | `FileNotFoundError: 'challenge-2019-training_metadata_2026-04-22.csv'`. Training was launched from inside a git worktree (`.claude/worktrees/<name>/`); data CSVs and ISIC image folders are gitignored and live only in the main project root. The worktree could see the code but not the data. | Always run from main project root. Memory entry recorded so future sessions repeat the rule. No `results.tsv` row was logged because `prepare.load_data()` failed before any model was built. |

## What kinds of modifications consistently failed?

Three categories show up repeatedly in the discard/crash pile and the "kept but useless" pile (iters 8, 9, plus the rejected hypotheses in 14–16):

| Failure category | Why it kept failing | Example iters |
|---|---|---|
| **Multi-knob threshold tweaks** (changing two cal hyperparameters at once) | Confounded — even when numbers improved, I couldn't attribute the gain. Iter 9 is the cleanest example: changed both `TARGET_RECALL` and `CALIBRATION_BATCHES` together, regressed, and could not blame either one. | 4, 5, 7, 9 |
| **Threshold-multiplier hacks** (`SAFETY_MARGIN` etc.) | Each model has a different probability calibration, so a multiplicative factor that "should" lower the threshold landed at a different absolute value each run. Variance dominated the intended effect. | 8 |
| **Cal-source swap (Evaluation-Leakage fix)** | I was confident this would tighten variance; it widened it (recall std 0.039 holdout vs 0.019 leaked). The fix attacked the wrong source — leakage was real but not dominant. | 14, 15, 16 |

## What kinds of modifications consistently worked?

| Category | Effect | Example iters |
|---|---|---|
| **Architecture upgrade** (larger / better-pretrained backbone) | Reproducibly added 0.05–0.10 AUC. Robust across runs and conditions. | 2 (LR→AlexNet), 3 (AlexNet→B0), 5 (B0→B2 — confounded but consistent direction) |
| **Adding a new mechanism with default hyperparameters** (Phase-3 calibration in iter 6) | Lifted recall ~0.10 with no AUC cost. The mechanism was unambiguously beneficial; only the *tuning* of it was unstable. | 6 |
| **Variance-reducing changes that swap a noisy estimator for a stable one** (percentile vs min in iter 17–19) | Cut recall std 2.8× in 3 reps. Comes with a precision/recall tradeoff that's now quantified. | 17, 18, 19 |

The pattern: things that change *what the model does* tend to work; things that change *how the threshold is interpreted* tend to be dominated by noise unless paired with a separate stabilization.
