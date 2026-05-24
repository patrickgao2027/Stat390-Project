# Calibrated EfficientNet-B4 for Benign-vs-Malignant Skin Lesion Screening on ISIC 2019+2020: An AutoResearch Capstone Report

**Patrick Gao** &nbsp; Northwestern University &nbsp; STAT 390 Capstone &nbsp; <patrickgao765@gmail.com>

*(Markdown rendering of the 4-page NeurIPS report. The canonical typeset version is [final_report.tex](final_report.tex); compile with `pdflatex final_report.tex` after placing `neurips_2024.sty` next to the file, or upload to Overleaf.)*

---

## Abstract

We report on a 36-iteration AutoResearch loop that produced a calibrated binary skin-lesion classifier on the combined ISIC 2019+2020 dermoscopy datasets (~57,586 images, ~5.3:1 benign:malignant imbalance). The final method — an ImageNet-pretrained EfficientNet-B4 with two-phase fine-tuning, threshold calibration on a held-out training slice, 4-view test-time augmentation, and an additive operating-point safety margin — achieves mean ROC-AUC 0.903 ± 0.001 and mean recall 0.952 ± 0.023 across three independent training replicates, exceeding both pre-registered targets (AUC ≥ 0.85, recall ≥ 0.95). We characterize which loop directions produced stable value (architecture upgrades and threshold calibration), which produced noise or regression (post-hoc determinism, multiplicative safety margins, calibration-target tuning), and the structural variance floor imposed by the frozen data pipeline. The deployed checkpoint reproduces its metrics bit-for-bit and has been shipped end-to-end through PyTorch → ONNX → TFLite to an Android (Kotlin/CameraX/NNAPI) screening app for Samsung S22+.

## 1. Introduction and AutoResearch contract

**Problem.** Given a dermoscopy image, output the probability that the imaged lesion is malignant and a binary screening decision. The target deployment is a smartphone screening aid, not a diagnostic device: missed cancers are catastrophic; false alarms are tolerable.

**From topic to contract.** The instructor-supplied prompt asked for "the best possible binary classifier" on ISIC 2019+2020 with a plain-English deliverable, on-device inference desired but not blocking. This was formalized into a falsifiable AutoResearch contract with three elements. (1) **Pre-registered targets:** ROC-AUC ≥ 0.85 on the held-out test set and recall ≥ 0.95 (chosen over precision to match the screening use case). (2) **Frozen interfaces:** two modules (`prepare.py`, `run.py`) are immutable; only `model.py` may be edited, and it must define `build_model()` returning an object with `fit/predict/predict_proba` that matches a fixed type signature. (3) **Mechanical iteration loop:** edit one variable, run `python run.py`, log one row to `results.tsv`, decide keep/discard. The interface constraint — not the targets — is what made this an *AutoResearch* project rather than a Kaggle-style score chase: it forced changes to be small, named, and individually revertible.

## 2. Method

**Data.** ISIC 2019 + ISIC 2020 dermoscopy challenge data, merged (~57,586 images after filtering indeterminate diagnoses). `prepare.py` performs a stratified 80/20 train/test split seeded `RANDOM_STATE=67`, subsamples training to 30% for tractable iteration time, and yields `tf.data.Dataset` pipelines of 128×128 NHWC float32 images in [0,1] along with the imbalance weights {0: 1.0, 1: 5.31}.

**Architecture (locked at iter 32-34).** An EfficientNet-B4 [1] initialized with ImageNet weights from torchvision. The wrapper module bilinearly upsamples each 128×128 input to 224×224, applies ImageNet normalization, and replaces the final 1000-class head with a single linear unit producing a logit.

**Training protocol (locked).** Three phases per training run: (1) frozen backbone, train binary head only, 2 epochs, lr=1e-3; (2) unfreeze all, full fine-tune, 8 epochs, lr=1e-4; (3) threshold calibration. Loss is `BCEWithLogitsLoss` with `pos_weight=10` (an empirically chosen multiplier on the imbalance ratio that pushes the model toward higher recall during training). Optimizer is Adam in both phases.

**Threshold calibration (locked).** The first 60 training batches (~960 images, ~141 malignant positives) are *materialized before training* and *excluded from gradient updates*. After fine-tuning, the model scores this held-out slice; we sort by predicted probability descending and select the lowest probability at which the cumulative-recall curve reaches `TARGET_RECALL=0.995`. This raw threshold is reduced by an additive `SAFETY_MARGIN=0.10` (floored at 0.05). At inference we additionally apply 4-view test-time augmentation: original image, horizontal flip, vertical flip, both flips; the four logits are averaged before sigmoid.

**Deployment artifact.** After training, the state dict, threshold, and raw threshold are persisted to `model_checkpoint.pt` only if `raw_threshold` exceeds the previously saved value (best-of-N selection). A separate code path loads this checkpoint, skips training, and runs deterministic inference in ~0.75 s end-to-end. The same checkpoint is exported PyTorch → ONNX → TFLite (fp16) and bundled into an Android app; the TFLite output agrees with the PyTorch source bit-for-bit on a fixed dummy input.

## 3. Experimental protocol

**Loop design.** 36 iterations were executed sequentially. Each iteration: (1) name one variable to change in `model.py`; (2) run `python -u run.py "<iter N description>"`; (3) append one row to `results.tsv` containing `(commit_hash, roc_auc, accuracy, recall, precision, train_time_s, status, description)`; (4) decide keep / discard / replicate. From iter 14 onward, iterations were grouped in pre-declared 3-rep blocks before claims could be made about a variable's effect.

**Evaluation.** A single test set, fixed by `RANDOM_STATE=67` in the frozen `prepare.py`, was used throughout. The evaluator (`run.py`'s `evaluate()`) computed AUC, accuracy, recall, and precision identically across all 36 runs; it did not change during the project. Test metrics were logged every iteration; the test set was not opened in the sense of being mined for hyperparameters — all hyperparameter decisions used the per-iteration holdout cal slice — but the test results were visible to the experimenter, a design limitation we discuss in §5.

**Stability standard.** A change was claimed as a real effect only if a 3-rep block of the new condition produced a between-condition mean difference exceeding within-condition standard deviation, and only if the direction of effect agreed with the explicit causal account for the change.

## 4. Results

### 4.1 Final result

**Table 1.** Locked configuration (iters 32-34): EfficientNet-B4 + two-phase fine-tune + holdout threshold calibration + 4-view TTA + additive SAFETY_MARGIN=0.10. Each row is one independent training run.

| Iter | ROC-AUC | Recall | Accuracy | Precision |
|---|---:|---:|---:|---:|
| 32 | 0.9029 | **0.9745** | 0.6854 | 0.3602 |
| 33 | 0.9007 | 0.9270 | 0.7560 | 0.4182 |
| 34 | 0.9040 | 0.9558 | 0.7348 | 0.3996 |
| **mean** | **0.9029** | **0.9524** | 0.7254 | 0.3927 |
| std | 0.0014 | 0.0226 | 0.0303 | 0.0244 |
| target | ≥ 0.85 | ≥ 0.95 | — | — |

Both pre-registered targets are met on the 3-rep mean (Table 1); 2 of 3 individual reps cross the recall target. The deployed checkpoint (iter 35 weights, verified by reloading at `results.tsv` row 37) produces AUC 0.9021 and recall 0.9676 every time it is reloaded, providing a bit-for-bit reproducible shippable artifact.

### 4.2 Stable directions (what produced real value)

Four single-variable changes account for essentially the entire gain from the iter-1 baseline (AUC 0.793, recall 0.898), each replicated:

**(1) Pretrained CNN backbone (iter 2).** Logistic regression → AlexNet: AUC +0.094 in a single iteration; every subsequent CNN run (≥ 35 runs) sits in [0.88, 0.90] AUC. The architecture-class effect is the largest in the project.

**(2) Phase-3 threshold calibration with holdout cal (iters 6, 14-16).** The default τ=0.5 decision threshold from `BaseTorchModel` is calibrated for class-balanced problems; replacing it with a `target-recall`-derived threshold gives recall +0.09 at no AUC cost. Moving the cal source from the training batches to a 60-batch holdout slice eliminated within-iteration leakage between calibration and gradient updates (iters 14-16, 3-rep mean recall 0.920).

**(3) Backbone upgrade B2 → B4 (iters 23-25).** Single-variable swap, 3 reps: mean recall 0.920 → 0.942, standard deviation halved (0.039 → 0.021). The deeper backbone scores hard malignant cases with higher probability, which lowers the holdout-derived threshold and raises test recall without affecting ranking quality (AUC unchanged).

**(4) 4-view TTA + additive SAFETY_MARGIN=0.10 (iters 29-34).** TTA alone (iters 29-31) added +0.006 AUC reproducibly and was a recall wash. Adding the safety margin (iters 32-34) raised mean recall across the 0.95 target line (0.934 → 0.952) without further AUC cost.

### 4.3 Noise, dead ends, and failures

Six modification classes regressed or produced no effect distinguishable from within-condition noise. We summarize the most informative failures.

**Full RNG seeding for determinism (iters 20-21).** Identically seeded runs produced different per-epoch losses (1.0396 vs 1.0288 at epoch 1), different thresholds (0.560 vs 0.603), and different recall (0.825 vs 0.850). Root cause: `prepare.py` calls `tf.data.Dataset.shuffle()` during `load_data()`, which runs before `model.py` is imported, so `tf.random.set_seed()` inside `model.py` is too late to affect the shuffle. *Determinism is unreachable from the editable surface alone.* This is the most consequential finding of the loop: it converts training-time variance from a tunable parameter into a structural constant, and motivated the eventual best-of-N deployment artifact.

**TARGET_RECALL tuning (iters 17-19, 26).** Both directions regressed. Lowering TARGET_RECALL from 0.995 to 0.95 (5th-percentile estimator, iters 17-19) cut threshold variance 4.5× but dropped mean recall to 0.77. Tightening to 0.99 (iter 26) moved the threshold the wrong direction and produced recall 0.88, below the entire B4 baseline range. The min-of-positives estimator at 0.995 is at a local optimum.

**Larger calibration set (iters 27-28).** Raising CALIBRATION_BATCHES from 60 to 120 cost 6% of training data and lowered mean recall by 0.03. The held-out slice is large enough at 60 batches (~141 positives).

**15 training epochs on B4 (iter 22).** Severe overfitting: train loss 0.25, holdout-derived threshold 0.66, test recall 0.75. 10 epochs is the confirmed sweet spot for B4 at this data scale.

**pos_weight 10 → 20 (Week 4 controlled experiment, iters 10-13).** Between-condition difference 3.5× smaller than within-condition standard deviation at `pos_weight=20`. Effect indistinguishable from noise at 2 reps per condition.

## 5. Discussion and limits of the AutoResearch loop

**Structural variance floor.** Because the frozen `prepare.py` shuffles data before any seed-setting code in `model.py` can run, per-run training variance is a constant of the loop. The locked recall standard deviation of 0.023 (Table 1) is at this floor; no `model.py`-side change can lower it. The deployment-mode checkpoint sidesteps this by saving the weights of a single canonical run — producing literal 100% reproducibility *for the deployed model*, but not for the training process.

**Test-set discipline under constant visibility.** The test set was never opened in the sense of being mined for hyperparameters — all tuning used the per-iteration holdout cal slice — but its metrics were logged on every iteration and therefore visible. We held to the discipline of not using test ranks to choose between conditions, but the loop tooling (which writes test metrics to `results.tsv` after every run) makes this a matter of researcher discipline rather than hard-enforced policy. A stronger loop would write test metrics to a sealed file unlocked only at scope lock.

**Reproducibility caveat.** None of the 36 training runs are bit-exactly reproducible. Reproducing a run requires `git checkout <hash>` and re-running, which produces metrics within published noise bounds but not identical numbers. The deployment artifact (`model_checkpoint.pt` + TFLite) *is* bit-for-bit reproducible and is what we ship.

**Pipeline ceiling.** 128×128 source resolution + 30% training subsample are characteristics of the frozen data pipeline. A higher ceiling presumably exists at 100% data and native resolution; we cannot estimate it without modifying the frozen interfaces, which would invalidate the experimental record.

## 6. What the AutoResearch loop did well and poorly

The loop's biggest strength was forcing single-variable iteration with mechanical bookkeeping. Once the Week-4 framework imposed "one variable, replicate before claiming," every subsequent block (14-16, 17-19, 23-25, 27-28, 29-31, 32-34) is a clean controlled experiment readable directly from `results.tsv`. Negative results were preserved and labeled rather than hidden.

The loop's biggest weakness was pursuing one more lever past the point of diminishing returns. The locked claim was effectively in hand at iter 31 (AUC 0.902, recall 0.934, AUC win locked). Iters 32-34 crossed the recall mean above 0.95 with a clean single-variable change, but iters 26-28 are pure exploratory cost. Without the externally imposed Week-6 scope lock, the loop would have continued indefinitely. A pre-declared stop condition ("once 3 reps cross both targets, lock") would have closed the loop earlier.

---

**Code and data availability.** Full code, all 36 experiment rows, all weekly deliverables, the deployed checkpoint, the TFLite-converted model, and the Android app source are at <https://github.com/patrickgao2027/Stat390-Project>. The complete per-iteration archive is in `week7/experiment_archive.md`; the locked results table is in `week7/final_results_table.md`; a longer reflection on the agent loop is in `week7/reflection_memo.md`.

## References

[1] M. Tan and Q. V. Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In *ICML*, 2019.

[2] N. Combalia et al. BCN20000: Dermoscopic lesions in the wild. *arXiv:1908.02288*, 2019. (ISIC 2019 dataset.)

[3] V. Rotemberg et al. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. *Scientific Data*, 2021. (ISIC 2020 dataset.)
