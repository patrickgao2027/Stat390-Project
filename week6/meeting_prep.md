# Week 6 Meeting Prep

Consolidates the three meeting artifacts called for in the Week 6 brief:
lightning-round answers, story-lock worksheet, and live scope-locking Q&A.

---

## Lightning Round (90 seconds, four answers)

**1. Current best direction (single direction committed to)**

EfficientNet-B4 with two-phase fine-tuning, holdout-calibrated decision threshold, 4-view test-time augmentation, and an additive operating-point safety margin of 0.10 — locked at iter 34.

**2. Supporting evidence**

Iters 32-34 (`results.tsv` rows 33-35) — three replicates of the locked config:
- ROC-AUC: 0.903 ± 0.001 (target 0.85 ✅)
- Recall: 0.952 ± 0.023 (target 0.95 ✅ on mean; 2/3 reps individually ≥ 0.95)
- Best single rep: iter 32 — AUC 0.903, recall 0.974

**3. What I am dropping (officially abandoned)**

- TARGET_RECALL ≠ 0.995 (iters 17-19 at 0.95, iter 26 at 0.99 — both regressed)
- CALIBRATION_BATCHES > 60 (iters 27-28 at 120 — recall mean dropped 0.03)
- > 10 epochs (iter 22 at 15 — overfit, recall 0.75)
- Full RNG determinism (iters 20-21 — unreachable from `model.py` because frozen `prepare.py` shuffles before import)
- Backbone change beyond B4 (B5/B6 — diminishing returns, ~1.5× compute)
- POS_WEIGHT > 10 (iters 11/13 on B2 — within within-condition noise)
- Mixup/CutMix, focal loss, AdamW, ensemble training, pipeline unfreezing — all out of scope
- TFLite deployment as graded deliverable — stays a stretch goal

**4. Only 2-3 things left between now and final presentation**

1. Plain-English summary memo + one Grad-CAM/saliency figure to prove lesion attention
2. Save-weights checkpoint + reproducibility appendix (one canonical training run with state_dict serialization)
3. Final presentation deck + dry-run

---

## Story-Lock Worksheet

**1. My project now shows that:**

A single EfficientNet-B4 with a calibrated, recall-biased operating point can classify benign vs. malignant skin lesions on ISIC 2019+2020 at ROC-AUC 0.90 and reach 95% recall on average — meeting both Week 0 success criteria from inside the frozen 30%-data, 128×128 source pipeline.

**2. The strongest evidence is:**

Three replicates of the identical locked config (iters 32-34) — AUC 0.903 ± 0.001 and recall 0.952 ± 0.023 — produced after 34 iterations of single-variable controlled experiments documented row-by-row in `results.tsv`.

**3. I am no longer doing:**

Architecture exploration, threshold-target search, calibration-set-size search, training-length tuning, full-determinism work, backbone scaling beyond B4, alternative losses, mixup/cutmix, ensembles, pipeline unfreezing. **Lever search is closed.**

**4. My final 2-week plan is:**

Week 7 = writeup, plots, Grad-CAM, weight-save for deployment determinism, presentation deck. Week 8 = optional ONNX/TFLite deployment stretch, presentation polish, submit on Day 14. (Full schedule in `week6/final_two_week_plan.md`.)

---

## Live Scope-Locking Q&A (five-question deep-dive)

**Q1. What is the strongest claim your evidence supports?**

"On ISIC 2019+2020 with the frozen 30%-subsample / 128×128-source pipeline, an EfficientNet-B4 with two-phase fine-tuning, holdout threshold calibration, 4-view TTA, and a 0.10 additive safety margin achieves mean ROC-AUC 0.903 ± 0.001 and mean recall 0.952 ± 0.023 across 3 replicates."

**Q2. What caused the gain?**

Four single-variable wins, each tied to a documented iteration block:
- Logistic regression → pretrained CNN backbone (iter 2): AUC +0.09
- Phase-3 threshold calibration with holdout cal set (iters 6, 14-16): recall +0.09 without AUC cost
- B2 → B4 backbone (iters 23-25): recall std halved (0.039 → 0.021)
- TTA + SAFETY_MARGIN=0.10 (iters 29-34): AUC +0.006 reproducibly, mean recall crossed 0.95

The cleanest controlled comparison driving the final lever (TTA off vs. on) is iters 23-25 vs 29-31; the final lever (SM 0 vs 0.10) is iters 29-31 vs 32-34.

**Q3. What are you explicitly dropping now?**

Listed in section "What I am dropping" above and in `week6/agent_strategy.md` "Dropped directions" table. The most recent additions to that list (added during Week 6 itself): TARGET_RECALL=0.99 (iter 26), CALIBRATION_BATCHES=120 (iters 27-28), and any further SAFETY_MARGIN escalation beyond 0.10.

**Q4. What must still be confirmed?**

Whether the saved-weight deployment artifact reproduces the iter-32-or-34 metrics bit-for-bit when reloaded for inference. This is the single open verification step in the Week 7 plan (Day 5). If it does, the deployed model has true 100% reproducibility; if it doesn't, the reduced claim "mean recall 0.952 across 3 reps" stands as-is.

**Q5. What will be in the final presentation?**

8-10 slides:
1. Problem framing (skin cancer screening; AUC ≥ 0.85, recall ≥ 0.95)
2. Data (ISIC 2019+2020, ~57K images, 5.3:1 imbalance, frozen 30% subsample)
3. Method (B4 + two-phase + holdout cal + TTA + safety margin)
4. Results table (AUC 0.903, recall 0.952) with the 34-iter trajectory plot
5. Ablation: the 4 wins and the 7 abandoned levers
6. Honest limits: variance source, precision tradeoff, pipeline ceiling
7. Grad-CAM figure proving the model attends to the lesion
8. Deployment artifact: saved-weights checkpoint + (stretch) TFLite demo
9. What's next (out of scope for this project)
10. Q&A
