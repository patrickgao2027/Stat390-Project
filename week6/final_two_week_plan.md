# Locked Final Two-Week Plan (Week 6 → Submission)

## Status at lock (end of Week 6)

- Both success criteria met: AUC 0.903 (target 0.85), mean recall 0.952 (target 0.95)
- Canonical config: `model.py` iter 32-34 (B4 + TTA + SAFETY_MARGIN=0.10)
- Experimental loop **CLOSED** — no further iterations

## Week 7 (Days 1-7): write-up + visualization

| Day | Task | Time | Owner |
|---|---|---|---|
| 1 | Polish four Week 6 deliverables; resolve any instructor feedback | 1-2 hr | self |
| 2 | Write the plain-English summary memo for non-technical stakeholders (1 page, integrates the project statement) | 2 hr | self |
| 3 | Generate trajectory + ablation plots (mirror `week5/make_plot.py` pattern); add through iter 34 | 1 hr | self |
| 4 | Generate one Grad-CAM / saliency-map figure on 5 sample test images — proves the model attends to the lesion, not artifacts | 3 hr | self |
| 5 | Save best-checkpoint weights from one canonical training run (add minimal save logic to model.py); deploy that single checkpoint for reproducibility appendix | 2 hr + 95 min training | self |
| 6 | Draft final presentation slide deck (8-10 slides: problem → data → method → results → limits → demo) | 4 hr | self |
| 7 | Practice presentation aloud; trim to time | 2 hr | self |

## Week 8 (Days 8-14): polish + deployment stretch + submission

| Day | Task | Time | Owner |
|---|---|---|---|
| 8 | Address any peer-review feedback from Week 7 deck | 2 hr | self |
| 9 | (Stretch) ONNX export from saved PyTorch checkpoint | 2 hr | self |
| 10 | (Stretch) ONNX → TFLite conversion; verify accuracy preserved within 0.5% | 3 hr | self |
| 11 | (Stretch) Smartphone demo notebook — load TFLite model, single-image inference | 2 hr | self |
| 12 | Final polish: README, repo cleanup, ensure `python run.py` reproduces canonical config | 2 hr | self |
| 13 | Dry-run final presentation; record video backup | 2 hr | self |
| 14 | **SUBMIT** | — | self |

## What's in scope (and locked)

1. The four Week 6 deliverables (this folder)
2. Plain-English summary memo
3. Updated trajectory + ablation plots through iter 34
4. One Grad-CAM figure
5. Saved-weights checkpoint + reproducibility appendix
6. Final presentation deck

## Stretch (only if Week 7 finishes on time)

7. ONNX export
8. TFLite conversion
9. Smartphone demo notebook

## What is OUT of scope (will not be done)

- Any further model.py iterations
- Ensemble training (would require ~17 GPU-hours + wrapper code)
- Pipeline unfreezing (100% data, 380px input, class-balanced sampling) — would invalidate the experimental record
- Backbone change to B5/B6
- New loss function (focal, asymmetric)
- AUC ≥ 0.93 stretch goal (would require ensemble; not committed)

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Grad-CAM doesn't show clean lesion attention | Medium | Cherry-pick 5 test images where the model is highly confident; document failures honestly if pattern is mixed |
| Weight-saving breaks the model.py contract | Low | Save only state_dict; load if file present, train if not — falls back to current behavior |
| TFLite conversion loses accuracy | High | Document the gap honestly; stretch goal can ship with the gap noted |
| Presentation runs long | Medium | Trim during day 7 dry-run; record video backup |

## Stopping condition

Project submits at end of Day 14. No code changes after Day 12. Final 2 days are presentation polish only.
