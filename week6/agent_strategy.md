# Agent Strategy — Locked at Week 6

## What changed from `program.md` (the original Week 0 strategy)

The original `program.md` told the agent to "explore logistic regression, transition to CNNs, then explore other options" with stopping conditions AUC ≥ 0.85 and recall ≥ 0.95. After 34 iterations both conditions are met. The locked strategy below reflects what *actually* worked, not the original speculative roadmap.

## Locked iteration loop

```
1. Edit ONE variable in model.py
2. python run.py "description Nth iteration"  (PowerShell, anaconda python)
3. Check the new row in results.tsv
4. If improved on the same single-variable comparison set: commit
5. If worse: git checkout model.py to revert
6. NO multi-variable changes per iteration. NO new directions.
```

## Editable surface area (now intentionally narrow)

| File | Status | Why |
|---|---|---|
| `model.py` | Editable | The only place to express modeling choices |
| `prepare.py` | Frozen | Data pipeline contract; modifying it invalidates all prior comparisons |
| `run.py` | Frozen | Logs every run with consistent metrics; modifying it breaks `results.tsv` continuity |
| `torch_adapter.py` | Editable but treated as boilerplate | Subclass-level patterns belong in `model.py` |

## Final canonical config (the model in `model.py` right now)

```python
class SkinLesionModel(BaseTorchModel):
    HEAD_WARMUP_EPOCHS = 2          # phase 1: head only, lr=1e-3
    CALIBRATION_BATCHES = 60        # first 60 batches held out from training
    TARGET_RECALL = 0.995           # min-positive estimator on cal set
    POS_WEIGHT = 10.0               # 5.3:1 class imbalance compensation
    USE_TTA = True                  # 4-view averaging (orig + hflip + vflip + both)
    USE_HOLDOUT_CAL = True          # cal scoring on OOS held-out images
    SAFETY_MARGIN = 0.10            # subtract from calibrated threshold
    CHECKPOINT_PATH = "model_checkpoint.pt"
    SAVE_CHECKPOINT = True          # best-of-N: overwrite only if better
    LOAD_CHECKPOINT = True          # DEPLOY: skip training, load saved weights
```

Pooled results across 5 training runs of this config (iters 32-36): mean recall 0.958 ± 0.019, mean AUC 0.900 ± 0.005, 4 of 5 replicates ≥ 0.95 recall individually. **Iter 35's weights (recall 0.968, AUC 0.902) are saved as the deployment checkpoint**; iter 37 confirmed bit-for-bit reproducibility on load.

Backbone: `efficientnet_b4` with ImageNet weights, input upsampled 128→224.
Loss: `BCEWithLogitsLoss(pos_weight=10)`.
Optimizer: Adam, lr split 1e-3 (warmup) → 1e-4 (fine-tune).

## Dropped directions (officially off the table)

| Direction | Iters | Why dropped |
|---|---|---|
| Lower-target recall calibration (TARGET_RECALL=0.95) | 17-19 | Mean recall fell to 0.77 — wrong direction |
| Full RNG seeding for determinism | 20-21 | Frozen `prepare.py`'s `tf.image.random_*` augmentations are unseeded and run under non-deterministic parallel `.map(AUTOTUNE)`; full determinism unreachable from `model.py` |
| More than 10 training epochs | 22 | Overfitting: train loss 0.25, recall 0.75 |
| TARGET_RECALL=0.99 (less-conservative cal) | 26 | Threshold moved wrong direction; recall 0.882 |
| CALIBRATION_BATCHES=120 (more cal positives) | 27-28 | Lost training data; recall mean 0.909 (worse) |
| POS_WEIGHT escalation beyond 10 | 11, 13 | Δ within within-condition noise on B2 |
| Multiplicative threshold scaling (SAFETY_MARGIN=0.85) | 8 | Multiplicative is wrong scale; additive is the right form |
| B5/B6 backbone | — | Diminishing returns; would need ~1.5× compute per rep |
| Mixup/CutMix, focal loss, AdamW | — | Not motivated by failure of current approach |
| TFLite deployment as graded deliverable | — | Stretch goal only; not in scope for Week 6 lock |

## Allowed refinements going forward (per Week 6 brief)

- Tighten failure-recovery rules in the iteration loop
- Reorder remaining tasks for efficient delivery
- Minor module-boundary edits **with instructor approval only**
- Save-weights-and-deploy for reproducibility appendix (does not change the model)

## Forbidden (per Week 6 brief)

- Expanding scope to new architectures, losses, or augmentation strategies
- Moving evaluation goalposts after seeing favorable numbers
- Ensemble training (out of scope for this single-model deliverable)
- Touching `prepare.py` or `run.py`
