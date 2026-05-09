"""
EDITABLE — modify this file each iteration.

Iteration 17: corrected priority-1 fix. The iter 14-16 holdout-cal experiment
REJECTED my original hypothesis: holdout cal *widened* recall std (0.039 vs
0.019 leaked) and threshold std (0.135 vs 0.047 leaked) instead of tightening
them. The dominant Signal Failure mechanism is NOT the leak — it's the
threshold estimator itself: `min(positive_probability)` over a small cal
sample is an order statistic that's intrinsically noisy.

Iter 17 changes ONE variable (single-variable controlled experiment):
TARGET_RECALL 0.995 -> 0.95. With 141 holdout positives, the threshold
becomes the 8th-lowest positive probability (~5th percentile) instead of
the absolute minimum. That's a much more stable order statistic.

Held fixed from iters 14-16: USE_HOLDOUT_CAL=True, POS_WEIGHT=10,
CALIBRATION_BATCHES=60, architecture, two-phase training, lr schedule.

Iters 17-19 are 3 replicates. Comparison set: iters 14-16 (same code with
TARGET_RECALL=0.995). If recall std drops from 0.039 to <0.02, the percentile
estimator is the right fix and the loop becomes interpretable.

Original iter-14 docstring (kept for context):

  Iteration 14: replace the leaky training-data calibration source with a
  held-out 10% validation slice. Hypothesis was that Evaluation Leakage was
  the dominant cause of Signal Failure. The 3-run holdout experiment ran
  at iters 14-16 and rejected this hypothesis.

Background: the Week-4 controlled experiment showed that within-condition
recall std at pw=20 was 0.047, larger than the +0.014 between-condition
effect of pos_weight. Failure memo identified the mechanism: Phase 3
calibration uses training batches the model has memorized, so cal-set
recall is trivially 1.0 in every run and the resulting threshold equals
"the lowest probability the model emits on positives it has already seen."
This causes a 0.06-0.09 train->test recall gap and the threshold dispersion
that drives Signal Failure.

Fix in iter 14: USE_HOLDOUT_CAL = True
  - Before training, materialize the first CALIBRATION_BATCHES batches of
    train_ds into numpy arrays. This is the held-out validation slice
    (~960 images, ~5.7% of train).
  - Train on train_ds.skip(CALIBRATION_BATCHES). The model never sees the
    held-out items.
  - In Phase 3, score the held-out arrays. Cal recall is now a real
    out-of-sample metric, and the threshold reflects the model's behavior
    on truly unseen positives — same statistical population as the test set.

Controlled experiment: cal_data_source in {train_leaked, holdout_val},
3 reps each. The 3 leaked-cal reps already exist (iters 7, 10, 12 — same
config: pw=10, target=0.995, batches=60, no SAFETY_MARGIN). Iters 14-16
are the 3 holdout-cal reps.

POS_WEIGHT reset to 10.0 to match iters 7/10/12 baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from torch_adapter import BaseTorchModel


class EfficientNetB2Binary(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, 1)
        self.backbone = backbone

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)


class SkinLesionModel(BaseTorchModel):
    HEAD_WARMUP_EPOCHS = 2
    CALIBRATION_BATCHES = 60
    TARGET_RECALL = 0.95     # Iter 17 single-var change: 0.995 -> 0.95 (min -> 5th-percentile)
    POS_WEIGHT = 10.0
    USE_HOLDOUT_CAL = True   # Held fixed from iters 14-16 (no longer the experimental axis)

    def _build_module(self):
        return EfficientNetB2Binary()

    def _build_optimizer(self, parameters):
        # Placeholder — fit() rebuilds the optimizer per phase.
        return torch.optim.Adam(parameters, lr=1e-4)

    def fit(self, train_ds, epochs=10, steps_per_epoch=None, class_weight=None):
        boosted = {0: 1.0, 1: self.POS_WEIGHT}
        backbone = self.module.backbone

        # Holdout cal: materialize first CALIBRATION_BATCHES batches before training,
        # then train on the rest (model never sees the held-out items).
        held_out_x = held_out_y = None
        if self.USE_HOLDOUT_CAL:
            x_chunks, y_chunks = [], []
            for i, (images, y) in enumerate(train_ds):
                if i >= self.CALIBRATION_BATCHES:
                    break
                x_chunks.append(images.numpy())
                y_chunks.append(y.numpy())
            held_out_x = np.concatenate(x_chunks, axis=0)
            held_out_y = np.concatenate(y_chunks, axis=0).astype(int)
            print(f"Holdout cal: materialized {len(held_out_x)} images "
                  f"({int(held_out_y.sum())} positive) from first "
                  f"{self.CALIBRATION_BATCHES} batches; excluding from training.")
            train_ds_for_fit = train_ds.skip(self.CALIBRATION_BATCHES)
            if steps_per_epoch is not None:
                steps_per_epoch = max(steps_per_epoch - self.CALIBRATION_BATCHES, 1)
        else:
            train_ds_for_fit = train_ds
            print("Leaked cal: Phase 3 will score the same train batches the model trained on.")

        # Phase 1: freeze conv stack, train head only at higher lr.
        for p in backbone.features.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(backbone.classifier.parameters(), lr=1e-3)
        print(f"Phase 1: head warmup ({self.HEAD_WARMUP_EPOCHS} epochs at lr=1e-3)")
        super().fit(train_ds_for_fit, epochs=self.HEAD_WARMUP_EPOCHS,
                    steps_per_epoch=steps_per_epoch, class_weight=boosted)

        # Phase 2: unfreeze everything, full fine-tune at lower lr.
        for p in backbone.features.parameters():
            p.requires_grad = True
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-4)
        remaining = epochs - self.HEAD_WARMUP_EPOCHS
        print(f"Phase 2: full fine-tune ({remaining} epochs at lr=1e-4)")
        super().fit(train_ds_for_fit, epochs=remaining,
                    steps_per_epoch=steps_per_epoch, class_weight=boosted)

        # Phase 3: calibrate threshold to TARGET_RECALL on the cal sample.
        self.module.eval()
        probs_chunks, labels_chunks = [], []
        with torch.no_grad():
            if self.USE_HOLDOUT_CAL:
                # Score the materialized OOS held-out images (model never trained on these).
                bs = 16
                for s in range(0, len(held_out_x), bs):
                    batch = held_out_x[s:s+bs]
                    x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
                    logits = self.module(x).squeeze(-1)
                    probs_chunks.append(torch.sigmoid(logits).cpu().numpy())
                p = np.concatenate(probs_chunks)
                y_arr = held_out_y
                cal_label = "holdout"
            else:
                # Legacy leaky path: score training batches the model already memorized.
                for i, (images, y) in enumerate(train_ds):
                    if i >= self.CALIBRATION_BATCHES:
                        break
                    x = self._to_torch_images(images, self.device)
                    logits = self.module(x).squeeze(-1)
                    probs_chunks.append(torch.sigmoid(logits).cpu().numpy())
                    labels_chunks.append(y.numpy())
                p = np.concatenate(probs_chunks)
                y_arr = np.concatenate(labels_chunks).astype(int)
                cal_label = "leaked"

        order = np.argsort(-p)
        p_sorted = p[order]
        y_sorted = y_arr[order]
        total_pos = max(int(y_arr.sum()), 1)
        recall_curve = np.cumsum(y_sorted) / total_pos
        idx = int(np.searchsorted(recall_curve, self.TARGET_RECALL))
        idx = min(idx, len(p_sorted) - 1)
        self.threshold = float(p_sorted[idx])
        print(f"Phase 3 ({cal_label}): threshold = {self.threshold:.4f} "
              f"(cal recall = {recall_curve[idx]:.3f}, "
              f"pos = {total_pos}, total = {len(p)})")

        return self

    def predict(self, x_ds):
        proba = self.predict_proba(x_ds)
        t = getattr(self, "threshold", 0.5)
        return (proba[:, 1] >= t).astype(np.int64)


def build_model():
    return SkinLesionModel()
