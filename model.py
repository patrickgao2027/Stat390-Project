"""
EDITABLE — modify this file each iteration.

Iters 29-31 (B4 + TTA): AUC +0.006 reproducibly (0.902 across all 3 reps,
above every non-TTA run). Recall mean 0.934 ± 0.022 — slightly below
the pre-TTA baseline; TTA helps AUC but is a recall wash.

Pooled 6-rep B4 recall (iters 23-25 + 29-31): mean 0.938, std 0.020,
2/6 reps ≥ 0.95. Model-side noise dominates; no threshold lever and no
inference-side smoothing has shifted the mean above 0.95.

FINAL CONFIG (Week 6 lock): iters 32-34 — B4 + two-phase fine-tuning +
holdout threshold calibration + 4-view TTA + additive SAFETY_MARGIN=0.10.

Reproducible results across 3 reps:
  ROC-AUC: 0.903 ± 0.001
  Recall:  0.952 ± 0.023 (target 0.95 ✅, 2/3 reps individually cross)
  Best single rep: iter 32 — AUC 0.903, recall 0.974

The lever search (iters 1-34) is officially closed; remaining variance
is model-side and only addressable via ensemble or weight-saving (out of
scope for this single-model deliverable).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from torch_adapter import BaseTorchModel


class EfficientNetB4Binary(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
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
    CALIBRATION_BATCHES = 60     # reverted from 120: iters 27-28 confirmed worse (mean recall 0.909 vs 0.942)
    TARGET_RECALL = 0.995        # min positive on holdout (catches every cal positive)
    POS_WEIGHT = 10.0
    USE_TTA = True               # 4-view TTA (added iter 29) — gives reproducible AUC +0.006
    USE_HOLDOUT_CAL = True
    SAFETY_MARGIN = 0.10         # FINAL: iters 32-34, mean recall 0.952, AUC 0.903 reproducibly

    def _build_module(self):
        return EfficientNetB4Binary()

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
        # If USE_TTA, both calibration and inference use averaged logits across
        # 4 flip views — keeps cal and test on the same probability distribution.
        self.module.eval()
        probs_chunks, labels_chunks = [], []
        with torch.no_grad():
            if self.USE_HOLDOUT_CAL:
                # Score the materialized OOS held-out images (model never trained on these).
                bs = 16
                for s in range(0, len(held_out_x), bs):
                    batch = held_out_x[s:s+bs]
                    x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
                    logits = self._tta_logits(x) if self.USE_TTA else self.module(x).squeeze(-1)
                    probs_chunks.append(torch.sigmoid(logits).cpu().numpy())
                p = np.concatenate(probs_chunks)
                y_arr = held_out_y
                cal_label = "holdout+tta" if self.USE_TTA else "holdout"
            else:
                # Legacy leaky path: score training batches the model already memorized.
                for i, (images, y) in enumerate(train_ds):
                    if i >= self.CALIBRATION_BATCHES:
                        break
                    x = self._to_torch_images(images, self.device)
                    logits = self._tta_logits(x) if self.USE_TTA else self.module(x).squeeze(-1)
                    probs_chunks.append(torch.sigmoid(logits).cpu().numpy())
                    labels_chunks.append(y.numpy())
                p = np.concatenate(probs_chunks)
                y_arr = np.concatenate(labels_chunks).astype(int)
                cal_label = "leaked+tta" if self.USE_TTA else "leaked"

        order = np.argsort(-p)
        p_sorted = p[order]
        y_sorted = y_arr[order]
        total_pos = max(int(y_arr.sum()), 1)
        recall_curve = np.cumsum(y_sorted) / total_pos
        idx = int(np.searchsorted(recall_curve, self.TARGET_RECALL))
        idx = min(idx, len(p_sorted) - 1)
        raw_threshold = float(p_sorted[idx])
        # Additive safety margin: subtract a fixed offset to err toward recall.
        # Floor at 0.05 so the threshold can't collapse to ~0 and flag everything.
        self.threshold = max(raw_threshold - self.SAFETY_MARGIN, 0.05)
        print(f"Phase 3 ({cal_label}): raw_threshold = {raw_threshold:.4f}, "
              f"margin = {self.SAFETY_MARGIN:.3f}, "
              f"final threshold = {self.threshold:.4f} "
              f"(cal recall = {recall_curve[idx]:.3f}, "
              f"pos = {total_pos}, total = {len(p)})")

        return self

    def _tta_logits(self, x):
        """4-view TTA: average logits across original, hflip, vflip, both flips.
        Smooths per-image probabilities → lower variance threshold + recall."""
        views = [
            x,
            torch.flip(x, dims=[3]),       # horizontal flip
            torch.flip(x, dims=[2]),       # vertical flip
            torch.flip(x, dims=[2, 3]),    # both
        ]
        return sum(self.module(v).squeeze(-1) for v in views) / 4.0

    @torch.no_grad()
    def predict_proba(self, x_ds):
        if not self.USE_TTA:
            return super().predict_proba(x_ds)
        self.module.eval()
        chunks = []
        for batch in x_ds:
            x = self._to_torch_images(batch, self.device)
            logits = self._tta_logits(x)
            p = torch.sigmoid(logits).cpu().numpy()
            chunks.append(np.stack([1.0 - p, p], axis=1))
        return np.concatenate(chunks, axis=0)

    def predict(self, x_ds):
        proba = self.predict_proba(x_ds)
        t = getattr(self, "threshold", 0.5)
        return (proba[:, 1] >= t).astype(np.int64)


def build_model():
    return SkinLesionModel()
