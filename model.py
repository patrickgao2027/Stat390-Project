"""
EDITABLE — modify this file each iteration.

Iteration 7: same architecture as iter 5/6 (EfficientNet-B2 + 224 upscale +
two-phase + pos_weight=10), tightening the post-training threshold calibration
to close the train->test recall gap observed in iter 6.

Iter 6 ran at TARGET_RECALL=0.97 with 30 calibration batches and produced
test recall = 0.884 (calibration recall ~0.97 — a train->test gap of ~0.09).
Iter 7 tightens the target to 0.995 and doubles the calibration sample to 60
batches for a less noisy threshold estimate. Architecture, optimizer, loss,
and pos_weight are unchanged so this isolates the threshold effect.

ROC-AUC target (>=0.85) has been met since iter 2 — peaks at 0.897. The
unmet goal is recall >= 0.95. predict_proba() is untouched, so ROC-AUC
should be unaffected; only predict()'s decision boundary moves.
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
    TARGET_RECALL = 0.995

    def _build_module(self):
        return EfficientNetB2Binary()

    def _build_optimizer(self, parameters):
        # Placeholder — fit() rebuilds the optimizer per phase.
        return torch.optim.Adam(parameters, lr=1e-4)

    def fit(self, train_ds, epochs=10, steps_per_epoch=None, class_weight=None):
        boosted = {0: 1.0, 1: 10.0}
        backbone = self.module.backbone

        # Phase 1: freeze conv stack, train head only at higher lr.
        for p in backbone.features.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(backbone.classifier.parameters(), lr=1e-3)
        print(f"Phase 1: head warmup ({self.HEAD_WARMUP_EPOCHS} epochs at lr=1e-3)")
        super().fit(train_ds, epochs=self.HEAD_WARMUP_EPOCHS,
                    steps_per_epoch=steps_per_epoch, class_weight=boosted)

        # Phase 2: unfreeze everything, full fine-tune at lower lr.
        for p in backbone.features.parameters():
            p.requires_grad = True
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-4)
        remaining = epochs - self.HEAD_WARMUP_EPOCHS
        print(f"Phase 2: full fine-tune ({remaining} epochs at lr=1e-4)")
        super().fit(train_ds, epochs=remaining,
                    steps_per_epoch=steps_per_epoch, class_weight=boosted)

        # Phase 3: calibrate threshold so calibration recall >= TARGET_RECALL.
        self.module.eval()
        probs, labels = [], []
        with torch.no_grad():
            for i, (images, y) in enumerate(train_ds):
                if i >= self.CALIBRATION_BATCHES:
                    break
                x = self._to_torch_images(images, self.device)
                logits = self.module(x).squeeze(-1)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y.numpy())
        p = np.concatenate(probs)
        y_arr = np.concatenate(labels).astype(int)
        order = np.argsort(-p)
        p_sorted = p[order]
        y_sorted = y_arr[order]
        total_pos = max(int(y_arr.sum()), 1)
        recall_curve = np.cumsum(y_sorted) / total_pos
        idx = int(np.searchsorted(recall_curve, self.TARGET_RECALL))
        idx = min(idx, len(p_sorted) - 1)
        self.threshold = float(p_sorted[idx])
        print(f"Phase 3: calibrated threshold = {self.threshold:.4f} "
              f"(calibration recall = {recall_curve[idx]:.3f}, "
              f"pos = {total_pos}, total = {len(p)})")

        return self

    def predict(self, x_ds):
        proba = self.predict_proba(x_ds)
        t = getattr(self, "threshold", 0.5)
        return (proba[:, 1] >= t).astype(np.int64)


def build_model():
    return SkinLesionModel()
