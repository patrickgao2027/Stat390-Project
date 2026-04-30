"""
EDITABLE — modify this file each iteration.

Iteration 5: EfficientNet-B2 + 224x224 input upscale + two-phase training,
keeping pos_weight=10 from iteration 4. Three coordinated changes aimed at
lifting every metric simultaneously:
  1. Backbone B0 -> B2 (~5.3M -> ~9.1M params): more capacity for fine-grained
     dermoscopy features.
  2. Two-phase fit(): epochs 1-2 freeze backbone and train only the new
     Linear(1408, 1) head at lr=1e-3 (head warmup), then epochs 3-10 unfreeze
     and full fine-tune at lr=1e-4. Prevents the random-init head from sending
     destructive gradients through the pretrained features in early steps.
  3. pos_weight=10 retained to keep recall pressure on the malignant class.
"""

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

        return self


def build_model():
    return SkinLesionModel()
