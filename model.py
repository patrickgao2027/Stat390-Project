"""
EDITABLE — modify this file each iteration.

Iteration: EfficientNet-B0 + 224x224 input upscale + boosted pos_weight (10.0).
Two targeted changes vs the prior iteration, both aimed at closing the recall
gap (current 0.83, target >= 0.95):
  1. Upscale 128x128 -> 224x224 in forward(). Pretrained ImageNet weights were
     trained at 224; running at 128 cramps spatial features and caps AUC.
  2. Override fit() to use pos_weight=10.0 instead of the auto-computed 5.31.
     The auto value matches class frequency; pushing it higher biases the
     model toward malignant predictions during training, lifting recall at
     the default 0.5 threshold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from torch_adapter import BaseTorchModel


class EfficientNetB0Binary(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
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
    def _build_module(self):
        return EfficientNetB0Binary()

    def _build_optimizer(self, parameters):
        trainable = [p for p in parameters if p.requires_grad]
        return torch.optim.Adam(trainable, lr=1e-4)

    def fit(self, train_ds, epochs=10, steps_per_epoch=None, class_weight=None):
        boosted = {0: 1.0, 1: 10.0}
        return super().fit(train_ds, epochs, steps_per_epoch, boosted)


def build_model():
    return SkinLesionModel()
