"""
EDITABLE — modify this file each iteration.

Iteration 6: switch to a different architecture family — pretrained ResNet-18.

Why ResNet: iterations 2-5 stayed in MBConv-style architectures (AlexNet aside,
which was a stepping stone). EfficientNet plateaued around AUC ~0.89 / recall
~0.83. ResNet's pure residual blocks (no depthwise, no squeeze-excitation)
give the model a genuinely different inductive bias to pick up signal with.

Why ResNet-18 specifically: smallest ResNet that fits the GPU-bound training
budget on a 4050 (~95 min estimated). Larger ResNets (50+) would push past
3 hours per iteration.

Other settings held constant vs iteration 4 (the reference best result) to
isolate the architecture effect:
  - 224x224 input upscale (ResNet is 224-native too)
  - pos_weight=10 retained
  - lr=1e-4 Adam, no LR schedule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from torch_adapter import BaseTorchModel


class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # ResNet's final classifier is a single Linear at .fc (not Sequential like EfficientNet)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)


class SkinLesionModel(BaseTorchModel):
    def _build_module(self):
        return ResNet18Binary()

    def _build_optimizer(self, parameters):
        trainable = [p for p in parameters if p.requires_grad]
        return torch.optim.Adam(trainable, lr=1e-4)

    def fit(self, train_ds, epochs=10, steps_per_epoch=None, class_weight=None):
        boosted = {0: 1.0, 1: 10.0}
        return super().fit(train_ds, epochs, steps_per_epoch, boosted)


def build_model():
    return SkinLesionModel()
