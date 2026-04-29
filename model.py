"""
EDITABLE — modify this file each iteration.

Iteration: pretrained EfficientNet-B0 (full fine-tune). Smaller and stronger
than AlexNet, and TFLite-friendly for the deployment stretch goal.
ImageNet normalization is applied inside the module since prepare.py only
scales to [0, 1].

build_model() returns an instance satisfying run.py's contract via
BaseTorchModel (see torch_adapter.py).
"""

import torch
import torch.nn as nn
from torchvision import models

from torch_adapter import BaseTorchModel


class EfficientNetB0Binary(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # classifier is Sequential(Dropout(0.2), Linear(1280, 1000)) — keep dropout, swap head
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, 1)
        self.backbone = backbone

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)


class SkinLesionModel(BaseTorchModel):
    def _build_module(self):
        return EfficientNetB0Binary()

    def _build_optimizer(self, parameters):
        trainable = [p for p in parameters if p.requires_grad]
        return torch.optim.Adam(trainable, lr=1e-4)


def build_model():
    return SkinLesionModel()
