"""
EDITABLE — modify this file each iteration.

Iteration: pretrained AlexNet (full fine-tune), replacing the 1000-way
classifier with a single binary logit. ImageNet normalization is applied
inside the module since prepare.py only scales to [0, 1].

build_model() returns an instance satisfying run.py's contract via
BaseTorchModel (see torch_adapter.py).
"""

import torch
import torch.nn as nn
from torchvision import models

from torch_adapter import BaseTorchModel


class AlexNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        backbone.classifier[6] = nn.Linear(4096, 1)
        self.backbone = backbone

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)


class SkinLesionModel(BaseTorchModel):
    def _build_module(self):
        return AlexNetBinary()

    def _build_optimizer(self, parameters):
        trainable = [p for p in parameters if p.requires_grad]
        return torch.optim.Adam(trainable, lr=1e-4)


def build_model():
    return SkinLesionModel()
