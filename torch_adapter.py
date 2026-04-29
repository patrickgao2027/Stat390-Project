"""
Reusable PyTorch adapter — lives outside model.py so the auto-research loop
can keep rewriting model.py without re-deriving wrapper boilerplate.

Subclass BaseTorchModel in model.py:
    - override _build_module() to return your nn.Module (input: [N, 3, H, W]
      float tensor in [0, 1]; output: [N] or [N, 1] logits)
    - optionally override _build_optimizer() and _build_scheduler()

The adapter then satisfies run.py's contract:
    fit(train_ds, epochs, steps_per_epoch, class_weight)
    predict(x_ds)         -> int labels
    predict_proba(x_ds)   -> [n, 2], col 1 = P(malignant)
"""

import numpy as np
import torch
import torch.nn as nn


class BaseTorchModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = self._build_module().to(self.device)
        self.optimizer = self._build_optimizer(self.module.parameters())

    def _build_module(self) -> nn.Module:
        raise NotImplementedError("Subclass must return an nn.Module producing logits.")

    def _build_optimizer(self, parameters):
        trainable = [p for p in parameters if p.requires_grad]
        return torch.optim.Adam(trainable, lr=1e-3)

    @staticmethod
    def _to_torch_images(tf_batch, device):
        # tf yields NHWC float32 in [0, 1]; torch wants NCHW
        arr = tf_batch.numpy()
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    def fit(self, train_ds, epochs=10, steps_per_epoch=None, class_weight=None):
        pos_weight = None
        if class_weight is not None:
            pos_weight = torch.tensor([float(class_weight[1])], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(epochs):
            self.module.train()
            running_loss, n_batches = 0.0, 0
            for step, (images, labels) in enumerate(train_ds):
                if steps_per_epoch is not None and step >= steps_per_epoch:
                    break
                x = self._to_torch_images(images, self.device)
                y = torch.from_numpy(labels.numpy()).float().to(self.device)

                self.optimizer.zero_grad()
                logits = self.module(x).squeeze(-1)
                loss = criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n_batches += 1
            avg = running_loss / max(n_batches, 1)
            print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}")
        return self

    @torch.no_grad()
    def predict_proba(self, x_ds):
        self.module.eval()
        chunks = []
        for batch in x_ds:
            x = self._to_torch_images(batch, self.device)
            logits = self.module(x).squeeze(-1)
            p = torch.sigmoid(logits).cpu().numpy()
            chunks.append(np.stack([1.0 - p, p], axis=1))
        return np.concatenate(chunks, axis=0)

    def predict(self, x_ds):
        proba = self.predict_proba(x_ds)
        return (proba[:, 1] >= 0.5).astype(np.int64)
