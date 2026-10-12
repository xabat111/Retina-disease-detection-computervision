"""Focal Loss para clasificación binaria desbalanceada."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal Loss binaria sobre logits (compatible con salida de 1 neurona).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma > 0 reduce el peso de ejemplos fáciles.
    - alpha en (0, 1) da más peso a la clase positiva si hay desbalance.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits / targets: (batch,) o (batch, 1)
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        # p_t: probabilidad de la clase verdadera
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()
