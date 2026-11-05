"""Pérdidas para clasificación multietiqueta desbalanceada.

El baseline (notebook 02) sigue usando BCE con `pos_weight` por clase.
El ajuste fino (notebooks 05-06) usa Focal Loss multilabel; el desbalance se pliega
en el escalar `alpha_pos` y no se combina con `pos_weight`.

Modos de `pos_weight` (TrainConfig.pos_weight_mode):
  - capped:       neg/pos acotado a [1, max_pos_weight] (BaselineV2, default)
  - uncapped:     neg/pos con techo alto (max_pos_weight_uncapped) por estabilidad numérica
  - soft_cap:     misma razón, comprimida suavemente hacia max_pos_weight (asíntota)
  - class_balanced: pesos por número efectivo (Cui et al.), sin cap 20
  - iterative:    arranque frequency (soft/raised) + refinamiento en val cada N épocas
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PosWeightMode = Literal["capped", "uncapped", "soft_cap", "class_balanced", "iterative"]


def raw_neg_pos_ratio(train_targets: np.ndarray) -> np.ndarray:
    """Razón negativos/positivos por clase (solo train). Sin clip."""
    positives = np.asarray(train_targets, dtype=np.float64).sum(axis=0)
    negatives = len(train_targets) - positives
    return negatives / np.maximum(positives, 1.0)


def soft_cap_weights(raw: np.ndarray, cap: float) -> np.ndarray:
    """Comprime w>=1 hacia [1, cap) con tanh (asíntota en `cap`).

    En w=cap (hard) → ~0.76·cap + 0.24; en w≫cap → ~cap. Evita el corte
    abrupto en 20 que deja a `otros` exactamente en el techo.
    """
    raw = np.maximum(np.asarray(raw, dtype=np.float64), 1.0)
    excess = raw - 1.0
    scale = max(cap - 1.0, 1e-6)
    return 1.0 + scale * np.tanh(excess / scale)


def compute_pos_weights(
    train_targets: np.ndarray,
    max_pos_weight: float,
    *,
    mode: PosWeightMode = "capped",
    max_pos_weight_uncapped: float = 100.0,
    cb_beta: float = 0.9999,
) -> torch.Tensor:
    """Peso de positivos por clase a partir del desbalance de train.

    `capped` (default BaselineV2): clip(neg/pos, 1, max_pos_weight).
    """
    raw = raw_neg_pos_ratio(train_targets)
    if mode in ("capped", "iterative"):
        # iterative arranca como capped o soft según TrainConfig; aquí capped.
        weights = np.clip(raw, 1.0, max_pos_weight)
    elif mode == "uncapped":
        weights = np.clip(raw, 1.0, max_pos_weight_uncapped)
    elif mode == "soft_cap":
        weights = soft_cap_weights(raw, max_pos_weight)
    elif mode == "class_balanced":
        weights = class_balanced_pos_weights(train_targets, beta=cb_beta)
    else:
        raise ValueError(f"pos_weight mode desconocido: {mode!r}")
    return torch.tensor(weights, dtype=torch.float32)


def class_balanced_pos_weights(
    train_targets: np.ndarray,
    beta: float = 0.9999,
    *,
    min_weight: float = 1.0,
    rescale_median_to: float | None = None,
) -> np.ndarray:
    """Pesos por número efectivo (Cui et al., CVPR 2019), sin cap arbitrario a 20.

    E_n = (1 − β^n) / (1 − β); w ∝ 1/E_n. Se reescala para que la mediana
    coincida con `rescale_median_to` (o con la mediana de neg/pos) y min >= 1.
    """
    n = np.maximum(np.asarray(train_targets, dtype=np.float64).sum(axis=0), 1.0)
    beta = float(np.clip(beta, 0.0, 1.0 - 1e-12))
    effective = (1.0 - np.power(beta, n)) / (1.0 - beta)
    inv = 1.0 / np.maximum(effective, 1e-12)
    raw = raw_neg_pos_ratio(train_targets)
    target_med = float(np.median(raw)) if rescale_median_to is None else float(rescale_median_to)
    inv_med = float(np.median(inv))
    weights = inv * (target_med / max(inv_med, 1e-12))
    return np.maximum(weights, min_weight)


def initial_pos_weights_for_mode(
    train_targets: np.ndarray,
    max_pos_weight: float,
    mode: PosWeightMode,
    *,
    max_pos_weight_uncapped: float = 100.0,
    iterative_init: Literal["soft_cap", "uncapped", "capped"] = "soft_cap",
    iterative_init_cap: float = 40.0,
    cb_beta: float = 0.9999,
) -> torch.Tensor:
    """Inicialización de pos_weight según modo (iterative usa soft_cap@40 por defecto)."""
    if mode == "iterative":
        raw = raw_neg_pos_ratio(train_targets)
        if iterative_init == "uncapped":
            w = np.clip(raw, 1.0, max_pos_weight_uncapped)
        elif iterative_init == "capped":
            w = np.clip(raw, 1.0, max_pos_weight)
        else:
            w = soft_cap_weights(raw, iterative_init_cap)
        return torch.tensor(w, dtype=torch.float32)
    return compute_pos_weights(
        train_targets,
        max_pos_weight,
        mode=mode,
        max_pos_weight_uncapped=max_pos_weight_uncapped,
        cb_beta=cb_beta,
    )


def adapt_pos_weights(
    current: np.ndarray | torch.Tensor,
    val_ap: np.ndarray,
    *,
    val_recall: np.ndarray | None = None,
    base: np.ndarray | torch.Tensor | None = None,
    eta: float = 0.15,
    momentum: float = 0.7,
    anchor: float = 0.05,
    ap_blend: float = 0.7,
    w_min: float = 1.0,
    w_max: float = 50.0,
) -> np.ndarray:
    """Una ronda de adaptación usando señal de validación (AP y opcionalmente recall).

    Algoritmo (amortiguado, multiplicativo):
      1. deficit_c = max(0, median(signal) - signal_c) / (median + eps)
         surplus_c = max(0, signal_c - median) / (1 - median + eps)
         con signal = ap_blend·AP + (1-ap_blend)·recall (si hay recall).
      2. w_prop = current · exp(eta · (deficit − surplus))
      3. EMA: w = momentum·current + (1−momentum)·w_prop
      4. Ancla al peso inicial: w = (1−anchor)·w + anchor·base
      5. clip [w_min, w_max]

    Sube peso en clases con AP/recall bajo (p. ej. `otros`) y baja un poco
    las fáciles saturadas, sin explosiones gracias a eta/momentum/cap.
    """
    w = np.asarray(current, dtype=np.float64).copy()
    ap = np.asarray(val_ap, dtype=np.float64)
    if val_recall is not None:
        rec = np.asarray(val_recall, dtype=np.float64)
        signal = ap_blend * np.nan_to_num(ap, nan=0.0) + (1.0 - ap_blend) * np.nan_to_num(
            rec, nan=0.0
        )
    else:
        signal = np.nan_to_num(ap, nan=0.0)

    med = float(np.nanmedian(signal))
    eps = 1e-6
    deficit = np.maximum(0.0, med - signal) / (med + eps)
    surplus = np.maximum(0.0, signal - med) / (max(1.0 - med, 0.0) + eps)
    log_update = eta * (deficit - surplus)
    # limitar paso multiplicativo por ronda (~±35% con eta=0.15 y gap extremo)
    log_update = np.clip(log_update, -0.35, 0.35)
    proposed = w * np.exp(log_update)
    adapted = momentum * w + (1.0 - momentum) * proposed
    if base is not None:
        b = np.asarray(base, dtype=np.float64)
        adapted = (1.0 - anchor) * adapted + anchor * b
    return np.clip(adapted, w_min, w_max)


def set_bce_pos_weight(loss_fn: nn.BCEWithLogitsLoss, weights: torch.Tensor | np.ndarray) -> None:
    """Actualiza in-place el buffer `pos_weight` de BCEWithLogitsLoss."""
    tensor = torch.as_tensor(weights, dtype=torch.float32, device=loss_fn.pos_weight.device)
    if tensor.shape != loss_fn.pos_weight.shape:
        raise ValueError(
            f"pos_weight shape {tuple(tensor.shape)} != {tuple(loss_fn.pos_weight.shape)}"
        )
    loss_fn.pos_weight.copy_(tensor)


def build_loss(pos_weights: torch.Tensor) -> nn.BCEWithLogitsLoss:
    """BCE con logits y `pos_weight` por clase (cada etiqueta es una decisión binaria)."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


class MultilabelFocalLoss(nn.Module):
    """Focal Loss por etiqueta: BCE-with-logits × (1 − p_t)^γ × α_t.

    Para cada etiqueta de forma independiente (sigmoid, no softmax):

    - p = σ(logit), p_t = p si y=1 y 1−p si y=0
    - el factor (1−p_t)^γ baja el peso de los ejemplos fáciles
    - α_t = α_pos si y=1 y (1−α_pos) si y=0

    `alpha_pos` es un **escalar global** (no un vector por clase): el desbalance se
    corrige buscando α_pos en Optuna (p. ej. 0.25–0.75) en lugar de un `pos_weight`
    aparte. γ típico: 1–3 (RetinaNet usa 2).
    """

    def __init__(self, gamma: float = 2.0, alpha_pos: float = 0.25) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma debe ser >= 0, recibido {gamma}")
        if not 0.0 < alpha_pos < 1.0:
            raise ValueError(f"alpha_pos debe estar en (0, 1), recibido {alpha_pos}")
        self.gamma = float(gamma)
        self.alpha_pos = float(alpha_pos)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_per_label = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        modulating_factor = (1.0 - p_t).clamp(min=0.0).pow(self.gamma)
        alpha_t = self.alpha_pos * targets + (1.0 - self.alpha_pos) * (1.0 - targets)
        return (alpha_t * modulating_factor * bce_per_label).mean()


def build_focal_loss(gamma: float, alpha_pos: float) -> MultilabelFocalLoss:
    """Focal Loss multilabel usada en el ajuste fino (no en el baseline de solo cabeza)."""
    return MultilabelFocalLoss(gamma=gamma, alpha_pos=alpha_pos)
