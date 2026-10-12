"""Métricas e evaluación en validación / test."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.recall_selection import compute_recall_selection_metrics


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Devuelve (probabilidades, etiquetas, loss_media opcional)."""
    model.eval()
    all_probabilities: list[float] = []
    all_labels: list[float] = []
    running_loss = 0.0
    num_batches = 0

    for images, labels in tqdm(data_loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            running_loss += criterion(logits, labels_device).item()
            num_batches += 1

        probabilities = torch.sigmoid(logits.view(-1)).cpu().numpy()
        all_probabilities.extend(probabilities.tolist())
        all_labels.extend(labels.view(-1).cpu().numpy().tolist())

    mean_loss = running_loss / max(num_batches, 1) if criterion is not None else None
    return (
        np.asarray(all_probabilities, dtype=np.float64),
        np.asarray(all_labels, dtype=np.float64),
        mean_loss,
    )


def compute_classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """AUC, F1, accuracy y matriz de confusión a umbral fijo."""
    predictions = (probabilities >= threshold).astype(int)
    labels_int = labels.astype(int)

    # AUC requiere ambas clases presentes
    if len(np.unique(labels_int)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels_int, probabilities))

    metrics: dict[str, Any] = {
        "auc": auc,
        "f1": float(f1_score(labels_int, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels_int, predictions)),
        "confusion_matrix": confusion_matrix(labels_int, predictions).tolist(),
        "threshold": threshold,
        "n_samples": int(len(labels_int)),
        "n_positive": int(labels_int.sum()),
        "n_negative": int((1 - labels_int).sum()),
    }
    return metrics


def select_threshold_youden(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Umbral que maximiza J = TPR - FPR sobre las predicciones dadas.

    Se calcula SOLO con validación y se aplica después a test. Youden es preferible a
    maximizar F1 con clases desbalanceadas (79 % positivos), donde F1 empuja el umbral
    hacia "predecir todo positivo".
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels.astype(int), probabilities
    )
    best_index = int(np.argmax(true_positive_rate - false_positive_rate))
    return float(thresholds[best_index])


def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Evalúa un DataLoader completo y devuelve métricas (+ loss si hay criterion)."""
    probabilities, labels, mean_loss = collect_predictions(
        model, data_loader, device, criterion=criterion
    )
    metrics = compute_classification_metrics(probabilities, labels, threshold=threshold)
    # Criterios independientes del umbral para elegir checkpoints (ver src/recall_selection.py)
    metrics.update(compute_recall_selection_metrics(probabilities, labels))
    if mean_loss is not None:
        metrics["loss"] = float(mean_loss)
    return metrics


def format_metrics(metrics: dict[str, Any], split_name: str) -> str:
    """Cadena legible para logs."""
    return (
        f"[{split_name}] "
        f"AUC={metrics['auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"CM={metrics['confusion_matrix']} "
        f"(n={metrics['n_samples']}, pos={metrics['n_positive']})"
    )
