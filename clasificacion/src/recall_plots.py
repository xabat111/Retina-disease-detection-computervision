"""Figuras del análisis de recall: curvas vs umbral, ROC/PR y matrices de confusión."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def _save_figure(figure: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=110, bbox_inches="tight")


def plot_threshold_sweep(
    threshold_table: pd.DataFrame,
    chosen_threshold: float,
    target_recall: float,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Recall, especificidad, precisión, F1 y accuracy frente al umbral de decisión."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for metric_name in ("recall", "specificity", "precision", "f1", "accuracy"):
        axis.plot(threshold_table["threshold"], threshold_table[metric_name], label=metric_name)
    axis.axhline(target_recall, color="gray", linestyle=":", label=f"recall objetivo = {target_recall:.2f}")
    axis.axvline(
        chosen_threshold,
        color="black",
        linestyle="--",
        label=f"umbral elegido (val) = {chosen_threshold:.3f}",
    )
    axis.set(title=title, xlabel="Umbral de decisión", ylabel="Métrica", ylim=(0, 1.02))
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_roc_and_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    operating_points: dict[str, tuple[float, float, float]],
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """ROC y precisión-recall de varias configuraciones sobre un mismo split.

    `curves`: nombre -> (probabilidades, etiquetas).
    `operating_points`: nombre -> (umbral, recall, especificidad) en este split; se marca ese
    punto sobre ambas curvas.
    """
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, (probabilities, labels) in curves.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(labels.astype(int), probabilities)
        precision, recall, _ = precision_recall_curve(labels.astype(int), probabilities)
        roc_line = roc_axis.plot(false_positive_rate, true_positive_rate, label=name)[0]
        pr_axis.plot(recall, precision, label=name, color=roc_line.get_color())

        if name in operating_points:
            threshold, point_recall, point_specificity = operating_points[name]
            point_precision = _precision_at(probabilities, labels, threshold)
            roc_axis.scatter(
                1 - point_specificity, point_recall, color=roc_line.get_color(), marker="o", zorder=5
            )
            pr_axis.scatter(
                point_recall, point_precision, color=roc_line.get_color(), marker="o", zorder=5
            )

    roc_axis.plot([0, 1], [0, 1], color="gray", linestyle=":")
    roc_axis.set(title="ROC (punto = umbral elegido)", xlabel="1 - especificidad", ylabel="Recall")
    pr_axis.set(title="Precisión-Recall (punto = umbral elegido)", xlabel="Recall", ylabel="Precisión")
    for axis in (roc_axis, pr_axis):
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def _precision_at(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    predictions = probabilities >= threshold
    predicted_positives = predictions.sum()
    if predicted_positives == 0:
        return float("nan")
    return float((predictions & (labels.astype(int) == 1)).sum() / predicted_positives)


def plot_confusion_matrices(
    confusion_matrices: dict[str, list[list[int]]],
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Matrices de confusión [[TN, FP], [FN, TP]] lado a lado."""
    figure, axes = plt.subplots(1, len(confusion_matrices), figsize=(4.2 * len(confusion_matrices), 4))
    axes = np.atleast_1d(axes)
    for axis, (name, matrix) in zip(axes, confusion_matrices.items()):
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row][column], ha="center", va="center", fontsize=14)
        axis.set(
            title=name,
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["sano", "riesgo"],
            yticklabels=["sano", "riesgo"],
            xlabel="Predicho",
            ylabel="Real",
        )
    figure.suptitle(title)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure
