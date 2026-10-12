"""Gráficas de la comparación de arquitecturas (barras, ROC, curvas, matrices de confusión)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_curve

from src.comparison import EvaluatedModel, summarize_split

BAR_METRIC_COLUMNS = ["AUC test", "F1 test", "Accuracy test", "Precision test", "Recall test"]


def _save_figure(figure, save_path: Path | None) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=110, bbox_inches="tight")


def plot_metric_bars(comparison_table: pd.DataFrame, save_path: Path | None = None):
    """Barras agrupadas: una barra por modelo para cada métrica de test."""
    model_names = comparison_table["modelo"].tolist()
    bar_width = 0.8 / len(model_names)
    positions = np.arange(len(BAR_METRIC_COLUMNS))

    figure, axis = plt.subplots(figsize=(11, 4.5))
    for model_index, model_name in enumerate(model_names):
        values = comparison_table.loc[model_index, BAR_METRIC_COLUMNS].astype(float).to_numpy()
        axis.bar(positions + model_index * bar_width, values, bar_width, label=model_name)
    axis.set_xticks(positions + bar_width * (len(model_names) - 1) / 2)
    axis.set_xticklabels([column.replace(" test", "") for column in BAR_METRIC_COLUMNS])
    axis.set(ylim=(0, 1.05), ylabel="Valor (test)", title="Métricas en test (umbral elegido en validación)")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_roc_curves(
    evaluated_models: list[EvaluatedModel],
    split_name: str = "test",
    save_path: Path | None = None,
):
    """Curvas ROC superpuestas de todos los modelos."""
    figure, axis = plt.subplots(figsize=(6, 5.5))
    for evaluated in evaluated_models:
        labels = getattr(evaluated, f"{split_name}_labels").astype(int)
        probabilities = getattr(evaluated, f"{split_name}_probabilities")
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
        auc = summarize_split(probabilities, labels, evaluated.threshold)["auc"]
        axis.plot(
            false_positive_rate, true_positive_rate, label=f"{evaluated.entry.display_name} (AUC={auc:.3f})"
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
             title=f"Curvas ROC ({split_name})")
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_training_curves(evaluated_models: list[EvaluatedModel], save_path: Path | None = None):
    """Loss de train/val y AUC de val por época (solo modelos con historial guardado)."""
    with_history = [evaluated for evaluated in evaluated_models if evaluated.history]
    if not with_history:
        return None

    figure, (loss_axis, auc_axis) = plt.subplots(1, 2, figsize=(13, 4.5))
    for evaluated in with_history:
        epochs = [epoch["epoch"] for epoch in evaluated.history]
        name = evaluated.entry.display_name
        train_line = loss_axis.plot(
            epochs, [e["train_loss"] for e in evaluated.history], linestyle="--", alpha=0.7
        )[0]
        loss_axis.plot(
            epochs, [e["val_loss"] for e in evaluated.history], color=train_line.get_color(), label=name
        )
        auc_axis.plot(epochs, [e["val_auc"] for e in evaluated.history], marker="o", label=name)
    loss_axis.set(title="Pérdida focal (discontinua = train, continua = val)", xlabel="Época", ylabel="Loss")
    auc_axis.set(title="AUC de validación por época", xlabel="Época", ylabel="AUC val")
    loss_axis.legend(fontsize=8)
    auc_axis.legend(fontsize=8)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_confusion_matrices(
    evaluated_models: list[EvaluatedModel],
    save_path: Path | None = None,
):
    """Matriz de confusión en test de cada modelo (umbral elegido en validación)."""
    figure, axes = plt.subplots(1, len(evaluated_models), figsize=(3.2 * len(evaluated_models), 3.4))
    axes = np.atleast_1d(axes)
    for axis, evaluated in zip(axes, evaluated_models):
        summary = summarize_split(evaluated.test_probabilities, evaluated.test_labels, evaluated.threshold)
        matrix = np.array([[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=12)
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["sano", "riesgo"],
                 yticklabels=["sano", "riesgo"], xlabel="Predicho", ylabel="Real")
        axis.set_title(evaluated.entry.display_name, fontsize=9)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_study_histories(studies: dict[str, optuna.Study], save_path: Path | None = None):
    """Mejor AUC de val. alcanzado por cada trial (completos y podados) en cada arquitectura."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for architecture, study in studies.items():
        trial_numbers = [t.number for t in study.trials if t.intermediate_values]
        best_values = [max(t.intermediate_values.values()) for t in study.trials if t.intermediate_values]
        axis.plot(trial_numbers, best_values, marker="o", label=architecture)
    axis.set(title="Optuna: mejor AUC de val. por trial (régimen reducido)", xlabel="Trial", ylabel="AUC val.")
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure
