"""Figuras del proyecto. Todas devuelven la figura y la guardan si se indica `save_path`."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from .labels import LABEL_NAMES


def _finalize(figure: Figure, save_path: Path | None) -> Figure:
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=130, bbox_inches="tight")
    return figure


def plot_label_counts(prevalence_table: pd.DataFrame, save_path: Path | None = None) -> Figure:
    """Positivos por etiqueta y split (escala logarítmica por el fuerte desbalance)."""
    count_columns = ["n_train", "n_val", "n_test"]
    axis = prevalence_table[count_columns].plot.bar(figsize=(11, 4), logy=True)
    axis.set_ylabel("nº de imágenes (escala log)")
    axis.set_title("Positivos por etiqueta y split")
    return _finalize(axis.get_figure(), save_path)


def plot_cooccurrence(matrix: pd.DataFrame, save_path: Path | None = None) -> Figure:
    """Mapa de calor de coocurrencia de etiquetas (la diagonal son los positivos de cada una)."""
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(np.log1p(matrix.to_numpy()), cmap="viridis")
    axis.set_xticks(range(len(matrix)), matrix.columns, rotation=60, ha="right")
    axis.set_yticks(range(len(matrix)), matrix.index)
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            axis.text(column, row, int(matrix.iat[row, column]), ha="center", va="center",
                      color="white", fontsize=7)
    axis.set_title("Coocurrencia de etiquetas (color = log(1+n))")
    figure.colorbar(image, ax=axis, fraction=0.046)
    return _finalize(figure, save_path)


def plot_training_curves(history: pd.DataFrame, best_epoch: int, save_path: Path | None = None) -> Figure:
    figure, (loss_axis, score_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(history["epoch"], history["train_loss"], label="train")
    loss_axis.plot(history["epoch"], history["val_loss"], label="val")
    loss_axis.set(xlabel="época", ylabel="BCE ponderada", title="Pérdida")
    loss_axis.legend()
    score_axis.plot(history["epoch"], history["val_macro_auroc"], color="tab:green")
    score_axis.axvline(best_epoch, linestyle="--", color="gray", label=f"mejor época ({best_epoch})")
    score_axis.set(xlabel="época", ylabel="AUROC macro (val)", title="Validación")
    score_axis.legend()
    return _finalize(figure, save_path)


def plot_metric_bars(
    metrics_table: pd.DataFrame,
    metric_names: tuple[str, ...] = ("auroc", "average_precision", "recall"),
    save_path: Path | None = None,
) -> Figure:
    """Barras por grupo con IC 95 %; un asterisco marca grupos con pocos positivos."""
    groups = metrics_table.loc[LABEL_NAMES]
    figure, axis = plt.subplots(figsize=(12, 4.5))
    width = 0.8 / len(metric_names)
    positions = np.arange(len(groups))
    for offset, name in enumerate(metric_names):
        values = groups[name].to_numpy(dtype=float)
        error = np.vstack(
            [values - groups[f"{name}_ic_low"], groups[f"{name}_ic_high"] - values]
        ).clip(min=0)
        axis.bar(positions + offset * width, values, width, yerr=np.nan_to_num(error),
                 capsize=2, label=name)
    labels = [f"{name}{'' if reliable else ' *'}" for name, reliable in
              zip(groups.index, groups["metricas_fiables"])]
    axis.set_xticks(positions + width * (len(metric_names) - 1) / 2, labels, rotation=40, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_title("Métricas por grupo (barras de error = IC 95 % bootstrap; * = pocos positivos)")
    axis.legend()
    return _finalize(figure, save_path)


def plot_confusion_matrices(
    matrices: np.ndarray, label_names: list[str] = LABEL_NAMES, save_path: Path | None = None
) -> Figure:
    """Matrices de confusión 2x2 por grupo en una cuadrícula."""
    n_columns = 5
    n_rows = int(np.ceil(len(label_names) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(3 * n_columns, 3 * n_rows))
    for axis, name, matrix in zip(axes.ravel(), label_names, matrices):
        axis.imshow(matrix, cmap="Blues")
        for truth in range(2):
            for predicted in range(2):
                axis.text(predicted, truth, matrix[truth, predicted], ha="center", va="center")
        axis.set_xticks([0, 1], ["pred 0", "pred 1"])
        axis.set_yticks([0, 1], ["real 0", "real 1"])
        axis.set_title(name, fontsize=9)
    for axis in axes.ravel()[len(label_names):]:
        axis.axis("off")
    return _finalize(figure, save_path)


def plot_screening_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    operating_points: dict[str, float],
    save_path: Path | None = None,
) -> Figure:
    """Curva ROC sano-vs-enfermo con los puntos de operación (umbrales elegidos en validación)."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(false_positive_rate, true_positive_rate)
    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    for name, threshold in operating_points.items():
        point = np.argmin(np.abs(thresholds - threshold))
        axis.scatter(false_positive_rate[point], true_positive_rate[point], zorder=3, label=name)
    axis.set(xlabel="1 - especificidad", ylabel="sensibilidad", title="Sano vs. enfermo (test)")
    axis.legend()
    return _finalize(figure, save_path)


def plot_xai_cases(rows: list[dict], column_titles: list[str], title: str,
                   save_path: Path | None = None) -> Figure:
    """Una fila por caso: imagen original y sus mapas de explicación superpuestos.

    Cada fila es un dict con `images` (lista de arrays RGB, una por columna) y `caption`.
    """
    figure, axes = plt.subplots(len(rows), len(column_titles),
                                figsize=(3.2 * len(column_titles), 3.3 * len(rows)), squeeze=False)
    for row_axes, row in zip(axes, rows):
        for column, (axis, image) in enumerate(zip(row_axes, row["images"])):
            axis.imshow(image)
            axis.axis("off")
            if column == 0:
                axis.set_title(row["caption"], fontsize=8)
    for axis, column_title in zip(axes[0], column_titles):
        axis.annotate(column_title, (0.5, 1.18), xycoords="axes fraction", ha="center", fontsize=10)
    figure.suptitle(title, y=1.0)
    return _finalize(figure, save_path)


def plot_finetune_history(history: pd.DataFrame, best_epoch: int, save_path: Path | None = None) -> Figure:
    """Curvas del ajuste fino: la selección es argmin(val_loss); val mAP es secundario."""
    figure, (loss_axis, score_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(history["epoch"], history["train_loss"], label="train")
    loss_axis.plot(history["epoch"], history["val_loss"], label="val")
    loss_axis.axvline(best_epoch, linestyle="--", color="gray", label=f"mejor val_loss ({best_epoch})")
    loss_axis.set(xlabel="época", ylabel="Focal Loss", title="Pérdida (criterio de selección)")
    loss_axis.legend()
    score_axis.plot(history["epoch"], history["val_mAP"], color="tab:green", label="val mAP")
    score_axis.axvline(best_epoch, linestyle="--", color="gray", label=f"mejor val_loss ({best_epoch})")
    score_axis.set(xlabel="época", ylabel="val mAP", title="Validación (métrica secundaria)")
    score_axis.legend()
    return _finalize(figure, save_path)


def plot_per_label_curves(per_label_history: pd.DataFrame, save_path: Path | None = None) -> Figure:
    """AP y F1 (umbral 0.5) por etiqueta a lo largo de las épocas de validación."""
    figure, axes = plt.subplots(2, 5, figsize=(14, 6), sharex=True, sharey=True)
    for axis, label in zip(axes.ravel(), LABEL_NAMES):
        subset = per_label_history[per_label_history["label"] == label]
        axis.plot(subset["epoch"], subset["average_precision"], label="AP")
        axis.plot(subset["epoch"], subset["f1"], label="F1@0.5")
        axis.set_title(label, fontsize=8)
        axis.set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].set_xlabel("época")
    axes[1, 0].set_ylabel("val")
    figure.suptitle("Evolución por etiqueta (val). El mAP medio puede subir mientras una clase empeora.")
    return _finalize(figure, save_path)


def plot_headline_comparison(headline_table: pd.DataFrame, save_path: Path | None = None) -> Figure:
    """Barras de F1 macro/micro y AP macro (test) entre baseline y ajuste fino."""
    test_rows = headline_table[headline_table["split"] == "test"]
    metrics = ["f1_macro", "f1_micro", "ap_macro"]
    figure, axis = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(metrics))
    width = 0.35
    for offset, (_, row) in enumerate(test_rows.iterrows()):
        axis.bar(positions + offset * width, [row[name] for name in metrics], width, label=row["modelo"])
    axis.set_xticks(positions + width * (len(test_rows) - 1) / 2, metrics)
    axis.set_ylim(0, 1.05)
    axis.set_title("Protocolo de informe (test): F1 y AP, no AUROC macro")
    axis.legend()
    return _finalize(figure, save_path)
