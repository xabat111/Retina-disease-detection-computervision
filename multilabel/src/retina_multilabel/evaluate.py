"""Evaluación: umbrales por clase (solo validación), métricas e intervalos bootstrap."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .labels import HEALTHY_INDEX, LABEL_NAMES

METRIC_NAMES = ["auroc", "average_precision", "recall", "precision", "f1", "specificity"]
AGGREGATE_ROWS = ["macro", "micro"]
DEFAULT_THRESHOLD = 0.5


# --------------------------------------------------------------------------- umbrales


THRESHOLD_CRITERIA = ("f1", "f2", "auprc_recall", "hybrid_error")

# Labels where auprc_recall historically picked near-zero thresholds → flood of FPs.
HYBRID_F1_LABELS = frozenset({"excavacion_disco", "otras_nervio_optico"})
# Catch-all rare class: auprc_recall kept threshold high → under-recall.
HYBRID_F2_LABELS = frozenset({"otros"})
# Floor so auprc_recall cannot collapse to "almost always positive".
AUPRC_RECALL_MIN_THRESHOLD = 0.05


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, criterion: str = "f1") -> float:
    """Umbral sobre validación. Criterios: f1, f2, auprc_recall o hybrid_error.

    Si la clase no tiene positivos o negativos no se puede calibrar y se devuelve 0.5.
    """
    if criterion == "auprc_recall":
        return choose_threshold_auprc_recall(y_true, y_score)
    if criterion == "hybrid_error":
        # Sin nombre de etiqueta: caer a auprc_recall con suelo.
        return max(choose_threshold_auprc_recall(y_true, y_score), AUPRC_RECALL_MIN_THRESHOLD)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return DEFAULT_THRESHOLD
    beta_squared = {"f1": 1.0, "f2": 4.0}[criterion]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]  # el último punto no tiene umbral
    f_score = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall + 1e-12)
    return float(thresholds[np.argmax(f_score)])


def choose_threshold_auprc_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    beta: float = 2.0,
    min_precision_of_auprc: float = 0.35,
    min_f1_of_best: float = 0.70,
) -> float:
    """Umbral por etiqueta para datos desbalanceados: prioriza recall sin hundir el resto.

    Usa AUPRC (AP) y AUROC como anclas de calidad del ranking:
    - Objetivo principal: maximizar F_beta (beta=2 → recall cuenta el doble).
    - Suelo de precisión: max(prevalencia, min_precision_of_auprc × AUPRC) para no
      predecir casi todo positivo en clases raras.
    - Suelo de F1: al menos min_f1_of_best × mejor F1 de la curva (no sacrificar
      equilibrio total por un recall extremo).
    - Si AUROC o AUPRC son muy bajos (<0.55 / <prevalencia), se vuelve más
      conservador (beta→1.5) porque el ranking no es fiable.
    """
    n_positive = float(y_true.sum())
    if n_positive == 0 or n_positive == len(y_true):
        return DEFAULT_THRESHOLD

    auprc = _safe_score(average_precision_score, y_true, y_score)
    auroc = _safe_score(roc_auc_score, y_true, y_score)
    prevalence = n_positive / len(y_true)
    effective_beta = beta
    if (
        (not np.isnan(auroc) and auroc < 0.55)
        or (not np.isnan(auprc) and auprc < max(prevalence, 0.05))
    ):
        effective_beta = min(beta, 1.5)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) == 0:
        return DEFAULT_THRESHOLD

    beta_squared = effective_beta**2
    f_beta = (
        (1 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall + 1e-12)
    )
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_f1 = float(np.nanmax(f1)) if np.isfinite(f1).any() else 0.0

    auprc_floor = 0.0 if np.isnan(auprc) else float(auprc)
    precision_floor = max(prevalence, min_precision_of_auprc * auprc_floor)
    f1_floor = min_f1_of_best * best_f1 if best_f1 > 0 else 0.0

    feasible = (precision >= precision_floor) & (f1 >= f1_floor)
    if not np.any(feasible):
        feasible = precision >= precision_floor
    if not np.any(feasible):
        feasible = np.ones(len(thresholds), dtype=bool)

    # Entre candidatos factibles: F2 (o F_beta), desempate a mayor recall y luego F1.
    scores = np.where(feasible, f_beta, -np.inf)
    order = np.lexsort((-f1, -recall, -scores))
    return float(thresholds[order[0]])


def choose_threshold_for_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float
) -> float:
    """Mayor umbral cuya sensibilidad sea >= `target_recall` (máxima especificidad posible).

    Sirve para fijar un punto de operación de cribado comparable entre modelos; usar SOLO
    validación.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall debe estar en (0, 1], recibido: {target_recall}")
    n_positive = int(y_true.sum())
    if n_positive == 0 or n_positive == len(y_true):
        return DEFAULT_THRESHOLD
    sorted_thresholds = np.sort(np.unique(y_score))[::-1]
    for threshold in sorted_thresholds:
        if ((y_score >= threshold) & (y_true == 1)).sum() / n_positive >= target_recall:
            return float(threshold)
    return float(sorted_thresholds[-1])


def choose_threshold_f1_with_recall_floor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.86,
) -> float:
    """Maximiza F1 entre umbrales que alcanzan el recall mínimo en validación."""
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall debe estar en (0, 1], recibido: {target_recall}")
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return DEFAULT_THRESHOLD

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    eligible = np.flatnonzero(recall >= target_recall)
    if eligible.size == 0:
        return choose_threshold_for_recall(y_true, y_score, target_recall)

    best_f1 = np.nanmax(f1[eligible])
    tied = eligible[np.isclose(f1[eligible], best_f1)]
    # In ties use the higher, more conservative threshold.
    return float(thresholds[tied[-1]])


def choose_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    criterion: str = "f1",
    label_names: list[str] | None = None,
) -> list[float]:
    """Elige umbral por clase. Con `hybrid_error` + `label_names` aplica reglas por etiqueta."""
    if criterion not in THRESHOLD_CRITERIA:
        raise ValueError(f"criterion desconocido: {criterion}. Opciones: {THRESHOLD_CRITERIA}")
    n_classes = targets.shape[1]
    if criterion != "hybrid_error":
        return [
            choose_threshold(targets[:, c], probabilities[:, c], criterion)
            for c in range(n_classes)
        ]

    names = list(label_names) if label_names is not None else [f"class_{i}" for i in range(n_classes)]
    if len(names) != n_classes:
        raise ValueError(f"label_names length {len(names)} != n_classes {n_classes}")
    out: list[float] = []
    for class_index, name in enumerate(names):
        y_true = targets[:, class_index]
        y_score = probabilities[:, class_index]
        if name in HYBRID_F1_LABELS:
            out.append(choose_threshold(y_true, y_score, "f1"))
        elif name in HYBRID_F2_LABELS:
            out.append(choose_threshold(y_true, y_score, "f2"))
        else:
            out.append(
                max(choose_threshold_auprc_recall(y_true, y_score), AUPRC_RECALL_MIN_THRESHOLD)
            )
    return out


def per_label_ranking_diagnostics(
    targets: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    """AUPRC y AUROC por etiqueta (sin umbral); útil para interpretar el ajuste de umbrales."""
    rows = []
    for class_index, label in enumerate(LABEL_NAMES):
        y_true = targets[:, class_index]
        y_score = probabilities[:, class_index]
        rows.append(
            {
                "label": label,
                "n_positivos": int(y_true.sum()),
                "prevalence": float(y_true.mean()),
                "auprc": _safe_score(average_precision_score, y_true, y_score),
                "auroc": _safe_score(roc_auc_score, y_true, y_score),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- métricas


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def _safe_score(metric_function, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC/AP solo están definidos si hay positivos y negativos."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(metric_function(y_true, y_score))


def binary_metric_vector(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """[AUROC, AP, recall, precision, F1, especificidad] de un problema binario (NaN si indefinido)."""
    true_positive = float(((y_pred == 1) & (y_true == 1)).sum())
    false_positive = float(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = float(((y_pred == 0) & (y_true == 1)).sum())
    true_negative = float(((y_pred == 0) & (y_true == 0)).sum())
    return np.array(
        [
            _safe_score(roc_auc_score, y_true, y_score),
            _safe_score(average_precision_score, y_true, y_score),
            _safe_ratio(true_positive, true_positive + false_negative),
            _safe_ratio(true_positive, true_positive + false_positive),
            _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
            _safe_ratio(true_negative, true_negative + false_positive),
        ]
    )


def multilabel_metric_matrix(
    targets: np.ndarray, probabilities: np.ndarray, thresholds: list[float]
) -> np.ndarray:
    """Matriz (C + 2, 6): una fila por clase y después macro y micro."""
    predictions = assign_predictions(probabilities, thresholds)
    per_class = np.stack(
        [
            binary_metric_vector(targets[:, c], probabilities[:, c], predictions[:, c])
            for c in range(targets.shape[1])
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # columnas todo-NaN
        macro = np.nanmean(per_class, axis=0)
    micro = binary_metric_vector(targets.ravel(), probabilities.ravel(), predictions.ravel())
    return np.vstack([per_class, macro, micro])


def macro_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """AUROC macro sobre las clases evaluables (criterio de parada temprana del baseline)."""
    scores = [
        _safe_score(roc_auc_score, targets[:, c], probabilities[:, c])
        for c in range(targets.shape[1])
    ]
    return float(np.nanmean(scores)) if not np.all(np.isnan(scores)) else float("nan")


def macro_average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """mAP: media de la AP por etiqueta, ignorando clases sin positivos o sin negativos."""
    scores = [
        _safe_score(average_precision_score, targets[:, c], probabilities[:, c])
        for c in range(targets.shape[1])
    ]
    return float(np.nanmean(scores)) if not np.all(np.isnan(scores)) else float("nan")


def per_label_average_precision(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.array(
        [
            _safe_score(average_precision_score, targets[:, c], probabilities[:, c])
            for c in range(targets.shape[1])
        ]
    )


def per_label_f1_at_threshold(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """F1 por etiqueta con un umbral fijo (0.5 al entrenar; los umbrales finales se eligen en val)."""
    predictions = (probabilities >= threshold).astype(int)
    scores = []
    for class_index in range(targets.shape[1]):
        metrics = binary_metric_vector(
            targets[:, class_index], probabilities[:, class_index], predictions[:, class_index]
        )
        scores.append(metrics[METRIC_NAMES.index("f1")])
    return np.array(scores)


def subset_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Fracción de imágenes con el vector de 10 etiquetas exactamente correcto."""
    return float((predictions == targets).all(axis=1).mean())


def hamming_loss(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Fracción de bits (imagen × etiqueta) incorrectos."""
    return float((predictions != targets).mean())


def secondary_multilabel_scores(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Subset accuracy y Hamming loss, comparados con predecir todo ceros."""
    zeros = np.zeros_like(predictions)
    return {
        "subset_accuracy": subset_accuracy(targets, predictions),
        "subset_accuracy_all_zeros": subset_accuracy(targets, zeros),
        "hamming_loss": hamming_loss(targets, predictions),
        "hamming_loss_all_zeros": hamming_loss(targets, zeros),
    }


def bootstrap_confidence_bounds(
    metric_function,
    n_rows: int,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentiles bootstrap de `metric_function(indices)` remuestreando filas con reemplazo."""
    rng = np.random.default_rng(seed)
    resampled = np.stack(
        [metric_function(rng.integers(0, n_rows, n_rows)) for _ in range(n_resamples)]
    )
    alpha = (1 - confidence_level) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lower = np.nanpercentile(resampled, 100 * alpha, axis=0)
        upper = np.nanpercentile(resampled, 100 * (1 - alpha), axis=0)
    return lower, upper


def build_metrics_table(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
    n_resamples: int,
    confidence_level: float,
    seed: int,
    min_positives: int,
) -> pd.DataFrame:
    """Tabla por grupo + macro/micro con cada métrica y su IC bootstrap [ci_low, ci_high]."""
    point = multilabel_metric_matrix(targets, probabilities, thresholds)
    lower, upper = bootstrap_confidence_bounds(
        lambda idx: multilabel_metric_matrix(targets[idx], probabilities[idx], thresholds),
        len(targets),
        n_resamples,
        confidence_level,
        seed,
    )
    index = LABEL_NAMES + AGGREGATE_ROWS
    table = pd.DataFrame(index=index)
    n_positive = targets.sum(axis=0).astype(int)
    table["n_positivos"] = list(n_positive) + [np.nan, np.nan]
    table["umbral"] = list(thresholds) + [np.nan, np.nan]
    for column, name in enumerate(METRIC_NAMES):
        table[name] = point[:, column]
        table[f"{name}_ic_low"] = lower[:, column]
        table[f"{name}_ic_high"] = upper[:, column]
    predictions = assign_predictions(probabilities, thresholds)

    def accuracy_vector(indices: np.ndarray) -> np.ndarray:
        matches = targets[indices].astype(bool) == predictions[indices].astype(bool)
        per_label = matches.mean(axis=0)
        return np.concatenate([per_label, [per_label.mean(), matches.mean()]])

    accuracy = accuracy_vector(np.arange(len(targets)))
    accuracy_low, accuracy_high = bootstrap_confidence_bounds(
        accuracy_vector, len(targets), n_resamples, confidence_level, seed
    )
    table["accuracy"] = accuracy
    table["accuracy_ic_low"] = accuracy_low
    table["accuracy_ic_high"] = accuracy_high
    table["metricas_fiables"] = [bool(n >= min_positives) for n in n_positive] + [np.nan, np.nan]
    return table


def format_metrics_table(table: pd.DataFrame) -> pd.DataFrame:
    """Versión legible: 'valor [ic_low–ic_high]' por métrica."""
    formatted = table[["n_positivos", "umbral", "metricas_fiables"]].copy()
    for name in (*METRIC_NAMES, "accuracy"):
        formatted[name] = [
            f"{value:.3f} [{low:.2f}–{high:.2f}]"
            for value, low, high in zip(
                table[name], table[f"{name}_ic_low"], table[f"{name}_ic_high"]
            )
        ]
    return formatted


# --------------------------------------------------------------------------- sano vs. enfermo


def disease_scores(probabilities: np.ndarray) -> np.ndarray:
    """Puntuación de enfermedad derivada = 1 - P(sano)."""
    return 1.0 - probabilities[:, HEALTHY_INDEX]


def disease_targets(targets: np.ndarray) -> np.ndarray:
    return 1 - targets[:, HEALTHY_INDEX].astype(int)


def evaluate_disease_screening(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Métricas binarias sano-vs-enfermo (positivo = enfermo) con IC bootstrap."""
    y_true = disease_targets(targets)
    y_score = disease_scores(probabilities)

    def metrics_for(indices: np.ndarray) -> np.ndarray:
        return binary_metric_vector(
            y_true[indices], y_score[indices], (y_score[indices] >= threshold).astype(int)
        )

    all_rows = np.arange(len(y_true))
    point = metrics_for(all_rows)
    lower, upper = bootstrap_confidence_bounds(
        metrics_for, len(y_true), n_resamples, confidence_level, seed
    )
    table = pd.DataFrame({"valor": point, "ic_low": lower, "ic_high": upper}, index=METRIC_NAMES)
    table.attrs["threshold"] = threshold
    return table


def confusion_matrices(
    targets: np.ndarray, probabilities: np.ndarray, thresholds: list[float]
) -> np.ndarray:
    """Matrices 2x2 [[TN, FP], [FN, TP]] por clase, forma (C, 2, 2)."""
    predictions = assign_predictions(probabilities, thresholds)
    matrices = np.zeros((targets.shape[1], 2, 2), dtype=int)
    for c in range(targets.shape[1]):
        for truth in (0, 1):
            for predicted in (0, 1):
                matrices[c, truth, predicted] = int(
                    ((targets[:, c] == truth) & (predictions[:, c] == predicted)).sum()
                )
    return matrices


def build_summary_table(
    metrics_by_split: dict[str, pd.DataFrame], screening_by_split: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Resumen macro/micro y sano-vs-enfermo en una sola tabla (valor + IC 95 %).

    Las claves de `screening_by_split` describen split y punto de operación, p. ej. "test [F2]".
    """
    rows = {}
    for split, table in metrics_by_split.items():
        for aggregate in AGGREGATE_ROWS:
            for name in (*METRIC_NAMES, "accuracy"):
                rows[f"{split} | {aggregate} | {name}"] = {
                    "valor": table.loc[aggregate, name],
                    "ic_low": table.loc[aggregate, f"{name}_ic_low"],
                    "ic_high": table.loc[aggregate, f"{name}_ic_high"],
                }
    for key, table in screening_by_split.items():
        for name in ("auroc", "recall", "specificity", "precision", "f1"):
            rows[f"{key} | sano_vs_enfermo | {name}"] = table.loc[name].to_dict()
    return pd.DataFrame.from_dict(rows, orient="index")


HEADLINE_METRIC_NAMES = ("f1", "average_precision", "precision", "recall", "specificity")


def f2_from_precision_recall(precision: float, recall: float) -> float:
    """F2 a partir de P y R (el recall cuenta el doble). Es la fórmula de las tablas de ranking."""
    return float((5.0 * precision * recall) / (4.0 * precision + recall + 1e-12))


def healthy_disease_conflict_mask(
    predictions: np.ndarray, healthy_index: int = HEALTHY_INDEX
) -> np.ndarray:
    """True en las filas donde `sano` y al menos una clase de enfermedad salen positivas."""
    predicted = np.asarray(predictions)
    if predicted.ndim != 2:
        raise ValueError(f"predictions debe ser (N, C); recibido {predicted.shape}")
    healthy_positive = predicted[:, healthy_index] == 1
    disease_positive = np.delete(predicted, healthy_index, axis=1).any(axis=1)
    return healthy_positive & disease_positive


def resolve_healthy_disease_conflict(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    healthy_index: int = HEALTHY_INDEX,
) -> np.ndarray:
    """Si `sano` y alguna enfermedad superan el umbral, se queda solo el lado de mayor probabilidad.

    En las etiquetas de entrenamiento `sano` es exclusivo de los 9 grupos de patología.
    Los umbrales por etiqueta son independientes, así que una imagen puede disparar ambos.
    Esta regla no toca las puntuaciones (AUROC / AUPRC): solo el vector binario usado para
    F1, precisión, recall, especificidad, subset accuracy y Hamming.

    Por imagen, si hay conflicto:
    - si max(prob. de enfermedades ya positivas) > P(sano) → se apaga `sano` y se
      conservan las detecciones de enfermedad;
    - si P(sano) es la más alta (incluido el empate) → se apagan todas las enfermedades
      y se conserva solo `sano`.

    `otros` cuenta como enfermedad en la comparación; su marca de métricas no fiables
    no cambia.
    """
    scores = np.asarray(probabilities, dtype=np.float64)
    resolved = np.array(predictions, dtype=int, copy=True)
    if scores.shape != resolved.shape:
        raise ValueError(
            f"probabilities {scores.shape} y predictions {resolved.shape} deben coincidir"
        )
    if resolved.ndim != 2:
        raise ValueError(f"se espera (N, C); recibido {resolved.shape}")

    conflict = healthy_disease_conflict_mask(resolved, healthy_index)
    if not np.any(conflict):
        return resolved

    disease_active = resolved.astype(bool)
    disease_active[:, healthy_index] = False
    masked_disease_scores = np.where(disease_active, scores, -np.inf)
    max_disease_score = masked_disease_scores.max(axis=1)
    healthy_score = scores[:, healthy_index]

    disease_wins = conflict & (max_disease_score > healthy_score)
    healthy_wins = conflict & ~disease_wins

    resolved[disease_wins, healthy_index] = 0
    if np.any(healthy_wins):
        resolved[healthy_wins] = 0
        resolved[healthy_wins, healthy_index] = 1
    return resolved


def assign_predictions(probabilities: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Binariza con umbrales por etiqueta y resuelve el conflicto sano / enfermedad.

    Si el número de columnas no es el de `LABEL_NAMES` (p. ej. experimento sin `sano`),
    se omite la regla de exclusividad sano/enfermedad.
    """
    predictions = (np.asarray(probabilities) >= np.asarray(thresholds)).astype(int)
    if predictions.shape[1] != len(LABEL_NAMES):
        return predictions
    return resolve_healthy_disease_conflict(np.asarray(probabilities), predictions)


def build_protocol_headline(
    metrics_table: pd.DataFrame,
    secondary: dict[str, float],
    screening: pd.DataFrame,
    model_name: str,
    split: str,
) -> pd.Series:
    """Fila-resumen del protocolo pedido: F1 macro/micro de cabecera, AP, secundarias.

    El AUROC macro no se destaca (sí aparece por etiqueta en `metrics_*.csv`).
    """
    row = {
        "modelo": model_name,
        "split": split,
        "f1_macro": metrics_table.loc["macro", "f1"],
        "f1_macro_ic_low": metrics_table.loc["macro", "f1_ic_low"],
        "f1_macro_ic_high": metrics_table.loc["macro", "f1_ic_high"],
        "f1_micro": metrics_table.loc["micro", "f1"],
        "f1_micro_ic_low": metrics_table.loc["micro", "f1_ic_low"],
        "f1_micro_ic_high": metrics_table.loc["micro", "f1_ic_high"],
        "ap_macro": metrics_table.loc["macro", "average_precision"],
        "ap_macro_ic_low": metrics_table.loc["macro", "average_precision_ic_low"],
        "ap_macro_ic_high": metrics_table.loc["macro", "average_precision_ic_high"],
        "ap_micro": metrics_table.loc["micro", "average_precision"],
        "accuracy_macro": metrics_table.loc["macro", "accuracy"],
        "accuracy_macro_ic_low": metrics_table.loc["macro", "accuracy_ic_low"],
        "accuracy_macro_ic_high": metrics_table.loc["macro", "accuracy_ic_high"],
        "accuracy_micro": metrics_table.loc["micro", "accuracy"],
        "accuracy_micro_ic_low": metrics_table.loc["micro", "accuracy_ic_low"],
        "accuracy_micro_ic_high": metrics_table.loc["micro", "accuracy_ic_high"],
        "subset_accuracy": secondary["subset_accuracy"],
        "subset_accuracy_all_zeros": secondary["subset_accuracy_all_zeros"],
        "hamming_loss": secondary["hamming_loss"],
        "hamming_loss_all_zeros": secondary["hamming_loss_all_zeros"],
        "sano_vs_enfermo_f1": screening.loc["f1", "valor"],
        "sano_vs_enfermo_recall": screening.loc["recall", "valor"],
        "sano_vs_enfermo_specificity": screening.loc["specificity", "valor"],
        "sano_vs_enfermo_auroc": screening.loc["auroc", "valor"],
    }
    return pd.Series(row)


def evaluate_split_protocol(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
    screening_threshold: float,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    min_positives: int,
    model_name: str,
    split: str,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Evalúa un split con umbrales ya fijados en validación (el test no los elige)."""
    metrics = build_metrics_table(
        targets, probabilities, thresholds, n_resamples, confidence_level, seed, min_positives
    )
    predictions = assign_predictions(probabilities, thresholds)
    secondary = secondary_multilabel_scores(targets.astype(int), predictions)
    screening = evaluate_disease_screening(
        targets, probabilities, screening_threshold, n_resamples, confidence_level, seed
    )
    headline = build_protocol_headline(metrics, secondary, screening, model_name, split)
    return {
        "metrics": metrics,
        "headline": headline,
        "secondary": pd.Series(secondary, name=f"{model_name}_{split}"),
        "screening": screening,
        "confusion": confusion_matrices(targets, probabilities, thresholds),
    }


def labels_that_worsened(
    baseline_metrics: pd.DataFrame,
    finetuned_metrics: pd.DataFrame,
    metric_name: str = "f1",
) -> pd.DataFrame:
    """Etiquetas cuyo valor puntual en test baja respecto al baseline (incluye `otros`)."""
    rows = []
    for label in LABEL_NAMES:
        before = float(baseline_metrics.loc[label, metric_name])
        after = float(finetuned_metrics.loc[label, metric_name])
        rows.append(
            {
                "etiqueta": label,
                f"{metric_name}_baseline": before,
                f"{metric_name}_finetune": after,
                "delta": after - before,
                "metricas_fiables": bool(finetuned_metrics.loc[label, "metricas_fiables"]),
            }
        )
    table = pd.DataFrame(rows).sort_values("delta")
    return table[table["delta"] < 0].reset_index(drop=True)
