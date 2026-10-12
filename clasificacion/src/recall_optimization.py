"""Selección de umbral orientada a recall, métricas por punto de operación e intervalos bootstrap.

Contexto médico: un falso negativo (enfermo sin detectar) es peor que un falso positivo, así que
se fija un recall mínimo de la clase positiva (Disease_Risk = 1) y, entre los umbrales que lo
cumplen, se prefiere el que más especificidad conserva. TODO lo que se elige aquí (umbral,
miembros del ensemble) se decide con VALIDACIÓN; test solo se usa para informar.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

METRIC_NAMES = ("recall", "specificity", "precision", "accuracy", "f1", "auc")


def compute_operating_point_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    """Recall, especificidad, precisión, accuracy, F1, AUC y matriz de confusión a un umbral.

    Positivo = Disease_Risk = 1. La matriz de confusión sigue el formato de sklearn
    [[TN, FP], [FN, TP]]. El AUC no depende del umbral.
    """
    labels_int = labels.astype(int)
    predictions = probabilities >= threshold
    positives = labels_int == 1

    true_positives = int(np.sum(predictions & positives))
    false_negatives = int(np.sum(~predictions & positives))
    false_positives = int(np.sum(predictions & ~positives))
    true_negatives = int(np.sum(~predictions & ~positives))

    def safe_ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    recall = safe_ratio(true_positives, true_positives + false_negatives)
    precision = safe_ratio(true_positives, true_positives + false_positives)
    f1 = safe_ratio(2 * true_positives, 2 * true_positives + false_positives + false_negatives)
    auc = (
        float(roc_auc_score(labels_int, probabilities))
        if len(np.unique(labels_int)) == 2
        else float("nan")
    )
    return {
        "threshold": float(threshold),
        "recall": recall,
        "specificity": safe_ratio(true_negatives, true_negatives + false_positives),
        "precision": precision,
        "accuracy": safe_ratio(true_positives + true_negatives, len(labels_int)),
        "f1": f1,
        "auc": auc,
        "confusion_matrix": [
            [true_negatives, false_positives],
            [false_negatives, true_positives],
        ],
    }


def sweep_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Tabla recall / especificidad / precisión / F1 / accuracy en función del umbral."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)
    rows = []
    for threshold in thresholds:
        metrics = compute_operating_point_metrics(probabilities, labels, float(threshold))
        rows.append({name: metrics[name] for name in ("threshold", *METRIC_NAMES[:-1])})
    return pd.DataFrame(rows)


def select_threshold_for_recall(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
) -> float:
    """Mayor umbral cuyo recall sobre estas predicciones es >= `target_recall`.

    El recall solo baja al subir el umbral, y la especificidad solo sube; por tanto el mayor
    umbral que aún cumple el recall mínimo es el que más especificidad conserva. Es exacto:
    se queda con la k-ésima probabilidad positiva más alta, k = ceil(target * n_positivos).
    Debe llamarse SOLO con validación.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall debe estar en (0, 1], recibido {target_recall}")
    positive_probabilities = np.sort(probabilities[labels.astype(int) == 1])[::-1]
    required_true_positives = int(np.ceil(target_recall * len(positive_probabilities) - 1e-9))
    return float(positive_probabilities[required_true_positives - 1])


def select_threshold_fn_below_fp(
    probabilities: np.ndarray,
    labels: np.ndarray,
    confidence: float = 0.9,
    n_resamples: int = 1000,
    seed: int = 42,
) -> float:
    """Mayor umbral con FN < FP en al menos `confidence` de las remuestras bootstrap de validación.

    Con `confidence` = 0.5 equivale (aprox.) al umbral donde se cruzan FN y FP en la propia
    validación. Con un valor mayor se exige un margen: con 640 imágenes el cruce observado tiene
    ruido de varios casos y un umbral justo en el cruce dejaría FN ≈ FP al aplicarlo a otras
    imágenes. Los falsos negativos bajan y los falsos positivos suben al bajar el umbral, así
    que el mayor umbral que cumple la condición es el que más especificidad conserva.
    Debe llamarse SOLO con validación.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence debe estar en (0, 1), recibido {confidence}")
    candidate_thresholds = np.linspace(0.0, 1.0, 401)
    random_generator = np.random.default_rng(seed)
    n_samples = len(labels)
    labels_int = labels.astype(int)
    fewer_fn_than_fp_count = np.zeros(len(candidate_thresholds))

    for _ in range(n_resamples):
        resample_indices = random_generator.integers(0, n_samples, size=n_samples)
        resample_probabilities = probabilities[resample_indices]
        resample_positives = labels_int[resample_indices] == 1
        predicted_positive = resample_probabilities[:, None] >= candidate_thresholds[None, :]
        false_negatives = (~predicted_positive & resample_positives[:, None]).sum(axis=0)
        false_positives = (predicted_positive & ~resample_positives[:, None]).sum(axis=0)
        fewer_fn_than_fp_count += false_negatives < false_positives

    fraction_fewer_fn = fewer_fn_than_fp_count / n_resamples
    meets_confidence = np.flatnonzero(fraction_fewer_fn >= confidence)
    return float(candidate_thresholds[meets_confidence.max()])


def bootstrap_metric_intervals(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Intervalo percentil bootstrap de cada métrica con el umbral FIJO (elegido en validación).

    Se remuestrean las imágenes de test con reemplazo; cuantifica la incertidumbre por el
    tamaño finito del test (solo 134 negativos), no la de elegir el umbral.
    """
    resample_metrics = _bootstrap_resample_metrics(
        [probabilities], labels, [threshold], n_resamples, seed
    )[0]
    return _percentile_intervals(resample_metrics, confidence_level)


def bootstrap_metric_differences(
    probabilities_after: np.ndarray,
    threshold_after: float,
    probabilities_before: np.ndarray,
    threshold_before: float,
    labels: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Diferencia (después - antes) de cada métrica: (valor observado, IC inferior, IC superior).

    Ambos puntos de operación se evalúan sobre las MISMAS remuestras (bootstrap pareado), así
    que el IC de la diferencia es mucho más estrecho que comparar dos IC independientes.
    Si el IC contiene el 0, no hay evidencia de que la métrica haya cambiado.
    """
    after_metrics, before_metrics = _bootstrap_resample_metrics(
        [probabilities_after, probabilities_before],
        labels,
        [threshold_after, threshold_before],
        n_resamples,
        seed,
    )
    observed_after = compute_operating_point_metrics(probabilities_after, labels, threshold_after)
    observed_before = compute_operating_point_metrics(probabilities_before, labels, threshold_before)
    differences = {}
    for metric_name in METRIC_NAMES:
        resample_differences = after_metrics[metric_name] - before_metrics[metric_name]
        low, high = _percentile_intervals({metric_name: resample_differences}, confidence_level)[
            metric_name
        ]
        differences[metric_name] = (
            float(observed_after[metric_name] - observed_before[metric_name]),
            low,
            high,
        )
    return differences


def _bootstrap_resample_metrics(
    probabilities_per_model: list[np.ndarray],
    labels: np.ndarray,
    thresholds: list[float],
    n_resamples: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """Métricas de cada modelo sobre `n_resamples` remuestras compartidas (mismos índices)."""
    random_generator = np.random.default_rng(seed)
    n_samples = len(labels)
    collected = [{name: [] for name in METRIC_NAMES} for _ in probabilities_per_model]

    for _ in range(n_resamples):
        resample_indices = random_generator.integers(0, n_samples, size=n_samples)
        resample_labels = labels[resample_indices]
        if len(np.unique(resample_labels)) < 2:
            continue
        for model_index, (probabilities, threshold) in enumerate(
            zip(probabilities_per_model, thresholds)
        ):
            metrics = compute_operating_point_metrics(
                probabilities[resample_indices], resample_labels, threshold
            )
            for name in METRIC_NAMES:
                collected[model_index][name].append(metrics[name])

    return [
        {name: np.asarray(values) for name, values in model_metrics.items()}
        for model_metrics in collected
    ]


def _percentile_intervals(
    resample_metrics: dict[str, np.ndarray],
    confidence_level: float,
) -> dict[str, tuple[float, float]]:
    tail = (1.0 - confidence_level) / 2.0 * 100.0
    return {
        name: (
            float(np.nanpercentile(values, tail)),
            float(np.nanpercentile(values, 100.0 - tail)),
        )
        for name, values in resample_metrics.items()
    }


def rank_ensemble_subsets(
    member_probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
) -> pd.DataFrame:
    """AUC de validación de TODOS los subconjuntos no vacíos de miembros (media de probabilidades).

    Con pocos miembros (<= 6) es exhaustivo y barato. Se ordena por AUC de validación.
    """
    member_names = list(member_probabilities)
    rows = []
    for subset_size in range(1, len(member_names) + 1):
        for subset in combinations(member_names, subset_size):
            averaged = np.mean([member_probabilities[name] for name in subset], axis=0)
            rows.append(
                {
                    "miembros": subset,
                    "n_miembros": subset_size,
                    "val_auc": float(roc_auc_score(labels.astype(int), averaged)),
                }
            )
    return pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)


def choose_ensemble_subset(
    ranking: pd.DataFrame,
    auc_tolerance: float = 0.002,
) -> tuple[str, ...]:
    """Subconjunto con más miembros cuyo AUC de validación está a `auc_tolerance` del mejor.

    Con 640 imágenes de validación, diferencias de AUC < 0.002 son ruido; a igualdad práctica
    se prefiere el ensemble más grande (más estable y menos sobreajustado a validación).
    """
    best_auc = ranking["val_auc"].max()
    near_best = ranking[ranking["val_auc"] >= best_auc - auc_tolerance]
    largest_first = near_best.sort_values(["n_miembros", "val_auc"], ascending=[False, False])
    return tuple(largest_first.iloc[0]["miembros"])
