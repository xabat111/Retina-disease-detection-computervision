"""Resúmenes de puntos de operación (métricas + IC bootstrap) listos para tablas y para JSON."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.recall_optimization import (
    METRIC_NAMES,
    bootstrap_metric_intervals,
    compute_operating_point_metrics,
)


def summarize_operating_point(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    n_resamples: int,
    seed: int,
    with_intervals: bool = True,
) -> dict:
    """Métricas a un umbral dado y, opcionalmente, su IC 95 % por bootstrap."""
    summary = compute_operating_point_metrics(probabilities, labels, threshold)
    if with_intervals:
        intervals = bootstrap_metric_intervals(
            probabilities, labels, threshold, n_resamples=n_resamples, seed=seed
        )
        summary["confidence_intervals_95"] = {
            name: [low, high] for name, (low, high) in intervals.items()
        }
    return summary


def miss_rate_by_disease(
    labels_csv_path: Path,
    probabilities: np.ndarray,
    threshold: float,
    min_cases: int = 5,
) -> pd.DataFrame:
    """Tasa de falsos negativos por enfermedad concreta (columnas de RFMiD) en los positivos.

    Las filas del CSV siguen el mismo orden que el DataFrame de `build_split_dataframe`.
    Una imagen con varias enfermedades cuenta en cada una; los positivos sin ninguna columna
    activa se agrupan en 'otra / sin etiqueta específica'. Solo informativo (test).
    """
    labels_table = pd.read_csv(labels_csv_path)
    positives = labels_table["Disease_Risk"].to_numpy() == 1
    missed = positives & (probabilities < threshold)
    disease_columns = [name for name in labels_table.columns if name not in ("ID", "Disease_Risk")]

    rows = []
    for disease_name in disease_columns:
        has_disease = labels_table[disease_name].to_numpy() == 1
        n_cases = int(has_disease.sum())
        if n_cases < min_cases:
            continue
        rows.append(
            {
                "enfermedad": disease_name,
                "casos": n_cases,
                "falsos_negativos": int((has_disease & missed).sum()),
            }
        )
    no_specific_label = positives & (labels_table[disease_columns].sum(axis=1).to_numpy() == 0)
    if no_specific_label.any():
        rows.append(
            {
                "enfermedad": "otra / sin etiqueta específica",
                "casos": int(no_specific_label.sum()),
                "falsos_negativos": int((no_specific_label & missed).sum()),
            }
        )
    table = pd.DataFrame(rows)
    table["tasa_fn"] = (table["falsos_negativos"] / table["casos"]).round(3)
    return table.sort_values("tasa_fn", ascending=False).reset_index(drop=True)


def build_summary_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """Tabla 'valor [IC95 %]' por configuración (una fila por configuración)."""
    rows = []
    for configuration_name, summary in summaries.items():
        row = {"configuración": configuration_name, "umbral": round(summary["threshold"], 3)}
        for metric_name in METRIC_NAMES:
            value = summary[metric_name]
            intervals = summary.get("confidence_intervals_95")
            if intervals is None:
                row[metric_name] = f"{value:.3f}"
            else:
                low, high = intervals[metric_name]
                row[metric_name] = f"{value:.3f} [{low:.3f}, {high:.3f}]"
        true_negatives, false_positives = summary["confusion_matrix"][0]
        false_negatives, true_positives = summary["confusion_matrix"][1]
        row["TN / FP / FN / TP"] = (
            f"{true_negatives} / {false_positives} / {false_negatives} / {true_positives}"
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("configuración")
