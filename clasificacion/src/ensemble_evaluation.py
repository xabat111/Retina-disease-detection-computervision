"""Utilidades para evaluar modelos y ensembles con TTA a partir de probabilidades cacheadas.

Flujo (el mismo de los notebooks 05, 06 y 07): probabilidades por vista TTA y por modelo ->
media sobre vistas -> media sobre miembros del ensemble -> umbral elegido en validación -> métricas
en test. Todo lo que se elige (miembros, umbral) usa SOLO validación; test únicamente informa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.recall_optimization import (
    compute_operating_point_metrics,
    select_threshold_fn_below_fp,
    select_threshold_for_recall,
)
from src.tta_inference import (
    DEFAULT_TTA_VIEW_NAMES,
    EnsembleMember,
    average_views,
    get_member_view_probabilities,
)

SPLIT_NAMES = ("val", "test")

# Reglas de umbral comparadas en los notebooks 06 y 07 (ambas se calculan SOLO con validación)
RECALL_TARGET_RULE = "recall val >= 0.91"
FN_BELOW_FP_RULE = "FN < FP (90 % bootstrap val)"


def load_member_views(
    members: list[EnsembleMember],
    dataframes: dict[str, pd.DataFrame],
    cache_dir: Path,
    device: torch.device,
    view_names: tuple[str, ...] = DEFAULT_TTA_VIEW_NAMES,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """{miembro: {split: {vista: P(riesgo)}}}; solo calcula lo que falte en la caché de `cache_dir`."""
    return {
        member.name: {
            split: get_member_view_probabilities(
                member, split, dataframes[split], cache_dir, device, view_names=view_names
            )
            for split in SPLIT_NAMES
        }
        for member in members
    }


def average_over_views(
    member_views: dict[str, dict[str, dict[str, np.ndarray]]],
    use_tta: bool,
) -> dict[str, dict[str, np.ndarray]]:
    """{miembro: {split: P(riesgo)}} sin TTA (vista identity) o con la media de las vistas TTA."""
    view_names = DEFAULT_TTA_VIEW_NAMES if use_tta else ("identity",)
    return {
        name: {split: average_views(views[split], view_names) for split in SPLIT_NAMES}
        for name, views in member_views.items()
    }


def average_over_members(
    probabilities: dict[str, dict[str, np.ndarray]],
    member_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """{split: media de P(riesgo) de los miembros indicados}."""
    return {
        split: np.mean([probabilities[name][split] for name in member_names], axis=0)
        for split in SPLIT_NAMES
    }


def choose_operating_thresholds(
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    validation_recall_goal: float,
    seed: int,
) -> dict[str, float]:
    """Umbral de cada regla de decisión, calculado solo con validación."""
    return {
        RECALL_TARGET_RULE: select_threshold_for_recall(
            validation_probabilities, validation_labels, validation_recall_goal
        ),
        FN_BELOW_FP_RULE: select_threshold_fn_below_fp(
            validation_probabilities, validation_labels, seed=seed
        ),
    }


def build_model_comparison_table(
    probabilities_by_model: dict[str, dict[str, np.ndarray]],
    labels_by_split: dict[str, np.ndarray],
    rule_name: str,
    validation_recall_goal: float,
    seed: int,
) -> pd.DataFrame:
    """Una fila por modelo con el umbral de `rule_name` (elegido en validación) aplicado a val y test."""
    rows = []
    for model_name, probabilities in probabilities_by_model.items():
        threshold = choose_operating_thresholds(
            probabilities["val"], labels_by_split["val"], validation_recall_goal, seed
        )[rule_name]
        val_metrics = compute_operating_point_metrics(probabilities["val"], labels_by_split["val"], threshold)
        test_metrics = compute_operating_point_metrics(probabilities["test"], labels_by_split["test"], threshold)
        (_, val_fp), (val_fn, _) = val_metrics["confusion_matrix"]
        (test_tn, test_fp), (test_fn, test_tp) = test_metrics["confusion_matrix"]
        rows.append(
            {
                "modelo": model_name,
                "umbral": round(threshold, 3),
                "AUC val": val_metrics["auc"],
                "AUC test": test_metrics["auc"],
                "val FN / FP": f"{val_fn} / {val_fp}",
                "test FN / FP": f"{test_fn} / {test_fp}",
                "test recall": test_metrics["recall"],
                "test especificidad": test_metrics["specificity"],
                "test precisión": test_metrics["precision"],
                "test accuracy": test_metrics["accuracy"],
                "test F1": test_metrics["f1"],
                "test TN/FP/FN/TP": f"{test_tn}/{test_fp}/{test_fn}/{test_tp}",
            }
        )
    return pd.DataFrame(rows).set_index("modelo")


def summary_for_json(summary: dict) -> dict:
    """Resumen de `summarize_operating_point` con la matriz de confusión aplanada (TN, FP, FN, TP)."""
    (true_negatives, false_positives), (false_negatives, true_positives) = summary["confusion_matrix"]
    flattened = {name: value for name, value in summary.items() if name != "confusion_matrix"}
    flattened["confusion_matrix_TN_FP_FN_TP"] = [
        true_negatives,
        false_positives,
        false_negatives,
        true_positives,
    ]
    return flattened
