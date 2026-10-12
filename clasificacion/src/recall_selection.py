"""Métricas de validación orientadas a recall, independientes del umbral, para elegir checkpoints.

Un criterio como AUC mide el ranking global, pero el objetivo clínico vive en una zona concreta
de la curva ROC (alto recall con la menor pérdida de especificidad). Estas métricas se calculan
recorriendo TODOS los umbrales posibles sobre las probabilidades de validación, de modo que el
checkpoint elegido no depende de un umbral fijo (0.5 está mal calibrado cuando la loss pesa
más a los positivos).

Se calculan SOLO con validación; aquí no hay nada que use test.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

# Especificidad mínima exigida para "recall a especificidad fija"
SPECIFICITY_FLOOR = 0.85
# F-beta con beta=2 pondera el recall 4 veces más que la precisión
F_BETA = 2.0
# El AUC solo desempata entre épocas con la misma especificidad (diferencias de 1/134 son ruido)
AUC_TIE_BREAK_WEIGHT = 1e-3

RECALL_SELECTION_METRIC_NAMES = ("best_f2", "recall_at_spec_85", "spec_at_fn_lt_fp")


def operating_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conteos (TP, FP, FN, TN) para cada umbral distinto.

    El punto k-ésimo corresponde a predecir positivas las k probabilidades más altas
    (k = 0 ... n). Las probabilidades empatadas comparten umbral, así que solo se conservan
    los cortes entre valores distintos.
    """
    labels_int = labels.astype(int)
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels_int[order]

    true_positives = np.concatenate([[0], np.cumsum(sorted_labels)])
    false_positives = np.concatenate([[0], np.cumsum(1 - sorted_labels)])
    is_valid_cut = np.concatenate(
        [[True], sorted_probabilities[1:] != sorted_probabilities[:-1], [True]]
    )

    n_positive = int(labels_int.sum())
    n_negative = len(labels_int) - n_positive
    true_positives = true_positives[is_valid_cut]
    false_positives = false_positives[is_valid_cut]
    false_negatives = n_positive - true_positives
    true_negatives = n_negative - false_positives
    return true_positives, false_positives, false_negatives, true_negatives


def compute_recall_selection_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Tres criterios de selección orientados a recall (mayor es mejor en los tres).

    - best_f2: mejor F2 sobre todos los umbrales (recall pesa más que precisión).
    - recall_at_spec_85: mayor recall con especificidad >= 0.85.
    - spec_at_fn_lt_fp: mayor especificidad entre los umbrales con FN < FP, que es el objetivo
      del proyecto (menos falsos negativos que falsos positivos) medido en su coste: cuánta
      especificidad hay que sacrificar para conseguirlo. Se desempata con el AUC.
    """
    labels_int = labels.astype(int)
    if len(np.unique(labels_int)) < 2:
        return {name: float("nan") for name in RECALL_SELECTION_METRIC_NAMES}

    true_positives, false_positives, false_negatives, true_negatives = operating_curve(
        probabilities, labels_int
    )
    n_positive = int(labels_int.sum())
    n_negative = len(labels_int) - n_positive
    recall = true_positives / n_positive
    specificity = true_negatives / n_negative

    beta_squared = F_BETA**2
    f_beta = (1 + beta_squared) * true_positives / np.maximum(
        (1 + beta_squared) * true_positives + beta_squared * false_negatives + false_positives, 1
    )

    meets_specificity = specificity >= SPECIFICITY_FLOOR
    recall_at_specificity = float(recall[meets_specificity].max()) if meets_specificity.any() else 0.0

    # Siempre existe el umbral que predice todo positivo (FN = 0 < FP = n_negative)
    fewer_fn_than_fp = false_negatives < false_positives
    specificity_at_fn_lt_fp = float(specificity[fewer_fn_than_fp].max())
    auc = float(roc_auc_score(labels_int, probabilities))

    return {
        "best_f2": float(f_beta.max()),
        "recall_at_spec_85": recall_at_specificity,
        "spec_at_fn_lt_fp": specificity_at_fn_lt_fp + AUC_TIE_BREAK_WEIGHT * auc,
    }
