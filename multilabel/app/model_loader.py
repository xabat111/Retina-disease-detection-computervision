"""Carga del ensemble odir7 (app_v2/ensemble.py) y metricas de test por clase (sin Streamlit)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import settings


@dataclass
class DeployedModel:
    ensemble: object
    device: torch.device
    label_names: list[str]
    thresholds: np.ndarray
    test_metrics: pd.DataFrame
    unreliable_labels: set[str]

    @property
    def image_size(self) -> int:
        return settings.DISPLAY_IMAGE_SIZE


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def compute_test_metrics(manifest: dict, thresholds: np.ndarray) -> pd.DataFrame:
    """Precision/recall/AUROC por clase del ensemble en TEST (npz guardados + clustered_test.csv)."""
    labels = pd.read_csv(settings.TEST_LABELS_CSV)
    prob = np.mean([np.load(settings.TEST_PREDICTIONS_DIR / m["run"] / "test_predictions.npz")["probabilities"]
                    for m in manifest["members"]], 0)
    rows = {}
    for j, label in enumerate(settings.LABEL_NAMES):
        y = labels[label].to_numpy().astype(int); pred = prob[:, j] >= thresholds[j]
        tp = int((pred & (y == 1)).sum())
        rows[label] = dict(precision=tp / max(pred.sum(), 1), recall=tp / max(y.sum(), 1),
                           auroc=_auroc(y, prob[:, j]), n_positivos=int(y.sum()))
    return pd.DataFrame.from_dict(rows, orient="index")


def load_deployed_model() -> DeployedModel:
    if not settings.MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {settings.MANIFEST_PATH}")
    from app.ensemble import MANIFEST, THRESHOLDS, Ensemble  # backend unico, sin duplicar codigo

    if list(MANIFEST["labels"]) != settings.LABEL_NAMES:
        raise ValueError("Label order in manifest differs from settings.LABEL_NAMES")
    thresholds = np.asarray(THRESHOLDS, dtype=np.float64)
    metrics = compute_test_metrics(MANIFEST, thresholds)
    unreliable = set(metrics.index[metrics["n_positivos"] < settings.UNRELIABLE_MIN_TEST_POSITIVES])
    return DeployedModel(ensemble=Ensemble(), device=torch.device("cpu"), label_names=list(settings.LABEL_NAMES),
                         thresholds=thresholds, test_metrics=metrics, unreliable_labels=unreliable)
