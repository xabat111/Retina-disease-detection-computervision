"""Offline threshold calibration / ensembling on saved predictions.

Protocol
--------
* Thresholds are **always** fitted on VAL predictions (``val_predictions.npz``) and
  applied to TEST. Nothing is fitted on TEST labels.
* Runs trained before ``val_predictions.npz`` existed (v1) fall back to a 2-fold
  *cross-calibration on TEST*: thresholds fitted on fold A are applied to fold B and
  vice versa, so every test sample is scored with thresholds that never saw its
  label. This is an **estimate** and is flagged as ``mode=test_crosscal``.

Criteria
--------
``recall_floor``   max F1 s.t. recall >= floor (v1 default, floor=0.80)
``fbeta``          max F_beta (beta>1 favours recall; 1.5–2 is a softer floor)
``joint_floor``    max F1 s.t. recall >= r_floor AND precision >= p_floor; if
                   infeasible relax to recall-only, then to plain F1.
``f1``             max F1 (no constraint)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from retina_multilabel.evaluate import (  # noqa: E402
    choose_threshold_f1_with_recall_floor,
    choose_threshold_for_recall,
)
from paper_train.train_pipeline import compute_metrics  # noqa: E402

DEFAULT_THRESHOLD = 0.5


@dataclass
class Preds:
    probabilities: np.ndarray  # (n, C) all label columns
    targets: np.ndarray
    label_names: list[str]
    active_mask: np.ndarray
    source: str = ""

    @property
    def active_names(self) -> list[str]:
        return [n for n, a in zip(self.label_names, self.active_mask) if a]


@dataclass
class Criterion:
    kind: str
    recall_floor: float = 0.80
    beta: float = 1.0
    precision_floor: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        if self.kind == "recall_floor":
            return f"recall_floor_{self.recall_floor:.2f}"
        if self.kind == "fbeta":
            return f"fbeta_{self.beta:g}"
        if self.kind == "joint_floor":
            return f"joint_r{self.recall_floor:.2f}_p{self.precision_floor:.2f}"
        if self.kind == "f1":
            return "f1"
        raise ValueError(self.kind)


def load_preds(run_dir: Path, split: str) -> Preds | None:
    path = run_dir / f"{split}_predictions.npz"
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    return Preds(
        probabilities=np.asarray(z["probabilities"], dtype=np.float64),
        targets=np.asarray(z["targets"], dtype=np.float32),
        label_names=[str(x) for x in z["label_names"].tolist()],
        active_mask=np.asarray(z["active_mask"], dtype=bool),
        source=str(path),
    )


# ----------------------------------------------------------------------------- thresholds


def _pr_curve(y: np.ndarray, s: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y, s)
    return precision[:-1], recall[:-1], thresholds


def threshold_fbeta(y: np.ndarray, s: np.ndarray, beta: float) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return DEFAULT_THRESHOLD
    p, r, t = _pr_curve(y, s)
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r + 1e-12)
    best = np.nanmax(f)
    tied = np.flatnonzero(np.isclose(f, best))
    return float(t[tied[-1]])  # conservative (higher) threshold on ties


def threshold_joint_floor(
    y: np.ndarray, s: np.ndarray, recall_floor: float, precision_floor: float
) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return DEFAULT_THRESHOLD
    p, r, t = _pr_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    ok = np.flatnonzero((r >= recall_floor) & (p >= precision_floor))
    if ok.size == 0:
        # relax: recall-only (v1 behaviour), then plain F1
        ok = np.flatnonzero(r >= recall_floor)
        if ok.size == 0:
            return choose_threshold_for_recall(y, s, recall_floor)
    best = np.nanmax(f1[ok])
    tied = ok[np.isclose(f1[ok], best)]
    return float(t[tied[-1]])


def choose_threshold(y: np.ndarray, s: np.ndarray, crit: Criterion) -> float:
    if crit.kind == "recall_floor":
        return choose_threshold_f1_with_recall_floor(y, s, crit.recall_floor)
    if crit.kind == "fbeta":
        return threshold_fbeta(y, s, crit.beta)
    if crit.kind == "joint_floor":
        return threshold_joint_floor(y, s, crit.recall_floor, crit.precision_floor)
    if crit.kind == "f1":
        return threshold_fbeta(y, s, 1.0)
    raise ValueError(crit.kind)


def fit_thresholds(val: Preds, crit: Criterion) -> list[float]:
    thr: list[float] = []
    for i, active in enumerate(val.active_mask):
        if not active:
            continue
        thr.append(choose_threshold(val.targets[:, i], val.probabilities[:, i], crit))
    return thr


# ----------------------------------------------------------------------------- evaluation


def evaluate_with_thresholds(test: Preds, thresholds: list[float]) -> dict[str, Any]:
    return compute_metrics(
        test.targets,
        test.probabilities,
        test.label_names,
        test.active_mask,
        thresholds=thresholds,
    )


def _crosscal_folds(n: int, seed: int, k: int = 2) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[f::k] for f in range(k)]


def calibrate_and_evaluate(
    val: Preds | None,
    test: Preds,
    crit: Criterion,
    *,
    seed: int = 0,
) -> tuple[dict[str, Any], str, list[float] | None]:
    """Returns (test_metrics, mode, thresholds). mode in {"val", "test_crosscal"}."""
    if val is not None:
        thr = fit_thresholds(val, crit)
        return evaluate_with_thresholds(test, thr), "val", thr

    # Cross-calibration on TEST (estimate): threshold each fold with the other fold.
    folds = _crosscal_folds(len(test.targets), seed)
    thr_matrix = np.zeros_like(test.probabilities)
    for f_idx, fold in enumerate(folds):
        other = np.concatenate([folds[j] for j in range(len(folds)) if j != f_idx])
        fit = Preds(
            test.probabilities[other],
            test.targets[other],
            test.label_names,
            test.active_mask,
        )
        thr = fit_thresholds(fit, crit)
        full_thr = np.full(test.probabilities.shape[1], DEFAULT_THRESHOLD)
        full_thr[test.active_mask] = thr
        thr_matrix[fold] = full_thr
    # Per-sample thresholds → shift probabilities so a global 0.5 reproduces the decision
    shifted = np.clip(test.probabilities - thr_matrix + 0.5, 0.0, 1.0)
    shifted_preds = Preds(shifted, test.targets, test.label_names, test.active_mask)
    metrics = evaluate_with_thresholds(
        shifted_preds, [0.5] * int(test.active_mask.sum())
    )
    # Ranking metrics must come from the untouched probabilities
    raw = evaluate_with_thresholds(test, [0.5] * int(test.active_mask.sum()))
    metrics["ap_macro"], metrics["auroc_macro"] = raw["ap_macro"], raw["auroc_macro"]
    for row_m, row_r in zip(metrics["per_label"], raw["per_label"]):
        row_m["auroc"], row_m["average_precision"] = row_r["auroc"], row_r["average_precision"]
        row_m["threshold"] = float("nan")
    metrics["threshold"] = "per_sample_crosscal"
    metrics["thresholds"] = None
    return metrics, "test_crosscal", None


# ----------------------------------------------------------------------------- ensembling


def average_preds(members: list[Preds], weights: list[float] | None = None) -> Preds:
    if not members:
        raise ValueError("no members")
    ref = members[0]
    for m in members[1:]:
        if m.label_names != ref.label_names:
            raise ValueError(f"label mismatch: {m.source} vs {ref.source}")
        if m.targets.shape != ref.targets.shape or not np.array_equal(m.targets, ref.targets):
            raise ValueError(f"target order mismatch: {m.source} vs {ref.source}")
    w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    probs = sum(wi * m.probabilities for wi, m in zip(w, members))
    return Preds(
        probabilities=np.asarray(probs),
        targets=ref.targets.copy(),
        label_names=list(ref.label_names),
        active_mask=ref.active_mask.copy(),
        source="+".join(Path(m.source).parent.name for m in members),
    )


def default_criteria() -> list[Criterion]:
    return [
        Criterion("recall_floor", recall_floor=0.80),  # v1 baseline
        Criterion("recall_floor", recall_floor=0.70),
        Criterion("fbeta", beta=2.0),
        Criterion("fbeta", beta=1.5),
        Criterion("joint_floor", recall_floor=0.70, precision_floor=0.40),
        Criterion("joint_floor", recall_floor=0.75, precision_floor=0.35),
        Criterion("f1"),
    ]
