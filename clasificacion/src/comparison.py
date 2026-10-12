"""Orquestación de la comparación de arquitecturas: Optuna, entrenamiento final y evaluación.

Reglas anti-fuga:
- Optuna puntúa SOLO con validación (`create_objective`).
- El umbral de decisión se elige SOLO con validación (Youden) y se aplica a test sin cambios.
- Test se usa únicamente para informar métricas y para elegir los ejemplos de Grad-CAM.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, TrainConfig
from src.evaluate import collect_predictions, compute_classification_metrics, select_threshold_youden
from src.model import count_trainable_parameters
from src.models_zoo import (
    config_for_architecture,
    get_architecture_spec,
    load_architecture_checkpoint,
    max_unfrozen_blocks,
)
from src.train import train_model
from src.tuning import (
    ZOO_BASELINE_PARAMS,
    TuningData,
    apply_zoo_hyperparameters,
    count_finished_trials,
    create_objective,
    suggest_zoo_hyperparameters,
)

EVALUATION_BATCH_SIZE = 32


@dataclass(frozen=True)
class ArtifactPaths:
    """Carpetas de salida; el smoke test usa una raíz distinta para no pisar artefactos reales."""

    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def optuna(self) -> Path:
        return self.root / "optuna"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def gradcam(self) -> Path:
        return self.root / "gradcam"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    def create_directories(self) -> None:
        for directory in (self.checkpoints, self.optuna, self.logs, self.gradcam, self.figures):
            directory.mkdir(parents=True, exist_ok=True)


def subsample_dataframe(dataframe: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Submuestreo estratificado (solo para el smoke test)."""
    if fraction >= 1.0:
        return dataframe
    subset, _ = train_test_split(
        dataframe, train_size=fraction, stratify=dataframe["disease_risk"], random_state=seed
    )
    return subset.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Optuna
# --------------------------------------------------------------------------------------
def run_architecture_study(
    architecture: str,
    tuning_data: TuningData,
    base_config: TrainConfig,
    n_trials: int,
    epochs_per_trial: int,
    optuna_dir: Path,
    timeout_seconds: float | None = None,
) -> optuna.Study:
    """Estudio Optuna propio de la arquitectura (SQLite reanudable) maximizando el AUC de val."""
    optuna_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{(optuna_dir / f'compare_{architecture}.db').as_posix()}"
    study = optuna.create_study(
        study_name=f"compare_{architecture}",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(ZOO_BASELINE_PARAMS)  # trial 0 = referencia en el mismo régimen

    remaining_trials = n_trials - count_finished_trials(study)
    print(
        f"[{architecture}] trials terminados: {count_finished_trials(study)} | "
        f"pendientes: {max(remaining_trials, 0)}"
    )
    if remaining_trials > 0:
        objective = create_objective(
            tuning_data,
            config_for_architecture(base_config, architecture),
            epochs_per_trial,
            suggest_fn=partial(
                suggest_zoo_hyperparameters, max_unfrozen_blocks=max_unfrozen_blocks(architecture)
            ),
            apply_fn=apply_zoo_hyperparameters,
        )
        study.optimize(objective, n_trials=remaining_trials, timeout=timeout_seconds)
    return study


def save_best_params(
    architecture: str,
    study: optuna.Study,
    optuna_dir: Path,
    search_regime: dict,
) -> dict:
    """Guarda `compare_<modelo>_best_params.json` y devuelve el diccionario de mejores parámetros."""
    payload = {
        "architecture": architecture,
        "best_params": study.best_params,
        "best_val_auc_reduced_regime": study.best_value,
        "best_trial_number": study.best_trial.number,
        "search_regime": search_regime,
    }
    (optuna_dir / f"compare_{architecture}_best_params.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return study.best_params


# --------------------------------------------------------------------------------------
# Entrenamiento final
# --------------------------------------------------------------------------------------
def train_architecture(
    architecture: str,
    best_params: dict,
    final_data: TuningData,
    base_config: TrainConfig,
    final_epochs: int,
    early_stopping_patience: int,
    paths: ArtifactPaths,
    force_retrain: bool = False,
) -> list[dict]:
    """Reentrena con los mejores parámetros; checkpoint y curvas por AUC de val. Reanudable."""
    checkpoint_name = f"compare_{architecture}.pt"
    history_path = paths.logs / f"compare_{architecture}_history.json"
    paths.create_directories()

    if not force_retrain and (paths.checkpoints / checkpoint_name).is_file() and history_path.is_file():
        print(f"[{architecture}] checkpoint existente, se omite el entrenamiento.")
        return json.loads(history_path.read_text(encoding="utf-8"))

    final_config = apply_zoo_hyperparameters(
        config_for_architecture(
            base_config,
            architecture,
            epochs=final_epochs,
            early_stopping_patience=early_stopping_patience,
            image_size=final_data.image_size,
            checkpoint_dir=paths.checkpoints,
            best_checkpoint_name=checkpoint_name,
        ),
        best_params,
    )
    train_loader, val_loader = final_data.make_loaders(final_config.batch_size, final_config.num_workers)

    history: list[dict] = []
    train_model(train_loader, val_loader, final_config, history=history)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


# --------------------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------------------
@dataclass
class ModelEntry:
    """Un modelo a evaluar: arquitectura + checkpoint + política de congelación usada."""

    key: str
    display_name: str
    architecture: str
    checkpoint_path: Path
    unfrozen_blocks: int
    history_path: Path | None = None


@dataclass
class EvaluatedModel:
    """Predicciones por split y umbral elegido en validación."""

    entry: ModelEntry
    val_probabilities: np.ndarray
    val_labels: np.ndarray
    test_probabilities: np.ndarray
    test_labels: np.ndarray
    threshold: float
    trainable_parameters: int
    total_parameters: int
    seconds_per_epoch: float
    history: list[dict] = field(default_factory=list)


def make_eval_loader(dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=EVALUATION_BATCH_SIZE, shuffle=False, num_workers=0)


def _load_history(history_path: Path | None) -> list[dict]:
    if history_path is None or not history_path.is_file():
        return []
    return json.loads(history_path.read_text(encoding="utf-8"))


def _mean_epoch_seconds(history: list[dict]) -> float:
    seconds = [epoch["epoch_seconds"] for epoch in history if "epoch_seconds" in epoch]
    return float(np.mean(seconds)) if seconds else math.nan


def evaluate_entry(
    entry: ModelEntry,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> EvaluatedModel:
    """Carga el checkpoint, predice val y test, y fija el umbral de Youden SOLO con val."""
    model = load_architecture_checkpoint(
        entry.architecture, entry.checkpoint_path, device, unfrozen_blocks=entry.unfrozen_blocks
    )
    val_probabilities, val_labels, _ = collect_predictions(model, val_loader, device)
    test_probabilities, test_labels, _ = collect_predictions(model, test_loader, device)
    trainable, total = count_trainable_parameters(model)
    history = _load_history(entry.history_path)
    return EvaluatedModel(
        entry=entry,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        test_probabilities=test_probabilities,
        test_labels=test_labels,
        threshold=select_threshold_youden(val_probabilities, val_labels),
        trainable_parameters=trainable,
        total_parameters=total,
        seconds_per_epoch=_mean_epoch_seconds(history),
        history=history,
    )


def find_resnet152_references(reference_root: Path | None = None) -> list[ModelEntry]:
    """ResNet-152 ya entrenado (baseline y, si existe, el ajustado en 02.1); no se reentrena."""
    root = reference_root or PROJECT_ROOT / "artifacts"
    entries: list[ModelEntry] = []

    baseline_checkpoint = root / "checkpoints" / "best_resnet152_disease_risk.pt"
    if baseline_checkpoint.is_file():
        entries.append(
            ModelEntry(
                key="resnet152_baseline",
                display_name="ResNet-152 (baseline)",
                architecture="resnet152",
                checkpoint_path=baseline_checkpoint,
                unfrozen_blocks=1,
            )
        )

    tuned_checkpoint = root / "checkpoints" / "best_resnet152_disease_risk_optuna.pt"
    best_params_path = root / "optuna" / "best_params.json"
    if tuned_checkpoint.is_file() and best_params_path.is_file():
        tuned_params = json.loads(best_params_path.read_text(encoding="utf-8"))["best_params"]
        entries.append(
            ModelEntry(
                key="resnet152_optuna",
                display_name="ResNet-152 (Optuna)",
                architecture="resnet152",
                checkpoint_path=tuned_checkpoint,
                unfrozen_blocks=2 if tuned_params.get("unfreeze_layer3") else 1,
                history_path=root / "logs" / "tuned_history.json",
            )
        )
    return entries


def build_new_model_entries(architectures: list[str], paths: ArtifactPaths) -> list[ModelEntry]:
    """Entradas de los modelos entrenados por este notebook (usa su best_params.json)."""
    entries = []
    for architecture in architectures:
        best_params_path = paths.optuna / f"compare_{architecture}_best_params.json"
        best_params = json.loads(best_params_path.read_text(encoding="utf-8"))["best_params"]
        entries.append(
            ModelEntry(
                key=architecture,
                display_name=get_architecture_spec(architecture).display_name,
                architecture=architecture,
                checkpoint_path=paths.checkpoints / f"compare_{architecture}.pt",
                unfrozen_blocks=best_params["unfrozen_blocks"],
                history_path=paths.logs / f"compare_{architecture}_history.json",
            )
        )
    return entries


def summarize_split(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """AUC, F1, accuracy, precisión, recall, especificidad y matriz de confusión a un umbral."""
    metrics = compute_classification_metrics(probabilities, labels, threshold=threshold)
    (true_negatives, false_positives), (false_negatives, true_positives) = metrics["confusion_matrix"]

    def safe_ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "auc": metrics["auc"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "precision": safe_ratio(true_positives, true_positives + false_positives),
        "recall": safe_ratio(true_positives, true_positives + false_negatives),
        "specificity": safe_ratio(true_negatives, true_negatives + false_positives),
        "tn": true_negatives,
        "fp": false_positives,
        "fn": false_negatives,
        "tp": true_positives,
    }


def build_comparison_table(evaluated_models: list[EvaluatedModel]) -> pd.DataFrame:
    """Tabla val + test por modelo (umbral de val), ordenada por AUC de test."""
    rows = []
    for evaluated in evaluated_models:
        val_summary = summarize_split(evaluated.val_probabilities, evaluated.val_labels, evaluated.threshold)
        test_summary = summarize_split(evaluated.test_probabilities, evaluated.test_labels, evaluated.threshold)
        rows.append(
            {
                "modelo": evaluated.entry.display_name,
                "umbral (val, Youden)": evaluated.threshold,
                "AUC val": val_summary["auc"],
                "AUC test": test_summary["auc"],
                "F1 test": test_summary["f1"],
                "Accuracy test": test_summary["accuracy"],
                "Precision test": test_summary["precision"],
                "Recall test": test_summary["recall"],
                "Especificidad test": test_summary["specificity"],
                "TN": test_summary["tn"],
                "FP": test_summary["fp"],
                "FN": test_summary["fn"],
                "TP": test_summary["tp"],
                "params entrenables [M]": evaluated.trainable_parameters / 1e6,
                "params totales [M]": evaluated.total_parameters / 1e6,
                "s/época": evaluated.seconds_per_epoch,
            }
        )
    return pd.DataFrame(rows).sort_values("AUC test", ascending=False).reset_index(drop=True)
