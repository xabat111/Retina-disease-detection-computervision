"""Entrenamiento reanudable de los modelos del notebook 07 (compartido con el script de fondo)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.cached_datasets import CachedSplits
from src.hard_example_training import train_with_hard_examples
from src.zoo_recall_config import (
    build_zoo_train_config,
    history_path,
    recall_recipe,
)


def history_from_checkpoint_metadata(checkpoint_path: Path) -> list[dict]:
    """Una fila de historial a partir de los metadatos del mejor checkpoint.

    Sirve cuando el `.pt` existe pero se perdió el JSON de entrenamiento (p. ej. traslado
    de carpetas a mitad de un run).
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    val_metrics = metadata.get("val_metrics") or {}
    if not val_metrics:
        return []
    return [
        {
            "epoch": int(metadata.get("epoch", 1)),
            "train_loss": float(metadata.get("train_loss", float("nan"))),
            "val_loss": float(val_metrics.get("loss", float("nan"))),
            "val_auc": float(val_metrics.get("auc", float("nan"))),
            "val_f1": float(val_metrics.get("f1", float("nan"))),
            "val_accuracy": float(val_metrics.get("accuracy", float("nan"))),
            "val_best_f2": float(val_metrics.get("best_f2", float("nan"))),
            "val_recall_at_spec_85": float(val_metrics.get("recall_at_spec_85", float("nan"))),
            "val_spec_at_fn_lt_fp": float(val_metrics.get("spec_at_fn_lt_fp", float("nan"))),
            "learning_rate": float("nan"),
            "epoch_seconds": float("nan"),
            "n_hard_positives": 0,
            "n_hard_negatives": 0,
            "mean_positive_weight": float("nan"),
            "mean_negative_weight": float("nan"),
            "capped_fraction": float("nan"),
            "from_checkpoint_metadata": True,
        }
    ]


def train_architecture_if_missing(
    architecture: str,
    splits: CachedSplits,
    val_loader: DataLoader,
) -> list[dict]:
    """Entrena la arquitectura solo si falta su checkpoint; nunca sobrescribe uno existente.

    Si el checkpoint ya existe, se omite el entrenamiento y se devuelve el historial
    guardado. Si falta el JSON pero el `.pt` trae metadatos de validación, se reconstruye
    una fila mínima y se persiste el JSON para las siguientes ejecuciones.
    """
    config = build_zoo_train_config(architecture)
    checkpoint_path = config.checkpoint_dir / config.best_checkpoint_name
    saved_history_path = history_path(architecture)

    if checkpoint_path.is_file():
        print(f"Checkpoint ya existe, se omite entrenamiento: {checkpoint_path}")
        if saved_history_path.is_file():
            return json.loads(saved_history_path.read_text(encoding="utf-8"))
        history = history_from_checkpoint_metadata(checkpoint_path)
        if history:
            saved_history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"Historial reconstruido desde metadatos del checkpoint: {saved_history_path}")
        else:
            print(f"AVISO: sin historial ni metadatos útiles para {architecture}")
        return history

    print(f"\n===== {architecture} =====", flush=True)
    _, hard_config = recall_recipe()
    history: list[dict] = []
    train_with_hard_examples(
        splits.train_dataset,
        val_loader,
        config,
        hard_config,
        warm_start_checkpoint=None,
        history=history,
    )
    saved_history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
