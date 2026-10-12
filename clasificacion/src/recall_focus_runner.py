"""Entrenamiento reanudable de las variantes del notebook 06 (compartido con el script de fondo)."""

from __future__ import annotations

import json

from torch.utils.data import DataLoader

from src.cached_datasets import CachedSplits
from src.hard_example_training import train_with_hard_examples
from src.recall_focus_config import (
    CHECKPOINT_DIR,
    WARM_START_CHECKPOINT,
    RecallFocusVariant,
    build_variant_train_config,
)


def train_variant_if_missing(
    variant: RecallFocusVariant,
    splits: CachedSplits,
    val_loader: DataLoader,
) -> list[dict]:
    """Entrena la variante si no existe su checkpoint + historial; devuelve el historial.

    Nunca sobrescribe: si el checkpoint existe y falta el historial, `train_with_hard_examples`
    lanza FileExistsError y hay que revisarlo a mano.
    """
    checkpoint_path = CHECKPOINT_DIR / variant.checkpoint_name
    if checkpoint_path.is_file() and variant.history_path.is_file():
        print(f"[{variant.name}] ya entrenada, se carga el historial.")
        return json.loads(variant.history_path.read_text(encoding="utf-8"))

    print(f"\n===== {variant.name}: {variant.description} =====", flush=True)
    history: list[dict] = []
    train_with_hard_examples(
        splits.train_dataset,
        val_loader,
        build_variant_train_config(variant),
        variant.hard_config,
        warm_start_checkpoint=WARM_START_CHECKPOINT,
        history=history,
    )
    variant.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
