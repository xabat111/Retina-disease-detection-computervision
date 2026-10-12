"""Entrena ResNet-50, DenseNet-121, EfficientNet-B0 y ConvNeXt-Tiny SIN Optuna (notebook 07), en segundo plano.

Sin recorte (letterbox 224), con la receta centrada en recall del notebook 06. Reanudable: lo ya
entrenado se omite y nunca se sobrescribe un checkpoint.

Uso:  python -u scripts/train_zoo_recall.py > artifacts/logs/zoo_recall_train.out 2>&1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
# Al redirigir la salida a un fichero, Windows usaría cp1252 y fallaría con caracteres como "→"
sys.stdout.reconfigure(encoding="utf-8")

from torch.utils.data import DataLoader  # noqa: E402

from src.cached_datasets import build_cached_letterbox_splits  # noqa: E402
from src.zoo_recall_config import ARCHITECTURES, build_zoo_train_config  # noqa: E402
from src.zoo_recall_runner import train_architecture_if_missing  # noqa: E402

EVALUATION_BATCH_SIZE = 32


def main() -> None:
    splits = build_cached_letterbox_splits(build_zoo_train_config(ARCHITECTURES[0]))
    val_loader = DataLoader(splits.val_dataset, batch_size=EVALUATION_BATCH_SIZE, shuffle=False)
    for architecture in ARCHITECTURES:
        train_architecture_if_missing(architecture, splits, val_loader)


if __name__ == "__main__":
    main()
