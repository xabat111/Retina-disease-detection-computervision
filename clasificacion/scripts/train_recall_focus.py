"""Fine-tuning centrado en recall del ResNet-152 (diseño 2x2 del notebook 06), en segundo plano.

Entrena cada variante de `src/recall_focus_config.VARIANTS` partiendo del mejor ResNet-152
individual y guarda checkpoint + historial. Lo ya entrenado se omite (reanudable) y NUNCA se
sobrescribe un checkpoint. Tarda ~1-1.3 h por variante en CPU (4 épocas, ~17-20 min/época).

Uso:  python -u scripts/train_recall_focus.py > artifacts/logs/recall_focus_train.out 2>&1
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
from src.recall_focus_config import VARIANTS, build_variant_train_config  # noqa: E402
from src.recall_focus_runner import train_variant_if_missing  # noqa: E402

TRAINING_ORDER = ("control", "both", "reweight", "recallloss")
EVALUATION_BATCH_SIZE = 32


def main() -> None:
    splits = build_cached_letterbox_splits(build_variant_train_config(VARIANTS[TRAINING_ORDER[0]]))
    val_loader = DataLoader(splits.val_dataset, batch_size=EVALUATION_BATCH_SIZE, shuffle=False)
    for variant_name in TRAINING_ORDER:
        train_variant_if_missing(VARIANTS[variant_name], splits, val_loader)


if __name__ == "__main__":
    main()
