"""Calcula (y guarda en caché) las probabilidades TTA de cada miembro en val y test.

Es la parte cara del notebook 05 en CPU; correrla aparte permite dejarla en segundo plano:
    python scripts/precompute_recall_predictions.py
Si la caché ya existe para un miembro/split/vista, no se recalcula.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))

from src.config import default_config
from src.data import build_split_dataframe
from src.recall_config import PREDICTION_CACHE_DIR, existing_checkpoint_members
from src.train import get_device
from src.tta_inference import get_member_view_probabilities


def main() -> None:
    config = default_config()
    split_dataframes = {
        "val": build_split_dataframe(config.val_labels_csv, config.val_images_dir),
        "test": build_split_dataframe(config.test_labels_csv, config.test_images_dir),
    }
    device = get_device()
    for member in existing_checkpoint_members():
        for split_name, dataframe in split_dataframes.items():
            get_member_view_probabilities(
                member, split_name, dataframe, PREDICTION_CACHE_DIR, device
            )
    print("Predicciones TTA en caché:", PREDICTION_CACHE_DIR, flush=True)


if __name__ == "__main__":
    main()
