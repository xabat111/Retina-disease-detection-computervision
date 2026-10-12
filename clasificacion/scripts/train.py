"""Entrypoint CLI: entrenamiento (+ evaluación opcional en test).

Uso desde la raíz del proyecto:
    python scripts/train.py
    python scripts/train.py --epochs 5 --batch-size 8
    python scripts/train.py --eval-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Permite `python scripts/train.py` sin instalar el paquete
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cache local (en este entorno ~/.cache no es un directorio usable)
_torch_home = PROJECT_ROOT / "artifacts" / "torch_cache"
_torch_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(_torch_home))

from src.config import TrainConfig, default_config
from src.data import PREPROCESSING_OPTIONS, create_dataloaders
from src.evaluate import evaluate_loader, format_metrics
from src.train import (
    SELECTION_METRICS,
    get_device,
    load_best_checkpoint_for_eval,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena ResNet-152 para Disease_Risk (RFMiD)."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_OPTIONS,
        default=None,
        help="Preprocesado espacial (por defecto el de TrainConfig: random_crop).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=sorted(SELECTION_METRICS),
        default=None,
        help="Métrica de validación para el mejor checkpoint y el early stopping "
        "(por defecto la de TrainConfig: auc).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Solo evaluar el mejor checkpoint en val/test (sin entrenar).",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="No evaluar el split de test al final.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.image_size is not None:
        config.image_size = args.image_size
    if args.seed is not None:
        config.seed = args.seed
    if args.preprocessing is not None:
        config.preprocessing = args.preprocessing
    if args.selection_metric is not None:
        config.selection_metric = args.selection_metric
    return config


def print_class_balance(split_name: str, dataframe) -> None:
    counts = dataframe["disease_risk"].value_counts().sort_index()
    proportions = dataframe["disease_risk"].value_counts(normalize=True).sort_index()
    print(f"{split_name}: n={len(dataframe)}")
    print(f"  conteos: {counts.to_dict()}")
    print(f"  proporciones: {proportions.round(3).to_dict()}")


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(default_config(), args)

    print("Cargando datos RFMiD...")
    (
        train_loader,
        val_loader,
        test_loader,
        train_dataframe,
        val_dataframe,
        test_dataframe,
    ) = create_dataloaders(config)

    print_class_balance("train", train_dataframe)
    print_class_balance("val", val_dataframe)
    print_class_balance("test", test_dataframe)

    device = get_device()

    if args.eval_only:
        print("Modo eval-only: cargando mejor checkpoint...")
        model = load_best_checkpoint_for_eval(config, device)
    else:
        print("Iniciando entrenamiento...")
        train_model(train_loader, val_loader, config)
        model = load_best_checkpoint_for_eval(config, device)

    val_metrics = evaluate_loader(model, val_loader, device)
    print(format_metrics(val_metrics, "val"))

    if not args.skip_test:
        test_metrics = evaluate_loader(model, test_loader, device)
        print(format_metrics(test_metrics, "test"))


if __name__ == "__main__":
    main()
