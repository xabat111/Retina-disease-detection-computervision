"""Búsqueda de hiperparámetros con Optuna (solo train para ajustar, solo val para puntuar).

Por coste de cómputo (ResNet-152 en CPU) cada trial usa un régimen REDUCIDO:
subconjunto estratificado de train, imagen más pequeña y pocas épocas. Las imágenes se
preprocesan una sola vez y se guardan en RAM para no decodificar PNGs grandes en cada época.
El split de test no se toca en ningún punto de este módulo.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import optuna
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.config import TrainConfig
from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_split_dataframe,
    get_train_transforms,
)
from src.evaluate import evaluate_loader
from src.losses import BinaryFocalLoss
from src.models_zoo import build_model_from_config
from src.train import (
    create_optimizer,
    create_scheduler,
    get_device,
    set_seed,
    step_scheduler,
    train_one_epoch,
)

# Los crops de train se hacen sobre una copia con el lado corto = 1.5 x image_size
TRAIN_CACHE_SCALE = 1.5

BASELINE_PARAMS = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
    "batch_size": 16,
    "unfreeze_layer3": False,
    "lr_scheduler": "none",
}


class InMemoryImageDataset(Dataset):
    """Imágenes ya reescaladas (PIL) en RAM + etiqueta; aplica `transform` al vuelo."""

    def __init__(self, images: list[Image.Image], labels: list[int], transform) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        return image, torch.tensor(float(self.labels[index]), dtype=torch.float32)


def _load_resized_images(
    image_paths: list[str],
    resize_fn: Callable[[Image.Image], Image.Image],
    description: str,
) -> list[Image.Image]:
    images = []
    for image_path in tqdm(image_paths, desc=description, leave=False):
        with Image.open(image_path) as raw_image:
            images.append(resize_fn(raw_image.convert("RGB")))
    return images


def _resize_keeping_aspect(image: Image.Image, short_side: int) -> Image.Image:
    width, height = image.size
    scale = short_side / min(width, height)
    return image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)


@dataclass
class TuningData:
    """Datasets en RAM compartidos por todos los trials."""

    train_dataset: InMemoryImageDataset
    val_dataset: InMemoryImageDataset
    image_size: int

    def make_loaders(self, batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, val_loader


def build_tuning_data(
    config: TrainConfig,
    image_size: int,
    train_fraction: float,
    val_fraction: float = 1.0,
) -> TuningData:
    """Prepara train (subconjunto estratificado) y val en RAM (completo por defecto). No lee test.

    `val_fraction` < 1 solo se usa en pruebas rápidas (smoke test); el tuning real usa val completo.
    """
    train_dataframe = build_split_dataframe(config.train_labels_csv, config.train_images_dir)
    val_dataframe = build_split_dataframe(config.val_labels_csv, config.val_images_dir)

    if train_fraction < 1.0:
        train_dataframe, _ = train_test_split(
            train_dataframe,
            train_size=train_fraction,
            stratify=train_dataframe["disease_risk"],
            random_state=config.seed,
        )
    train_dataframe = train_dataframe.reset_index(drop=True)
    if val_fraction < 1.0:
        val_dataframe, _ = train_test_split(
            val_dataframe,
            train_size=val_fraction,
            stratify=val_dataframe["disease_risk"],
            random_state=config.seed,
        )
    val_dataframe = val_dataframe.reset_index(drop=True)

    train_images = _load_resized_images(
        train_dataframe["image_path"].tolist(),
        lambda image: _resize_keeping_aspect(image, int(TRAIN_CACHE_SCALE * image_size)),
        "cache train",
    )
    # Misma transformación determinista que get_eval_transforms, aplicada una sola vez
    val_images = _load_resized_images(
        val_dataframe["image_path"].tolist(),
        lambda image: image.resize((image_size, image_size), Image.BILINEAR),
        "cache val",
    )

    eval_tensor_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    return TuningData(
        train_dataset=InMemoryImageDataset(
            train_images,
            train_dataframe["disease_risk"].tolist(),
            get_train_transforms(image_size),
        ),
        val_dataset=InMemoryImageDataset(
            val_images,
            val_dataframe["disease_risk"].tolist(),
            eval_tensor_transform,
        ),
        image_size=image_size,
    )


def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda (justificado en el notebook 03)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
        "focal_alpha": trial.suggest_float("focal_alpha", 0.15, 0.75),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "unfreeze_layer3": trial.suggest_categorical("unfreeze_layer3", [False, True]),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["none", "cosine", "plateau"]),
    }


def apply_hyperparameters(config: TrainConfig, params: dict) -> TrainConfig:
    """Copia de `config` con los hiperparámetros dados (sin mutar el original)."""
    return replace(
        config,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        focal_gamma=params["focal_gamma"],
        focal_alpha=params["focal_alpha"],
        batch_size=params["batch_size"],
        unfreeze_layer3=params["unfreeze_layer3"],
        lr_scheduler=params["lr_scheduler"],
    )


def suggest_zoo_hyperparameters(trial: optuna.Trial, max_unfrozen_blocks: int) -> dict:
    """Mismo espacio que `suggest_hyperparameters`, pero con `unfrozen_blocks` (1..N) en vez de layer3."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
        "focal_alpha": trial.suggest_float("focal_alpha", 0.15, 0.75),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "unfrozen_blocks": trial.suggest_int("unfrozen_blocks", 1, max_unfrozen_blocks),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["none", "cosine", "plateau"]),
    }


def apply_zoo_hyperparameters(config: TrainConfig, params: dict) -> TrainConfig:
    """Equivalente a `apply_hyperparameters` para las arquitecturas de `src/models_zoo.py`."""
    return replace(
        config,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        focal_gamma=params["focal_gamma"],
        focal_alpha=params["focal_alpha"],
        batch_size=params["batch_size"],
        unfrozen_blocks=params["unfrozen_blocks"],
        lr_scheduler=params["lr_scheduler"],
    )


# Primer trial de cada estudio del zoo: configuración "baseline" con solo el último grupo entrenable
ZOO_BASELINE_PARAMS = {
    **{name: value for name, value in BASELINE_PARAMS.items() if name != "unfreeze_layer3"},
    "unfrozen_blocks": 1,
}


def cache_eval_dataset(dataframe: pd.DataFrame, image_size: int) -> InMemoryImageDataset:
    """Dataset determinista (resize cuadrado, sin augmentation) en RAM para val/test.

    Equivale a `get_eval_transforms`, pero decodifica cada PNG una sola vez. Quien lo llame
    decide qué split pasa; este módulo nunca lo invoca con test.
    """
    images = _load_resized_images(
        dataframe["image_path"].tolist(),
        lambda image: image.resize((image_size, image_size), Image.BILINEAR),
        "cache eval",
    )
    eval_tensor_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    return InMemoryImageDataset(
        images, dataframe["disease_risk"].tolist(), eval_tensor_transform
    )


def create_objective(
    tuning_data: TuningData,
    base_config: TrainConfig,
    epochs_per_trial: int,
    suggest_fn: Callable[[optuna.Trial], dict] = suggest_hyperparameters,
    apply_fn: Callable[[TrainConfig, dict], TrainConfig] = apply_hyperparameters,
) -> Callable[[optuna.Trial], float]:
    """Devuelve la función objetivo: mejor AUC de VALIDACIÓN alcanzado en el trial.

    `suggest_fn` / `apply_fn` permiten reutilizarla con otras arquitecturas; la arquitectura
    la fija `base_config.backbone_name`. Por defecto reproduce la búsqueda del ResNet-152.
    """
    device = get_device()

    def objective(trial: optuna.Trial) -> float:
        config = apply_fn(base_config, suggest_fn(trial))
        set_seed(config.seed)
        train_loader, val_loader = tuning_data.make_loaders(
            config.batch_size, config.num_workers
        )

        model = build_model_from_config(config).to(device)
        criterion = BinaryFocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config, total_epochs=epochs_per_trial)

        best_validation_auc = -float("inf")
        for epoch in range(1, epochs_per_trial + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_loader(model, val_loader, device, criterion=criterion)
            validation_auc = float(val_metrics["auc"])
            step_scheduler(scheduler, validation_auc)

            best_validation_auc = max(best_validation_auc, validation_auc)
            print(
                f"[trial {trial.number}] epoch {epoch}/{epochs_per_trial} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"val_auc={validation_auc:.4f}"
            )

            trial.report(validation_auc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_validation_auc

    return objective


def build_final_config(base_config: TrainConfig, best_params: dict, checkpoint_name: str) -> TrainConfig:
    """Config de entrenamiento completo con los mejores hiperparámetros y checkpoint propio."""
    tuned_config = apply_hyperparameters(base_config, best_params)
    return replace(tuned_config, best_checkpoint_name=checkpoint_name)


def count_finished_trials(study: optuna.Study) -> int:
    """Trials ya terminados (completos o podados); sirve para reanudar el estudio."""
    finished_states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(trial.state in finished_states for trial in study.trials)


def trials_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Tabla resumen de trials (valor, estado, duración y parámetros)."""
    table = study.trials_dataframe(attrs=("number", "value", "state", "duration", "params"))
    return table.rename(columns=lambda name: name.replace("params_", ""))
