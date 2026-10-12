"""Medición del tiempo por época en CPU, para dimensionar el nº de épocas antes de entrenar."""

from __future__ import annotations

import time
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.losses import BinaryFocalLoss
from src.models_zoo import build_model_from_config
from src.train import create_optimizer, get_device, set_seed
from src.tuning import InMemoryImageDataset

WARMUP_STEPS = 1
MEASURED_STEPS = 3
EVALUATION_BATCH_SIZE = 32


def estimate_epoch_minutes(
    config: TrainConfig,
    train_dataset: InMemoryImageDataset,
    n_validation_images: int,
) -> dict[str, float]:
    """Minutos estimados de una época (train completo + validación) midiendo unos pocos pasos.

    Se mide un paso de entrenamiento (forward + backward + optimizador) con el batch real de
    `config.batch_size` y la inferencia por imagen en modo eval; luego se escala al tamaño del
    train y de la validación. Los pesos son aleatorios (`pretrained=False`): el tiempo no depende
    de ellos.
    """
    set_seed(config.seed)
    device = get_device()
    model = build_model_from_config(replace(config, pretrained=False)).to(device)
    criterion = BinaryFocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    optimizer = create_optimizer(model, config)

    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    model.train()
    for step in range(WARMUP_STEPS + MEASURED_STEPS):
        if step == WARMUP_STEPS:
            start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        criterion(model(images), labels).backward()
        optimizer.step()
    seconds_per_train_step = (time.time() - start_time) / MEASURED_STEPS

    model.eval()
    evaluation_images = images.repeat(EVALUATION_BATCH_SIZE // len(images) + 1, 1, 1, 1)[:EVALUATION_BATCH_SIZE]
    with torch.no_grad():
        model(evaluation_images)  # calentamiento
        start_time = time.time()
        model(evaluation_images)
    seconds_per_evaluated_image = (time.time() - start_time) / EVALUATION_BATCH_SIZE

    train_steps_per_epoch = len(train_dataset) / config.batch_size
    return {
        "s/paso de train": seconds_per_train_step,
        "ms/imagen (eval)": 1000 * seconds_per_evaluated_image,
        "min/época": (
            train_steps_per_epoch * seconds_per_train_step
            + n_validation_images * seconds_per_evaluated_image
        )
        / 60,
    }
