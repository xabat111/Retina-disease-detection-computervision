"""Bucle de entrenamiento y validación para Disease_Risk."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, TrainConfig
from src.evaluate import evaluate_loader
from src.losses import BinaryFocalLoss
from src.model import (
    count_trainable_parameters,
    describe_freeze_policy,
    save_model,
)
from src.models_zoo import build_model_from_config


# Métricas admitidas para elegir el mejor checkpoint y parar el entrenamiento:
# nombre -> (clave en `val_metrics`, True si "mayor es mejor").
# "val_loss" se minimiza: es la pérdida de validación, que sí mide generalización
# (la pérdida de train seleccionaría la época más sobreajustada).
SELECTION_METRICS: dict[str, tuple[str, bool]] = {
    "auc": ("auc", True),
    "f1": ("f1", True),
    "val_loss": ("loss", False),
    # Orientadas a recall, independientes del umbral (ver src/recall_selection.py)
    "best_f2": ("best_f2", True),
    "recall_at_spec_85": ("recall_at_spec_85", True),
    "spec_at_fn_lt_fp": ("spec_at_fn_lt_fp", True),
}


def resolve_selection_metric(selection_metric: str) -> tuple[str, bool]:
    """Devuelve (clave en val_metrics, mayor_es_mejor) o lanza ValueError si no existe."""
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric desconocida: {selection_metric!r}. "
            f"Valores admitidos: {sorted(SELECTION_METRICS)}"
        )
    return SELECTION_METRICS[selection_metric]


def _ensure_torch_home() -> None:
    """Usa cache bajo artifacts/ si TORCH_HOME no está definido."""
    if "TORCH_HOME" in os.environ:
        return
    torch_home = PROJECT_ROOT / "artifacts" / "torch_cache"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)


def set_seed(seed: int) -> None:
    """Fija semillas para reproducibilidad razonable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_optimizer(model: nn.Module, config: TrainConfig) -> AdamW:
    """AdamW solo sobre parámetros con requires_grad=True."""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_scheduler(
    optimizer: AdamW,
    config: TrainConfig,
    total_epochs: int | None = None,
):
    """Scheduler de LR opcional (se avanza una vez por época).

    - "none": devuelve None (comportamiento del baseline).
    - "cosine": decae de lr a ~0 a lo largo de `total_epochs`.
    - "plateau": reduce el LR a la mitad si la métrica de selección no mejora en 2 épocas
      (AUC de validación, a maximizar; con selection_metric="val_loss", la pérdida de
      validación, a minimizar, para ser coherente con el criterio de checkpoint).
    """
    total_epochs = total_epochs or config.epochs
    if config.lr_scheduler == "none":
        return None
    if config.lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_epochs)
    if config.lr_scheduler == "plateau":
        plateau_mode = "min" if config.selection_metric == "val_loss" else "max"
        return ReduceLROnPlateau(optimizer, mode=plateau_mode, factor=0.5, patience=2)
    raise ValueError(f"lr_scheduler desconocido: {config.lr_scheduler}")


def step_scheduler(scheduler, monitored_value: float) -> None:
    """Avanza el scheduler; ReduceLROnPlateau necesita la métrica monitorizada."""
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(monitored_value)
    else:
        scheduler.step()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Una época de entrenamiento; devuelve loss media."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in tqdm(train_loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


def _describe_freeze_policy_for_metadata(config: TrainConfig) -> str:
    """Texto de la política de congelación que se guarda junto al checkpoint."""
    if config.backbone_name != "resnet152":
        return f"last {config.unfrozen_blocks} block group(s)+head trainable; rest frozen"
    if config.unfreeze_layer3:
        return "layer3+layer4+fc trainable; rest frozen"
    return "layer4+fc trainable; rest frozen"


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    history: list[dict] | None = None,
) -> nn.Module:
    """Entrena el modelo y guarda el mejor checkpoint según métrica de validación.

    Si se pasa una lista en `history`, se le añade un diccionario por época
    (epoch, train_loss, val_loss, val_auc, ...) para poder graficar las curvas.
    """
    selection_key, higher_is_better = resolve_selection_metric(config.selection_metric)

    _ensure_torch_home()
    set_seed(config.seed)
    device = get_device()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config).to(device)
    trainable, total = count_trainable_parameters(model)

    print(f"Dispositivo: {device}")
    if config.backbone_name == "resnet152":
        print(f"Backbone: ResNet-152 (no existe ResNet-150 en torchvision)")
    else:
        print(f"Backbone: {config.backbone_name} (bloques finales entrenables: {config.unfrozen_blocks})")
    print(f"Épocas: {config.epochs}")
    print(f"Parámetros entrenables: {trainable:,} / {total:,}")
    print("Política de congelación:")
    print(describe_freeze_policy(model))

    criterion = BinaryFocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    best_metric = -float("inf") if higher_is_better else float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = config.checkpoint_dir / config.best_checkpoint_name

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_loader(model, val_loader, device, criterion=criterion)
        val_loss = float(val_metrics["loss"])

        selection_value = float(val_metrics[selection_key])
        if history is not None:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": float(val_metrics["auc"]),
                    "val_f1": float(val_metrics["f1"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.time() - epoch_start_time,
                }
            )
        # El plateau monitoriza la misma métrica que elige el checkpoint solo con val_loss;
        # con "auc"/"f1" se mantiene el comportamiento original (AUC).
        scheduler_value = val_loss if config.selection_metric == "val_loss" else float(val_metrics["auc"])
        step_scheduler(scheduler, scheduler_value)
        # Formato explícito: Epoch 03/20 | train_loss=... | val_loss=... | métricas
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"AUC={val_metrics['auc']:.4f} | "
            f"F1={val_metrics['f1']:.4f} | "
            f"Acc={val_metrics['accuracy']:.4f}"
        )

        is_improvement = (
            selection_value > best_metric if higher_is_better else selection_value < best_metric
        )
        if is_improvement:
            best_metric = selection_value
            epochs_without_improvement = 0
            save_model(
                model,
                best_checkpoint_path,
                metadata={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "selection_metric": config.selection_metric,
                    "selection_value": best_metric,
                    "val_metrics": val_metrics,
                    "backbone": config.backbone_name,
                    "freeze_policy": _describe_freeze_policy_for_metadata(config),
                    "lr_scheduler": config.lr_scheduler,
                    "preprocessing": config.preprocessing,
                },
            )
            print(
                f"  -> Nuevo mejor modelo "
                f"({config.selection_metric}={best_metric:.4f}) "
                f"guardado en {best_checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f"Early stopping en época {epoch} "
                    f"(paciencia={config.early_stopping_patience})"
                )
                break

    print(f"Mejor {config.selection_metric} de validación: {best_metric:.4f}")
    return model


def load_best_checkpoint_for_eval(
    config: TrainConfig,
    device: torch.device | None = None,
) -> nn.Module:
    """Carga el mejor checkpoint guardado durante el entrenamiento."""
    from src.model import load_model

    if device is None:
        device = get_device()
    checkpoint_path = config.checkpoint_dir / config.best_checkpoint_name
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"No se encontró checkpoint: {checkpoint_path}")
    return load_model(checkpoint_path, config, device)
