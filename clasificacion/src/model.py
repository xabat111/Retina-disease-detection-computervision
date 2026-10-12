"""ResNet-152 ImageNet para Disease_Risk binario, con fine-tuning parcial."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights

from src.config import TrainConfig


def build_resnet152(config: TrainConfig) -> nn.Module:
    """Construye ResNet-152 pretrained con cabeza binaria.

    Nota: no existe ResNet-150 en torchvision; usamos ResNet-152.

    Política de congelación (si freeze_backbone_except_layer4=True):
    - Congelado: conv1, bn1, layer1, layer2, layer3 (features tempranos/medios).
    - Entrenable: layer4 (bloques residuales finales) + fc (clasificador).
    """
    weights = ResNet152_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = models.resnet152(weights=weights)

    # Sustituir fc: 1 logit para clasificación binaria con Focal/BCEWithLogits
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.num_classes)

    if config.freeze_backbone_except_layer4:
        _freeze_early_layers(model, unfreeze_layer3=config.unfreeze_layer3)

    return model


def _freeze_early_layers(model: nn.Module, unfreeze_layer3: bool = False) -> None:
    """Congela todo excepto layer4 y fc (y layer3 si se pide)."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_blocks = [model.layer4]
    if unfreeze_layer3:
        trainable_blocks.append(model.layer3)

    for block in trainable_blocks:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Devuelve (entrenables, totales)."""
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def describe_freeze_policy(model: nn.Module) -> str:
    """Resumen legible de qué módulos tienen parámetros entrenables."""
    lines: list[str] = []
    for module_name, module in model.named_children():
        trainable = any(parameter.requires_grad for parameter in module.parameters())
        status = "entrenable" if trainable else "congelado"
        if sum(1 for _ in module.parameters()) == 0:
            status = "sin parámetros"
        lines.append(f"  - {module_name}: {status}")
    return "\n".join(lines)


def save_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Guarda state_dict y metadatos opcionales."""
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)


def load_model(
    checkpoint_path: str | Path,
    config: TrainConfig,
    device: torch.device,
) -> nn.Module:
    """Carga pesos en un ResNet-152 con la misma arquitectura de entrenamiento."""
    # Los pesos vienen del checkpoint: no hace falta descargar los de ImageNet
    inference_config = replace(config, pretrained=False)
    model = build_resnet152(inference_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
