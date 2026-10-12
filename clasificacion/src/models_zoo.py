"""Arquitecturas torchvision (ImageNet) para Disease_Risk binario, con política de congelación común.

Cada arquitectura se describe con un `ArchitectureSpec` (solo datos, rutas de submódulos):
- `block_group_paths`: grupos de bloques entrenables, ordenados del FINAL de la red hacia atrás.
  Con `unfrozen_blocks=n` se descongelan los n primeros grupos + la cabeza. Cada grupo es una
  "etapa" completa junto con la capa de downsampling que la precede (misma idea que layer4 en ResNet).
- `head_path`: módulo clasificador (siempre entrenable).
- `gradcam_layer_path`: última capa con mapa espacial 2D, la que se engancha para Grad-CAM.

`src/model.py` (ResNet-152 del baseline) no se modifica: `build_model_from_config` delega en él
cuando `backbone_name == "resnet152"` para conservar exactamente su comportamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from src.config import TrainConfig
from src.model import build_resnet152, count_trainable_parameters

# Mismo conjunto de pesos para todas: IMAGENET1K_V1 (el que usa el ResNet-152 del baseline)
PRETRAINED_WEIGHTS_NAME = "IMAGENET1K_V1"


@dataclass(frozen=True)
class ArchitectureSpec:
    """Descripción declarativa de una arquitectura y su política de fine-tuning."""

    name: str  # nombre en torchvision.models
    display_name: str
    final_linear_path: str  # Linear final que se sustituye por 1 logit
    head_path: str  # módulo clasificador completo (entrenable siempre)
    block_group_paths: tuple[tuple[str, ...], ...]  # del final hacia atrás
    gradcam_layer_path: str
    note: str


ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "resnet152": ArchitectureSpec(
        name="resnet152",
        display_name="ResNet-152",
        final_linear_path="fc",
        head_path="fc",
        block_group_paths=(("layer4",), ("layer3",)),
        gradcam_layer_path="layer4.2",
        note="Referencia del proyecto (baseline).",
    ),
    "resnet50": ArchitectureSpec(
        name="resnet50",
        display_name="ResNet-50",
        final_linear_path="fc",
        head_path="fc",
        block_group_paths=(("layer4",), ("layer3",)),
        gradcam_layer_path="layer4.2",
        note="Misma familia residual que el baseline, ~2.3x menos profunda.",
    ),
    "densenet121": ArchitectureSpec(
        name="densenet121",
        display_name="DenseNet-121",
        final_linear_path="classifier",
        head_path="classifier",
        block_group_paths=(
            ("features.norm5", "features.denseblock4", "features.transition3"),
            ("features.denseblock3", "features.transition2"),
        ),
        gradcam_layer_path="features.norm5",
        note="Conexiones densas: reutiliza rasgos de baja y alta resolución.",
    ),
    "efficientnet_b0": ArchitectureSpec(
        name="efficientnet_b0",
        display_name="EfficientNet-B0",
        final_linear_path="classifier.1",
        head_path="classifier",
        block_group_paths=(
            ("features.8", "features.7"),
            ("features.6",),
            ("features.5", "features.4"),
        ),
        gradcam_layer_path="features.8",
        note="Muy ligera (~5 M parámetros); escalado compuesto.",
    ),
    "convnext_tiny": ArchitectureSpec(
        name="convnext_tiny",
        display_name="ConvNeXt-Tiny",
        final_linear_path="classifier.2",
        head_path="classifier",
        block_group_paths=(
            ("features.7", "features.6"),
            ("features.5", "features.4"),
        ),
        gradcam_layer_path="features.7",
        note="CNN moderna (receta tipo Transformer, LayerNorm); mapa 2D válido para Grad-CAM.",
    ),
}


def get_architecture_spec(architecture: str) -> ArchitectureSpec:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Arquitectura desconocida: {architecture!r}. Disponibles: {sorted(ARCHITECTURES)}"
        )
    return ARCHITECTURES[architecture]


def max_unfrozen_blocks(architecture: str) -> int:
    """Número de grupos de bloques que puede descongelar Optuna."""
    return len(get_architecture_spec(architecture).block_group_paths)


def _replace_submodule(model: nn.Module, path: str, new_module: nn.Module) -> None:
    """Sustituye `model.<path>` (admite índices de Sequential, p. ej. 'classifier.1')."""
    parent_path, _, child_name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, child_name, new_module)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def apply_freeze_policy(model: nn.Module, spec: ArchitectureSpec, unfrozen_blocks: int) -> None:
    """Congela todo salvo los `unfrozen_blocks` últimos grupos de bloques y la cabeza."""
    if not 1 <= unfrozen_blocks <= len(spec.block_group_paths):
        raise ValueError(
            f"{spec.name}: unfrozen_blocks debe estar en [1, {len(spec.block_group_paths)}], "
            f"recibido {unfrozen_blocks}"
        )
    _set_trainable(model, False)
    for group_paths in spec.block_group_paths[:unfrozen_blocks]:
        for path in group_paths:
            _set_trainable(model.get_submodule(path), True)
    _set_trainable(model.get_submodule(spec.head_path), True)


def build_architecture(
    architecture: str,
    unfrozen_blocks: int = 1,
    pretrained: bool = True,
    num_classes: int = 1,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Modelo torchvision con cabeza de `num_classes` logits y política de congelación aplicada."""
    spec = get_architecture_spec(architecture)
    weights = PRETRAINED_WEIGHTS_NAME if pretrained else None
    model = models.get_model(spec.name, weights=weights)

    in_features = model.get_submodule(spec.final_linear_path).in_features
    _replace_submodule(model, spec.final_linear_path, nn.Linear(in_features, num_classes))

    if freeze_backbone:
        apply_freeze_policy(model, spec, unfrozen_blocks)
    return model


def build_model_from_config(config: TrainConfig) -> nn.Module:
    """Punto de entrada único para train/tuning: elige la arquitectura según `config.backbone_name`."""
    if config.backbone_name == "resnet152":
        # Camino original del proyecto (usa config.unfreeze_layer3), sin cambios
        return build_resnet152(config)
    return build_architecture(
        config.backbone_name,
        unfrozen_blocks=config.unfrozen_blocks,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        freeze_backbone=config.freeze_backbone_except_layer4,
    )


def get_gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Capa convolucional final sobre la que se calcula Grad-CAM."""
    return model.get_submodule(get_architecture_spec(architecture).gradcam_layer_path)


def load_architecture_checkpoint(
    architecture: str,
    checkpoint_path: str | Path,
    device: torch.device,
    unfrozen_blocks: int = 1,
) -> nn.Module:
    """Carga un checkpoint de `save_model` (sirve también para el ResNet-152 del baseline).

    Los pesos vienen del checkpoint: no hace falta descargar ImageNet. `unfrozen_blocks` no
    afecta a la inferencia; solo sirve para que el recuento de parámetros entrenables sea correcto.
    """
    model = build_architecture(architecture, unfrozen_blocks=unfrozen_blocks, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def describe_architectures(architectures: list[str]) -> list[dict]:
    """Filas resumen (parámetros por política de congelación y capa Grad-CAM) para el notebook."""
    rows = []
    for architecture in architectures:
        spec = get_architecture_spec(architecture)
        row = {
            "modelo": spec.display_name,
            "capa Grad-CAM": spec.gradcam_layer_path,
        }
        for unfrozen_blocks in range(1, len(spec.block_group_paths) + 1):
            model = build_architecture(architecture, unfrozen_blocks, pretrained=False)
            trainable, total = count_trainable_parameters(model)
            row[f"entrenables (bloques={unfrozen_blocks}) [M]"] = round(trainable / 1e6, 2)
            row["parámetros totales [M]"] = round(total / 1e6, 2)
        rows.append(row)
    return rows


def config_for_architecture(base_config: TrainConfig, architecture: str, **overrides) -> TrainConfig:
    """Copia de `base_config` apuntando a otra arquitectura (sin mutar el original)."""
    get_architecture_spec(architecture)
    return replace(base_config, backbone_name=architecture, **overrides)
