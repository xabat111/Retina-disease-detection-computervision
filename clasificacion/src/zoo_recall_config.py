"""Configuración centralizada del notebook 07: cuatro arquitecturas SIN Optuna, sin recorte, centradas en recall.

Los hiperparámetros son arbitrarios pero razonables (valores habituales de fine-tuning de cada
familia), no buscados con validación. La receta de pérdida y muestreo es la que mejor funcionó
en el notebook 06 (`chosen_recall_recipe_name()`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from src.config import PROJECT_ROOT, TrainConfig, default_config
from src.data import PREPROCESSING_LETTERBOX
from src.recall_focus_config import (
    RECALL_FOCAL_ALPHA,
    SELECTION_METRIC,
    SUMMARY_PATH as RECALL_FOCUS_SUMMARY_PATH,
    VARIANTS,
    build_variant_train_config,
)

CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "artifacts" / "logs"
ZOO_DIR = PROJECT_ROOT / "artifacts" / "model_zoo_recall"
PREDICTION_CACHE_DIR = ZOO_DIR / "predictions"
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures" / "model_zoo_recall"
GRADCAM_DIR = PROJECT_ROOT / "artifacts" / "gradcam" / "model_zoo_recall"
SUMMARY_PATH = ZOO_DIR / "summary.json"

ARCHITECTURES = ("resnet50", "densenet121", "efficientnet_b0", "convnext_tiny")

# Receta por defecto si 06 aún no ha elegido variante (combina las dos ideas).
# `recall_recipe()` la sustituye por la ganadora de validación cuando existe summary.json.
DEFAULT_RECALL_RECIPE_VARIANT = "both"
# Desde ImageNet las primeras predicciones no informan sobre qué es difícil: 2 épocas uniformes
ZOO_UNIFORM_WARMUP_EPOCHS = 2

# Presupuesto: mismo nº de épocas para todos (comparabilidad); ver la medición en el notebook
ZOO_EPOCHS = 12
ZOO_BATCH_SIZE = 16
ZOO_LR_SCHEDULER = "cosine"


@dataclass(frozen=True)
class ZooModelSettings:
    """Hiperparámetros elegidos a mano para una arquitectura y por qué."""

    learning_rate: float
    weight_decay: float
    unfrozen_blocks: int
    rationale: str

    def checkpoint_name(self, architecture: str) -> str:
        return f"recallfocus_{architecture}_nocrop.pt"


ZOO_SETTINGS: dict[str, ZooModelSettings] = {
    "resnet50": ZooModelSettings(
        2e-4, 1e-4, 2,
        "Misma familia que el ResNet-152 (lr de Optuna 4.6e-4, aquí algo menor y con cosine); layer3+layer4 entrenables.",
    ),
    "densenet121": ZooModelSettings(
        2e-4, 1e-4, 2,
        "Receta estándar de fine-tuning con AdamW; denseblock3+4 entrenables (el 4 incluye norm5).",
    ),
    "efficientnet_b0": ZooModelSettings(
        3e-4, 1e-4, 3,
        "Modelo muy pequeño (5 M) y barato: tolera lr mayor y se descongelan 3 grupos de bloques.",
    ),
    "convnext_tiny": ZooModelSettings(
        1e-4, 1e-2, 2,
        "Receta ConvNeXt: lr bajo y weight decay alto (1e-2) con AdamW; etapas 3+4 entrenables.",
    ),
}


def chosen_recall_recipe_name() -> str:
    """Receta de 06 para el zoo: la ganadora si era recall-centrada; si no, `both`.

    En 06 el control (sin re-ponderación ni alpha alto) empató/ganó porque el ResNet-152
    ya estaba convergido. Los modelos del zoo parten de ImageNet: se aplica la receta
    recall-centrada (re-ponderación + alpha 0.6), que es lo pedido en la parte 3.
    """
    if RECALL_FOCUS_SUMMARY_PATH.is_file():
        summary = json.loads(RECALL_FOCUS_SUMMARY_PATH.read_text(encoding="utf-8"))
        chosen_name = summary.get("chosen_variant_validation")
        if chosen_name in VARIANTS:
            variant = VARIANTS[chosen_name]
            if variant.hard_config.is_active or variant.focal_alpha >= RECALL_FOCAL_ALPHA - 1e-9:
                return chosen_name
    return DEFAULT_RECALL_RECIPE_VARIANT


def recall_recipe():
    """(focal_alpha, HardExampleConfig) de la variante ganadora del notebook 06, para este notebook."""
    variant = VARIANTS[chosen_recall_recipe_name()]
    hard_config = replace(variant.hard_config, uniform_warmup_epochs=ZOO_UNIFORM_WARMUP_EPOCHS)
    return variant.focal_alpha, hard_config


def build_zoo_train_config(architecture: str) -> TrainConfig:
    """Config de entrenamiento de una arquitectura: letterbox 224, mismo augmentation que los nocrop."""
    settings = ZOO_SETTINGS[architecture]
    focal_alpha, _ = recall_recipe()
    base_config = default_config()
    return replace(
        base_config,
        backbone_name=architecture,
        preprocessing=PREPROCESSING_LETTERBOX,
        pretrained=True,
        unfrozen_blocks=settings.unfrozen_blocks,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        batch_size=ZOO_BATCH_SIZE,
        lr_scheduler=ZOO_LR_SCHEDULER,
        epochs=ZOO_EPOCHS,
        early_stopping_patience=ZOO_EPOCHS,
        focal_alpha=focal_alpha,
        # Mismo gamma que el ResNet-152 ajustado con Optuna (1.36)
        focal_gamma=build_variant_train_config(VARIANTS[chosen_recall_recipe_name()]).focal_gamma,
        selection_metric=SELECTION_METRIC,
        best_checkpoint_name=settings.checkpoint_name(architecture),
    )


def history_path(architecture: str):
    return LOG_DIR / f"recallfocus_{architecture}_history.json"
