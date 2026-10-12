"""Configuración centralizada del fine-tuning centrado en recall (notebook 06).

Se parte del mejor ResNet-152 individual (`best_resnet152_disease_risk_optuna_nocrop.pt`, ajustado
con Optuna y sin recorte aleatorio) y se afina pocas épocas con distintas combinaciones de las dos
ideas (re-ponderación por época y loss orientada a recall). Los cuatro experimentos forman un
diseño 2x2, de modo que se puede atribuir el efecto de cada idea por separado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from src.config import PROJECT_ROOT, TrainConfig, default_config
from src.data import PREPROCESSING_LETTERBOX
from src.hard_example_training import HardExampleConfig

CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "artifacts" / "logs"
RECALL_FOCUS_DIR = PROJECT_ROOT / "artifacts" / "recall_focus"
PREDICTION_CACHE_DIR = RECALL_FOCUS_DIR / "predictions"
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures" / "recall_focus"
GRADCAM_DIR = PROJECT_ROOT / "artifacts" / "gradcam" / "recall_focus"
SUMMARY_PATH = RECALL_FOCUS_DIR / "summary.json"
SELECTED_GRADCAM_IMAGES_PATH = PROJECT_ROOT / "artifacts" / "gradcam" / "selected_images.json"

# Punto de partida: el mejor ResNet-152 individual (no se modifica, solo se lee)
WARM_START_CHECKPOINT = CHECKPOINT_DIR / "best_resnet152_disease_risk_optuna_nocrop.pt"
OPTUNA_BEST_PARAMS_PATH = PROJECT_ROOT / "artifacts" / "optuna" / "best_params.json"

FINE_TUNE_EPOCHS = 4
# El Optuna original usó lr=4.6e-4 desde ImageNet; para afinar un modelo ya convergido se
# baja a 1e-4 con decaimiento cosine hacia 0 en las pocas épocas disponibles.
FINE_TUNE_LEARNING_RATE = 1e-4
# Criterio de checkpoint: especificidad que hay que sacrificar para tener FN < FP (ver recall_selection)
SELECTION_METRIC = "spec_at_fn_lt_fp"

# Loss orientada a recall: alpha pondera a los POSITIVOS (la original, 0.26, pesaba más a los
# negativos). 0.6 da a un positivo ~1.5x el peso de un negativo.
ORIGINAL_FOCAL_ALPHA = 0.26
RECALL_FOCAL_ALPHA = 0.6
NO_REWEIGHTING = HardExampleConfig(weighting_strength=0.0)
EPOCH_REWEIGHTING = HardExampleConfig(weighting_strength=3.0, positive_boost=2.0, max_sample_weight=6.0)


@dataclass(frozen=True)
class RecallFocusVariant:
    """Un experimento del diseño 2x2: qué loss y qué muestreo usa."""

    name: str
    description: str
    focal_alpha: float
    hard_config: HardExampleConfig

    @property
    def checkpoint_name(self) -> str:
        return f"best_resnet152_recallfocus_{self.name}_nocrop.pt"

    @property
    def history_path(self):
        return LOG_DIR / f"recallfocus_{self.name}_history.json"


VARIANTS: dict[str, RecallFocusVariant] = {
    variant.name: variant
    for variant in (
        RecallFocusVariant(
            "control", "control: loss original, muestreo uniforme", ORIGINAL_FOCAL_ALPHA, NO_REWEIGHTING
        ),
        RecallFocusVariant(
            "reweight", "solo re-ponderación por época", ORIGINAL_FOCAL_ALPHA, EPOCH_REWEIGHTING
        ),
        RecallFocusVariant(
            "recallloss", "solo loss orientada a recall (alpha 0.6)", RECALL_FOCAL_ALPHA, NO_REWEIGHTING
        ),
        RecallFocusVariant(
            "both", "re-ponderación + loss orientada a recall", RECALL_FOCAL_ALPHA, EPOCH_REWEIGHTING
        ),
    )
}


def build_variant_train_config(variant: RecallFocusVariant) -> TrainConfig:
    """Config del fine-tuning: hiperparámetros de Optuna, lr reducido, letterbox y nuevo criterio."""
    optuna_params = json.loads(OPTUNA_BEST_PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    return replace(
        default_config(),
        preprocessing=PREPROCESSING_LETTERBOX,
        pretrained=False,  # los pesos vienen del checkpoint de partida
        unfreeze_layer3=optuna_params["unfreeze_layer3"],
        weight_decay=optuna_params["weight_decay"],
        focal_gamma=optuna_params["focal_gamma"],
        focal_alpha=variant.focal_alpha,
        batch_size=optuna_params["batch_size"],
        learning_rate=FINE_TUNE_LEARNING_RATE,
        lr_scheduler="cosine",
        epochs=FINE_TUNE_EPOCHS,
        early_stopping_patience=FINE_TUNE_EPOCHS,
        selection_metric=SELECTION_METRIC,
        best_checkpoint_name=variant.checkpoint_name,
    )
