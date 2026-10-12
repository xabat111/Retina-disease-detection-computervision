"""Configuración centralizada del análisis de recall (notebook 05): rutas, objetivos y miembros."""

from __future__ import annotations

from src.config import PROJECT_ROOT
from src.data import PREPROCESSING_LETTERBOX, PREPROCESSING_RANDOM_CROP
from src.tta_inference import EnsembleMember

# Objetivo clínico: recall de la clase enfermo en validación Y test cercano a 0.90
TARGET_RECALL = 0.90
# Margen de seguridad al elegir el umbral en validación: el recall de test fluctúa ~±2 puntos
# alrededor del de validación (506 positivos), así que se apunta algo por encima del objetivo.
VALIDATION_RECALL_MARGIN = 0.01

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42

CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RECALL_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "recall"
PREDICTION_CACHE_DIR = RECALL_ARTIFACTS_DIR / "predictions"
RECALL_FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures" / "recall"
RECALL_GRADCAM_DIR = PROJECT_ROOT / "artifacts" / "gradcam" / "recall"
FINAL_SUMMARY_PATH = RECALL_ARTIFACTS_DIR / "final_summary.json"

# Punto de partida con el que se compara todo: modelo ajustado nocrop (02.1 nocrop)
REFERENCE_MEMBER_NAME = "ajustado_nocrop"


def existing_checkpoint_members() -> list[EnsembleMember]:
    """Los cuatro ResNet-152 ya entrenados, cada uno con SU preprocesado.

    Los de `random_crop` se evalúan con resize cuadrado (como en su validación original) y los
    nocrop con letterbox; mezclar preprocesados en un ensemble es válido porque cada modelo ve
    la entrada con la que fue entrenado.
    """
    candidates = [
        EnsembleMember(
            "ajustado_nocrop",
            CHECKPOINT_DIR / "best_resnet152_disease_risk_optuna_nocrop.pt",
            PREPROCESSING_LETTERBOX,
        ),
        EnsembleMember(
            "baseline_nocrop",
            CHECKPOINT_DIR / "best_resnet152_disease_risk_nocrop.pt",
            PREPROCESSING_LETTERBOX,
        ),
        EnsembleMember(
            "ajustado_original",
            CHECKPOINT_DIR / "best_resnet152_disease_risk_optuna.pt",
            PREPROCESSING_RANDOM_CROP,
        ),
        EnsembleMember(
            "baseline_original",
            CHECKPOINT_DIR / "best_resnet152_disease_risk.pt",
            PREPROCESSING_RANDOM_CROP,
        ),
    ]
    return [member for member in candidates if member.checkpoint_path.is_file()]
