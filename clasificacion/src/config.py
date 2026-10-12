"""Hiperparámetros y rutas del pipeline RFMiD Disease_Risk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Código y artifacts: un nivel por encima de src/ (p. ej. retina_clasificacion/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_RFMID_LABELS_MARKER = Path("Training_Set") / "Training_Set" / "RFMiD_Training_Labels.csv"
# Distingue este proyecto de siblings como retina_multilabel/ (que también tiene src/)
_PROJECT_MARKER = Path("src") / "config.py"


def find_project_root_from_cwd(start: Path | None = None) -> Path:
    """Localiza retina_clasificacion/ desde el cwd del notebook o un padre.

    Busca en el cwd, sus padres y los hijos inmediatos del cwd (caso típico:
    Jupyter arranca en `Retina/` y el código está en `Retina/retina_clasificacion/`).
    El marcador es `src/config.py` + `requirements.txt`, no solo `src/`.
    """
    cwd = (start or Path.cwd()).resolve()
    candidates: list[Path] = [cwd, *cwd.parents]
    if cwd.is_dir():
        candidates.extend(sorted(path for path in cwd.iterdir() if path.is_dir()))

    for directory in candidates:
        if (directory / _PROJECT_MARKER).is_file() and (directory / "requirements.txt").is_file():
            return directory

    raise FileNotFoundError(
        "No se encontró la raíz del proyecto (src/config.py + requirements.txt). "
        f"cwd={cwd}"
    )


def resolve_data_root(project_root: Path = PROJECT_ROOT) -> Path:
    """Carpeta que contiene Training_Set / Evaluation_Set / Test_Set.

    Tras mover el código a `retina_clasificacion/`, los CSV e imágenes RFMiD pueden
    seguir en el padre (`Retina/`). Se busca primero junto al código y, si no, un nivel arriba.
    """
    if (project_root / _RFMID_LABELS_MARKER).is_file():
        return project_root
    parent_root = project_root.parent
    if (parent_root / _RFMID_LABELS_MARKER).is_file():
        return parent_root
    return project_root


DATA_ROOT = resolve_data_root(PROJECT_ROOT)


@dataclass
class TrainConfig:
    """Configuración centralizada de entrenamiento y datos."""

    # Rutas: artifacts junto al código; datos RFMiD en DATA_ROOT (puede ser el padre)
    project_root: Path = PROJECT_ROOT
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "checkpoints")

    # Datos RFMiD (misma lógica que notebooks/01_build_datasets.ipynb)
    train_labels_csv: Path = field(
        default_factory=lambda: DATA_ROOT
        / "Training_Set"
        / "Training_Set"
        / "RFMiD_Training_Labels.csv"
    )
    train_images_dir: Path = field(
        default_factory=lambda: DATA_ROOT
        / "Training_Set"
        / "Training_Set"
        / "Training"
    )
    val_labels_csv: Path = field(
        default_factory=lambda: DATA_ROOT
        / "Evaluation_Set"
        / "Evaluation_Set"
        / "RFMiD_Validation_Labels.csv"
    )
    val_images_dir: Path = field(
        default_factory=lambda: DATA_ROOT
        / "Evaluation_Set"
        / "Evaluation_Set"
        / "Validation"
    )
    test_labels_csv: Path = field(
        default_factory=lambda: DATA_ROOT
        / "Test_Set"
        / "Test_Set"
        / "RFMiD_Testing_Labels.csv"
    )
    test_images_dir: Path = field(
        default_factory=lambda: DATA_ROOT / "Test_Set" / "Test_Set" / "Test"
    )

    # Modelo
    # Nota: el usuario pidió "ResNet-150"; no existe en torchvision.
    # Usamos ResNet-152 (ImageNet pretrained), la variante más cercana.
    backbone_name: str = "resnet152"
    num_classes: int = 1  # 1 logit + BCEWithLogits / Focal binario
    pretrained: bool = True

    # Congelación: backbone temprano congelado; layer4 + fc entrenables
    freeze_backbone_except_layer4: bool = True
    # Si True (y el backbone está congelado), layer3 también se entrena (usado por Optuna)
    unfreeze_layer3: bool = False
    # Solo para arquitecturas de src/models_zoo.py (no ResNet-152): nº de grupos de bloques
    # finales entrenables (ver ArchitectureSpec.block_group_paths)
    unfrozen_blocks: int = 1

    # Imagen / DataLoader
    image_size: int = 224
    # Preprocesado espacial: "random_crop" (comportamiento original: RandomResizedCrop en
    # train y resize cuadrado en val/test) | "letterbox" (resize conservando la proporción
    # + relleno negro a cuadrado, sin recorte aleatorio, igual en train y eval)
    preprocessing: str = "random_crop"
    batch_size: int = 16
    num_workers: int = 0  # 0 es más fiable en Windows
    pin_memory: bool = True

    # Entrenamiento
    seed: int = 42
    epochs: int = 20  # EPOCHS por defecto (pedido del usuario)
    learning_rate: float = 1e-4  # LR para capas descongeladas (layer4 + fc)
    weight_decay: float = 1e-4
    # Scheduler de LR por época: "none" (baseline) | "cosine" | "plateau"
    lr_scheduler: str = "none"
    # Focal Loss: gamma > 0 enfoca ejemplos difíciles; alpha equilibra clases
    focal_gamma: float = 2.0
    focal_alpha: float | None = 0.25  # peso de la clase positiva (disease_risk=1)

    # Selección de modelo: AUC es más robusta ante desbalance que accuracy
    # "auc" | "f1" (se maximizan) | "val_loss" (pérdida de validación, se minimiza).
    # Gobierna el mejor checkpoint y el early stopping (ver src/train.SELECTION_METRICS).
    selection_metric: str = "auc"
    early_stopping_patience: int = 7

    # Checkpoint
    best_checkpoint_name: str = "best_resnet152_disease_risk.pt"


def default_config() -> TrainConfig:
    """Devuelve la configuración por defecto del proyecto."""
    return TrainConfig()
