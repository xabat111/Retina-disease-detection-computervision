"""Inferencia con Test-Time Augmentation (TTA) y caché de probabilidades por modelo.

Cada modelo se evalúa con SU preprocesado de entrenamiento (`random_crop` -> resize cuadrado,
`letterbox` -> letterbox sin recorte) y, sobre esa imagen ya preprocesada, se aplican vistas
deterministas (volteo horizontal, rotaciones pequeñas con relleno negro, igual que en el
`RandomRotation` de train). Las probabilidades de cada vista se guardan por separado para poder
promediarlas después sin repetir la inferencia, que en CPU es lo caro.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as transforms_functional
from tqdm import tqdm

from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING_LETTERBOX,
    PREPROCESSING_OPTIONS,
    LetterboxResize,
)
from src.models_zoo import load_architecture_checkpoint

# Vistas TTA: nombre -> transformación PIL -> PIL. "identity" es la predicción sin TTA.
# Solo transformaciones que no cambian el diagnóstico (sin flip vertical: la orientación
# del fondo de ojo importa, igual que en el entrenamiento).
TTA_VIEWS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "identity": lambda image: image,
    "hflip": transforms_functional.hflip,
    "rotate_+10": lambda image: transforms_functional.rotate(image, 10.0, fill=0),
    "rotate_-10": lambda image: transforms_functional.rotate(image, -10.0, fill=0),
}
DEFAULT_TTA_VIEW_NAMES = tuple(TTA_VIEWS)

_NORMALIZE_TENSOR = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
)


@dataclass(frozen=True)
class EnsembleMember:
    """Un modelo entrenado y el preprocesado con el que debe evaluarse."""

    name: str
    checkpoint_path: Path
    preprocessing: str
    image_size: int = 224
    # Clave de `src/models_zoo.ARCHITECTURES`; por defecto el ResNet-152 de los notebooks 02-05
    architecture: str = "resnet152"

    def __post_init__(self) -> None:
        if self.preprocessing not in PREPROCESSING_OPTIONS:
            raise ValueError(
                f"{self.name}: preprocesado desconocido {self.preprocessing!r}. "
                f"Opciones: {PREPROCESSING_OPTIONS}"
            )


def load_preprocessed_images(
    dataframe: pd.DataFrame,
    preprocessing: str,
    image_size: int,
) -> list[Image.Image]:
    """Decodifica cada imagen UNA vez y aplica el paso espacial determinista de val/test."""
    if preprocessing == PREPROCESSING_LETTERBOX:
        spatial_step = LetterboxResize(image_size)
    else:
        spatial_step = transforms.Resize((image_size, image_size))

    images = []
    for image_path in tqdm(dataframe["image_path"], desc=f"imágenes ({preprocessing})", leave=False):
        with Image.open(image_path) as raw_image:
            images.append(spatial_step(raw_image.convert("RGB")))
    return images


@torch.inference_mode()
def predict_view_probabilities(
    model: nn.Module,
    preprocessed_images: list[Image.Image],
    view_names: tuple[str, ...],
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """P(riesgo) de cada imagen para cada vista TTA: {vista: array (n_imágenes,)}."""
    model.eval()
    probabilities_by_view: dict[str, np.ndarray] = {}
    for view_name in view_names:
        apply_view = TTA_VIEWS[view_name]
        view_probabilities: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(preprocessed_images), batch_size), desc=f"vista {view_name}", leave=False
        ):
            batch_images = preprocessed_images[start : start + batch_size]
            batch = torch.stack([_NORMALIZE_TENSOR(apply_view(image)) for image in batch_images])
            logits = model(batch.to(device)).view(-1)
            view_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        probabilities_by_view[view_name] = np.concatenate(view_probabilities).astype(np.float64)
    return probabilities_by_view


def _load_cached_views(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.is_file():
        return {}
    with np.load(cache_path) as cached:
        return {view_name: cached[view_name] for view_name in cached.files}


def get_member_view_probabilities(
    member: EnsembleMember,
    split_name: str,
    dataframe: pd.DataFrame,
    cache_dir: Path,
    device: torch.device,
    view_names: tuple[str, ...] = DEFAULT_TTA_VIEW_NAMES,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Probabilidades por vista de un miembro en un split, con caché en disco.

    La caché se invalida a mano borrando el .npz (p. ej. si se reentrena el checkpoint).
    Solo se calculan las vistas que falten.
    """
    cache_path = cache_dir / f"{member.name}_{split_name}.npz"
    cached_views = _load_cached_views(cache_path)
    missing_views = tuple(name for name in view_names if name not in cached_views)

    if missing_views:
        print(f"[{member.name} / {split_name}] calculando vistas: {', '.join(missing_views)}")
        model = load_architecture_checkpoint(member.architecture, member.checkpoint_path, device)
        preprocessed_images = load_preprocessed_images(
            dataframe, member.preprocessing, member.image_size
        )
        cached_views.update(
            predict_view_probabilities(model, preprocessed_images, missing_views, device, batch_size)
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **cached_views)
        del model, preprocessed_images
    return {name: cached_views[name] for name in view_names}


def average_views(
    probabilities_by_view: dict[str, np.ndarray],
    view_names: tuple[str, ...],
) -> np.ndarray:
    """Media de probabilidades de las vistas indicadas."""
    return np.mean([probabilities_by_view[name] for name in view_names], axis=0)
