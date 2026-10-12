"""Datasets letterbox con las imágenes ya reescaladas en RAM (y cacheadas en disco como .npy).

El letterbox es determinista y es el primer paso de las transformaciones de train, val y test
(`src/data.py`). Aplicarlo una sola vez evita decodificar PNGs de 2144x1424 en cada época, que
es una parte importante del coste en CPU, y deja EXACTAMENTE las mismas entradas: el resto de
augmentations (flip, rotación, ColorJitter) se aplica igual sobre la imagen ya reescalada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import PROJECT_ROOT, TrainConfig
from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING_LETTERBOX,
    LetterboxResize,
    build_split_dataframe,
    get_train_transforms,
)
from src.tuning import InMemoryImageDataset

LETTERBOX_CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache"


@dataclass
class CachedSplits:
    """Datasets en RAM y DataFrames (mismo orden que los CSV) de train / val / test."""

    train_dataset: InMemoryImageDataset
    val_dataset: InMemoryImageDataset
    test_dataset: InMemoryImageDataset
    train_dataframe: pd.DataFrame
    val_dataframe: pd.DataFrame
    test_dataframe: pd.DataFrame


def load_or_build_letterbox_array(
    split_name: str,
    dataframe: pd.DataFrame,
    image_size: int,
    cache_dir: Path = LETTERBOX_CACHE_DIR,
) -> np.ndarray:
    """Array uint8 (N, size, size, 3) con el letterbox de cada imagen; se cachea en disco."""
    cache_path = cache_dir / f"letterbox{image_size}_{split_name}.npy"
    if cache_path.is_file():
        cached_array = np.load(cache_path)
        if len(cached_array) == len(dataframe):
            return cached_array

    letterbox = LetterboxResize(image_size)
    letterboxed_images = []
    for image_path in tqdm(dataframe["image_path"], desc=f"letterbox {split_name}", leave=False):
        with Image.open(image_path) as raw_image:
            letterboxed_images.append(np.asarray(letterbox(raw_image.convert("RGB"))))
    image_array = np.stack(letterboxed_images)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, image_array)
    return image_array


def _to_pil_list(image_array: np.ndarray) -> list[Image.Image]:
    return [Image.fromarray(image) for image in image_array]


def build_cached_letterbox_splits(config: TrainConfig) -> CachedSplits:
    """Datasets letterbox equivalentes a `create_dataloaders(config)` pero sin decodificar PNGs."""
    if config.preprocessing != PREPROCESSING_LETTERBOX:
        raise ValueError("build_cached_letterbox_splits solo sirve para preprocessing='letterbox'")

    dataframes = {
        "train": build_split_dataframe(config.train_labels_csv, config.train_images_dir),
        "val": build_split_dataframe(config.val_labels_csv, config.val_images_dir),
        "test": build_split_dataframe(config.test_labels_csv, config.test_images_dir),
    }
    # Train: todas las transformaciones menos el letterbox inicial (ya aplicado en la caché)
    train_transform = transforms.Compose(
        get_train_transforms(config.image_size, PREPROCESSING_LETTERBOX).transforms[1:]
    )
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )

    datasets = {}
    for split_name, dataframe in dataframes.items():
        image_array = load_or_build_letterbox_array(split_name, dataframe, config.image_size)
        datasets[split_name] = InMemoryImageDataset(
            _to_pil_list(image_array),
            dataframe["disease_risk"].tolist(),
            train_transform if split_name == "train" else eval_transform,
        )
    return CachedSplits(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        train_dataframe=dataframes["train"],
        val_dataframe=dataframes["val"],
        test_dataframe=dataframes["test"],
    )
