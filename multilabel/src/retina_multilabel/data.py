"""Datos: tablas multietiqueta, preprocesado (letterbox), Dataset y DataLoader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

from .config import ProjectConfig, TrainConfig
from .labels import (
    IDENTIFIER_COLUMN,
    LABEL_NAMES,
    ConsistencyReport,
    build_label_frame,
)
from .utils import locate_artifact, resolve_dataset_search_roots

SPLITS = ("train", "val", "test")
LABEL_FILE_NAMES = {
    "train": "RFMiD_Training_Labels.csv",
    "val": "RFMiD_Validation_Labels.csv",
    "test": "RFMiD_Testing_Labels.csv",
}
IMAGE_FOLDER_NAMES = {"train": "Training", "val": "Validation", "test": "Test"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MULTILABEL_FILE_TEMPLATE = "multilabel_{split}.csv"


@dataclass(frozen=True)
class SplitFiles:
    labels_csv: Path
    image_dir: Path


# --------------------------------------------------------------------------- localización


def _contains_images(directory: Path) -> bool:
    return any(path.suffix.lower() in IMAGE_EXTENSIONS for path in directory.iterdir())


def _find_image_dir(labels_csv: Path, folder_name: str, search_root: Path) -> Path:
    """Busca la carpeta de imágenes junto al CSV y, si no, en todo el dataset."""
    sibling = labels_csv.parent / folder_name
    if sibling.is_dir() and _contains_images(sibling):
        return sibling
    for candidate in sorted(search_root.rglob(folder_name)):
        if candidate.is_dir() and _contains_images(candidate):
            return candidate
    raise FileNotFoundError(
        f"No se encontró la carpeta de imágenes '{folder_name}' junto a {labels_csv} "
        f"ni dentro de {search_root}."
    )


def locate_split_files(split: str, search_roots: dict[str, Path] | None = None) -> SplitFiles:
    """Localiza el CSV de etiquetas y la carpeta de imágenes de un split oficial."""
    search_roots = search_roots or resolve_dataset_search_roots()
    search_root = search_roots[split]
    matches = sorted(search_root.rglob(LABEL_FILE_NAMES[split]))
    if not matches:
        raise FileNotFoundError(
            f"No se encontró '{LABEL_FILE_NAMES[split]}' dentro de {search_root}. "
            "En Kaggle comprueba que has añadido el dataset RFMiD "
            "(p. ej. andrewmvd/retinal-disease-classification)."
        )
    labels_csv = matches[0]
    image_dir = _find_image_dir(labels_csv, IMAGE_FOLDER_NAMES[split], search_root)
    return SplitFiles(labels_csv=labels_csv, image_dir=image_dir)


def _index_images(image_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _attach_image_paths(labels: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Añade `image_path` (resuelto por ID) y falla si falta alguna imagen."""
    image_index = _index_images(image_dir)
    paths = [image_index.get(str(identifier)) for identifier in labels[IDENTIFIER_COLUMN]]
    missing = [str(i) for i, p in zip(labels[IDENTIFIER_COLUMN], paths) if p is None]
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} imágenes en {image_dir}: {missing[:5]}...")
    attached = labels.copy()
    attached.insert(1, "image_path", [str(path) for path in paths])
    return attached


# --------------------------------------------------------------------------- tablas


def _subsample_for_smoke_test(table: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(table) <= max_rows:
        return table
    return table.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def build_multilabel_tables(
    config: ProjectConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, ConsistencyReport]]:
    """Construye las tablas train/val/test (ID, image_path, 10 etiquetas) desde el dataset RFMiD."""
    search_roots = resolve_dataset_search_roots()
    tables: dict[str, pd.DataFrame] = {}
    reports: dict[str, ConsistencyReport] = {}
    for index, split in enumerate(SPLITS):
        files = locate_split_files(split, search_roots)
        labels, report = build_label_frame(pd.read_csv(files.labels_csv))
        table = _attach_image_paths(labels, files.image_dir)
        if config.data.max_rows_per_split is not None:
            table = _subsample_for_smoke_test(
                table, config.data.max_rows_per_split[index], config.seed
            )
        tables[split] = table
        reports[split] = report
    return tables, reports


def save_multilabel_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    saved_paths = []
    for split, table in tables.items():
        path = output_dir / MULTILABEL_FILE_TEMPLATE.format(split=split)
        table.to_csv(path, index=False)
        saved_paths.append(path)
    return saved_paths


def load_multilabel_tables(
    config: ProjectConfig, output_dir: Path
) -> dict[str, pd.DataFrame]:
    """Carga las tablas guardadas por el notebook 01 o, si no existen, las reconstruye.

    Las rutas de imagen se vuelven a resolver por ID porque cambian entre sesiones de Kaggle.
    """
    try:
        saved = {
            split: pd.read_csv(
                locate_artifact(MULTILABEL_FILE_TEMPLATE.format(split=split), output_dir)
            )
            for split in SPLITS
        }
    except FileNotFoundError:
        print("No se encontraron los CSV del notebook 01: se reconstruyen desde RFMiD.")
        return build_multilabel_tables(config)[0]

    search_roots = resolve_dataset_search_roots()
    tables = {}
    for split, table in saved.items():
        image_dir = locate_split_files(split, search_roots).image_dir
        tables[split] = _attach_image_paths(table.drop(columns="image_path", errors="ignore"), image_dir)
    return tables


def label_matrix(
    table: pd.DataFrame, label_names: list[str] | None = None
) -> np.ndarray:
    """Matriz (N, C) de etiquetas binarias (por defecto las 10 de `LABEL_NAMES`)."""
    names = label_names if label_names is not None else LABEL_NAMES
    return table[list(names)].to_numpy(dtype=np.float32)


def build_prevalence_table(
    tables: dict[str, pd.DataFrame], min_positives: int
) -> pd.DataFrame:
    """Positivos y prevalencia por etiqueta y split, con aviso de métricas poco fiables."""
    rows = {}
    for label in LABEL_NAMES:
        row = {}
        for split in SPLITS:
            positives = int(tables[split][label].sum())
            row[f"n_{split}"] = positives
            row[f"%_{split}"] = round(100 * positives / len(tables[split]), 1)
        row["metricas_fiables"] = row["n_val"] >= min_positives and row["n_test"] >= min_positives
        rows[label] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def build_cooccurrence_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """Nº de imágenes en las que coinciden cada par de etiquetas (diagonal = positivos)."""
    matrix = label_matrix(table)
    return pd.DataFrame(matrix.T @ matrix, index=LABEL_NAMES, columns=LABEL_NAMES).astype(int)


# --------------------------------------------------------------------------- preprocesado


def letterbox_image(image: Image.Image, size: int) -> np.ndarray:
    """Redimensiona conservando la proporción y rellena con negro hasta un cuadrado.

    No recorta: el fondo de los fondos de ojo ya es negro y así no se pierde periferia.
    Devuelve un array uint8 (size, size, 3).
    """
    image = image.convert("RGB")
    scale = size / max(image.size)
    new_width = max(1, round(image.width * scale))
    new_height = max(1, round(image.height * scale))
    resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC, reducing_gap=3.0)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
    return np.array(canvas)  # copia escribible (np.asarray daría un array de solo lectura)


class _LetterboxPathDataset(Dataset):
    """Lee y aplica letterbox a una lista de rutas (para paralelizar la caché)."""

    def __init__(self, image_paths: list[str], size: int) -> None:
        self.image_paths = image_paths
        self.size = size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.image_paths[index]) as image:
            array = letterbox_image(image, self.size)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def cache_letterboxed_images(
    image_paths: list[str], size: int, num_workers: int, description: str = "imágenes"
) -> torch.Tensor:
    """Lee todas las imágenes una sola vez y las guarda como tensor uint8 (N, 3, size, size).

    Las imágenes originales son muy grandes; con el backbone congelado solo hace falta
    preprocesarlas una vez y aplicar los aumentos sobre la versión de 224x224.
    """
    loader = DataLoader(
        _LetterboxPathDataset(image_paths, size), batch_size=8, num_workers=num_workers
    )
    batches = [batch for batch in tqdm(loader, desc=f"Preprocesando {description}")]
    return torch.cat(batches)


CROP_TRANSFORM_NAMES = (
    "RandomResizedCrop",
    "RandomCrop",
    "CenterCrop",
    "FiveCrop",
    "TenCrop",
    "RandomSizedCrop",
)


def crop_transform_types() -> tuple[type, ...]:
    """Clases de recorte de torchvision presentes en esta instalación."""
    return tuple(
        transform_type
        for name in CROP_TRANSFORM_NAMES
        if (transform_type := getattr(T, name, None)) is not None
    )


def collect_transform_types(transform: object) -> set[type]:
    """Tipos de transformaciones, incluyendo las anidadas en `Compose`."""
    found = {type(transform)}
    nested = getattr(transform, "transforms", None)
    if nested:
        for child in nested:
            found.update(collect_transform_types(child))
    return found


def build_train_augmentation(train_config: TrainConfig) -> T.Compose:
    """Aumentos ligeros sobre tensores uint8: volteo, rotación pequeña y color.

    No hay recorte (`RandomResizedCrop` / `RandomCrop`): el letterbox ya dejó un
    cuadrado 224×224 conservando toda la imagen.
    """
    strength = train_config.color_jitter_strength
    return T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomRotation(train_config.rotation_degrees, fill=0),
            T.ColorJitter(brightness=strength, contrast=strength, saturation=strength),
        ]
    )


class FundusDataset(Dataset):
    """Imágenes ya preprocesadas (uint8) + etiquetas, con aumentos opcionales."""

    def __init__(
        self,
        images: torch.Tensor,
        targets: np.ndarray,
        transform: T.Compose | None = None,
    ) -> None:
        self.images = images
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def make_train_loader(
    images: torch.Tensor, targets: np.ndarray, config: ProjectConfig
) -> DataLoader:
    dataset = FundusDataset(images, targets, build_train_augmentation(config.train))
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
