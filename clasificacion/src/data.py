"""Dataset, transforms y DataLoaders para RFMiD Disease_Risk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import TrainConfig

IMAGE_EXTENSION = ".png"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Valores válidos de TrainConfig.preprocessing
PREPROCESSING_RANDOM_CROP = "random_crop"
PREPROCESSING_LETTERBOX = "letterbox"
PREPROCESSING_OPTIONS = (PREPROCESSING_RANDOM_CROP, PREPROCESSING_LETTERBOX)


def build_split_dataframe(labels_csv: Path, images_dir: Path) -> pd.DataFrame:
    """Carga etiquetas RFMiD y resuelve la ruta de cada imagen de fondo de ojo."""
    labels = pd.read_csv(labels_csv)

    image_paths = [
        str(images_dir / f"{image_id}{IMAGE_EXTENSION}")
        for image_id in labels["ID"]
    ]

    return pd.DataFrame(
        {
            "image_path": image_paths,
            "disease_risk": labels["Disease_Risk"].astype(int),
        }
    )


class LetterboxResize:
    """Escala la imagen para que su lado largo mida `image_size` y rellena con negro hasta cuadrado.

    A diferencia de `Resize((s, s))` no deforma la proporción, y a diferencia de un recorte
    aleatorio no pierde periferia, disco óptico ni mácula: el ojo siempre se ve entero.
    """

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.image_size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized_image = image.resize((resized_width, resized_height), Image.BILINEAR)

        square_canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        offset = (
            (self.image_size - resized_width) // 2,
            (self.image_size - resized_height) // 2,
        )
        square_canvas.paste(resized_image, offset)
        return square_canvas

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(image_size={self.image_size})"


def _validate_preprocessing(preprocessing: str) -> None:
    if preprocessing not in PREPROCESSING_OPTIONS:
        raise ValueError(
            f"preprocessing desconocido: {preprocessing!r}. Opciones: {PREPROCESSING_OPTIONS}"
        )


def get_train_transforms(
    image_size: int,
    preprocessing: str = PREPROCESSING_RANDOM_CROP,
) -> transforms.Compose:
    """Augmentations solo en train. Sin flip vertical (orientación del fundus importa).

    `preprocessing` solo cambia el primer paso (recorte aleatorio vs letterbox);
    el resto de augmentations es idéntico en ambos modos.
    """
    _validate_preprocessing(preprocessing)
    if preprocessing == PREPROCESSING_LETTERBOX:
        spatial_step = LetterboxResize(image_size)
    else:
        spatial_step = transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0))

    return transforms.Compose(
        [
            spatial_step,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(
    image_size: int,
    preprocessing: str = PREPROCESSING_RANDOM_CROP,
) -> transforms.Compose:
    """Resize determinista para validación y test (sin augmentations).

    Con `preprocessing="letterbox"` usa el mismo letterbox que train (sin partes aleatorias).
    """
    _validate_preprocessing(preprocessing)
    if preprocessing == PREPROCESSING_LETTERBOX:
        resize_step = LetterboxResize(image_size)
    else:
        resize_step = transforms.Resize((image_size, image_size))

    return transforms.Compose(
        [
            resize_step,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class FundusDiseaseRiskDataset(Dataset):
    """Dataset binario: imagen de fundus → disease_risk (0/1)."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(float(row["disease_risk"]), dtype=torch.float32)
        return image, label


def create_dataloaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye DataFrames y DataLoaders train / val / test."""
    train_dataframe = build_split_dataframe(config.train_labels_csv, config.train_images_dir)
    val_dataframe = build_split_dataframe(config.val_labels_csv, config.val_images_dir)
    test_dataframe = build_split_dataframe(config.test_labels_csv, config.test_images_dir)

    train_dataset = FundusDiseaseRiskDataset(
        train_dataframe,
        transform=get_train_transforms(config.image_size, config.preprocessing),
    )
    val_dataset = FundusDiseaseRiskDataset(
        val_dataframe,
        transform=get_eval_transforms(config.image_size, config.preprocessing),
    )
    test_dataset = FundusDiseaseRiskDataset(
        test_dataframe,
        transform=get_eval_transforms(config.image_size, config.preprocessing),
    )

    use_pin_memory = config.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataframe,
        val_dataframe,
        test_dataframe,
    )
