"""Comprueba visualmente el preprocesado letterbox frente al recorte aleatorio original.

Genera `artifacts/preprocessing_check_letterbox.png` con, para varias imágenes de validación:
original | train con RandomResizedCrop (original) | train con letterbox | eval con letterbox.

Uso desde la raíz del proyecto:
    python scripts/check_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import default_config
from src.data import (
    PREPROCESSING_LETTERBOX,
    PREPROCESSING_RANDOM_CROP,
    LetterboxResize,
    build_split_dataframe,
    get_eval_transforms,
    get_train_transforms,
)
from src.explain import denormalize_image

NUM_EXAMPLES = 4
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "preprocessing_check_letterbox.png"
SEED = 0


def check_letterbox_numerically(image: Image.Image, image_size: int) -> None:
    """Verifica forma de salida, proporción conservada y relleno negro."""
    letterboxed = LetterboxResize(image_size)(image)
    pixels = np.asarray(letterboxed)
    width, height = image.size

    assert letterboxed.size == (image_size, image_size), letterboxed.size

    # Región ocupada por la imagen: filas/columnas con algún píxel no negro en la imagen original
    resized_width = round(width * image_size / max(width, height))
    resized_height = round(height * image_size / max(width, height))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    padding_mask = np.ones((image_size, image_size), dtype=bool)
    padding_mask[top : top + resized_height, left : left + resized_width] = False
    assert pixels[padding_mask].max(initial=0) == 0, "el relleno no es negro"

    original_ratio = width / height
    resized_ratio = resized_width / resized_height
    print(
        f"  original {width}x{height} (ratio {original_ratio:.3f}) -> "
        f"contenido {resized_width}x{resized_height} (ratio {resized_ratio:.3f}) "
        f"en lienzo {image_size}x{image_size}; relleno negro OK"
    )


def main() -> None:
    config = default_config()
    dataframe = build_split_dataframe(config.val_labels_csv, config.val_images_dir)
    # Mezcla sana y riesgo para ver ambas clases
    examples = dataframe.groupby("disease_risk").head(NUM_EXAMPLES // 2)
    examples = examples.sort_values("disease_risk").reset_index(drop=True)

    train_random_crop = get_train_transforms(config.image_size, PREPROCESSING_RANDOM_CROP)
    train_letterbox = get_train_transforms(config.image_size, PREPROCESSING_LETTERBOX)
    eval_letterbox = get_eval_transforms(config.image_size, PREPROCESSING_LETTERBOX)

    row_titles = [
        "original",
        "train: RandomResizedCrop (original)",
        "train: letterbox + aug",
        "eval: letterbox",
    ]
    figure, axes = plt.subplots(
        len(row_titles), len(examples), figsize=(3.2 * len(examples), 3.2 * len(row_titles))
    )

    torch.manual_seed(SEED)
    print("Comprobación numérica del letterbox:")
    for column, row in examples.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        check_letterbox_numerically(image, config.image_size)

        panels = [
            np.asarray(image),
            denormalize_image(train_random_crop(image)),
            denormalize_image(train_letterbox(image)),
            denormalize_image(eval_letterbox(image)),
        ]
        for row_index, panel in enumerate(panels):
            axis = axes[row_index, column]
            axis.imshow(panel)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(row_titles[row_index], fontsize=9)
        axes[0, column].set_title(
            f"{Path(row['image_path']).name} | riesgo={row['disease_risk']}", fontsize=9
        )

    figure.suptitle(
        f"Preprocesado a {config.image_size}px: el letterbox mantiene el ojo completo", fontsize=12
    )
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=110, bbox_inches="tight")
    print(f"Figura guardada en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
