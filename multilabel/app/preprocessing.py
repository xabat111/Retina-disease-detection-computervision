"""Decodificacion y preprocesado: resize bicubico plano (igual que en entrenamiento odir7, sin letterbox)."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except Exception as error:
        raise ValueError(f"The file is not a readable image ({type(error).__name__}).") from error


def preprocess_fundus_image(image: Image.Image, image_size: int) -> torch.Tensor:
    """Resize bicubico a image_size^2 -> tensor uint8 (3, H, W) (solo para mostrar / Grad-CAM)."""
    array = np.array(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC), dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
