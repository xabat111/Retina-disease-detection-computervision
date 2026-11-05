"""Image augmentation module (IAM) from PMC9230753.

Weighted enhancement: I_weight = α·I_org + β·I_blur + γ
with α=4, β=-4, γ=128, Gaussian kernel 63×63, σ=10.
Uses PIL + torchvision (no OpenCV dependency).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def weighted_enhancement(
    image: Image.Image | np.ndarray,
    *,
    alpha: float = 4.0,
    beta: float = -4.0,
    gamma: float = 128.0,
    ksize: int = 63,
    sigma: float = 10.0,
) -> Image.Image:
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        arr = image.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pil = Image.fromarray(arr)

    # Approximate Gaussian blur; PIL radius ≈ σ
    blur = pil.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    org = np.asarray(pil, dtype=np.float32)
    blu = np.asarray(blur, dtype=np.float32)
    # ksize retained for API parity with the paper (PIL uses sigma)
    _ = ksize
    out = alpha * org + beta * blu + gamma
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
