"""Grad-CAM manual (hooks de PyTorch) para el modelo binario Disease_Risk."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING_RANDOM_CROP,
    get_eval_transforms,
)


class GradCAM:
    """Grad-CAM sobre una capa convolucional de un modelo con una sola salida (logit).

    CAM = ReLU( sum_k  alpha_k * A_k ),  alpha_k = media espacial de dLogit/dA_k.
    El objetivo es el propio logit: resalta las regiones que AUMENTAN la probabilidad
    de Disease_Risk = 1.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.layer4[-1]
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output: torch.Tensor) -> None:
        self._activations = output.detach()
        # El gradiente se captura con un hook sobre el propio tensor de activación
        output.register_hook(lambda gradient: setattr(self, "_gradients", gradient.detach()))

    def remove_hooks(self) -> None:
        self._handle.remove()

    def __call__(self, image_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """Devuelve (mapa CAM en [0, 1] con el tamaño de la imagen, probabilidad)."""
        device = next(self.model.parameters()).device
        self.model.eval()
        # requires_grad en la entrada garantiza grafo aunque el backbone esté congelado
        batch = image_tensor.unsqueeze(0).to(device).requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        logit = self.model(batch).view(-1)[0]
        logit.backward()

        channel_weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((channel_weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalización min-max por imagen: solo informa de la forma relativa del mapa
        cam_range = cam.max() - cam.min()
        cam = (cam - cam.min()) / cam_range if cam_range > 0 else np.zeros_like(cam)
        probability = float(torch.sigmoid(logit.detach()).item())
        return cam, probability


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Tensor normalizado (3, H, W) -> imagen RGB float (H, W, 3) en [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image_tensor.detach().cpu() * std + mean).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Mezcla el mapa de calor (colormap jet) con la imagen RGB original."""
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1.0 - alpha) * image_rgb + alpha * heatmap, 0.0, 1.0)


def load_eval_tensor(
    image_path: str,
    image_size: int,
    preprocessing: str = PREPROCESSING_RANDOM_CROP,
) -> torch.Tensor:
    """Carga una imagen con la misma transformación determinista de val/test."""
    image = Image.open(image_path).convert("RGB")
    return get_eval_transforms(image_size, preprocessing)(image)


def select_example_indices(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    examples_per_group: int = 4,
) -> dict[str, list[int]]:
    """Elige ejemplos TP / TN / FP / FN, los más confiados de cada grupo.

    Los más confiados muestran lo que el modelo considera evidencia clara;
    en FP/FN son los errores "más seguros" del modelo, los más informativos.
    """
    predictions = probabilities >= threshold
    positives = labels.astype(bool)

    groups = {
        "TP": predictions & positives,
        "TN": ~predictions & ~positives,
        "FP": predictions & ~positives,
        "FN": ~predictions & positives,
    }
    selected: dict[str, list[int]] = {}
    for group_name, mask in groups.items():
        indices = np.flatnonzero(mask)
        # Confianza = distancia de la probabilidad a la clase opuesta
        confidence = probabilities[indices] if group_name in ("TP", "FP") else 1 - probabilities[indices]
        ordered = indices[np.argsort(-confidence)]
        selected[group_name] = ordered[:examples_per_group].tolist()
    return selected


def compute_gradcam_samples(
    model: nn.Module,
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    image_size: int,
    examples_per_group: int = 4,
    preprocessing: str = PREPROCESSING_RANDOM_CROP,
) -> list[dict]:
    """Calcula Grad-CAM para los ejemplos seleccionados (TP, TN, FP, FN).

    `preprocessing` debe coincidir con el usado al entrenar/evaluar el modelo
    (`config.preprocessing`) para que el mapa se calcule sobre la misma entrada.
    """
    selected = select_example_indices(probabilities, labels, threshold, examples_per_group)
    grad_cam = GradCAM(model)
    samples: list[dict] = []
    try:
        for group_name, indices in selected.items():
            for index in indices:
                image_path = dataframe.iloc[index]["image_path"]
                image_tensor = load_eval_tensor(image_path, image_size, preprocessing)
                cam, probability = grad_cam(image_tensor)
                samples.append(
                    {
                        "group": group_name,
                        "image_name": Path(image_path).name,
                        "image": denormalize_image(image_tensor),
                        "cam": cam,
                        "probability": probability,
                        "true_label": int(labels[index]),
                    }
                )
    finally:
        grad_cam.remove_hooks()
    return samples


def plot_gradcam_grid(
    samples: list[dict],
    title: str,
    save_path: str | Path | None = None,
    columns: int = 2,
):
    """Rejilla de pares (original | superposición Grad-CAM) con probabilidad y etiqueta."""
    if not samples:
        return None

    rows = int(np.ceil(len(samples) / columns))
    figure, axes = plt.subplots(rows, columns * 2, figsize=(4.0 * columns * 2, 4.0 * rows))
    axes = np.atleast_2d(axes)

    for axis in axes.ravel():
        axis.axis("off")

    for position, sample in enumerate(samples):
        row, pair = divmod(position, columns)
        original_axis = axes[row, pair * 2]
        overlay_axis = axes[row, pair * 2 + 1]
        original_axis.imshow(sample["image"])
        original_axis.set_title(f"{sample['group']} | {sample['image_name']}", fontsize=9)
        overlay_axis.imshow(overlay_heatmap(sample["image"], sample["cam"]))
        overlay_axis.set_title(
            f"P(riesgo)={sample['probability']:.3f} | real={sample['true_label']}",
            fontsize=9,
        )

    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=110, bbox_inches="tight")
    return figure
