"""Explicabilidad para ViT: attention rollout, Grad-CAM sobre tokens e Integrated Gradients."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps

from .config import ExplainConfig
from .model import FrozenBackboneClassifier

try:  # captum es opcional
    from captum.attr import IntegratedGradients
except ImportError:  # pragma: no cover - depende del entorno
    IntegratedGradients = None


def is_integrated_gradients_available() -> bool:
    return IntegratedGradients is not None


def _min_max_normalize(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap - heatmap.min()
    return heatmap / (heatmap.max() + 1e-8)


class ViTExplainer:
    """Mapas de explicación de un `FrozenBackboneClassifier` para una imagen cada vez.

    Todos los métodos reciben la imagen ya normalizada (1, 3, H, W) y devuelven un mapa
    (grid, grid) con valores en [0, 1] (14x14 para ViT-L/16 a 224 px), salvo Integrated
    Gradients, que devuelve un mapa del tamaño de la imagen.
    """

    def __init__(self, model: FrozenBackboneClassifier, config: ExplainConfig) -> None:
        self.model = model.eval()
        self.config = config
        backbone = model.backbone
        self.grid_size = tuple(backbone.patch_embed.grid_size)
        self.num_prefix_tokens = getattr(backbone, "num_prefix_tokens", 1)
        self.uses_cls_token = model.bundle.uses_cls_token

    # --------------------------------------------------------------- attention rollout

    def attention_rollout(self, normalized_image: torch.Tensor) -> torch.Tensor:
        """Attention rollout (Abnar & Zuidema) sobre las últimas capas.

        Se promedian las cabezas, se suma la conexión residual (0.5·A + 0.5·I), se normalizan
        las filas y se multiplican las matrices capa a capa. Con [CLS] se lee la fila del
        token [CLS]; con pooling medio (RETFound) se promedia la fila de todos los parches.
        """
        blocks = list(self.model.backbone.blocks)[-self.config.rollout_layers :]
        attention_maps: list[torch.Tensor] = []
        previous_fused_flags = [getattr(block.attn, "fused_attn", None) for block in blocks]
        hooks = []
        for block in blocks:
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False  # el kernel fusionado no expone la atención
            hooks.append(
                block.attn.attn_drop.register_forward_hook(
                    lambda module, inputs, output: attention_maps.append(inputs[0].detach())
                )
            )
        try:
            with torch.no_grad():
                self.model(normalized_image)
        finally:
            for hook in hooks:
                hook.remove()
            for block, flag in zip(blocks, previous_fused_flags):
                if flag is not None:
                    block.attn.fused_attn = flag

        n_tokens = attention_maps[0].shape[-1]
        identity = torch.eye(n_tokens, device=attention_maps[0].device)
        rollout = identity.clone()
        for layer_attention in attention_maps:
            attention = layer_attention[0].mean(dim=0)
            attention = 0.5 * attention + 0.5 * identity
            attention = attention / attention.sum(dim=-1, keepdim=True)
            rollout = attention @ rollout

        prefix = self.num_prefix_tokens
        if self.uses_cls_token:
            patch_relevance = rollout[0, prefix:]
        else:
            patch_relevance = rollout[prefix:, prefix:].mean(dim=0)
        return _min_max_normalize(patch_relevance.reshape(self.grid_size)).cpu()

    # --------------------------------------------------------------- Grad-CAM

    def grad_cam(self, normalized_image: torch.Tensor, class_index: int) -> torch.Tensor:
        """Grad-CAM sobre los tokens que entran en la atención del último bloque transformer.

        Se usa la salida de `norm1` del último bloque (entrada de su atención) y no la salida
        del bloque: con pooling por [CLS], los tokens de parche que salen del último bloque no
        influyen en el logit y su gradiente sería exactamente cero.
        Los pesos de cada canal son el gradiente medio del logit de la clase sobre los tokens
        de parche; el mapa es ReLU(Σ_canal peso · activación) reorganizado en rejilla 14x14.
        """
        captured: dict[str, torch.Tensor] = {}
        hook = self.model.backbone.blocks[-1].norm1.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("tokens", output)
        )
        image = normalized_image.clone().requires_grad_(True)  # activa el grafo (backbone congelado)
        try:
            logits = self.model(image)
        finally:
            hook.remove()
        tokens = captured["tokens"]
        gradients = torch.autograd.grad(logits[0, class_index], tokens)[0]

        prefix = self.num_prefix_tokens
        channel_weights = gradients[0, prefix:].mean(dim=0)
        relevance = F.relu((tokens[0, prefix:] * channel_weights).sum(dim=-1)).detach()
        return _min_max_normalize(relevance.reshape(self.grid_size)).cpu()

    # --------------------------------------------------------------- Integrated Gradients

    def integrated_gradients(self, normalized_image: torch.Tensor, class_index: int) -> torch.Tensor:
        """Integrated Gradients (captum) con línea base negra; devuelve |atribución| por píxel."""
        if IntegratedGradients is None:
            raise ImportError("captum no está instalado (pip install captum).")
        black_baseline = self.model.normalize(torch.zeros_like(normalized_image, dtype=torch.uint8))
        explainer = IntegratedGradients(lambda inputs: self.model(inputs)[:, class_index])
        attribution = explainer.attribute(
            normalized_image,
            baselines=black_baseline,
            n_steps=self.config.integrated_gradients_steps,
            internal_batch_size=4,
        )
        pixel_relevance = attribution.abs().sum(dim=1)[0].detach().cpu()
        clip_value = torch.quantile(pixel_relevance.flatten(), 0.99)
        return _min_max_normalize(pixel_relevance.clamp(max=clip_value))


BACKGROUND_INTENSITY_THRESHOLD = 12


def field_of_view_mask(image_uint8: np.ndarray) -> torch.Tensor:
    """Máscara (H, W) del campo de visión: píxeles no negros, algo dilatados para cerrar huecos."""
    foreground = torch.from_numpy(image_uint8.max(axis=2) > BACKGROUND_INTENSITY_THRESHOLD)
    dilated = F.max_pool2d(foreground[None, None].float(), kernel_size=9, stride=1, padding=4)
    return dilated[0, 0]


def overlay_heatmap(
    image_uint8: np.ndarray,
    heatmap: torch.Tensor,
    alpha: float = 0.45,
    mask_background: bool = True,
) -> np.ndarray:
    """Superpone un mapa [0,1] (cualquier resolución) sobre la imagen HWC uint8. Devuelve RGB [0,1].

    Con `mask_background` el mapa se limita al campo de visión y se renormaliza dentro de él:
    los ViT suelen concentrar atención en tokens "artefacto" del fondo negro, que no son
    informativos y taparían la señal real.
    """
    height, width = image_uint8.shape[:2]
    resized = F.interpolate(
        heatmap[None, None].float(), size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    blend_weight = torch.full((height, width), alpha)
    if mask_background:
        mask = field_of_view_mask(image_uint8)
        resized = resized * mask
        blend_weight = blend_weight * mask
    colored = colormaps["jet"](_min_max_normalize(resized).numpy())[..., :3]
    weight = blend_weight.numpy()[..., None]
    return (1 - weight) * image_uint8 / 255.0 + weight * colored


def select_xai_cases(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
    class_index: int,
    cases_per_outcome: int,
) -> list[tuple[int, str]]:
    """Índices de verdaderos positivos (más seguros) y falsos negativos (más fallados).

    Es una selección deliberadamente extrema (mejores aciertos / peores fallos) para ver
    qué mira el modelo en los casos más claros y más problemáticos; no es una muestra
    aleatoria representativa.
    """
    positives = np.where(targets[:, class_index] == 1)[0]
    class_probabilities = probabilities[positives, class_index]
    detected = class_probabilities >= thresholds[class_index]
    true_positives = positives[detected][np.argsort(-class_probabilities[detected])]
    false_negatives = positives[~detected][np.argsort(class_probabilities[~detected])]
    return [(int(i), "TP") for i in true_positives[:cases_per_outcome]] + [
        (int(i), "FN") for i in false_negatives[:cases_per_outcome]
    ]


def explain_case(
    explainer: ViTExplainer,
    image_uint8: torch.Tensor,
    class_index: int,
    include_integrated_gradients: bool,
) -> tuple[list[np.ndarray], list[str]]:
    """Imágenes RGB [0,1] (original + una por método) y sus títulos para una imagen (3, H, W)."""
    device = next(explainer.model.parameters()).device
    original = image_uint8.permute(1, 2, 0).numpy()
    normalized = explainer.model.normalize(image_uint8[None].to(device))

    images = [original / 255.0]
    titles = ["Original"]
    maps = [
        ("Attention rollout", explainer.attention_rollout(normalized)),
        ("Grad-CAM (entrada del último bloque)", explainer.grad_cam(normalized, class_index)),
    ]
    if include_integrated_gradients:
        maps.append(("Integrated Gradients", explainer.integrated_gradients(normalized, class_index)))
    for title, heatmap in maps:
        images.append(
            overlay_heatmap(original, heatmap, mask_background=explainer.config.mask_background)
        )
        titles.append(title)
    return images, titles


def explain_case_side_by_side(
    baseline_explainer: ViTExplainer,
    finetuned_explainer: ViTExplainer,
    image_uint8: torch.Tensor,
    class_index: int,
    include_integrated_gradients: bool,
) -> tuple[list[np.ndarray], list[str]]:
    """Misma imagen: original + rollout/Grad-CAM (y IG) del baseline y del modelo ajustado."""
    baseline_images, baseline_titles = explain_case(
        baseline_explainer, image_uint8, class_index, include_integrated_gradients
    )
    finetuned_images, _ = explain_case(
        finetuned_explainer, image_uint8, class_index, include_integrated_gradients
    )
    images = [baseline_images[0]]
    titles = ["Original"]
    for index, title in enumerate(baseline_titles[1:], start=1):
        images.append(baseline_images[index])
        titles.append(f"Baseline · {title}")
        images.append(finetuned_images[index])
        titles.append(f"Finetune · {title}")
    return images, titles
