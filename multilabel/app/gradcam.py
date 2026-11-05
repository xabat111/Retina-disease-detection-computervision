"""Grad-CAM (tokens de norm1 del ultimo bloque) sobre un miembro del ensemble; un forward, todas las clases."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import settings  # noqa: F401
from retina_multilabel.explain import overlay_heatmap


def compute_gradcam_maps(model, normalized_image: torch.Tensor, class_indices: list[int]) -> dict[int, np.ndarray]:
    prefix_tokens = getattr(model.backbone, "num_prefix_tokens", 1)
    captured: dict[str, torch.Tensor] = {}
    hook = model.backbone.blocks[-1].norm1.register_forward_hook(lambda m, i, o: captured.__setitem__("t", o))
    try:
        with torch.enable_grad():
            x = normalized_image.clone().requires_grad_(True)
            logits = model(x)
            tokens = captured["t"]
            maps: dict[int, np.ndarray] = {}
            n = tokens.shape[1] - prefix_tokens; side = int(round(n ** 0.5))
            for c in class_indices:
                g = torch.autograd.grad(logits[0, c], tokens, retain_graph=True)[0]
                w = g[0, prefix_tokens:].mean(dim=0)
                rel = F.relu((tokens[0, prefix_tokens:] * w).sum(dim=-1)).detach()
                maps[c] = _normalize_to_unit_range(rel[: side * side].reshape(side, side).cpu().numpy())
    finally:
        hook.remove()
    return maps


def _normalize_to_unit_range(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap - heatmap.min(); peak = heatmap.max()
    return heatmap / peak if peak > 1e-12 else np.zeros_like(heatmap)


def render_overlay(image_uint8: torch.Tensor, heatmap: np.ndarray, alpha: float, mask_background: bool) -> np.ndarray:
    original = image_uint8.permute(1, 2, 0).numpy()
    if not heatmap.any():
        return original.copy()
    blended = overlay_heatmap(original, torch.from_numpy(heatmap), alpha=alpha, mask_background=mask_background)
    return (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)
