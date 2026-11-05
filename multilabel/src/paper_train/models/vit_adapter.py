"""Vision Transformer adapter for the clustered multilabel task (same protocol as papers)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retina_multilabel.config import ModelConfig  # noqa: E402
from retina_multilabel.model import build_backbone, unfreeze_last_vit_blocks  # noqa: E402


class VitClusterHead(nn.Module):
    """Frozen or partially-unfrozen ViT/DINOv2 + linear multilabel head."""

    def __init__(self, backbone: nn.Module, num_features: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_features, num_classes))

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return self.head(feats)


def build_vit_cluster_model(
    num_classes: int,
    *,
    backbone_choice: str = "dinov2",
    freeze_backbone: bool = True,
    img_size: int = 392,
    n_unfrozen_blocks: int = 0,
) -> VitClusterHead:
    """Build ViT/DINOv2 multilabel model.

    DINOv2 native timm size is 518; BaselineV2 / paper-compare use 392
    (392÷14=28). Pass ``dynamic_img_size`` via ``build_backbone``.
    ``n_unfrozen_blocks>0`` fine-tunes only the last N blocks (+ final norm).
    """
    choice = backbone_choice if backbone_choice in {"dinov2", "public"} else "dinov2"
    cfg = replace(ModelConfig(), backbone_choice=choice, img_size=img_size)
    bundle = build_backbone(cfg, hf_token=None)
    backbone = bundle.backbone
    if not freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = True
    elif n_unfrozen_blocks > 0 and hasattr(backbone, "blocks"):
        unfreeze_last_vit_blocks(backbone, n_unfrozen_blocks)
    return VitClusterHead(backbone, bundle.num_features, num_classes)
