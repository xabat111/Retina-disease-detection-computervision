"""DINOv2 + residual channel/spatial token attention (BFPC-inspired, monocular).

Justification (PMC9230753 RAM → ViT tokens):
- Paper residual attention highlights lesion channels/regions on CNN feature maps.
- On a frozen DINOv2, the same idea applied to patch tokens improves localization of
  sparse findings → higher recall without unfreezing the full ViT-L backbone.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retina_multilabel.attention_pool import ResidualAttentionTokenHead  # noqa: E402
from retina_multilabel.config import ModelConfig  # noqa: E402
from retina_multilabel.model import build_backbone, unfreeze_last_vit_blocks  # noqa: E402


class VitResidualAttnModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: ResidualAttentionTokenHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("Expected ViT token tensor from forward_features")
        return self.head(tokens)


def build_vit_residual_attn(
    num_classes: int,
    *,
    img_size: int = 392,
    freeze_backbone: bool = True,
    n_unfrozen_blocks: int = 0,
    dropout: float = 0.2,
) -> VitResidualAttnModel:
    """``n_unfrozen_blocks>0`` fine-tunes the last N transformer blocks (+ final norm);
    ``freeze_backbone=False`` unfreezes everything. Default = frozen (v1)."""
    cfg = replace(ModelConfig(), backbone_choice="dinov2", img_size=img_size)
    bundle = build_backbone(cfg, hf_token=None)
    backbone = bundle.backbone
    if not freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = True
    elif n_unfrozen_blocks > 0:
        unfreeze_last_vit_blocks(backbone, n_unfrozen_blocks)
    else:
        for p in backbone.parameters():
            p.requires_grad = False
    # Need forward_features on the timm module (may be wrapped)
    inner = backbone
    if not hasattr(inner, "forward_features"):
        raise RuntimeError("DINOv2 backbone missing forward_features")
    head = ResidualAttentionTokenHead(bundle.num_features, num_classes, dropout=dropout)
    return VitResidualAttnModel(inner, head)
