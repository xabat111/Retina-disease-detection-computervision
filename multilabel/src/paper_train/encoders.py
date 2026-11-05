"""Frozen encoders for latent class centroids (ImageNet ResNet50 or DINOv2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms as T


@dataclass
class EncoderBundle:
    name: str
    model: nn.Module
    transform: T.Compose
    embed_dim: int
    img_size: int


class _ResNetEmbed(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))
        self.embed_dim = int(backbone.fc.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return z.flatten(1)


def build_encoder(name: str = "resnet50_imagenet", img_size: int = 224) -> EncoderBundle:
    name = name.lower().strip()
    if name in {"resnet50", "resnet50_imagenet"}:
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        backbone = tvm.resnet50(weights=weights)
        model = _ResNetEmbed(backbone)
        # ImageNet stats (meta keys vary across torchvision versions).
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        return EncoderBundle(
            name="resnet50_imagenet",
            model=model.eval(),
            transform=transform,
            embed_dim=model.embed_dim,
            img_size=img_size,
        )

    if name.startswith("dinov2"):
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("DINOv2 encoder requires timm") from exc
        model_name = "vit_base_patch14_dinov2.lvd142m"
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        cfg = backbone.pretrained_cfg
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        return EncoderBundle(
            name="dinov2_vitb14",
            model=backbone.eval(),
            transform=transform,
            embed_dim=int(backbone.num_features),
            img_size=img_size,
        )

    raise ValueError(f"Unknown encoder: {name}")
