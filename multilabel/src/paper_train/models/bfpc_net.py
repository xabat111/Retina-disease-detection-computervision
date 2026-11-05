"""BFPC-Net inspired by PMC9230753: ResNet50 + residual attention + multilabel head.

Paper uses binocular fusion on ODIR. RFMiD is monocular; we support:
- ``mode='single'``: one stream ResNet50+attention
- ``mode='dual_view'``: two augmented views fused (paper FFM with w=0.5/0.5)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ChannelAttention(nn.Module):
    """Paper Eq.7 simplified: GAP → FC → ReLU → scale channels."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        gap = F.adaptive_avg_pool2d(x, 1).view(b, c)
        scale = self.fc(gap).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Paper Eq.8: pool → conv → residual-style spatial gate."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        gate = self.act(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class ResidualAttentionBlock(nn.Module):
    """RAM: channel + spatial attention with residual add (paper Eq.9)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.ca(x)
        u_prime = self.sa(u)
        return F.relu(u_prime + x)


class BFPCNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, mode: str = "single") -> None:
        super().__init__()
        if mode not in {"single", "dual_view"}:
            raise ValueError(mode)
        self.mode = mode
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)
        # Keep conv stem … layer4 (drop avgpool + fc)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.ram = ResidualAttentionBlock(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.ram(feat)
        return self.pool(feat).flatten(1)

    def forward(self, x: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
        z1 = self.encode(x)
        if self.mode == "dual_view" and x2 is not None:
            z2 = self.encode(x2)
            z = 0.5 * z1 + 0.5 * z2  # paper FFM weights
        else:
            z = z1
        return self.head(z)


def build_bfpc_net(num_classes: int, *, pretrained: bool = True, mode: str = "single") -> BFPCNet:
    return BFPCNet(num_classes=num_classes, pretrained=pretrained, mode=mode)
