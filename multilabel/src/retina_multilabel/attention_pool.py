"""Cabeza DINOv2 inspirada en la atención residual y fusión del paper BFPC-Net.

RFMiD se trata como una imagen monocular; esta cabeza combina el token global CLS
con un resumen aprendido de los tokens espaciales, sin asumir que existe la foto
del otro ojo. El backbone ViT y su esquema de ajuste fino siguen siendo los mismos.
"""

from __future__ import annotations

import torch
from torch import nn

from .labels import PATHOLOGY_LABEL_NAMES
from .model import BackboneBundle, FinetuneClassifier


class ResidualAttentionTokenHead(nn.Module):
    """Atención de canal/espacio con salto residual y fusión global-local."""

    def __init__(
        self,
        width: int,
        num_classes: int,
        dropout: float = 0.2,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        hidden = max(width // reduction, 32)
        self.channel_mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        self.spatial_gate = nn.Linear(2, 1)
        self.token_score = nn.Linear(width, 1)
        self.output_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise ValueError(
                "La cabeza de atención espera tokens ViT (batch, 1+parches, canales)."
            )

        cls_token = tokens[:, 0]
        patches = tokens[:, 1:]

        # Channel attention from complementary average/max patch descriptors.
        channel_avg = patches.mean(dim=1)
        channel_max = patches.amax(dim=1)
        channel_weights = torch.sigmoid(
            self.channel_mlp(channel_avg) + self.channel_mlp(channel_max)
        ).unsqueeze(1)
        channel_features = patches * channel_weights

        # Spatial attention assigns a learned gate to each patch, then preserves
        # the original representation through a residual connection.
        spatial_descriptors = torch.stack(
            (channel_features.mean(dim=-1), channel_features.amax(dim=-1)), dim=-1
        )
        spatial_weights = torch.sigmoid(self.spatial_gate(spatial_descriptors))
        attended = patches + channel_features * spatial_weights

        pooling_weights = torch.softmax(self.token_score(attended).squeeze(-1), dim=1)
        local_feature = torch.sum(attended * pooling_weights.unsqueeze(-1), dim=1)
        fused_feature = cls_token + local_feature
        return self.classifier(self.dropout(self.output_norm(fused_feature)))


class ResidualAttentionFinetuneClassifier(FinetuneClassifier):
    """ViT-L parcial + cabeza de atención residual de una sola imagen."""

    checkpoint_head_type = "residual_attention_pool"

    def __init__(
        self,
        bundle: BackboneBundle,
        n_unfrozen_blocks: int,
        num_classes: int = len(PATHOLOGY_LABEL_NAMES),
        dropout: float = 0.2,
    ) -> None:
        super().__init__(bundle, n_unfrozen_blocks, num_classes)
        self.head = ResidualAttentionTokenHead(bundle.num_features, num_classes, dropout)

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(normalized_images)
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("El backbone debe devolver un tensor de tokens ViT.")
        return self.head(tokens)
