"""InceptionV3 multilabel adapter inspired by IEEE 10595740 (ODIR-5K).

Paper reports InceptionV3 / InceptionV3–ResNet50V2 with K-fold on ODIR.
Here: torchvision InceptionV3 ImageNet weights + linear multilabel head
for the clustered RFMiD label space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class InceptionOdirMultilabel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.inception_v3(weights=weights, aux_logits=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        if backbone.AuxLogits is not None:
            aux_in = backbone.AuxLogits.fc.in_features
            backbone.AuxLogits.fc = nn.Identity()
            self._aux_in = aux_in
        else:
            self._aux_in = None
        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)
        self.aux_head = nn.Linear(self._aux_in, num_classes) if self._aux_in else None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(x)
        if self.training and isinstance(out, tvm.inception.InceptionOutputs):
            main = self.head(out.logits)
            if self.aux_head is not None and out.aux_logits is not None:
                aux = self.aux_head(out.aux_logits)
                return main, aux
            return main
        # eval: InceptionOutputs or tensor
        if isinstance(out, tvm.inception.InceptionOutputs):
            feats = out.logits
        else:
            feats = out
        return self.head(feats)


def build_inception_odir(num_classes: int, *, pretrained: bool = True) -> InceptionOdirMultilabel:
    return InceptionOdirMultilabel(num_classes=num_classes, pretrained=pretrained)
