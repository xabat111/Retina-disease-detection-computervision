"""Approach A — frozen ResNet50 ImageNet embeddings for disease centroids."""

from __future__ import annotations

from pathlib import Path

from ..encoders import EncoderBundle, build_encoder
from .protocol import ClusteringResult, run_clustering_with_bundle


def build_baseline_bundle(img_size: int = 224) -> EncoderBundle:
    return build_encoder("resnet50_imagenet", img_size=img_size)


def run_baseline_clustering(
    output_dir: Path,
    *,
    img_size: int = 224,
    max_pos_per_class: int = 64,
    batch_size: int = 32,
    device: str = "auto",
    k_min: int = 6,
    k_max: int = 12,
    preferred_k_min: int = 8,
    preferred_k_max: int = 10,
) -> ClusteringResult:
    bundle = build_baseline_bundle(img_size=img_size)
    return run_clustering_with_bundle(
        output_dir,
        bundle,
        approach="A_baseline_resnet50_imagenet",
        max_pos_per_class=max_pos_per_class,
        batch_size=batch_size,
        device=device,
        k_min=k_min,
        k_max=k_max,
        preferred_k_min=preferred_k_min,
        preferred_k_max=preferred_k_max,
    )
