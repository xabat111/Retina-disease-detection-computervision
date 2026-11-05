"""Backward-compatible entry for Approach A (frozen ResNet50 ImageNet).

Prefer `src.clustering.baseline_resnet` / `src.clustering.protocol` for new code.
"""

from __future__ import annotations

from pathlib import Path

from .clustering.protocol import (
    ClusteringResult,
    apply_cluster_map_to_frame,
    build_cluster_map,
    select_k_and_cluster,
)
from .encoders import build_encoder


def compute_class_centroids(
    labels_csv: Path,
    image_dir: Path,
    *,
    encoder_name: str = "resnet50_imagenet",
    img_size: int = 224,
    max_pos_per_class: int = 64,
    batch_size: int = 32,
    device: str = "auto",
):
    from .clustering.protocol import compute_class_centroids as _compute

    bundle = build_encoder(encoder_name, img_size=img_size)
    kept, centroids, n_used = _compute(
        labels_csv,
        image_dir,
        bundle,
        max_pos_per_class=max_pos_per_class,
        batch_size=batch_size,
        device=device,
    )
    return kept, centroids, n_used, bundle


def run_clustering_pipeline(
    output_dir: Path,
    *,
    encoder_name: str = "resnet50_imagenet",
    img_size: int = 224,
    max_pos_per_class: int = 64,
    batch_size: int = 32,
    device: str = "auto",
    k_min: int = 6,
    k_max: int = 12,
    preferred_k_min: int = 8,
    preferred_k_max: int = 10,
) -> ClusteringResult:
    """Legacy API: writes labels next to output_dir (outputs/labels) for old layout."""
    from .clustering.protocol import run_clustering_with_bundle
    from .encoders import build_encoder as _build

    output_dir = Path(output_dir)
    # Preserve historical layout: outputs/clustering + sibling outputs/labels
    legacy_labels = output_dir.parent / "labels"
    bundle = _build(encoder_name, img_size=img_size)
    return run_clustering_with_bundle(
        output_dir,
        bundle,
        approach=f"A_baseline_{bundle.name}",
        max_pos_per_class=max_pos_per_class,
        batch_size=batch_size,
        device=device,
        k_min=k_min,
        k_max=k_max,
        preferred_k_min=preferred_k_min,
        preferred_k_max=preferred_k_max,
        labels_subdir=str(legacy_labels),
    )


__all__ = [
    "ClusteringResult",
    "apply_cluster_map_to_frame",
    "build_cluster_map",
    "compute_class_centroids",
    "run_clustering_pipeline",
    "select_k_and_cluster",
]
