"""Three-way latent clustering of RFMiD disease columns (A/B/C)."""

from .protocol import (
    ClusteringResult,
    apply_cluster_map_to_frame,
    build_cluster_map,
    compute_class_centroids,
    run_clustering_with_bundle,
    select_k_and_cluster,
)

__all__ = [
    "ClusteringResult",
    "apply_cluster_map_to_frame",
    "build_cluster_map",
    "compute_class_centroids",
    "run_clustering_with_bundle",
    "select_k_and_cluster",
]
