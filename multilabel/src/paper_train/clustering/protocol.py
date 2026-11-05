"""Shared centroid + agglomerative clustering protocol for approaches A/B/C."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..encoders import EncoderBundle
from ..rfmid_columns import META_COLUMNS, disease_columns_present

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retina_multilabel.data import locate_split_files  # noqa: E402
from retina_multilabel.utils import resolve_dataset_search_roots  # noqa: E402


@dataclass
class ClusteringResult:
    k: int
    silhouette: float
    labels: dict[str, int]
    centroids: np.ndarray
    class_names: list[str]
    encoder: str
    n_images_used: dict[str, int]


class _PositiveImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def index_images(image_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in image_dir.iterdir():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            out[path.stem] = path
    return out


@torch.inference_mode()
def embed_paths(
    paths: list[Path],
    bundle: EncoderBundle,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not paths:
        return np.zeros((0, bundle.embed_dim), dtype=np.float32)
    loader = DataLoader(
        _PositiveImageDataset(paths, bundle.transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    chunks: list[torch.Tensor] = []
    model = bundle.model.to(device).eval()
    for batch in loader:
        batch = batch.to(device)
        z = model(batch)
        z = torch.nn.functional.normalize(z.float(), dim=1)
        chunks.append(z.cpu())
    return torch.cat(chunks, dim=0).numpy()


def compute_class_centroids(
    labels_csv: Path,
    image_dir: Path,
    bundle: EncoderBundle,
    *,
    max_pos_per_class: int = 64,
    batch_size: int = 32,
    device: str = "auto",
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Mean L2-normalized embedding of up to N positives per disease column."""
    df = pd.read_csv(labels_csv)
    cols = disease_columns_present(list(df.columns))
    if not cols:
        raise RuntimeError(f"No RFMiD disease columns found in {labels_csv}")

    id_col = "ID" if "ID" in df.columns else df.columns[0]
    images = index_images(image_dir)
    torch_device = resolve_device(device)

    centroids: list[np.ndarray] = []
    kept: list[str] = []
    n_used: dict[str, int] = {}

    for col in tqdm(cols, desc="class centroids"):
        pos_ids = df.loc[df[col].astype(int) == 1, id_col].astype(str).tolist()
        paths = [images[i] for i in pos_ids if i in images][:max_pos_per_class]
        if len(paths) < 2:
            continue
        emb = embed_paths(paths, bundle, torch_device, batch_size)
        centroid = emb.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid.astype(np.float32))
        kept.append(col)
        n_used[col] = len(paths)

    if len(kept) < 3:
        raise RuntimeError("Fewer than 3 disease columns had enough positives for centroids.")

    return kept, np.stack(centroids, axis=0), n_used


def select_k_and_cluster(
    centroids: np.ndarray,
    class_names: list[str],
    *,
    k_min: int = 6,
    k_max: int = 12,
    preferred_k_min: int = 8,
    preferred_k_max: int = 10,
) -> tuple[int, np.ndarray, dict[int, float]]:
    """Agglomerative clustering on cosine distance; prefer silhouette in 8–10 band."""
    n = len(class_names)
    k_max = min(k_max, n - 1)
    k_min = max(2, min(k_min, k_max))
    dist = pairwise_distances(centroids, metric="cosine")
    scores: dict[int, float] = {}
    best_k = k_min
    best_score = -1.0
    best_labels: np.ndarray | None = None

    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(dist, labels, metric="precomputed"))
        scores[k] = sil
        in_band = preferred_k_min <= k <= preferred_k_max
        best_in_band = preferred_k_min <= best_k <= preferred_k_max
        better = False
        if best_labels is None:
            better = True
        elif in_band and not best_in_band:
            better = True
        elif in_band == best_in_band and sil > best_score:
            better = True
        if better:
            best_k, best_score, best_labels = k, sil, labels

    assert best_labels is not None
    return best_k, best_labels, scores


def build_cluster_map(
    class_names: list[str],
    cluster_ids: np.ndarray,
) -> dict[str, dict]:
    """column → cluster_id, plus reverse lists and human names."""
    forward = {name: int(cid) for name, cid in zip(class_names, cluster_ids)}
    reverse: dict[str, list[str]] = {}
    for name, cid in forward.items():
        reverse.setdefault(str(cid), []).append(name)
    cluster_names = {
        str(cid): "cluster_" + "_".join(members[:3]) + (f"_n{len(members)}" if len(members) > 3 else "")
        for cid, members in reverse.items()
    }
    return {
        "column_to_cluster_id": forward,
        "cluster_id_to_columns": reverse,
        "cluster_id_to_name": cluster_names,
        "cluster_names_ordered": [cluster_names[str(i)] for i in range(len(reverse))],
    }


def apply_cluster_map_to_frame(df: pd.DataFrame, cluster_map: dict) -> pd.DataFrame:
    """OR fine-grained columns into cluster multilabels."""
    id_col = "ID" if "ID" in df.columns else df.columns[0]
    col_to_id: dict[str, int] = cluster_map["column_to_cluster_id"]
    n_clusters = len(cluster_map["cluster_names_ordered"])
    names = cluster_map["cluster_names_ordered"]

    out = pd.DataFrame({id_col: df[id_col]})
    for cid in range(n_clusters):
        members = [c for c, i in col_to_id.items() if i == cid and c in df.columns]
        if members:
            out[names[cid]] = (df[members].astype(int).sum(axis=1) > 0).astype(int)
        else:
            out[names[cid]] = 0

    if "Disease_Risk" in df.columns:
        out["Disease_Risk"] = df["Disease_Risk"].astype(int)
    return out


def attach_rare_columns(
    cluster_map: dict,
    all_cols: list[str],
    n_used: dict[str, int],
) -> list[str]:
    """Attach columns skipped for rare positives to the largest cluster."""
    missing = [c for c in all_cols if c not in cluster_map["column_to_cluster_id"]]
    if not missing:
        return []
    sizes = {int(cid): len(cols) for cid, cols in cluster_map["cluster_id_to_columns"].items()}
    dump_cid = max(sizes, key=sizes.get)
    for col in missing:
        cluster_map["column_to_cluster_id"][col] = dump_cid
        cluster_map["cluster_id_to_columns"][str(dump_cid)].append(col)
        n_used[col] = 0
    return missing


def run_clustering_with_bundle(
    output_dir: Path,
    bundle: EncoderBundle,
    *,
    approach: str,
    max_pos_per_class: int = 64,
    batch_size: int = 32,
    device: str = "auto",
    k_min: int = 6,
    k_max: int = 12,
    preferred_k_min: int = 8,
    preferred_k_max: int = 10,
    labels_subdir: str | None = None,
    extra_meta: dict | None = None,
) -> ClusteringResult:
    """Full protocol: centroids → agglomerative → cluster_map + clustered CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels" if labels_subdir is None else Path(labels_subdir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    roots = resolve_dataset_search_roots()
    train_files = locate_split_files("train", roots)

    class_names, centroids, n_used = compute_class_centroids(
        train_files.labels_csv,
        train_files.image_dir,
        bundle,
        max_pos_per_class=max_pos_per_class,
        batch_size=batch_size,
        device=device,
    )
    k, labels, scores = select_k_and_cluster(
        centroids,
        class_names,
        k_min=k_min,
        k_max=k_max,
        preferred_k_min=preferred_k_min,
        preferred_k_max=preferred_k_max,
    )
    cluster_map = build_cluster_map(class_names, labels)

    df_train = pd.read_csv(train_files.labels_csv)
    all_cols = disease_columns_present(list(df_train.columns))
    missing = attach_rare_columns(cluster_map, all_cols, n_used)

    result = ClusteringResult(
        k=k,
        silhouette=float(scores.get(k, float("nan"))),
        labels=cluster_map["column_to_cluster_id"],
        centroids=centroids,
        class_names=class_names,
        encoder=bundle.name,
        n_images_used=n_used,
    )

    payload = {
        "approach": approach,
        "encoder": bundle.name,
        "k": k,
        "silhouette": result.silhouette,
        "silhouette_by_k": {str(kk): vv for kk, vv in scores.items()},
        "n_images_used": n_used,
        "skipped_rare_columns": missing,
        **cluster_map,
        "meta_columns": list(META_COLUMNS),
    }
    if extra_meta:
        payload["extra"] = extra_meta

    map_path = output_dir / "cluster_map.json"
    map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(output_dir / "centroids.npy", centroids)
    pd.DataFrame(
        {
            "column": class_names,
            "cluster_id": [int(x) for x in labels],
            "cluster_name": [cluster_map["cluster_id_to_name"][str(int(x))] for x in labels],
            "n_images_used": [n_used[c] for c in class_names],
        }
    ).to_csv(output_dir / "column_cluster_assignment.csv", index=False)

    for split in ("train", "val", "test"):
        files = locate_split_files(split, roots)
        raw = pd.read_csv(files.labels_csv)
        clustered = apply_cluster_map_to_frame(raw, cluster_map)
        clustered.to_csv(labels_dir / f"clustered_{split}.csv", index=False)

    summary = {
        "approach": approach,
        "k": k,
        "silhouette": result.silhouette,
        "encoder": bundle.name,
        "n_columns_clustered": len(class_names),
        "n_rare_attached": len(missing),
        "cluster_names": cluster_map["cluster_names_ordered"],
        "cluster_map": str(map_path),
        "labels_dir": str(labels_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result
