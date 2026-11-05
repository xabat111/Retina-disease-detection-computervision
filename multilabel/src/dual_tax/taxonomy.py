"""Resolve taxonomy paths (data/taxonomies/)."""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TAXONOMIES = {
    "odir7": REPO / "data" / "taxonomies" / "odir7",
    "cluster_c": REPO / "data" / "taxonomies" / "cluster_c",
}

def resolve_taxonomy(name: str) -> tuple[Path, Path]:
    key = name.lower().replace("-", "_")
    if key not in TAXONOMIES:
        raise SystemExit(f"Unknown taxonomy {name!r}. Choose: {list(TAXONOMIES)}")
    root = TAXONOMIES[key]
    cmap = root / "cluster_map.json"
    labels = root / "labels"
    if not cmap.is_file() or not (labels / "clustered_train.csv").is_file():
        raise SystemExit(f"Missing {root}. Run: python scripts/01_build_taxonomies.py")
    return cmap, labels
