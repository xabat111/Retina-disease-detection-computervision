#!/usr/bin/env python3
"""Verify the three paper taxonomies under data/taxonomies/.

Prints label lists and per-split positive counts. Exit 0 iff all shipped
files are present and have the expected shapes (7 / 10 / 14 labels,
1920/640/640 rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAX = ROOT / "data" / "taxonomies"
EXPECTED = {
    "odir7": 7,
    "cluster_c": 10,
    "supervised_k12_sano": 14,
}


def check_one(name: str, k: int, verbose: bool = True) -> bool:
    root = TAX / name
    cmap_p = root / "cluster_map.json"
    if not cmap_p.is_file():
        print("MISSING", cmap_p)
        return False
    cmap = json.loads(cmap_p.read_text(encoding="utf-8"))
    labels = cmap.get("cluster_names_ordered")
    if not labels:
        labels = [cmap["cluster_id_to_name"][str(i)] for i in range(cmap["k"])]
    if len(labels) != k:
        print(f"BAD {name}: expected {k} labels, got {len(labels)}")
        return False
    if verbose:
        print(f"\n== {name} ({k} labels) ==")
        print(" ", labels)
        col2 = cmap.get("column_to_cluster_id") or {}
        id2 = cmap.get("cluster_id_to_name") or {str(i): n for i, n in enumerate(labels)}
        print(f"  RFMiD columns mapped: {len(col2)}")
        if verbose and col2:
            # short preview
            preview = list(col2.items())[:5]
            for col, cid in preview:
                print(f"    {col} -> {id2.get(str(cid), cid)}")
            if len(col2) > 5:
                print(f"    ... ({len(col2) - 5} more)")
    ok = True
    for split, n_expect in (("train", 1920), ("val", 640), ("test", 640)):
        csv = root / "labels" / f"clustered_{split}.csv"
        if not csv.is_file():
            print("MISSING", csv)
            ok = False
            continue
        df = pd.read_csv(csv)
        if len(df) != n_expect:
            print(f"BAD {name}/{split}: rows={len(df)} expected {n_expect}")
            ok = False
        missing = [l for l in labels if l not in df.columns]
        if missing:
            print(f"BAD {name}/{split}: missing columns {missing}")
            ok = False
        if verbose and ok:
            pos = {l: int(df[l].sum()) for l in labels}
            print(f"  {split}: n={len(df)} positives={pos}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--rebuild-odir7",
        action="store_true",
        help="Not implemented here — shipped CSVs reproduce the paper. "
        "See notebooks/01_build_taxonomies.ipynb / retina_no_github builders.",
    )
    args = ap.parse_args()
    if args.rebuild_odir7:
        print(
            "Shipped CSVs are the paper source of truth. "
            "Encoder retrain scripts live under retina_no_github/compact_aside/.../training/",
            file=sys.stderr,
        )
        return 2
    if not TAX.is_dir():
        print("MISSING", TAX, file=sys.stderr)
        return 1
    # also require clustering_C_supervised centroids for Cluster-C provenance
    cc = TAX / "clustering_C_supervised"
    extra_ok = (cc / "centroids.npy").is_file() and (cc / "cluster_map.json").is_file()
    if not args.quiet:
        print("clustering_C_supervised centroids:", "OK" if extra_ok else "MISSING")
    ok = all(check_one(name, k, verbose=not args.quiet) for name, k in EXPECTED.items()) and extra_ok
    if ok:
        print("\nAll taxonomies OK.")
        return 0
    print("\nTaxonomy check FAILED.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
