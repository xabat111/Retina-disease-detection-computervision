"""RFMiD → ODIR-8 taxonomy (PMC9230753 / ODIR-5K labels).

Paper classes: N, D, G, C, A, H, M, O.

RFMiD has no dedicated hypertension (H) column; H is kept for protocol
parity but will be all-zeros unless a future source adds it.
Cataract (C) is proxied by media-opacity columns (MH/AH/VH) — RFMiD has
no explicit cataract label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Ordered like the paper / ODIR challenge
ODIR_LABEL_ORDER: tuple[str, ...] = ("N", "D", "G", "C", "A", "H", "M", "O")

ODIR_FULL_NAMES: dict[str, str] = {
    "N": "normal",
    "D": "diabetes",
    "G": "glaucoma",
    "C": "cataract_media_opacity",
    "A": "amd",
    "H": "hypertension",
    "M": "pathological_myopia",
    "O": "other",
}

# Fine-grained RFMiD columns → ODIR super-label (OR aggregation).
RFMID_TO_ODIR: dict[str, str] = {
    # D — diabetic retinopathy family
    "DR": "D",
    "EDN": "D",
    "CWS": "D",
    # G — glaucoma / optic-nerve cupping & related
    "ODC": "G",
    "ODP": "G",
    "ODE": "G",
    "AION": "G",
    # C — cataract proxied by media opacity / haze
    "MH": "C",
    "AH": "C",
    "VH": "C",
    # A — AMD + drusen
    "ARMD": "A",
    "DN": "A",
    # H — no RFMiD equivalent (explicitly empty)
    # M — myopia family
    "MYA": "M",
    "TSLN": "M",
    "TD": "M",
    # O — everything else listed in RFMiD schema
    "BRVO": "O",
    "ERM": "O",
    "LS": "O",
    "MS": "O",
    "CSR": "O",
    "CRVO": "O",
    "TV": "O",
    "ST": "O",
    "PT": "O",
    "RT": "O",
    "RS": "O",
    "CRS": "O",
    "RPEC": "O",
    "MHL": "O",
    "RP": "O",
    "CB": "O",
    "ODPM": "O",
    "PRH": "O",
    "MNF": "O",
    "HR": "O",
    "CRAO": "O",
    "CME": "O",
    "PTCR": "O",
    "CF": "O",
    "MCA": "O",
    "VS": "O",
    "BRAO": "O",
    "PLQ": "O",
    "HPED": "O",
    "CL": "O",
}


def odir_cluster_map_payload() -> dict:
    """Same JSON shape as latent cluster_map for shared training scripts."""
    reverse: dict[str, list[str]] = {lab: [] for lab in ODIR_LABEL_ORDER}
    forward: dict[str, int] = {}
    for col, lab in RFMID_TO_ODIR.items():
        cid = ODIR_LABEL_ORDER.index(lab)
        forward[col] = cid
        reverse[lab].append(col)

    cluster_id_to_columns = {str(i): reverse[lab] for i, lab in enumerate(ODIR_LABEL_ORDER)}
    cluster_id_to_name = {str(i): lab for i, lab in enumerate(ODIR_LABEL_ORDER)}
    return {
        "scheme": "odir8_pmc9230753",
        "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9230753/",
        "notes": {
            "H": "No RFMiD hypertension column — always 0",
            "C": "Proxied by MH/AH/VH (media opacity), not true cataract",
            "N": "Derived from Disease_Risk==0 (not from disease columns)",
            "G": "ODC+optic-nerve columns as glaucoma proxy",
        },
        "k": len(ODIR_LABEL_ORDER),
        "column_to_cluster_id": forward,
        "cluster_id_to_columns": cluster_id_to_columns,
        "cluster_id_to_name": cluster_id_to_name,
        "cluster_names_ordered": list(ODIR_LABEL_ORDER),
        "full_names": ODIR_FULL_NAMES,
    }


def apply_odir_map(df: pd.DataFrame) -> pd.DataFrame:
    """Build multilabel ODIR-8 frame from raw RFMiD labels CSV."""
    id_col = "ID" if "ID" in df.columns else df.columns[0]
    out = pd.DataFrame({id_col: df[id_col]})

    for lab in ODIR_LABEL_ORDER:
        if lab == "N":
            if "Disease_Risk" in df.columns:
                out["N"] = (df["Disease_Risk"].astype(int) == 0).astype(int)
            else:
                out["N"] = 0
            continue
        members = [c for c, l in RFMID_TO_ODIR.items() if l == lab and c in df.columns]
        if members:
            out[lab] = (df[members].astype(int).sum(axis=1) > 0).astype(int)
        else:
            out[lab] = 0

    if "Disease_Risk" in df.columns:
        out["Disease_Risk"] = df["Disease_Risk"].astype(int)
    return out


def prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab in ODIR_LABEL_ORDER:
        n = int(df[lab].sum()) if lab in df.columns else 0
        rows.append({"label": lab, "name": ODIR_FULL_NAMES[lab], "positives": n, "prevalence": n / max(len(df), 1)})
    return pd.DataFrame(rows)


def write_odir_labels(output_root: Path, search_roots: dict[str, Path] | None = None) -> dict:
    """Write cluster_map + clustered_{train,val,test}.csv under outputs/odir8_taxonomy/."""
    import sys

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "src"))
    from retina_multilabel.data import locate_split_files
    from retina_multilabel.utils import resolve_dataset_search_roots

    roots = search_roots or resolve_dataset_search_roots()
    out_dir = Path(output_root)
    labels_dir = out_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    payload = odir_cluster_map_payload()
    (out_dir / "cluster_map.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summaries = {}
    for split in ("train", "val", "test"):
        files = locate_split_files(split, roots)
        raw = pd.read_csv(files.labels_csv)
        mapped = apply_odir_map(raw)
        mapped.to_csv(labels_dir / f"clustered_{split}.csv", index=False)
        prev = prevalence_table(mapped)
        prev.to_csv(out_dir / f"prevalence_{split}.csv", index=False)
        summaries[split] = prev.to_dict(orient="records")

    summary = {
        "scheme": "odir8_pmc9230753",
        "k": 8,
        "labels": list(ODIR_LABEL_ORDER),
        "cluster_map": str(out_dir / "cluster_map.json"),
        "labels_dir": str(labels_dir),
        "prevalence": summaries,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
