"""RFMiD fine-grained disease columns (~40) used before latent clustering."""

from __future__ import annotations

# Official RFMiD disease columns (Training Labels schema, excl. ID / Disease_Risk).
RFMID_DISEASE_COLUMNS: tuple[str, ...] = (
    "DR",
    "ARMD",
    "MH",
    "DN",
    "MYA",
    "BRVO",
    "TSLN",
    "ERM",
    "LS",
    "MS",
    "CSR",
    "ODC",
    "CRVO",
    "TV",
    "AH",
    "ODP",
    "ODE",
    "ST",
    "AION",
    "PT",
    "RT",
    "RS",
    "CRS",
    "EDN",
    "RPEC",
    "MHL",
    "RP",
    "CWS",
    "CB",
    "ODPM",
    "PRH",
    "MNF",
    "HR",
    "CRAO",
    "TD",
    "CME",
    "PTCR",
    "CF",
    "VH",
    "MCA",
    "VS",
    "BRAO",
    "PLQ",
    "HPED",
    "CL",
)

META_COLUMNS: tuple[str, ...] = ("ID", "Disease_Risk")


def disease_columns_present(columns: list[str] | tuple[str, ...]) -> list[str]:
    """Intersect schema with RFMiD disease list (preserve RFMiD order)."""
    available = set(columns)
    return [c for c in RFMID_DISEASE_COLUMNS if c in available]
