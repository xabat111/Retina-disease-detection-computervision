"""Catalogo de imagenes TEST con sus etiquetas odir7 de referencia (clustered_test.csv)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import settings


@dataclass(frozen=True)
class SampleCatalog:
    table: pd.DataFrame

    def image_path(self, identifier: int) -> Path:
        return Path(self.table.loc[self.table["ID"] == identifier, "image_path"].iloc[0])

    def positive_labels(self, identifier: int) -> list[str]:
        row = self.table.loc[self.table["ID"] == identifier].iloc[0]
        return [l for l in settings.LABEL_NAMES if row[l] == 1]

    @property
    def identifiers(self) -> list[int]:
        return self.table["ID"].tolist()


def load_sample_catalog() -> SampleCatalog:
    if not settings.TEST_IMAGE_DIR.is_dir() or not settings.TEST_LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing {settings.TEST_IMAGE_DIR} or {settings.TEST_LABELS_CSV}")
    table = pd.read_csv(settings.TEST_LABELS_CSV)
    paths = {p.stem: p for p in settings.TEST_IMAGE_DIR.iterdir() if p.is_file()}
    table = table[table["ID"].astype(str).isin(paths)].copy()
    table.insert(1, "image_path", [str(paths[str(i)]) for i in table["ID"]])
    return SampleCatalog(table=table.reset_index(drop=True))
