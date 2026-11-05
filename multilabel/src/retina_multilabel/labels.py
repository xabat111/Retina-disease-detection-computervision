"""Definición de las 10 etiquetas (9 grupos de patología + sano) y construcción desde RFMiD."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

IDENTIFIER_COLUMN = "ID"
DISEASE_RISK_COLUMN = "Disease_Risk"
HEALTHY_LABEL = "sano"
OTHER_GROUP = "otros"

# ÚNICA fuente de verdad: cada grupo agrupa columnas de enfermedad de RFMiD.
# El grupo "otros" incluye además las 7 columnas que no aparecen a la vez en train y
# val (CB, ODPM, MNF, HR, MCA, HPED, CL) para no perder ninguna imagen enferma.
PATHOLOGY_GROUPS: dict[str, tuple[str, ...]] = {
    "retinopatia_diabetica": ("DR", "EDN", "CWS"),
    "opacidad_medios": ("MH", "AH", "VH"),
    "patologia_macular": ("ARMD", "DN", "ERM", "MS", "CSR", "MHL", "CME", "PT"),
    "excavacion_disco": ("ODC",),
    "otras_nervio_optico": ("ODP", "ODE", "AION"),
    "oclusiones_vasculares": ("BRVO", "CRVO", "CRAO", "BRAO", "TV", "VS", "ST"),
    "miopia": ("MYA", "TSLN", "TD"),
    "cicatrices_inflamacion": ("CRS", "RS", "LS", "RPEC", "RP", "PTCR"),
    OTHER_GROUP: ("RT", "PRH", "CF", "PLQ", "CB", "ODPM", "MNF", "HR", "MCA", "HPED", "CL"),
}

GROUP_DESCRIPTIONS: dict[str, str] = {
    "retinopatia_diabetica": "Retinopatía diabética y lesiones asociadas",
    "opacidad_medios": "Opacidad de medios / vítreo",
    "patologia_macular": "Patología macular",
    "excavacion_disco": "Excavación del disco (sospecha de glaucoma)",
    "otras_nervio_optico": "Otras alteraciones del nervio óptico",
    "oclusiones_vasculares": "Oclusiones y patología vascular",
    "miopia": "Miopía y signos asociados",
    "cicatrices_inflamacion": "Cicatrices, inflamación y cambios del epitelio",
    OTHER_GROUP: "Otros (pocos casos)",
    HEALTHY_LABEL: "Sano",
}

LABEL_NAMES: list[str] = [*PATHOLOGY_GROUPS, HEALTHY_LABEL]
# 9 grupos de patología (sin `sano`). El pipeline clásico de 10 etiquetas no cambia.
PATHOLOGY_LABEL_NAMES: list[str] = list(PATHOLOGY_GROUPS)
HEALTHY_INDEX: int = LABEL_NAMES.index(HEALTHY_LABEL)
COLUMN_TO_GROUP: dict[str, str] = {
    column: group for group, columns in PATHOLOGY_GROUPS.items() for column in columns
}


def active_label_names(*, drop_healthy: bool = False) -> list[str]:
    """Lista de etiquetas usadas en entrenamiento/evaluación."""
    return list(PATHOLOGY_LABEL_NAMES) if drop_healthy else list(LABEL_NAMES)


@dataclass
class ConsistencyReport:
    """Incidencias detectadas al construir las etiquetas de un split."""

    n_rows: int = 0
    n_risk_without_pathology_column: int = 0  # Disease_Risk=1 sin ninguna columna activa
    n_pathology_column_without_risk: int = 0  # columna activa con Disease_Risk=0
    unmapped_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        return (
            f"filas={self.n_rows} | Disease_Risk=1 sin columna activa -> 'otros': "
            f"{self.n_risk_without_pathology_column} | columna activa con Disease_Risk=0 "
            f"-> enfermo: {self.n_pathology_column_without_risk} | columnas no mapeadas "
            f"-> 'otros': {self.unmapped_columns or 'ninguna'} | columnas ausentes "
            f"-> 0: {self.missing_columns or 'ninguna'}"
        )


def build_label_frame(raw_labels: pd.DataFrame) -> tuple[pd.DataFrame, ConsistencyReport]:
    """Convierte el CSV original de RFMiD en las 10 etiquetas binarias.

    Reglas:
    - Etiqueta de un grupo = OR de sus columnas de enfermedad.
    - Columnas de enfermedad desconocidas -> "otros" (no se pierde ninguna imagen).
    - Disease_Risk=1 sin ninguna columna activa -> "otros".
    - Columna activa con Disease_Risk=0 -> se considera enfermo (la anotación explícita
      de patología prevalece sobre el indicador global).
    - "sano" = Disease_Risk=0 y ninguna columna activa; es exclusiva respecto al resto.
    """
    report = ConsistencyReport(n_rows=len(raw_labels))
    known_columns = {IDENTIFIER_COLUMN, DISEASE_RISK_COLUMN}
    disease_columns = [column for column in raw_labels.columns if column not in known_columns]
    report.unmapped_columns = [c for c in disease_columns if c not in COLUMN_TO_GROUP]
    report.missing_columns = [c for c in COLUMN_TO_GROUP if c not in disease_columns]

    labels = pd.DataFrame({IDENTIFIER_COLUMN: raw_labels[IDENTIFIER_COLUMN].to_numpy()})
    for group, columns in PATHOLOGY_GROUPS.items():
        present_columns = [c for c in columns if c in disease_columns]
        if group == OTHER_GROUP:
            present_columns += report.unmapped_columns
        group_active = raw_labels[present_columns].astype(bool).any(axis=1).to_numpy()
        labels[group] = group_active

    has_pathology_column = labels[list(PATHOLOGY_GROUPS)].any(axis=1)
    disease_risk = raw_labels[DISEASE_RISK_COLUMN].astype(bool).to_numpy()

    risk_without_column = disease_risk & ~has_pathology_column.to_numpy()
    report.n_risk_without_pathology_column = int(risk_without_column.sum())
    report.n_pathology_column_without_risk = int(
        (~disease_risk & has_pathology_column.to_numpy()).sum()
    )
    labels.loc[risk_without_column, OTHER_GROUP] = True
    labels[HEALTHY_LABEL] = ~disease_risk & ~has_pathology_column.to_numpy()

    labels[LABEL_NAMES] = labels[LABEL_NAMES].astype("int8")
    return labels, report
