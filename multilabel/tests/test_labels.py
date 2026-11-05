import pandas as pd

from retina_multilabel.labels import (
    COLUMN_TO_GROUP,
    HEALTHY_LABEL,
    LABEL_NAMES,
    PATHOLOGY_GROUPS,
    build_label_frame,
)

RFMID_DISEASE_COLUMNS = (
    "DR ARMD MH DN MYA BRVO TSLN ERM LS MS CSR ODC CRVO TV AH ODP ODE ST AION PT RT RS CRS "
    "EDN RPEC MHL RP CWS CB ODPM PRH MNF HR CRAO TD CME PTCR CF VH MCA VS BRAO PLQ HPED CL"
).split()


def test_every_rfmid_column_belongs_to_exactly_one_group():
    mapped_columns = [c for columns in PATHOLOGY_GROUPS.values() for c in columns]
    assert sorted(mapped_columns) == sorted(RFMID_DISEASE_COLUMNS)
    assert len(mapped_columns) == len(set(mapped_columns)) == len(COLUMN_TO_GROUP)


def test_there_are_ten_labels_with_healthy_last():
    assert len(LABEL_NAMES) == 10
    assert LABEL_NAMES[-1] == HEALTHY_LABEL


def _synthetic_raw_labels() -> pd.DataFrame:
    rows = [
        # ID, Disease_Risk, DR, ODC, CL, MYA
        (1, 0, 0, 0, 0, 0),  # sano
        (2, 1, 1, 0, 0, 0),  # retinopatía diabética
        (3, 1, 1, 1, 0, 0),  # coocurrencia DR + excavación
        (4, 1, 0, 0, 1, 0),  # CL -> otros
        (5, 1, 0, 0, 0, 0),  # riesgo sin columna -> otros
        (6, 0, 0, 0, 0, 1),  # columna activa sin riesgo -> enfermo (miopía)
    ]
    return pd.DataFrame(rows, columns=["ID", "Disease_Risk", "DR", "ODC", "CL", "MYA"])


def test_build_label_frame_rules_and_consistency_report():
    labels, report = build_label_frame(_synthetic_raw_labels())
    by_id = labels.set_index("ID")

    assert by_id.loc[1, HEALTHY_LABEL] == 1 and by_id.loc[1, LABEL_NAMES[:-1]].sum() == 0
    assert by_id.loc[3, "retinopatia_diabetica"] == 1 and by_id.loc[3, "excavacion_disco"] == 1
    assert by_id.loc[4, "otros"] == 1
    assert by_id.loc[5, "otros"] == 1
    assert by_id.loc[6, "miopia"] == 1 and by_id.loc[6, HEALTHY_LABEL] == 0
    assert report.n_risk_without_pathology_column == 1
    assert report.n_pathology_column_without_risk == 1


def test_healthy_is_exclusive_and_every_row_has_a_label():
    labels, _ = build_label_frame(_synthetic_raw_labels())
    assert (labels[LABEL_NAMES].sum(axis=1) >= 1).all()
    diseased = labels[labels[HEALTHY_LABEL] == 1]
    assert diseased[LABEL_NAMES[:-1]].sum().sum() == 0


def test_unknown_columns_go_to_other_group():
    raw = pd.DataFrame({"ID": [1, 2], "Disease_Risk": [1, 0], "NEW_DISEASE": [1, 0]})
    labels, report = build_label_frame(raw)
    assert report.unmapped_columns == ["NEW_DISEASE"]
    assert labels.loc[0, "otros"] == 1
