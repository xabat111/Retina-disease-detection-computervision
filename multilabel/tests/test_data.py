import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retina_multilabel.config import ProjectConfig, TrainConfig
from retina_multilabel.data import (
    IMAGE_FOLDER_NAMES,
    LABEL_FILE_NAMES,
    build_cooccurrence_matrix,
    build_multilabel_tables,
    build_prevalence_table,
    build_train_augmentation,
    collect_transform_types,
    crop_transform_types,
    letterbox_image,
    load_multilabel_tables,
    save_multilabel_tables,
)
from retina_multilabel.labels import LABEL_NAMES
from retina_multilabel.utils import DATA_ROOT_ENV_VARIABLE

DISEASE_COLUMNS = ["DR", "ODC", "CL"]


def test_letterbox_is_square_and_keeps_aspect_ratio():
    wide_image = Image.new("RGB", (400, 200), (255, 255, 255))
    result = letterbox_image(wide_image, 64)
    assert result.shape == (64, 64, 3) and result.dtype == np.uint8
    assert result[:16].sum() == 0 and result[-16:].sum() == 0  # bandas negras arriba y abajo
    assert result[32, 32].min() > 200  # el contenido queda centrado


def test_train_augmentation_contains_no_crop():
    transform = build_train_augmentation(TrainConfig())
    found_crop_types = collect_transform_types(transform).intersection(crop_transform_types())
    assert not found_crop_types
    type_names = {cls.__name__ for cls in collect_transform_types(transform)}
    assert "RandomResizedCrop" not in type_names
    assert "RandomCrop" not in type_names
    assert "RandomHorizontalFlip" in type_names
    assert "RandomRotation" in type_names
    assert "ColorJitter" in type_names


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch):
    """Dataset RFMiD diminuto con la misma estructura de carpetas que el original."""
    layout = {"train": "Training_Set", "val": "Evaluation_Set", "test": "Test_Set"}
    for split, outer in layout.items():
        folder = tmp_path / "rfmid" / outer / outer
        image_folder = folder / IMAGE_FOLDER_NAMES[split]
        image_folder.mkdir(parents=True)
        rows = [(i, int(i % 2 == 0), int(i % 4 == 0), 0, int(i == 2)) for i in range(1, 9)]
        pd.DataFrame(rows, columns=["ID", "Disease_Risk", *DISEASE_COLUMNS]).to_csv(
            folder / LABEL_FILE_NAMES[split], index=False
        )
        for identifier in range(1, 9):
            Image.new("RGB", (30, 20)).save(image_folder / f"{identifier}.png")
    monkeypatch.setenv(DATA_ROOT_ENV_VARIABLE, str(tmp_path / "rfmid"))
    return tmp_path


def test_build_save_and_reload_tables(synthetic_dataset):
    config = ProjectConfig()
    tables, reports = build_multilabel_tables(config)

    assert set(tables) == {"train", "val", "test"}
    assert all(len(table) == 8 for table in tables.values())
    assert list(tables["train"].columns) == ["ID", "image_path", *LABEL_NAMES]
    assert reports["train"].n_risk_without_pathology_column == 1  # solo el ID 6

    output_dir = synthetic_dataset / "out"
    output_dir.mkdir()
    save_multilabel_tables(tables, output_dir)
    reloaded = load_multilabel_tables(config, output_dir)
    pd.testing.assert_frame_equal(tables["val"], reloaded["val"], check_dtype=False)


def test_prevalence_and_cooccurrence(synthetic_dataset):
    tables, _ = build_multilabel_tables(ProjectConfig())
    prevalence = build_prevalence_table(tables, min_positives=3)
    assert list(prevalence.index) == LABEL_NAMES
    assert prevalence.loc["sano", "n_train"] == 4

    cooccurrence = build_cooccurrence_matrix(tables["train"])
    assert cooccurrence.shape == (10, 10)
    assert cooccurrence.loc["sano", "sano"] == 4
    assert cooccurrence.loc["sano", "retinopatia_diabetica"] == 0
