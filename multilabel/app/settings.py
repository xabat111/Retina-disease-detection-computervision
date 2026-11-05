"""Paths, constants and class metadata for the final app (Streamlit UI + odir7 ensemble)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# --------------------------------------------------------------------------- modelo desplegado

MODEL_ID = "odir7 ensemble (3x DINOv2 ViT-L residual-attn: 518/392/392)"
CHECKPOINT_DIR = Path(
    os.environ.get("RETINA_CKPT_DIR", PROJECT_ROOT / "checkpoints" / "final_20261030")
).expanduser().resolve()
MANIFEST_PATH = PROJECT_ROOT / "configs" / "ensemble_odir7.json"
TEST_PREDICTIONS_DIR = PROJECT_ROOT / "predictions" / "reference" / "odir7"
THRESHOLD_CRITERION = "fbeta2_val"
GRADCAM_MEMBER = 0  # vit_residual_attn_odir7_v2_518
DISPLAY_IMAGE_SIZE = 518

# --------------------------------------------------------------------------- inferencia

DEFAULT_TTA_PASSES = 4  # cada pase = 3 forwards ViT-L (ensemble); en CPU es lento
MIN_TTA_PASSES = 2
MAX_TTA_PASSES = 16
UNRELIABLE_MIN_TEST_POSITIVES = 20

# --------------------------------------------------------------------------- incertidumbre

HIGH_MIN_CERTAINTY_SCORE = 90.0
MEDIUM_MIN_CERTAINTY_SCORE = 60.0
INTERVAL_Z_SCORE = 1.96

# --------------------------------------------------------------------------- muestras de test

TEST_LABELS_CSV = PROJECT_ROOT / "data" / "taxonomies" / "odir7" / "labels" / "clustered_test.csv"
DATA_ROOT = Path(os.environ.get("RETINA_DATA_ROOT", PROJECT_ROOT.parents[1])).expanduser().resolve()
TEST_IMAGE_DIR = DATA_ROOT / "Test_Set" / "Test_Set" / "Test"

# --------------------------------------------------------------------------- nombres de clases

LABEL_NAMES = ["N", "D", "G", "C", "A", "M", "O"]
HEALTHY_LABEL = "N"
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "N": "Normal / healthy",
    "D": "Diabetic retinopathy",
    "G": "Glaucoma-suspect / optic disc",
    "C": "Media opacity (cataract proxy)",
    "A": "AMD",
    "M": "Myopia",
    "O": "Other",
}
