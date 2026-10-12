"""Reescribe la celda setup del nb07, actualiza el builder, ejecuta el setup y deja un status file."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB07 = ROOT / "notebooks" / "other_experiments" / "07_train_model_zoo_recall_nocrop.ipynb"
STATUS_PATH = ROOT / "artifacts" / "logs" / "nb07_setup_status.txt"
BUILDER = ROOT / "artifacts" / "logs" / "_build_nb07.py"

SETUP_SOURCE = r'''import json
import os
import sys
from pathlib import Path

print("=== NB07 SETUP v3 ===")
print("Si NO ves este banner, cierra el notebook sin guardar y vuelve a abrirlo (estabas con una celda antigua en memoria).")

# Localiza retina_clasificacion/ aunque el cwd del kernel sea el padre Retina/
_cwd = Path.cwd().resolve()
_candidates = [_cwd, *_cwd.parents]
if _cwd.is_dir():
    _candidates.extend(sorted(path for path in _cwd.iterdir() if path.is_dir()))
PROJECT_ROOT = next(
    directory
    for directory in _candidates
    if (directory / "src" / "config.py").is_file() and (directory / "requirements.txt").is_file()
)
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != PROJECT_ROOT]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.cached_datasets import build_cached_letterbox_splits
from src.comparison_plots import plot_metric_bars
from src.data import PREPROCESSING_LETTERBOX
from src.ensemble_evaluation import (
    FN_BELOW_FP_RULE,
    RECALL_TARGET_RULE,
    average_over_members,
    average_over_views,
    build_model_comparison_table,
    choose_operating_thresholds,
    load_member_views,
    summary_for_json,
)
from src.evaluate import select_threshold_youden
from src.gradcam_compare import GROUP_ORDER, compute_gradcam_for_selection, plot_gradcam_model_comparison
from src.hard_example_plots import plot_training_histories
from src.models_zoo import get_architecture_spec, load_architecture_checkpoint
from src.recall_config import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PREDICTION_CACHE_DIR as PREVIOUS_PREDICTION_CACHE_DIR,
    TARGET_RECALL,
    VALIDATION_RECALL_MARGIN,
    existing_checkpoint_members,
)
from src.recall_focus_config import (
    CHECKPOINT_DIR as RESNET152_CHECKPOINT_DIR,
    PREDICTION_CACHE_DIR as RECALL_FOCUS_CACHE_DIR,
    SELECTED_GRADCAM_IMAGES_PATH,
    SUMMARY_PATH as RECALL_FOCUS_SUMMARY_PATH,
    VARIANTS,
    WARM_START_CHECKPOINT,
)
from src.recall_optimization import (
    bootstrap_metric_differences,
    choose_ensemble_subset,
    compute_operating_point_metrics,
    rank_ensemble_subsets,
)
from src.recall_plots import plot_confusion_matrices, plot_roc_and_pr_curves
from src.recall_report import build_summary_table, summarize_operating_point
from src.train import get_device
from src.tta_inference import DEFAULT_TTA_VIEW_NAMES, EnsembleMember
from src.zoo_recall_config import (
    ARCHITECTURES,
    CHECKPOINT_DIR,
    FIGURES_DIR,
    GRADCAM_DIR,
    PREDICTION_CACHE_DIR,
    SUMMARY_PATH,
    ZOO_EPOCHS,
    ZOO_SETTINGS,
    build_zoo_train_config,
    chosen_recall_recipe_name,
    recall_recipe,
)
from src.config import DATA_ROOT, PROJECT_ROOT
from src.zoo_recall_runner import train_architecture_if_missing

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
device = get_device()
validation_recall_goal = TARGET_RECALL + VALIDATION_RECALL_MARGIN
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

legacy_checkpoints = PROJECT_ROOT.parent / "artifacts" / "checkpoints"
if legacy_checkpoints.is_dir() and any(legacy_checkpoints.glob("*.pt")):
    print(
        f"AVISO: hay .pt en la ruta antigua {legacy_checkpoints}. "
        f"Copialos a {CHECKPOINT_DIR} si quieres reutilizarlos."
    )

# Comprueba que torch pueda guardar en la carpeta nueva
probe_path = CHECKPOINT_DIR / "_write_probe.txt"
probe_path.write_text("ok", encoding="utf-8")
probe_path.unlink()

status_lines = [
    "NB07_SETUP_V3",
    f"cwd={Path.cwd().resolve()}",
    f"device={device}",
    f"PROJECT_ROOT={PROJECT_ROOT}",
    f"DATA_ROOT={DATA_ROOT}",
    f"checkpoints={CHECKPOINT_DIR}",
    f"train_labels={(DATA_ROOT / 'Training_Set' / 'Training_Set' / 'RFMiD_Training_Labels.csv').is_file()}",
    f"checkpoint_dir_writable=True",
]
status_path = PROJECT_ROOT / "artifacts" / "logs" / "nb07_setup_status.txt"
status_path.write_text("\n".join(status_lines) + "\n", encoding="utf-8")

print(f"device={device}")
print(f"PROJECT_ROOT={PROJECT_ROOT}")
print(f"DATA_ROOT={DATA_ROOT}")
print(f"checkpoints={CHECKPOINT_DIR}")
print(f"train_labels_ok={(DATA_ROOT / 'Training_Set' / 'Training_Set' / 'RFMiD_Training_Labels.csv').is_file()}")
print(f"status_file={status_path}")
print("=== NB07 SETUP v3 OK ===")
'''


def rewrite_nb07_setup() -> None:
    notebook = json.loads(NB07.read_text(encoding="utf-8"))
    # Primera celda de código = setup
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["source"] = SETUP_SOURCE.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            break
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    NB07.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"rewrote setup cell in {NB07}")


def patch_builder() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    old = 'print(f"device={device} | raíz={PROJECT_ROOT}")'
    new = '''print("=== NB07 SETUP v3 ===")
print(f"device={device}")
print(f"PROJECT_ROOT={PROJECT_ROOT}")
print(f"DATA_ROOT={DATA_ROOT}")
print(f"checkpoints={CHECKPOINT_DIR}")'''
    if old in text:
        BUILDER.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
        print("patched _build_nb07.py")
    else:
        print("_build_nb07.py: old print not found (ok if already patched)")


def patch_nb06_print() -> None:
    nb06 = ROOT / "notebooks" / "best_model" / "06_recall_focused_training_resnet152.ipynb"
    notebook = json.loads(nb06.read_text(encoding="utf-8"))
    changed = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if 'print(f"device={device} | raíz={PROJECT_ROOT}")' in src:
            src = src.replace(
                'print(f"device={device} | raíz={PROJECT_ROOT}")',
                'from src.config import DATA_ROOT, PROJECT_ROOT\n'
                'print(f"device={device}")\n'
                'print(f"PROJECT_ROOT={PROJECT_ROOT}")\n'
                'print(f"DATA_ROOT={DATA_ROOT}")',
            )
            cell["source"] = src.splitlines(keepends=True)
            changed += 1
        cell["outputs"] = []
        cell["execution_count"] = None
    if changed:
        nb06.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"nb06 print cells patched: {changed}")


def run_setup_smoke() -> None:
    # Simula cwd problemático (padre) y ejecuta el setup
    os.chdir(ROOT.parent)
    print(f"smoke cwd={Path.cwd()}")
    ns: dict = {"__name__": "__main__"}
    exec(compile(SETUP_SOURCE, str(NB07), "exec"), ns, ns)
    print("smoke status file:")
    print(STATUS_PATH.read_text(encoding="utf-8"))
    # Verifica rutas de un checkpoint del zoo
    cfg = ns["build_zoo_train_config"]("convnext_tiny")
    ckpt = cfg.checkpoint_dir / cfg.best_checkpoint_name
    print(f"convnext would save to: {ckpt}")
    assert str(ROOT) in str(ckpt), ckpt
    assert cfg.train_labels_csv.is_file(), cfg.train_labels_csv
    print("SMOKE OK")


if __name__ == "__main__":
    rewrite_nb07_setup()
    patch_builder()
    patch_nb06_print()
    run_setup_smoke()
