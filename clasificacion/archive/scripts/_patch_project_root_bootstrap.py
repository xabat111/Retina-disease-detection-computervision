"""Actualiza el bootstrap de PROJECT_ROOT en todos los notebooks y limpia outputs del 07."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = ROOT / "notebooks"

OLD_BLOCKS = [
    '''# Código: carpeta con src/ + requirements.txt (retina_clasificacion/)
# (funciona desde notebooks/, notebooks/best_model/ o notebooks/other_experiments/)
PROJECT_ROOT = next(
    directory
    for directory in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (directory / "src").is_dir() and (directory / "requirements.txt").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))''',
    '''# Código: carpeta con src/ + requirements.txt (retina_clasificacion/)
PROJECT_ROOT = next(
    directory
    for directory in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (directory / "src").is_dir() and (directory / "requirements.txt").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))''',
    '''# Raíz del proyecto: se sube desde el directorio actual hasta encontrar src/ y requirements.txt
# (funciona desde notebooks/, notebooks/best_model/ o notebooks/other_experiments/)
PROJECT_ROOT = next(
    directory
    for directory in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (directory / "src").is_dir() and (directory / "requirements.txt").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))''',
    '''# Raíz del proyecto: se sube desde el directorio actual hasta encontrar src/ y requirements.txt
# (funciona desde notebooks/, notebooks/best_model/ o notebooks/other_experiments/)
PROJECT_ROOT = next(
    directory
    for directory in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (directory / "src").is_dir() and (directory / "requirements.txt").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))''',
    '''# Raíz del proyecto: se sube hasta encontrar src/ y requirements.txt
PROJECT_ROOT = next(
    directory
    for directory in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (directory / "src").is_dir() and (directory / "requirements.txt").is_file()
)''',
]

NEW_BOOTSTRAP = '''# Localiza retina_clasificacion/ aunque el cwd del kernel sea el padre Retina/
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
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))'''

NEW_BOOTSTRAP_NO_TORCH = '''# Localiza retina_clasificacion/ aunque el cwd del kernel sea el padre Retina/
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
sys.path.insert(0, str(PROJECT_ROOT))'''

NB07_RELOAD_BLOCK = '''import importlib

# Tras mover el código a retina_clasificacion/, regenera rutas (mejor: Restart Kernel)
for module_name in (
    "src.config",
    "src.cached_datasets",
    "src.recall_config",
    "src.recall_focus_config",
    "src.zoo_recall_config",
    "src.zoo_recall_runner",
):
    importlib.reload(importlib.import_module(module_name))

from src.cached_datasets import build_cached_letterbox_splits
from src.config import DATA_ROOT, PROJECT_ROOT as CONFIG_PROJECT_ROOT
from src.recall_config import PREDICTION_CACHE_DIR as PREVIOUS_PREDICTION_CACHE_DIR, existing_checkpoint_members
from src.recall_focus_config import (
    CHECKPOINT_DIR as RESNET152_CHECKPOINT_DIR,
    PREDICTION_CACHE_DIR as RECALL_FOCUS_CACHE_DIR,
    SELECTED_GRADCAM_IMAGES_PATH,
    SUMMARY_PATH as RECALL_FOCUS_SUMMARY_PATH,
    VARIANTS,
    WARM_START_CHECKPOINT,
)
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
from src.zoo_recall_runner import train_architecture_if_missing

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
device = get_device()
validation_recall_goal = TARGET_RECALL + VALIDATION_RECALL_MARGIN
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"device={device}")
print(f"PROJECT_ROOT (código)={CONFIG_PROJECT_ROOT}")
print(f"DATA_ROOT (RFMiD)={DATA_ROOT}")
print(f"checkpoints={CHECKPOINT_DIR}")'''

NB07_CLEAN_TAIL = '''from src.config import DATA_ROOT, PROJECT_ROOT
from src.zoo_recall_config import CHECKPOINT_DIR, FIGURES_DIR, GRADCAM_DIR, PREDICTION_CACHE_DIR
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
        f"Cópialos a {CHECKPOINT_DIR} si quieres reutilizarlos."
    )

print(f"device={device}")
print(f"PROJECT_ROOT (código)={PROJECT_ROOT}")
print(f"DATA_ROOT (RFMiD)={DATA_ROOT}")
print(f"checkpoints={CHECKPOINT_DIR}")
print(f"train labels existen={ (DATA_ROOT / 'Training_Set' / 'Training_Set' / 'RFMiD_Training_Labels.csv').is_file() }")'''


def cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def set_cell_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def clear_code_outputs(notebook: dict) -> int:
    cleared = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            cell["outputs"] = []
            cell["execution_count"] = None
            cleared += 1
    return cleared


def patch_bootstrap(text: str) -> str:
    for old in OLD_BLOCKS:
        if old in text:
            replacement = NEW_BOOTSTRAP if "TORCH_HOME" in old else NEW_BOOTSTRAP_NO_TORCH
            # El bloque de 01 no tenía TORCH_HOME ni el comentario largo con other_experiments
            if "os.environ.setdefault" not in old and "TORCH_HOME" not in old:
                replacement = NEW_BOOTSTRAP_NO_TORCH
            return text.replace(old, replacement, 1)
    return text


def patch_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        original = cell_source(cell)
        updated = patch_bootstrap(original)
        if path.name.startswith("07_") and NB07_RELOAD_BLOCK in updated:
            updated = updated.replace(NB07_RELOAD_BLOCK, NB07_CLEAN_TAIL)
        if updated != original:
            set_cell_source(cell, updated)
            changed += 1

    cleared = 0
    if path.name.startswith("07_"):
        cleared = clear_code_outputs(notebook)

    if changed or cleared:
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(f"{path.relative_to(ROOT)}: celdas={changed}, outputs_limpiados={cleared}")


def main() -> None:
    for path in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
        patch_notebook(path)


if __name__ == "__main__":
    main()
