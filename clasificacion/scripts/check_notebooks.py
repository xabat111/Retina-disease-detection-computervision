"""Comprobación ligera de los notebooks (sin ejecutar entrenamientos).

Para cada notebook de `notebooks/`:
1. El JSON es válido y toda celda de código compila.
2. La primera celda de código (rutas + imports) se ejecuta con el directorio de trabajo puesto
   en la carpeta del notebook, que es lo que hace Jupyter; así se comprueba que `PROJECT_ROOT`
   se resuelve y que `import src...` funciona desde la nueva ubicación.
3. Las rutas de datos y checkpoints básicos existen.

Uso:  python scripts/check_notebooks.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

sys.path.insert(0, str(PROJECT_ROOT))
from src.config import DATA_ROOT  # noqa: E402

REQUIRED_DATA_PATHS = [
    DATA_ROOT / "Training_Set/Training_Set/RFMiD_Training_Labels.csv",
    DATA_ROOT / "Training_Set/Training_Set/Training",
    DATA_ROOT / "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv",
    DATA_ROOT / "Evaluation_Set/Evaluation_Set/Validation",
    DATA_ROOT / "Test_Set/Test_Set/RFMiD_Testing_Labels.csv",
    DATA_ROOT / "Test_Set/Test_Set/Test",
]
REQUIRED_PROJECT_PATHS = [
    PROJECT_ROOT / "artifacts/checkpoints/best_resnet152_disease_risk_optuna_nocrop.pt",
    PROJECT_ROOT / "artifacts/optuna/best_params.json",
    PROJECT_ROOT / "artifacts/gradcam/selected_images.json",
]


def _code_cells(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def check_notebook(notebook_path: Path) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacía si todo está bien)."""
    problems: list[str] = []
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = _code_cells(notebook)

    for cell_index, source in enumerate(code_cells):
        try:
            compile(source, f"{notebook_path.name}[code {cell_index}]", "exec")
        except SyntaxError as error:
            problems.append(f"celda de código {cell_index} no compila: {error}")

    first_cell = code_cells[0]
    previous_directory = Path.cwd()
    os.chdir(notebook_path.parent)
    try:
        namespace: dict = {"__name__": "__main__"}
        exec(compile(first_cell, notebook_path.name, "exec"), namespace)
        resolved_root = Path(namespace["PROJECT_ROOT"]).resolve()
        if resolved_root != PROJECT_ROOT:
            problems.append(f"PROJECT_ROOT={resolved_root} (esperado {PROJECT_ROOT})")
    except Exception as error:  # noqa: BLE001 - se informa cualquier fallo de la celda de setup
        problems.append(f"la celda de setup falla: {type(error).__name__}: {error}")
    finally:
        os.chdir(previous_directory)
    return problems


def main() -> int:
    missing_paths = [path for path in REQUIRED_DATA_PATHS + REQUIRED_PROJECT_PATHS if not path.exists()]
    for path in missing_paths:
        print(f"FALTA ruta: {path}")

    failures = 0
    for notebook_path in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in notebook_path.parts:
            continue
        problems = check_notebook(notebook_path)
        relative = notebook_path.relative_to(PROJECT_ROOT).as_posix()
        print(f"[{'OK' if not problems else 'FALLO'}] {relative}")
        for problem in problems:
            print(f"    - {problem}")
        failures += bool(problems)
    return int(failures > 0 or bool(missing_paths))


if __name__ == "__main__":
    raise SystemExit(main())
