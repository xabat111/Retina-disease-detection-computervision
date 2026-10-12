"""Reconstruye 04_compare_models.ipynb desde el volcado de fuentes (el .ipynb original se perdió).

El dump en artifacts/logs/_nb_dump.txt guarda el source de cada celda DESPUÉS del parche de
PROJECT_ROOT. El notebook 04 nunca se ejecutó (plantilla para rellenar tras Optuna), así que
no hay outputs que preservar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nbtools import markdown_cell, write_notebook

DUMP_PATH = Path("artifacts/logs/_nb_dump.txt")
OUTPUT_PATH = Path("notebooks/other_experiments/04_compare_models.ipynb")
MARKER = "########## notebooks\\other_experiments\\04_compare_models.ipynb"


def parse_notebook_dump(dump_text: str, marker: str) -> list[dict]:
    start = dump_text.find(marker)
    if start < 0:
        raise FileNotFoundError(f"No aparece {marker!r} en {DUMP_PATH}")
    rest = dump_text[start + len(marker) :]
    next_notebook = rest.find("\n########## ")
    section = rest if next_notebook < 0 else rest[:next_notebook]

    cells: list[dict] = []
    current_kind: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_kind is None:
            return
        source = "".join(current_lines).strip("\n")
        if current_kind == "markdown":
            cells.append(markdown_cell(source))
        else:
            from _nbtools import code_cell

            cells.append(code_cell(source))

    for line in section.splitlines(keepends=True):
        if line.startswith("--- [") and "] " in line:
            flush()
            current_kind = line.split("] ", 1)[1].strip()
            current_lines = []
            continue
        current_lines.append(line)
    flush()
    return cells


def main() -> None:
    dump_text = DUMP_PATH.read_text(encoding="utf-8")
    cells = parse_notebook_dump(dump_text, MARKER)
    if OUTPUT_PATH.is_file():
        raise FileExistsError(f"No se sobrescribe {OUTPUT_PATH}")
    write_notebook(OUTPUT_PATH, cells)
    notebook = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    print(f"escrito {OUTPUT_PATH} | {len(notebook['cells'])} celdas")


if __name__ == "__main__":
    main()
