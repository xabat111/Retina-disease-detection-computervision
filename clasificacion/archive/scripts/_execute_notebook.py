"""Ejecuta un notebook in-place (conserva outputs) con el kernel python3 del env dl."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = Path(sys.argv[1]).resolve()
TIMEOUT_SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 6 * 3600


def main() -> None:
    os.chdir(NOTEBOOK_PATH.parent)
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=TIMEOUT_SECONDS,
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
        allow_errors=False,
    )
    print(f"ejecutando {NOTEBOOK_PATH} (cwd={NOTEBOOK_PATH.parent})", flush=True)
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    n_errors = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                n_errors += 1
                print(f"ERROR celda {index}: {output.get('ename')}: {output.get('evalue')}", flush=True)
    print(f"escrito {NOTEBOOK_PATH} errores={n_errors}", flush=True)
    if n_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
