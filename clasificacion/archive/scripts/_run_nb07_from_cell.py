"""Reanuda la ejecución del notebook 07 desde una celda de código concreta."""

from __future__ import annotations

import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "other_experiments" / "07_train_model_zoo_recall_nocrop.ipynb"
LOG_PATH = ROOT / "artifacts" / "logs" / "nb07_exec.out"
ERR_PATH = ROOT / "artifacts" / "logs" / "nb07_exec.err"
START_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def main() -> int:
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: plt.close("all")

    try:
        from IPython.display import display as ipython_display
    except ImportError:
        def ipython_display(obj):  # noqa: ANN001
            print(obj)

    notebook = json.loads(NB_PATH.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__main__", "display": ipython_display}
    log_lines: list[str] = []

    def log(message: str) -> None:
        try:
            print(message, flush=True)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe, flush=True)
        log_lines.append(message)

    # Ejecuta setup + celdas previas necesarias si START_CELL > 1
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if index > START_CELL:
            break
        if index < START_CELL and index not in {1, 3, 5, 7, 9, 10}:
            # Para reanudar Grad-CAM/ensemble hace falta el estado previo
            if index < START_CELL:
                continue
        if index < START_CELL:
            # Re-ejecutar dependencias ligeras/medias siempre antes del target
            pass

    # Más simple: ejecutar todas las celdas de código desde 1, pero START_CELL
    # implica "empezar a fallar solo a partir de"; mejor ejecutar 1..end siempre
    # con caches. Aquí: ejecutar desde min(1, ...) todas, o desde START si
    # el usuario ya tiene namespace — no. Ejecutamos TODAS desde 1; START se
    # usa para log. Para reanudación real: ejecutar 1,3,5,7,9,10 y luego >=START.

    required_before = [1, 3, 5, 7, 9, 10]
    cells_to_run = [i for i in required_before if i < START_CELL] + [
        i
        for i, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code" and i >= START_CELL and "".join(cell.get("source", [])).strip()
    ]
    # unique preserve order
    seen: set[int] = set()
    ordered: list[int] = []
    for index in cells_to_run:
        if index not in seen:
            seen.add(index)
            ordered.append(index)

    for index in ordered:
        cell = notebook["cells"][index]
        source = "".join(cell.get("source", []))
        first_line = source.strip().splitlines()[0][:90]
        log(f"\n===== CELL {index}: {first_line} =====")
        try:
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(source, f"{NB_PATH.name}:cell{index}", "exec"), namespace, namespace)
            out = stdout_buf.getvalue()
            err = stderr_buf.getvalue()
            if out:
                log(out.rstrip())
            if err:
                log("[stderr]\n" + err.rstrip())
            cell["outputs"] = []
            if out:
                cell["outputs"].append(
                    {"output_type": "stream", "name": "stdout", "text": out.splitlines(keepends=True)}
                )
            cell["execution_count"] = index
            NB_PATH.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except Exception as exc:
            tb = traceback.format_exc()
            log(f"FAILED cell {index}: {type(exc).__name__}: {exc}")
            log(tb)
            ERR_PATH.write_text("\n".join(log_lines) + "\n" + tb, encoding="utf-8")
            LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            NB_PATH.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            return 1

    log("NB07 COMPLETE OK")
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
