"""Ejecuta el notebook 07 celda a celda, guarda outputs y aborta con traceback claro."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "other_experiments" / "07_train_model_zoo_recall_nocrop.ipynb"
LOG_PATH = ROOT / "artifacts" / "logs" / "nb07_exec.out"
ERR_PATH = ROOT / "artifacts" / "logs" / "nb07_exec.err"


def main() -> int:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Sin GUI: evita que plt.show() cuelgue en ejecución headless
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: plt.close("all")

    notebook = json.loads(NB_PATH.read_text(encoding="utf-8"))
    # IPython helpers usados por las celdas (display / plt.show)
    try:
        from IPython.display import display as ipython_display
    except ImportError:
        def ipython_display(obj):  # noqa: ANN001
            print(obj)

    namespace: dict = {
        "__name__": "__main__",
        "display": ipython_display,
        "get_ipython": lambda: None,
    }
    log_lines: list[str] = []

    def log(message: str) -> None:
        # Windows cp1252 no admite − (U+2212) ni acentos raros de tablas pandas.
        try:
            print(message, flush=True)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe, flush=True)
        log_lines.append(message)

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        first_line = source.strip().splitlines()[0][:90]
        log(f"\n===== CELL {index}: {first_line} =====")
        try:
            # Captura prints de la celda
            import io
            from contextlib import redirect_stdout, redirect_stderr

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
        except Exception as exc:
            tb = traceback.format_exc()
            log(f"FAILED cell {index}: {type(exc).__name__}: {exc}")
            log(tb)
            ERR_PATH.write_text("\n".join(log_lines) + "\n" + tb, encoding="utf-8")
            LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            # Guarda notebook parcialmente ejecutado
            NB_PATH.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            return 1

    NB_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log("NB07 COMPLETE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
