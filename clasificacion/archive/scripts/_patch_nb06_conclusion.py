"""Inserta/arregla la conclusión honesta del notebook 06 sin borrar outputs."""

from __future__ import annotations

import json
from pathlib import Path

PATH = Path("notebooks/best_model/06_recall_focused_training_resnet152.ipynb")

CONCLUSION = """## 11. Conclusión honesta

Resultado del intento (tras ejecutar las celdas anteriores; números en artifacts/recall_focus/summary.json):

- **Validación:** se alcanza FN < FP con el umbral elegido solo en validación.
- **Test:** **no** se alcanza FN < FP de forma estable (p. ej. el sistema nuevo con regla FN<FP quedó en FN 28 / FP 22 frente a 36 / 16 del FINAL de 05). Bajar más el umbral intercambia FN por FP, pero con 134 sanos el cruce FN≈FP en test no coincide con el de validación.
- **acc / F1 / AUC:** dentro de la tolerancia de 0.005 respecto al FINAL de 05 (a menudo ligeramente mejores en AUC/F1).
- **Variantes 2×2:** en validación ganó el **control** (seguir afinando 4 épocas sin re-ponderación ni alpha alto). La re-ponderación y la loss asimétrica no aportaron sobre un ResNet-152 ya convergido; sí son la receta que usa el zoo del notebook 07 al partir de ImageNet.

**Veredicto:** con letterbox + warm-start + ensemble/TTA/umbral de validación, **FN < FP en test y validación a la vez, sin perder ~0.005 de acc/F1/AUC, no se alcanzó de forma fiable**. Lo más cercano es FN 28 / FP 22 en test (mejor recall que 05, a costa de más FP). No hay fuga de test: umbral y miembros solo con validación.
"""


def main() -> None:
    notebook = json.loads(PATH.read_text(encoding="utf-8"))
    source_lines = CONCLUSION.splitlines(keepends=True)
    if source_lines and not source_lines[-1].endswith("\n"):
        source_lines[-1] += "\n"

    last_source = "".join(notebook["cells"][-1].get("source", []))
    if "Conclusión honesta" in last_source:
        notebook["cells"][-1]["source"] = source_lines
        notebook["cells"][-1]["cell_type"] = "markdown"
        notebook["cells"][-1]["metadata"] = {}
    else:
        notebook["cells"].append(
            {
                "cell_type": "markdown",
                "id": "concl06",
                "metadata": {},
                "source": source_lines,
            }
        )

    PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    verified = json.loads(PATH.read_text(encoding="utf-8"))
    text = "".join(verified["cells"][-1]["source"])
    assert "artifacts/recall_focus" in text, text[:200]
    assert "\x07" not in text
    print("OK cells=", len(verified["cells"]))
    print(text[:350])


if __name__ == "__main__":
    main()
