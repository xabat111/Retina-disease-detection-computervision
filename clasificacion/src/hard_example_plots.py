"""Figuras del entrenamiento con re-ponderación por época (notebooks 06 y 07)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_histories(
    histories: dict[str, list[dict]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Curvas por época de cada experimento: criterio de selección, AUC, loss y errores persistentes.

    Las épocas con `epoch == 0` (pesos de partida del warm start) se dibujan como punto de partida.
    """
    panels = [
        ("val_spec_at_fn_lt_fp", "Criterio de selección: especificidad con FN<FP (val)"),
        ("val_auc", "AUC de validación"),
        ("val_loss", "Loss focal de validación"),
        ("n_hard_positives", "Positivos que siguen mal (error suavizado > 0.5)"),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4))
    plotted_any = False
    for name, history in histories.items():
        if not history:
            continue
        table = pd.DataFrame(history).set_index("epoch")
        plotted_any = True
        for axis, (column, _) in zip(axes, panels):
            if column not in table.columns:
                continue
            axis.plot(table.index, table[column], marker="o", label=name)
    if not plotted_any:
        axes[0].text(0.5, 0.5, "Sin historiales de entrenamiento", ha="center", va="center")
    for axis, (_, title) in zip(axes, panels):
        axis.set(title=title, xlabel="Época")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=110, bbox_inches="tight")
    return figure
