"""Reorganización única: mueve figuras/Grad-CAM a results/ y archiva scripts auxiliares.

No borra checkpoints ni datasets. Idempotente: omite si el destino ya existe.
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# origen relativo -> destino relativo bajo PROJECT_ROOT
FIGURE_MOVES: list[tuple[str, str]] = [
    ("artifacts/figures/recall", "results/current/figures/recall_05"),
    ("artifacts/figures/recall_focus", "results/experiments/recall_focus/figures"),
    ("artifacts/figures/model_zoo_recall", "results/experiments/model_zoo_recall/figures"),
    ("artifacts/figures/tuned_training_curves.png", "results/baselines/figures/tuned_training_curves.png"),
    ("artifacts/figures/tuned_nocrop_training_curves.png", "results/baselines/figures/tuned_nocrop_training_curves.png"),
    ("artifacts/figures/tuned_confusion_matrix_test.png", "results/baselines/figures/tuned_confusion_matrix_test.png"),
    ("artifacts/figures/tuned_nocrop_confusion_matrix_test.png", "results/baselines/figures/tuned_nocrop_confusion_matrix_test.png"),
    ("artifacts/figures/preprocessing_check_letterbox.png", "results/archive/figures/preprocessing_check_letterbox.png"),
    ("artifacts/optuna/all_trials.png", "results/experiments/optuna/all_trials.png"),
    ("artifacts/optuna/optimization_history.png", "results/experiments/optuna/optimization_history.png"),
]

GRADCAM_MOVES: list[tuple[str, str]] = [
    ("artifacts/gradcam/recall", "results/current/gradcam/recall_05"),
    ("artifacts/gradcam/recall_focus", "results/experiments/recall_focus/gradcam"),
    ("artifacts/gradcam/model_zoo_recall", "results/experiments/model_zoo_recall/gradcam"),
    ("artifacts/gradcam/nocrop", "results/baselines/gradcam/nocrop"),
    ("artifacts/gradcam/baseline_TP.png", "results/baselines/gradcam/crop/baseline_TP.png"),
    ("artifacts/gradcam/baseline_TN.png", "results/baselines/gradcam/crop/baseline_TN.png"),
    ("artifacts/gradcam/baseline_FP.png", "results/baselines/gradcam/crop/baseline_FP.png"),
    ("artifacts/gradcam/baseline_FN.png", "results/baselines/gradcam/crop/baseline_FN.png"),
    ("artifacts/gradcam/tuned_TP.png", "results/baselines/gradcam/crop/tuned_TP.png"),
    ("artifacts/gradcam/tuned_TN.png", "results/baselines/gradcam/crop/tuned_TN.png"),
    ("artifacts/gradcam/tuned_FP.png", "results/baselines/gradcam/crop/tuned_FP.png"),
    ("artifacts/gradcam/tuned_FN.png", "results/baselines/gradcam/crop/tuned_FN.png"),
]

METRIC_COPIES: list[tuple[str, str]] = [
    ("artifacts/recall/final_summary.json", "results/current/metrics/final_summary_05_recall.json"),
    ("artifacts/recall_focus/summary.json", "results/experiments/recall_focus/summary.json"),
    ("artifacts/model_zoo_recall/summary.json", "results/experiments/model_zoo_recall/summary.json"),
    ("artifacts/optuna/best_params.json", "results/experiments/optuna/best_params.json"),
]


def move_path(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def copy_path(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    for rel_src, rel_dst in FIGURE_MOVES + GRADCAM_MOVES:
        move_path(PROJECT_ROOT / rel_src, PROJECT_ROOT / rel_dst)

    for rel_src, rel_dst in METRIC_COPIES:
        copy_path(PROJECT_ROOT / rel_src, PROJECT_ROOT / rel_dst)

    archive_scripts = PROJECT_ROOT / "archive" / "scripts"
    archive_scripts.mkdir(parents=True, exist_ok=True)
    logs_dir = PROJECT_ROOT / "artifacts" / "logs"
    for helper in logs_dir.glob("_*.py"):
        dest = archive_scripts / helper.name
        if not dest.exists():
            shutil.move(str(helper), str(dest))

    # Notebook 01 en carpeta de datos (skill 01_data)
    nb_src = PROJECT_ROOT / "notebooks" / "01_build_datasets.ipynb"
    nb_dst = PROJECT_ROOT / "notebooks" / "01_data" / "01_build_datasets.ipynb"
    if nb_src.exists() and not nb_dst.exists():
        nb_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(nb_src), str(nb_dst))

    print("Reorganización completada.")


if __name__ == "__main__":
    main()
