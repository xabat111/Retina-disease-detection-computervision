"""Grad-CAM comparado entre arquitecturas sobre EXACTAMENTE las mismas imágenes.

La selección (TP/TN/FP/FN más confiados de test) se hace con el ResNet-152 baseline y el mismo
procedimiento que `notebooks/other_experiments/02_train_resnet152.ipynb` (`select_example_indices`, umbral 0.5, 4 por grupo).
Se persiste en `selected_images.json`; si el archivo ya existe NO se recalcula, de modo que todos
los modelos (y futuras ejecuciones) usan siempre las mismas imágenes.

Nota: elegir ejemplos de test según el modelo es una decisión puramente ilustrativa; no se usa
para ajustar hiperparámetros ni umbrales.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

from src.config import DATA_ROOT, PROJECT_ROOT
from src.data import PREPROCESSING_RANDOM_CROP
from src.explain import (
    GradCAM,
    denormalize_image,
    load_eval_tensor,
    overlay_heatmap,
    select_example_indices,
)
from src.models_zoo import get_gradcam_target_layer

GROUP_ORDER = ("TP", "TN", "FP", "FN")
GROUP_TITLES = {
    "TP": "verdaderos positivos (riesgo detectado)",
    "TN": "verdaderos negativos (sano)",
    "FP": "falsos positivos (sano predicho como riesgo)",
    "FN": "falsos negativos (riesgo no detectado)",
}


def _relative_to_project(image_path: str) -> str:
    """Guarda rutas relativas a DATA_ROOT (RFMiD) o a PROJECT_ROOT si no aplica."""
    path = Path(image_path).resolve()
    for root in (DATA_ROOT, PROJECT_ROOT):
        try:
            return path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return Path(image_path).as_posix()


def _resolve_project_path(stored_path: str) -> Path:
    """Resuelve rutas relativas del JSON Grad-CAM tras el traslado a retina_clasificacion/.

    Las imágenes RFMiD viven en DATA_ROOT (a menudo el padre `Retina/`); el código en PROJECT_ROOT.
    """
    path = Path(stored_path)
    if path.is_absolute():
        if path.is_file():
            return path
        # Ruta absoluta antigua apuntando a PROJECT_ROOT/.../Test_Set → reintentar en DATA_ROOT
        try:
            relative = path.relative_to(PROJECT_ROOT.resolve())
            candidate = DATA_ROOT / relative
            if candidate.is_file():
                return candidate
        except ValueError:
            pass
        return path

    for root in (DATA_ROOT, PROJECT_ROOT):
        candidate = root / path
        if candidate.is_file():
            return candidate
    return DATA_ROOT / path


def load_or_create_selected_images(
    json_path: Path,
    test_dataframe: pd.DataFrame,
    reference_probabilities: np.ndarray,
    reference_labels: np.ndarray,
    reference_description: str,
    threshold: float = 0.5,
    examples_per_group: int = 4,
) -> dict[str, list[dict]]:
    """Devuelve {grupo: [{image_path, image_name, reference_probability, true_label}]}.

    Si `json_path` existe se lee tal cual (garantiza imágenes idénticas entre ejecuciones);
    si no, se calcula con las probabilidades del modelo de referencia y se guarda.
    """
    if json_path.is_file():
        stored = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"Selección existente reutilizada: {json_path}")
        return stored["groups"]

    selected_indices = select_example_indices(
        reference_probabilities, reference_labels, threshold, examples_per_group
    )
    groups: dict[str, list[dict]] = {}
    for group_name in GROUP_ORDER:
        groups[group_name] = [
            {
                "image_path": _relative_to_project(test_dataframe.iloc[index]["image_path"]),
                "image_name": Path(test_dataframe.iloc[index]["image_path"]).name,
                "reference_probability": float(reference_probabilities[index]),
                "true_label": int(reference_labels[index]),
            }
            for index in selected_indices[group_name]
        ]

    payload = {
        "description": (
            "Imágenes de test para comparar Grad-CAM entre modelos. Mismo criterio que "
            "other_experiments/02_train_resnet152.ipynb: ejemplos más confiados de cada grupo TP/TN/FP/FN."
        ),
        "reference_model": reference_description,
        "threshold": threshold,
        "examples_per_group": examples_per_group,
        "groups": groups,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Selección guardada en {json_path}")
    return groups


def compute_gradcam_for_selection(
    model: nn.Module,
    architecture: str,
    selected_images: dict[str, list[dict]],
    image_size: int,
    preprocessing: str = PREPROCESSING_RANDOM_CROP,
) -> dict[str, list[dict]]:
    """Grad-CAM (capa objetivo propia de la arquitectura) para cada imagen seleccionada.

    `preprocessing` debe ser el del entrenamiento del modelo (p. ej. "letterbox" en los *nocrop*).
    """
    grad_cam = GradCAM(model, target_layer=get_gradcam_target_layer(model, architecture))
    results: dict[str, list[dict]] = {}
    try:
        for group_name, images in selected_images.items():
            results[group_name] = []
            for image_info in images:
                image_tensor = load_eval_tensor(
                    str(_resolve_project_path(image_info["image_path"])), image_size, preprocessing
                )
                cam, probability = grad_cam(image_tensor)
                results[group_name].append(
                    {"image": denormalize_image(image_tensor), "cam": cam, "probability": probability}
                )
    finally:
        grad_cam.remove_hooks()
    return results


def plot_gradcam_model_comparison(
    group_name: str,
    selected_images: dict[str, list[dict]],
    gradcam_by_model: dict[str, dict[str, list[dict]]],
    thresholds: dict[str, float],
    save_path: Path | None = None,
):
    """Filas = imágenes del grupo; columnas = original | un mapa Grad-CAM por modelo.

    El título de cada mapa muestra P(riesgo) y la etiqueta real; va en verde si el modelo acierta
    con su umbral (elegido en validación) y en rojo si falla.
    """
    images = selected_images[group_name]
    if not images:
        return None

    model_names = list(gradcam_by_model)
    figure, axes = plt.subplots(
        len(images), len(model_names) + 1, figsize=(3.0 * (len(model_names) + 1), 3.1 * len(images))
    )
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.axis("off")

    for row, image_info in enumerate(images):
        first_model = model_names[0]
        original = gradcam_by_model[first_model][group_name][row]["image"]
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{image_info['image_name']} | real={image_info['true_label']}", fontsize=8)
        for column, model_name in enumerate(model_names, start=1):
            sample = gradcam_by_model[model_name][group_name][row]
            predicted_label = int(sample["probability"] >= thresholds[model_name])
            is_correct = predicted_label == image_info["true_label"]
            axes[row, column].imshow(overlay_heatmap(sample["image"], sample["cam"]))
            axes[row, column].set_title(
                f"{model_name}\nP={sample['probability']:.2f} | real={image_info['true_label']}",
                fontsize=8,
                color="tab:green" if is_correct else "tab:red",
            )

    figure.suptitle(f"Grad-CAM comparado - {group_name}: {GROUP_TITLES[group_name]}", fontsize=12)
    figure.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=110, bbox_inches="tight")
    return figure
