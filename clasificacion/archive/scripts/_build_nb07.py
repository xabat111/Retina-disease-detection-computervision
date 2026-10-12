"""Genera notebooks/other_experiments/07_train_model_zoo_recall_nocrop.ipynb.

Flujo UX (como 04, no un pipeline opaco de cachés):
  1) hiperparámetros fijos + datos
  2) entrenar zoo (reanudable)
  3) comparar: tabla val/test, ROC, Grad-CAM (mismas imágenes)
  4) opcional: ensemble + TTA (con markdown que explica cachés y vistas)
"""

import sys
from pathlib import Path

sys.path.insert(0, "artifacts/logs")
from _nbtools import SETUP_PROJECT_ROOT, code_cell, markdown_cell, write_notebook

OUTPUT_PATH = Path("notebooks/other_experiments/07_train_model_zoo_recall_nocrop.ipynb")
if len(sys.argv) > 1:
    OUTPUT_PATH = Path(sys.argv[1])

cells = []

cells.append(markdown_cell("""
# RFMiD — zoo de arquitecturas **sin Optuna** (letterbox + recall), comparación al estilo de `04`

Misma idea que `other_experiments/04_compare_models.ipynb` (tabla clara, ROC superpuestas, Grad-CAM lado a lado en **las mismas imágenes**), pero:

- **Sin Optuna:** hiperparámetros fijos razonables por familia (AdamW, cosine, batch 16).
- **Sin recorte:** `preprocessing="letterbox"` (el ojo entero), 224 px.
- **Receta de recall del notebook 06:** Focal Loss con `alpha` orientada a positivos, re-ponderación por época de ejemplos mal clasificados (más peso a FN) y checkpoint por `spec_at_fn_lt_fp`.

**Flujo del notebook**

1. Entrenar ResNet-50, DenseNet-121, EfficientNet-B0 y ConvNeXt-Tiny.
2. **Comparar** (sección principal, como 04): métricas val/test, ROC, Grad-CAM.
3. *(Opcional)* Ensemble + TTA al final, con explicación de qué son las cachés `.npz` y las vistas TTA.

Referencia de comparación: ResNet-152 ajustado letterbox (`best_resnet152_disease_risk_optuna_nocrop.pt`). Grad-CAM usa `artifacts/gradcam/selected_images.json` (mismas imágenes que el ResNet-152 del resto del proyecto).

**Sin fuga de datos:** train entrena; validación elige época y umbral; test solo informa.

**Expectativa honesta:** estos modelos parten de ImageNet (no de un fondo de ojo ya convergido) y con 1920 imágenes de train; no se espera que superen al ResNet-152 afinado. El interés es cuantificar el hueco y ver si aportan diversidad.
"""))

cells.append(code_cell("""
import json
import os
import sys
from pathlib import Path
""" + SETUP_PROJECT_ROOT + """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.cached_datasets import build_cached_letterbox_splits
from src.comparison_plots import plot_metric_bars
from src.data import PREPROCESSING_LETTERBOX
from src.ensemble_evaluation import (
    FN_BELOW_FP_RULE,
    RECALL_TARGET_RULE,
    average_over_members,
    average_over_views,
    build_model_comparison_table,
    choose_operating_thresholds,
    load_member_views,
    summary_for_json,
)
from src.evaluate import select_threshold_youden
from src.gradcam_compare import GROUP_ORDER, compute_gradcam_for_selection, plot_gradcam_model_comparison
from src.hard_example_plots import plot_training_histories
from src.models_zoo import get_architecture_spec, load_architecture_checkpoint
from src.recall_config import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PREDICTION_CACHE_DIR as PREVIOUS_PREDICTION_CACHE_DIR,
    TARGET_RECALL,
    VALIDATION_RECALL_MARGIN,
    existing_checkpoint_members,
)
from src.recall_focus_config import (
    CHECKPOINT_DIR as RESNET152_CHECKPOINT_DIR,
    PREDICTION_CACHE_DIR as RECALL_FOCUS_CACHE_DIR,
    SELECTED_GRADCAM_IMAGES_PATH,
    SUMMARY_PATH as RECALL_FOCUS_SUMMARY_PATH,
    VARIANTS,
    WARM_START_CHECKPOINT,
)
from src.recall_optimization import (
    bootstrap_metric_differences,
    choose_ensemble_subset,
    compute_operating_point_metrics,
    rank_ensemble_subsets,
)
from src.recall_plots import plot_confusion_matrices, plot_roc_and_pr_curves
from src.recall_report import build_summary_table, summarize_operating_point
from src.train import get_device
from src.tta_inference import DEFAULT_TTA_VIEW_NAMES, EnsembleMember
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
import importlib
import src.zoo_recall_runner as zoo_recall_runner

importlib.reload(zoo_recall_runner)
from src.zoo_recall_runner import train_architecture_if_missing

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
device = get_device()
validation_recall_goal = TARGET_RECALL + VALIDATION_RECALL_MARGIN
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("=== NB07 SETUP v3 ===")
print(f"device={device}")
print(f"PROJECT_ROOT={PROJECT_ROOT}")
print(f"DATA_ROOT={DATA_ROOT}")
print(f"checkpoints={CHECKPOINT_DIR}")
"""))

cells.append(markdown_cell("""
## 1. Arquitecturas e hiperparámetros fijos

Todos: AdamW, **batch 16**, scheduler **cosine**, imagen 224 px letterbox, backbone congelado salvo los últimos grupos de bloques + cabeza. La receta de loss/muestreo es la de recall del notebook 06 (variante recall-centrada; si 06 eligió solo el control, aquí se usa `both` porque estos modelos parten de ImageNet). **No hay búsqueda con validación.**
"""))

cells.append(code_cell("""
recipe_alpha, recipe_hard_config = recall_recipe()
architecture_table = pd.DataFrame(
    [
        {
            "modelo": get_architecture_spec(architecture).display_name,
            "clave": architecture,
            "lr": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "bloques descongelados": settings.unfrozen_blocks,
            "checkpoint": settings.checkpoint_name(architecture),
            "justificación": settings.rationale,
        }
        for architecture, settings in ZOO_SETTINGS.items()
    ]
)
display(architecture_table.set_index("modelo"))
print(
    f"Receta recall (variante '{chosen_recall_recipe_name()}'): focal_alpha={recipe_alpha}, "
    f"re-ponderación={recipe_hard_config}\\n"
    f"épocas={ZOO_EPOCHS} | criterio de checkpoint={build_zoo_train_config('resnet50').selection_metric}"
)
"""))

cells.append(markdown_cell("""
## 2. Datos

Mismas imágenes letterbox en RAM que en el notebook 06 (`artifacts/cache/letterbox224_*.npy`).
"""))

cells.append(code_cell("""
splits = build_cached_letterbox_splits(build_zoo_train_config("resnet50"))
val_labels = splits.val_dataframe["disease_risk"].to_numpy(dtype=float)
test_labels = splits.test_dataframe["disease_risk"].to_numpy(dtype=float)
labels_by_split = {"val": val_labels, "test": test_labels}
dataframes = {"val": splits.val_dataframe, "test": splits.test_dataframe}
val_loader = DataLoader(splits.val_dataset, batch_size=32, shuffle=False)
print({name: len(df) for name, df in dataframes.items()}, "train:", len(splits.train_dataframe))
"""))

cells.append(markdown_cell("""
## 3. Entrenamiento (reanudable)

`scripts/train_zoo_recall.py` puede entrenar en segundo plano. Si ya existen checkpoint + historial, esta celda solo los carga. **Nunca sobrescribe un checkpoint ajeno.**
"""))

cells.append(code_cell("""
histories = {architecture: train_architecture_if_missing(architecture, splits, val_loader) for architecture in ARCHITECTURES}

time_table = pd.DataFrame(
    [
        {
            "modelo": get_architecture_spec(architecture).display_name,
            "épocas": len(history),
            "s/época (train+val)": float(np.mean([row["epoch_seconds"] for row in history])),
            "horas totales": sum(row["epoch_seconds"] for row in history) / 3600,
            "mejor época (val)": max(history, key=lambda row: row["val_spec_at_fn_lt_fp"])["epoch"],
            "spec@FN<FP val": max(history, key=lambda row: row["val_spec_at_fn_lt_fp"])["val_spec_at_fn_lt_fp"],
            "AUC val (mejor)": max(history, key=lambda row: row["val_spec_at_fn_lt_fp"])["val_auc"],
        }
        for architecture, history in histories.items()
    ]
).set_index("modelo")
display(time_table.round(4))
print(f"Tiempo total de entrenamiento: {time_table['horas totales'].sum():.2f} h")

plot_training_histories(
    {get_architecture_spec(architecture).display_name: history for architecture, history in histories.items()},
    save_path=FIGURES_DIR / "training_curves.png",
)
plt.show()
"""))

cells.append(markdown_cell("""
## 4. Comparación (como el notebook 04)

Sección principal. Se evalúa cada modelo del zoo **sin TTA** (vista `identity`) más el ResNet-152 letterbox de referencia. El umbral de cada fila es **Youden sobre validación** (igual que en 04); test solo informa.

Las probabilidades se guardan en `artifacts/model_zoo_recall/predictions/*.npz` la primera vez que se calculan: así re-ejecutar esta sección no vuelve a pasar el modelo por CPU/GPU.
"""))

cells.append(code_cell("""
reference_member = EnsembleMember(
    "resnet152_optuna_nocrop",
    WARM_START_CHECKPOINT,
    PREPROCESSING_LETTERBOX,
    architecture="resnet152",
)
zoo_members = [
    EnsembleMember(
        f"zoo_{architecture}",
        CHECKPOINT_DIR / ZOO_SETTINGS[architecture].checkpoint_name(architecture),
        PREPROCESSING_LETTERBOX,
        architecture=architecture,
    )
    for architecture in ARCHITECTURES
]
comparison_members = [reference_member, *zoo_members]
missing = [member.name for member in comparison_members if not member.checkpoint_path.is_file()]
if missing:
    raise FileNotFoundError(
        "Faltan checkpoints para la comparación: " + ", ".join(missing) + ". "
        "Ejecuta la celda de entrenamiento (o scripts/train_zoo_recall.py) antes."
    )

# Solo la vista identity (= sin TTA). Los .npz viven en PREDICTION_CACHE_DIR.
plain_views = load_member_views(
    comparison_members, dataframes, PREDICTION_CACHE_DIR, device, view_names=("identity",)
)
plain_probabilities = average_over_views(plain_views, use_tta=False)

display_names = {
    "resnet152_optuna_nocrop": "ResNet-152 optuna nocrop (ref.)",
    **{f"zoo_{architecture}": get_architecture_spec(architecture).display_name for architecture in ARCHITECTURES},
}

comparison_rows = []
for member in comparison_members:
    name = member.name
    threshold = select_threshold_youden(plain_probabilities[name]["val"], val_labels)
    val_metrics = compute_operating_point_metrics(plain_probabilities[name]["val"], val_labels, threshold)
    test_metrics = compute_operating_point_metrics(plain_probabilities[name]["test"], test_labels, threshold)
    (test_tn, test_fp), (test_fn, test_tp) = test_metrics["confusion_matrix"]
    (_, val_fp), (val_fn, _) = val_metrics["confusion_matrix"]
    comparison_rows.append(
        {
            "modelo": display_names[name],
            "umbral (val, Youden)": threshold,
            "AUC val": val_metrics["auc"],
            "AUC test": test_metrics["auc"],
            "F1 test": test_metrics["f1"],
            "Accuracy test": test_metrics["accuracy"],
            "Precision test": test_metrics["precision"],
            "Recall test": test_metrics["recall"],
            "Especificidad test": test_metrics["specificity"],
            "val FN / FP": f"{val_fn} / {val_fp}",
            "TN": test_tn,
            "FP": test_fp,
            "FN": test_fn,
            "TP": test_tp,
        }
    )

comparison_table = pd.DataFrame(comparison_rows).sort_values("AUC test", ascending=False).reset_index(drop=True)
comparison_table.to_csv(FIGURES_DIR / "compare_metrics.csv", index=False)
display(comparison_table.round(4))
"""))

cells.append(code_cell("""
plot_metric_bars(comparison_table, save_path=FIGURES_DIR / "compare_metric_bars.png")
plt.show()

plot_roc_and_pr_curves(
    curves={
        display_names[name]: (plain_probabilities[name]["test"], test_labels)
        for name in display_names
    },
    operating_points={},
    title="ROC y Precisión-Recall en test (sin TTA; umbral Youden de validación)",
    save_path=FIGURES_DIR / "compare_roc_pr_test.png",
)
plt.show()

plot_confusion_matrices(
    {
        display_names[name]: compute_operating_point_metrics(
            plain_probabilities[name]["test"],
            test_labels,
            select_threshold_youden(plain_probabilities[name]["val"], val_labels),
        )["confusion_matrix"]
        for name in display_names
    },
    title="Matrices de confusión en test (umbral Youden de validación)",
    save_path=FIGURES_DIR / "compare_confusion_matrices_test.png",
)
plt.show()
"""))

cells.append(markdown_cell("""
### Grad-CAM en las mismas imágenes que el ResNet-152

Se reutiliza `artifacts/gradcam/selected_images.json` (TP/TN/FP/FN más confiados del baseline ResNet-152, umbral 0.5). Cada columna es un modelo; verde = acierta con su umbral de validación, rojo = falla.
"""))

cells.append(code_cell("""
selected_images = json.loads(SELECTED_GRADCAM_IMAGES_PATH.read_text(encoding="utf-8"))["groups"]
print({group: [image["image_name"] for image in images] for group, images in selected_images.items()})

gradcam_by_model = {}
decision_thresholds = {}
for member in comparison_members:
    display_name = display_names[member.name]
    model = load_architecture_checkpoint(member.architecture, member.checkpoint_path, device)
    gradcam_by_model[display_name] = compute_gradcam_for_selection(
        model, member.architecture, selected_images, member.image_size, PREPROCESSING_LETTERBOX
    )
    decision_thresholds[display_name] = select_threshold_youden(
        plain_probabilities[member.name]["val"], val_labels
    )
    del model

for group_name in GROUP_ORDER:
    plot_gradcam_model_comparison(
        group_name,
        selected_images,
        gradcam_by_model,
        decision_thresholds,
        save_path=GRADCAM_DIR / f"compare_{group_name}.png",
    )
    plt.show()
"""))

cells.append(markdown_cell("""
## 5. (Opcional) Ensemble + TTA

Esta sección **no es necesaria** para comparar arquitecturas; sirve si quieres ver si promediar modelos o vistas mejora el punto de operación FN/FP.

### Qué son las cachés `.npz`

Al inferir un modelo en CPU/GPU se guardan las **probabilidades** (`P(Disease_Risk)`) de cada imagen en ficheros `.npz` bajo `artifacts/model_zoo_recall/predictions/` (y, para los ResNet-152 previos, en `artifacts/recall/predictions/` o `artifacts/recall_focus/predictions/`). La siguiente vez se leen del disco: se evita repetir el forward, que es lo caro. No son pesos del modelo ni imágenes.

### Qué son las vistas TTA

TTA = *Test-Time Augmentation*. Sobre la imagen letterbox ya preprocesada se aplican transformaciones deterministas que no deberían cambiar el diagnóstico y se promedian las probabilidades:

| Vista | Transformación |
|---|---|
| `identity` | imagen tal cual (= predicción sin TTA) |
| `hflip` | volteo horizontal |
| `rotate_+10` / `rotate_-10` | rotación ±10° (relleno negro) |

Cada vista tiene su propio `.npz`. El ensemble promedia miembros; el TTA promedia vistas de un mismo miembro.
"""))

cells.append(code_cell("""
# Carga opcional: ResNet-152 de 05/06 + zoo, las 4 vistas TTA (reutiliza .npz si existen).
recall_focus_summary = json.loads(RECALL_FOCUS_SUMMARY_PATH.read_text(encoding="utf-8"))
resnet152_recall_name = f"recallfocus_{recall_focus_summary['chosen_variant_validation']}"
resnet152_recall_member = EnsembleMember(
    resnet152_recall_name,
    RESNET152_CHECKPOINT_DIR / VARIANTS[recall_focus_summary["chosen_variant_validation"]].checkpoint_name,
    PREPROCESSING_LETTERBOX,
)
previous_members = existing_checkpoint_members()
previous_member_names = tuple(member.name for member in previous_members)

all_views = {
    **load_member_views(previous_members, dataframes, PREVIOUS_PREDICTION_CACHE_DIR, device),
    **load_member_views([resnet152_recall_member], dataframes, RECALL_FOCUS_CACHE_DIR, device),
    **load_member_views(zoo_members, dataframes, PREDICTION_CACHE_DIR, device),
}
tta_probabilities = average_over_views(all_views, use_tta=True)

tta_display_names = {
    "ajustado_nocrop": "ResNet-152 ajustado (05)",
    resnet152_recall_name: f"ResNet-152 recall-focus (06: {recall_focus_summary['chosen_variant_validation']})",
    **{f"zoo_{architecture}": get_architecture_spec(architecture).display_name + " (07)" for architecture in ARCHITECTURES},
}

print("Modelos individuales con TTA; umbral FN < FP (elegido solo en validación)")
display(
    build_model_comparison_table(
        {tta_display_names[name]: tta_probabilities[name] for name in tta_display_names},
        labels_by_split,
        FN_BELOW_FP_RULE,
        validation_recall_goal,
        BOOTSTRAP_SEED,
    ).round(4)
)

plot_roc_and_pr_curves(
    curves={tta_display_names[name]: (tta_probabilities[name]["test"], test_labels) for name in tta_display_names},
    operating_points={},
    title="ROC / PR en test con TTA (sección opcional)",
    save_path=FIGURES_DIR / "roc_pr_test_tta_optional.png",
)
plt.show()
"""))

cells.append(code_cell("""
zoo_member_names = tuple(member.name for member in zoo_members)
ranking = rank_ensemble_subsets({name: probabilities["val"] for name, probabilities in tta_probabilities.items()}, val_labels)
validation_best_subset = choose_ensemble_subset(ranking)
print(f"Subconjunto elegido con VALIDACIÓN ({len(tta_probabilities)} miembros): {validation_best_subset}")

best_of_06_members = tuple(recall_focus_summary["ensemble_members"])
systems = {
    "Anterior (05): 4 ResNet-152": average_over_members(tta_probabilities, previous_member_names),
    "Mejor de 06": average_over_members(tta_probabilities, best_of_06_members),
    "Zoo: 4 modelos nuevos": average_over_members(tta_probabilities, zoo_member_names),
    "4 nuevos + ResNet-152 recall-focus": average_over_members(
        tta_probabilities, zoo_member_names + (resnet152_recall_name,)
    ),
    "Subconjunto elegido por AUC val": average_over_members(tta_probabilities, validation_best_subset),
}

configurations = {}
for system_name, probabilities in systems.items():
    thresholds = choose_operating_thresholds(probabilities["val"], val_labels, validation_recall_goal, BOOTSTRAP_SEED)
    for rule_name, threshold in thresholds.items():
        configurations[f"{system_name} | {rule_name}"] = (probabilities, threshold)

val_summaries, test_summaries = {}, {}
for label, (probabilities, threshold) in configurations.items():
    val_summaries[label] = summarize_operating_point(
        probabilities["val"], val_labels, threshold, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, with_intervals=False
    )
    test_summaries[label] = summarize_operating_point(
        probabilities["test"], test_labels, threshold, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )

for rule_name in (FN_BELOW_FP_RULE, RECALL_TARGET_RULE):
    print(f"TEST (IC 95 % bootstrap), regla: {rule_name}")
    display(build_summary_table({key: value for key, value in test_summaries.items() if key.endswith(rule_name)}))

new_label = f"4 nuevos + ResNet-152 recall-focus | {FN_BELOW_FP_RULE}"
best_of_06_label = f"Mejor de 06 | {FN_BELOW_FP_RULE}"
paired = bootstrap_metric_differences(
    configurations[new_label][0]["test"], configurations[new_label][1],
    configurations[best_of_06_label][0]["test"], configurations[best_of_06_label][1],
    test_labels, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
)
print("Diferencias pareadas (4 nuevos+RN152 rf − Mejor de 06), regla FN < FP")
display(
    pd.DataFrame(
        {metric: {"diferencia": difference, "IC95 inf": lo, "IC95 sup": hi} for metric, (difference, lo, hi) in paired.items()}
    ).T.round(4)
)
"""))

cells.append(markdown_cell("""
## 6. Resumen guardado y conclusiones

> Rellenar tras ejecutar con números reales. Plantilla:

**Comparación individual (sección 4, sin TTA)**
- Mejor zoo por AUC test: _…_ (¿cerca del ResNet-152 de referencia?)
- ¿Algún modelo aporta recall/especificidad distinto al ResNet-152?

**Grad-CAM**
- ¿Algún modelo se apoya en bordes/artefactos en lugar de lesiones?

**Ensemble opcional**
- ¿Mejora el zoo al ensemble de 06 en validación? ¿Se confirma en test?

**Limitaciones:** 12 épocas desde ImageNet; AUC con 134 negativos en validación es ruidoso; Grad-CAM es cualitativo.
"""))

cells.append(code_cell("""
individual_test_rows = {}
for member in comparison_members:
    name = member.name
    threshold = select_threshold_youden(plain_probabilities[name]["val"], val_labels)
    individual_test_rows[display_names[name]] = summary_for_json(
        summarize_operating_point(
            plain_probabilities[name]["test"], test_labels, threshold, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
        )
    ) | {"threshold_rule": "Youden (validation)", "threshold": threshold}

# La sección 5 (ensemble+TTA) es opcional: solo se serializa si se ejecutó.
optional_ensemble = {}
if "configurations" in globals() and "test_summaries" in globals():
    optional_ensemble = {
        "ensemble_members_validation_best": list(validation_best_subset),
        "thresholds_optional_ensemble": {label: configurations[label][1] for label in configurations},
        "val_metrics_optional_ensemble": {label: summary_for_json(s) for label, s in val_summaries.items()},
        "test_metrics_optional_ensemble": {label: summary_for_json(s) for label, s in test_summaries.items()},
        "paired_difference_optional": {
            metric: {"difference": difference, "ci95": [lo, hi]}
            for metric, (difference, lo, hi) in paired.items()
        },
    }

summary = {
    "goal": "entrenar zoo sin Optuna (letterbox + recall) y comparar como el notebook 04; ensemble+TTA opcional",
    "recipe": {
        "variant_from_06": chosen_recall_recipe_name(),
        "focal_alpha": recipe_alpha,
        "epochs": ZOO_EPOCHS,
    },
    "hyperparameters": {
        architecture: {
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "unfrozen_blocks": settings.unfrozen_blocks,
            "rationale": settings.rationale,
        }
        for architecture, settings in ZOO_SETTINGS.items()
    },
    "comparison_table_youden_no_tta": comparison_table.to_dict(orient="records"),
    "individual_models_youden_test": individual_test_rows,
    "tta_views": list(DEFAULT_TTA_VIEW_NAMES),
    "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "confidence_level": 0.95, "seed": BOOTSTRAP_SEED},
    **optional_ensemble,
}
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("Guardado en", SUMMARY_PATH)
"""))

write_notebook(OUTPUT_PATH, cells)
print("escrito", OUTPUT_PATH, len(cells), "celdas")
