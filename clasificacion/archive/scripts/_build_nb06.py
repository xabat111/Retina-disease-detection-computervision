import sys
from pathlib import Path

sys.path.insert(0, "artifacts/logs")
from _nbtools import SETUP_PROJECT_ROOT, code_cell, markdown_cell, write_notebook

OUTPUT_PATH = Path("notebooks/best_model/06_recall_focused_training_resnet152.ipynb")
if len(sys.argv) > 1:
    OUTPUT_PATH = Path(sys.argv[1])

cells = []

cells.append(markdown_cell("""
# RFMiD — último intento: menos falsos negativos que falsos positivos (`Disease_Risk`)

**Objetivo del usuario.** Que en test (y en validación) haya **menos falsos negativos que falsos positivos** (FN < FP) **sin perjudicar** accuracy, F1, AUC ni especificidad más que un mínimo. Referencia a mantener (test, sistema final de `05_improve_recall.ipynb`): FN 36 / FP 16, recall 0.929, especificidad 0.881, accuracy 0.919, F1 0.948, AUC 0.971.

**Tolerancia que se usa aquí (declarada antes de ver resultados):** accuracy, F1 y AUC de test pueden bajar como máximo **0.005** respecto al sistema de `05`. La especificidad **no** se acota (FN < FP con 506 enfermos y 134 sanos obliga a aceptar algún FP más), pero se reporta su caída explícitamente.

**Nota de análisis.** Con 506 positivos y 134 negativos, FN < FP con una especificidad alta exige un recall muy alto (≈ 0.96). Bajar el umbral intercambia FN por FP, y accuracy/F1 casi no se resienten porque el 79 % son enfermos; **la mejora real solo puede venir de ordenar mejor los positivos difíciles** (más AUC en la zona de alto recall). Si no se logra, se dirá.

## Respuestas a las dos preguntas

**(a) ¿Se puede dar más peso a las instancias mal clasificadas en cada época? Sí**, y es gratis: el forward de entrenamiento ya calcula la probabilidad de cada imagen. `src/hard_example_training.py` guarda el **error suavizado** de cada imagen (media móvil exponencial; error = probabilidad asignada a la clase equivocada) y la época siguiente se **muestrea** con un `WeightedRandomSampler` que repite más las imágenes que siguen mal, y más aún los falsos negativos (`positive_boost`). Para no memorizar ruido de etiquetas (enfermedades raras, imágenes ambiguas) el peso tiene un **tope** (`max_sample_weight`) y la media móvil suaviza un error puntual. La Focal Loss ya enfatiza ejemplos difíciles *dentro de cada batch*; esto añade énfasis *entre épocas* en los que siguen siendo difíciles.

**(b) Criterio centrado en recall.** (1) **Loss asimétrica**: la `BinaryFocalLoss` original usaba `alpha=0.26`, que pesa a los **negativos** (0.74) más que a los positivos; aquí se prueba `alpha=0.6` (un positivo pesa ~1.5 veces un negativo). (2) **Selección de checkpoint** con `spec_at_fn_lt_fp` (`src/recall_selection.py`): *la mayor especificidad entre los umbrales con FN < FP en validación*, es decir, justo el coste del objetivo; se calcula sobre todos los umbrales, sin depender de un 0.5 mal calibrado.

## Diseño experimental (2 × 2, todo desde el mejor ResNet-152 individual)

Warm start desde `best_resnet152_disease_risk_optuna_nocrop.pt` (no se modifica), `preprocessing="letterbox"`, hiperparámetros de Optuna salvo `lr` (1e-4 con *cosine*, ya que el modelo está convergido) y 4 épocas de fine-tuning, con `layer3 + layer4` entrenables como en el original. Cuatro variantes: **control** (nada nuevo, para medir el efecto de simplemente entrenar 4 épocas más), **reweight**, **recallloss** y **both**. La mejor variante **según validación** se añade al ensemble de `05` (TTA + umbral elegido en validación).

**Sin fuga:** entrenamiento solo con *train*; validación elige época, variante, miembros y umbral; test solo informa (IC 95 % bootstrap, 1000 remuestras, umbral fijo).
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
from src.gradcam_compare import (
    GROUP_ORDER,
    compute_gradcam_for_selection,
    plot_gradcam_model_comparison,
)
from src.hard_example_plots import plot_training_histories
from src.models_zoo import load_architecture_checkpoint
from src.recall_config import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FINAL_SUMMARY_PATH as PREVIOUS_SUMMARY_PATH,
    PREDICTION_CACHE_DIR as PREVIOUS_PREDICTION_CACHE_DIR,
    TARGET_RECALL,
    VALIDATION_RECALL_MARGIN,
    existing_checkpoint_members,
)
from src.recall_focus_config import (
    CHECKPOINT_DIR,
    FIGURES_DIR,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    GRADCAM_DIR,
    PREDICTION_CACHE_DIR,
    SELECTED_GRADCAM_IMAGES_PATH,
    SELECTION_METRIC,
    SUMMARY_PATH,
    VARIANTS,
    WARM_START_CHECKPOINT,
    build_variant_train_config,
)
from src.recall_focus_runner import train_variant_if_missing
from src.recall_optimization import (
    bootstrap_metric_differences,
    choose_ensemble_subset,
    compute_operating_point_metrics,
    rank_ensemble_subsets,
    sweep_thresholds,
)
from src.recall_plots import plot_confusion_matrices, plot_roc_and_pr_curves, plot_threshold_sweep
from src.recall_report import build_summary_table, miss_rate_by_disease, summarize_operating_point
from src.train import get_device
from src.tta_inference import DEFAULT_TTA_VIEW_NAMES, EnsembleMember

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
device = get_device()
validation_recall_goal = TARGET_RECALL + VALIDATION_RECALL_MARGIN
print(f"device={device} | raíz={PROJECT_ROOT}")
"""))

cells.append(markdown_cell("""
## 1. Variantes y presupuesto

`alpha` es el peso de la clase **positiva** en la Focal Loss (la original usaba 0.26, es decir, 0.74 para los negativos). `fuerza` y `boost_positivos` definen el peso de muestreo `min(tope, 1 + fuerza · error_suavizado · factor_clase)`; el factor de clase es `boost_positivos` para los enfermos (un FN cuesta más) y 1 para los sanos. `fuerza = 0` desactiva la re-ponderación. La primera época es siempre de muestreo uniforme (sirve para registrar los errores iniciales).
"""))

cells.append(code_cell("""
variant_table = pd.DataFrame(
    [
        {
            "variante": variant.name,
            "descripción": variant.description,
            "alpha (peso positivos)": variant.focal_alpha,
            "fuerza re-ponderación": variant.hard_config.weighting_strength,
            "boost positivos": variant.hard_config.positive_boost,
            "tope de peso": variant.hard_config.max_sample_weight,
        }
        for variant in VARIANTS.values()
    ]
).set_index("variante")
display(variant_table)

train_config = build_variant_train_config(VARIANTS["control"])
print(
    f"Warm start: {WARM_START_CHECKPOINT.name}\\n"
    f"épocas={FINE_TUNE_EPOCHS}, lr={FINE_TUNE_LEARNING_RATE} (cosine), batch={train_config.batch_size}, "
    f"gamma={train_config.focal_gamma:.2f}, weight_decay={train_config.weight_decay:.1e}, "
    f"layer3 entrenable={train_config.unfreeze_layer3}, preprocesado={train_config.preprocessing}, "
    f"criterio de checkpoint={SELECTION_METRIC}"
)
"""))

cells.append(markdown_cell("""
## 2. Datos

Las imágenes letterbox se reescalan **una sola vez** y se guardan en RAM/disco (`artifacts/cache/*.npy`): el letterbox es el primer paso determinista de la transformación, así que las entradas son **idénticas** a las de `create_dataloaders(config)` (comprobado tensor a tensor con la misma semilla) y solo se ahorra decodificar PNGs de 2144×1424 en cada época (~20 % menos de tiempo por época).
"""))

cells.append(code_cell("""
splits = build_cached_letterbox_splits(train_config)
val_labels = splits.val_dataframe["disease_risk"].to_numpy(dtype=float)
test_labels = splits.test_dataframe["disease_risk"].to_numpy(dtype=float)
labels_by_split = {"val": val_labels, "test": test_labels}
dataframes = {"val": splits.val_dataframe, "test": splits.test_dataframe}
val_loader = DataLoader(splits.val_dataset, batch_size=32, shuffle=False)
for name, dataframe in [("train", splits.train_dataframe), ("val", splits.val_dataframe), ("test", splits.test_dataframe)]:
    print(name, len(dataframe), dataframe["disease_risk"].value_counts().to_dict())
"""))

cells.append(markdown_cell("""
## 3. Entrenamiento (reanudable)

Cada variante tarda ~1 h en CPU (4 épocas de ~17 min). Si ya existen checkpoint e historial (`scripts/train_recall_focus.py` los deja entrenados en segundo plano) solo se cargan; **nunca se sobrescribe** un checkpoint. La época 0 del historial es el modelo de partida evaluado en validación, y no compite por ser el mejor checkpoint.
"""))

cells.append(code_cell("""
histories = {name: train_variant_if_missing(variant, splits, val_loader) for name, variant in VARIANTS.items()}

time_table = pd.DataFrame(
    [
        {
            "variante": name,
            "épocas entrenadas": len(history) - 1,
            "s/época (train + val)": np.mean([row["epoch_seconds"] for row in history if row["epoch"] >= 1]),
            "minutos totales": sum(row["epoch_seconds"] for row in history) / 60,
        }
        for name, history in histories.items()
    ]
).set_index("variante")
display(time_table.round(1))

plot_training_histories(histories, save_path=FIGURES_DIR / "training_curves.png")
plt.show()
"""))

cells.append(markdown_cell("""
## 4. Elección de la variante (solo validación)

Para cada variante se toma su mejor época según `spec_at_fn_lt_fp` (la misma métrica con la que se guardó su checkpoint). **La variante elegida es la de mayor valor en validación**; el control permite ver cuánto se debe a simplemente entrenar más. Con 134 negativos en validación, una diferencia de 1/134 ≈ 0.0075 en especificidad es **un solo caso**: las diferencias pequeñas son ruido.
"""))

cells.append(code_cell("""
partida = histories["control"][0]
selection_rows = [
    {
        "variante": "partida (época 0)",
        "mejor época": 0,
        "spec@FN<FP": partida["val_spec_at_fn_lt_fp"],
        "AUC val": partida["val_auc"],
        "recall@spec85": partida["val_recall_at_spec_85"],
        "F2 máx.": partida["val_best_f2"],
    }
]
for name, history in histories.items():
    best_row = max((row for row in history if row["epoch"] >= 1), key=lambda row: row["val_spec_at_fn_lt_fp"])
    selection_rows.append(
        {
            "variante": name,
            "mejor época": best_row["epoch"],
            "spec@FN<FP": best_row["val_spec_at_fn_lt_fp"],
            "AUC val": best_row["val_auc"],
            "recall@spec85": best_row["val_recall_at_spec_85"],
            "F2 máx.": best_row["val_best_f2"],
        }
    )
selection_table = pd.DataFrame(selection_rows).set_index("variante")
display(selection_table.round(4))

chosen_variant_name = selection_table.drop(index="partida (época 0)")["spec@FN<FP"].idxmax()
chosen_variant = VARIANTS[chosen_variant_name]
print(f"Variante elegida con VALIDACIÓN: {chosen_variant_name} ({chosen_variant.description})")
"""))

cells.append(markdown_cell("""
## 5. Probabilidades de cada modelo

Se evalúan los cuatro modelos afinados (solo vista original, sin TTA) y el de partida, para la comparación individual. Los cuatro ResNet-152 de `05` usan sus probabilidades ya cacheadas (`artifacts/recall/predictions/`, con TTA); la variante elegida se evalúa además con las 4 vistas TTA.
"""))

cells.append(code_cell("""
previous_members = existing_checkpoint_members()
previous_member_names = tuple(member.name for member in previous_members)
previous_views = load_member_views(previous_members, dataframes, PREVIOUS_PREDICTION_CACHE_DIR, device)

variant_members = {
    name: EnsembleMember(f"recallfocus_{name}", CHECKPOINT_DIR / variant.checkpoint_name, PREPROCESSING_LETTERBOX)
    for name, variant in VARIANTS.items()
}
identity_only = load_member_views(list(variant_members.values()), dataframes, PREDICTION_CACHE_DIR, device, ("identity",))
chosen_member = variant_members[chosen_variant_name]
chosen_views = load_member_views([chosen_member], dataframes, PREDICTION_CACHE_DIR, device)  # 4 vistas TTA

plain_probabilities = {
    **average_over_views(previous_views, use_tta=False),
    **average_over_views(identity_only, use_tta=False),
}
tta_probabilities = {**average_over_views(previous_views, use_tta=True), **average_over_views(chosen_views, use_tta=True)}

individual_models = {
    "partida: ajustado_nocrop": plain_probabilities["ajustado_nocrop"],
    **{f"{name}": plain_probabilities[member.name] for name, member in variant_members.items()},
}
for rule_name in (FN_BELOW_FP_RULE, RECALL_TARGET_RULE):
    print(f"Modelos individuales (sin TTA), umbral '{rule_name}' elegido en validación; test solo informa")
    display(
        build_model_comparison_table(
            individual_models, labels_by_split, rule_name, validation_recall_goal, BOOTSTRAP_SEED
        ).round(4)
    )
"""))

cells.append(markdown_cell("""
## 6. Ensemble con TTA (miembros elegidos con el AUC de validación)

Se parte de los 4 ResNet-152 de `05` y se añade la variante elegida (5 miembros → 31 subconjuntos). Como en `05`, entre los subconjuntos a menos de 0.002 de AUC de validación del mejor se prefiere el **más grande** (más estable). Se comparan tres sistemas:

- **Anterior (05):** los 4 ResNet-152 + TTA.
- **Nuevo solo:** la variante elegida + TTA.
- **Ensemble elegido:** el subconjunto escogido con validación + TTA.
"""))

cells.append(code_cell("""
chosen_name = chosen_member.name
ranking = rank_ensemble_subsets({name: p["val"] for name, p in tta_probabilities.items()}, val_labels)
display(ranking.head(6))
chosen_subset = choose_ensemble_subset(ranking)
print("Subconjunto elegido con VALIDACIÓN:", chosen_subset)

systems = {
    "Anterior (05): 4 ResNet-152 + TTA": average_over_members(tta_probabilities, previous_member_names),
    f"Nuevo solo ({chosen_variant_name}) + TTA": average_over_members(tta_probabilities, (chosen_name,)),
    "Ensemble elegido + TTA": average_over_members(tta_probabilities, chosen_subset),
}
"""))

cells.append(markdown_cell("""
## 7. Punto de operación (umbral elegido solo en validación)

Dos reglas, ambas calculadas con validación:
- **Recall objetivo** (la de `05`): mayor umbral con recall de validación ≥ 0.91.
- **FN < FP**: mayor umbral para el que FN < FP en **al menos el 90 % de las remuestras bootstrap de validación** (con 640 imágenes el cruce observado tiene ruido de varios casos; exigir un margen evita quedarse justo en FN ≈ FP). Es la regla declarada de antemano para el objetivo del usuario.

Primero se comprueba que se reproduce el umbral de `05` (0.3993).
"""))

cells.append(code_cell("""
previous_final_threshold = json.loads(PREVIOUS_SUMMARY_PATH.read_text(encoding="utf-8"))["decision_threshold"]
thresholds_by_system = {
    name: choose_operating_thresholds(probabilities["val"], val_labels, validation_recall_goal, BOOTSTRAP_SEED)
    for name, probabilities in systems.items()
}
reproduced = thresholds_by_system["Anterior (05): 4 ResNet-152 + TTA"][RECALL_TARGET_RULE]
print(f"Umbral recall-objetivo del sistema anterior: reproducido={reproduced:.5f} | guardado en 05={previous_final_threshold:.5f}")
assert abs(reproduced - previous_final_threshold) < 1e-9

previous_name = "Anterior (05): 4 ResNet-152 + TTA"
ensemble_name = "Ensemble elegido + TTA"
configurations = {
    f"{previous_name} | {RECALL_TARGET_RULE}  [= FINAL de 05]": (systems[previous_name], thresholds_by_system[previous_name][RECALL_TARGET_RULE]),
    f"{previous_name} | {FN_BELOW_FP_RULE}  [solo umbral]": (systems[previous_name], thresholds_by_system[previous_name][FN_BELOW_FP_RULE]),
    f"{ensemble_name} | {RECALL_TARGET_RULE}": (systems[ensemble_name], thresholds_by_system[ensemble_name][RECALL_TARGET_RULE]),
    f"{ensemble_name} | {FN_BELOW_FP_RULE}  [NUEVO]": (systems[ensemble_name], thresholds_by_system[ensemble_name][FN_BELOW_FP_RULE]),
}
new_final_label = f"{ensemble_name} | {FN_BELOW_FP_RULE}  [NUEVO]"
previous_final_label = f"{previous_name} | {RECALL_TARGET_RULE}  [= FINAL de 05]"
only_threshold_label = f"{previous_name} | {FN_BELOW_FP_RULE}  [solo umbral]"

val_summaries, test_summaries = {}, {}
for label, (probabilities, threshold) in configurations.items():
    val_summaries[label] = summarize_operating_point(
        probabilities["val"], val_labels, threshold, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, with_intervals=False
    )
    test_summaries[label] = summarize_operating_point(
        probabilities["test"], test_labels, threshold, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )
print("VALIDACIÓN (la que decide)")
display(build_summary_table(val_summaries))
"""))

cells.append(markdown_cell("""
## 8. Resultado en test (IC 95 % bootstrap, umbral fijado en validación)

La fila **«solo umbral»** es el control honesto: el mismo sistema de `05` con la regla FN < FP, sin ninguna mejora del modelo. Lo que el nuevo entrenamiento aporte debe verse **frente a esa fila**, no solo frente al sistema final de `05`.
"""))

cells.append(code_cell("""
print("TEST (IC 95 % bootstrap)")
display(build_summary_table(test_summaries))

differences_to_show = {
    "NUEVO - FINAL de 05": (new_final_label, previous_final_label),
    "NUEVO - solo umbral": (new_final_label, only_threshold_label),
}
paired_differences = {}
for title, (after_label, before_label) in differences_to_show.items():
    after_probabilities, after_threshold = configurations[after_label]
    before_probabilities, before_threshold = configurations[before_label]
    paired_differences[title] = bootstrap_metric_differences(
        after_probabilities["test"], after_threshold, before_probabilities["test"], before_threshold,
        test_labels, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
    )
    print(f"Diferencia {title} (test, bootstrap pareado)")
    display(
        pd.DataFrame(
            {m: {"diferencia": d, "IC95 inf": lo, "IC95 sup": hi} for m, (d, lo, hi) in paired_differences[title].items()}
        ).T.round(4)
    )
"""))

cells.append(code_cell("""
reference_test = test_summaries[previous_final_label]
final_test = test_summaries[new_final_label]
final_val = val_summaries[new_final_label]
tolerance = 0.005

(_, final_fp), (final_fn, _) = final_test["confusion_matrix"]
(_, val_fp), (val_fn, _) = final_val["confusion_matrix"]
checks = {
    "FN < FP en test": final_fn < final_fp,
    "FN < FP en validación": val_fn < val_fp,
    **{
        f"{metric} no baja más de {tolerance} (vs FINAL de 05)": final_test[metric] >= reference_test[metric] - tolerance
        for metric in ("accuracy", "f1", "auc")
    },
}
for description, passed in checks.items():
    print(("CUMPLE   " if passed else "NO CUMPLE"), "-", description)
print(
    f"Test: FN={final_fn}, FP={final_fp} (antes 36/16) | recall {final_test['recall']:.3f} (antes {reference_test['recall']:.3f}) | "
    f"especificidad {final_test['specificity']:.3f} (antes {reference_test['specificity']:.3f})"
)
"""))

cells.append(code_cell("""
SHORT_LABELS = ["05 final (recall>=0.91)", "05 + umbral FN<FP", "nuevo (recall>=0.91)", "NUEVO (FN<FP)"]
plot_confusion_matrices(
    {short_name: summary["confusion_matrix"] for short_name, summary in zip(SHORT_LABELS, test_summaries.values())},
    title="Matrices de confusión en test (filas: real, columnas: predicho)",
    save_path=FIGURES_DIR / "confusion_matrices_test.png",
)
plt.show()

for split_name, split_summaries in (("val", val_summaries), ("test", test_summaries)):
    plot_roc_and_pr_curves(
        curves={
            "Anterior (05)": (systems[previous_name][split_name], labels_by_split[split_name]),
            "Ensemble elegido": (systems[ensemble_name][split_name], labels_by_split[split_name]),
        },
        operating_points={
            "Anterior (05)": (
                configurations[previous_final_label][1],
                split_summaries[previous_final_label]["recall"],
                split_summaries[previous_final_label]["specificity"],
            ),
            "Ensemble elegido": (
                configurations[new_final_label][1],
                split_summaries[new_final_label]["recall"],
                split_summaries[new_final_label]["specificity"],
            ),
        },
        title=f"ROC y Precisión-Recall ({split_name}); puntos = umbrales elegidos en validación",
        save_path=FIGURES_DIR / f"roc_pr_{split_name}.png",
    )
    plt.show()

new_threshold = configurations[new_final_label][1]
for split_name, title_suffix in (("val", "validación (decide)"), ("test", "test (solo informativo)")):
    plot_threshold_sweep(
        sweep_thresholds(systems[ensemble_name][split_name], labels_by_split[split_name]),
        new_threshold, TARGET_RECALL,
        title=f"Métricas vs umbral - ensemble elegido, {title_suffix}",
        save_path=FIGURES_DIR / f"threshold_sweep_{split_name}.png",
    )
    plt.show()
"""))

cells.append(markdown_cell("""
### Contexto: FN y FP en test para todos los umbrales (solo informativo)

Esta tabla **no se usa para elegir nada** (usar test para elegir el umbral sería fuga); sirve para entender cuánto de lo observado se debe a que los 134 sanos de validación resultan más difíciles que los de test (en validación hay que bajar más el umbral para igualar FN y FP).
"""))

cells.append(code_cell("""
rows = []
for threshold in np.arange(0.20, 0.46, 0.025):
    row = {"umbral": round(float(threshold), 3)}
    for label, probabilities in (("anterior", systems[previous_name]), ("nuevo", systems[ensemble_name])):
        for split in ("val", "test"):
            (_, fp), (fn, _) = compute_operating_point_metrics(probabilities[split], labels_by_split[split], threshold)["confusion_matrix"]
            row[f"{label} {split} FN/FP"] = f"{fn}/{fp}"
    rows.append(row)
display(pd.DataFrame(rows).set_index("umbral"))
"""))

cells.append(markdown_cell("""
## 9. Falsos negativos que quedan y Grad-CAM

Grad-CAM sobre **las mismas imágenes** que en el resto del proyecto (`artifacts/gradcam/selected_images.json`: TP/TN/FP/FN más confiados del ResNet-152 baseline), con el modelo de partida y la variante elegida, cada uno con su umbral de validación (regla FN < FP). Verde = acierta, rojo = falla.
"""))

cells.append(code_cell("""
final_predictions = systems[ensemble_name]["test"] >= new_threshold
false_negative_mask = (test_labels == 1) & ~final_predictions
print(f"FN del sistema nuevo en test: {int(false_negative_mask.sum())} de {int(test_labels.sum())} enfermos")
previous_final_predictions = systems[previous_name]["test"] >= configurations[previous_final_label][1]
previous_fn_mask = (test_labels == 1) & ~previous_final_predictions
print(
    f"FN que el sistema nuevo recupera respecto al FINAL de 05: {int((previous_fn_mask & ~false_negative_mask).sum())} | "
    f"FN nuevos que antes se acertaban: {int((false_negative_mask & ~previous_fn_mask).sum())}"
)
display(miss_rate_by_disease(train_config.test_labels_csv, systems[ensemble_name]["test"], new_threshold))
"""))

cells.append(code_cell("""
selected_images = json.loads(SELECTED_GRADCAM_IMAGES_PATH.read_text(encoding="utf-8"))["groups"]
gradcam_models = {
    "ResNet-152 partida": ("ajustado_nocrop", {m.name: m for m in previous_members}["ajustado_nocrop"].checkpoint_path),
    f"ResNet-152 recall-focus ({chosen_variant_name})": (chosen_name, chosen_member.checkpoint_path),
}
gradcam_by_model, gradcam_thresholds = {}, {}
for display_name, (member_name, checkpoint_path) in gradcam_models.items():
    model = load_architecture_checkpoint("resnet152", checkpoint_path, device)
    gradcam_by_model[display_name] = compute_gradcam_for_selection(
        model, "resnet152", selected_images, train_config.image_size, PREPROCESSING_LETTERBOX
    )
    gradcam_thresholds[display_name] = choose_operating_thresholds(
        plain_probabilities[member_name]["val"], val_labels, validation_recall_goal, BOOTSTRAP_SEED
    )[FN_BELOW_FP_RULE]
    del model

for group_name in GROUP_ORDER:
    plot_gradcam_model_comparison(
        group_name, selected_images, gradcam_by_model, gradcam_thresholds,
        save_path=GRADCAM_DIR / f"compare_{group_name}.png",
    )
    plt.show()
"""))

cells.append(markdown_cell("""
## 10. Resumen guardado
"""))

cells.append(code_cell("""
summary = {
    "goal": "FN < FP en test y validación con caída mínima de accuracy, F1, AUC",
    "tolerance_vs_previous_final": tolerance,
    "answer_a_per_epoch_reweighting": "posible y gratis: error suavizado por imagen + WeightedRandomSampler (src/hard_example_training.py)",
    "answer_b_recall_centered_criterion": "focal_alpha orientada a positivos + checkpoint por spec_at_fn_lt_fp (src/recall_selection.py)",
    "warm_start": WARM_START_CHECKPOINT.name,
    "fine_tune": {"epochs": FINE_TUNE_EPOCHS, "learning_rate": FINE_TUNE_LEARNING_RATE, "scheduler": "cosine"},
    "variants": {
        name: {
            "description": variant.description,
            "focal_alpha": variant.focal_alpha,
            "weighting_strength": variant.hard_config.weighting_strength,
            "checkpoint": variant.checkpoint_name,
            "best_epoch_validation": selection_table.loc[name, "mejor época"].item(),
            "val_spec_at_fn_lt_fp": selection_table.loc[name, "spec@FN<FP"].item(),
            "val_auc": selection_table.loc[name, "AUC val"].item(),
        }
        for name, variant in VARIANTS.items()
    },
    "chosen_variant_validation": chosen_variant_name,
    "ensemble_members": list(chosen_subset),
    "tta_views": list(DEFAULT_TTA_VIEW_NAMES),
    "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "confidence_level": 0.95, "seed": BOOTSTRAP_SEED},
    "decision_rules_selected_on": "validation only",
    "thresholds": {label: configurations[label][1] for label in configurations},
    "val_metrics": {label: summary_for_json(s) for label, s in val_summaries.items()},
    "test_metrics": {label: summary_for_json(s) for label, s in test_summaries.items()},
    "paired_differences_test": {
        title: {metric: {"difference": d, "ci95": [lo, hi]} for metric, (d, lo, hi) in diffs.items()}
        for title, diffs in paired_differences.items()
    },
    "checks": {description: bool(passed) for description, passed in checks.items()},
    "false_negatives_test": int(false_negative_mask.sum()),
    "false_positives_test": int(final_fp),
}
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("Guardado en", SUMMARY_PATH)
"""))

write_notebook(OUTPUT_PATH, cells)
print("escrito", OUTPUT_PATH, len(cells), "celdas")

