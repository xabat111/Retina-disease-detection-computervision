# Notebooks — RFMiD `Disease_Risk`

Orden de ejecución y carpetas. Cada notebook resuelve la raíz del proyecto subiendo directorios hasta encontrar `src/` (funciona desde `notebooks/`, `best_model/` u `other_experiments/`).

## Orden de ejecución

1. **`01_data/01_build_datasets.ipynb`** — DataFrames train / val / test. No entrena.
2. **`best_model/02_train_resnet152_nocrop.ipynb`** — baseline ResNet-152 **sin recorte** (letterbox), checkpoint por `val_loss`.
3. **`best_model/02.1_train_resnet152_nocrop.ipynb`** — mismos hiperparámetros que Optuna del notebook 03, letterbox. Produce `best_resnet152_disease_risk_optuna_nocrop.pt`.
4. **`best_model/05_improve_recall.ipynb`** — umbral, TTA y ensemble sobre los ResNet-152 ya entrenados. **Sistema de referencia** (test): FN 36, FP 16, recall 0.929, spec 0.881, acc 0.919, F1 0.948, AUC 0.971.
5. **`best_model/06_recall_focused_training_resnet152.ipynb`** — fine-tuning centrado en recall (re-ponderación por época + loss/checkpoint orientados a FN) partiendo del checkpoint de 02.1 nocrop; se combina con el ensemble+TTA de 05. Umbral y miembros **solo con validación**.
6. **`other_experiments/07_train_model_zoo_recall_nocrop.ipynb`** — ResNet-50, DenseNet-121, EfficientNet-B0 y ConvNeXt-Tiny **sin Optuna**, letterbox, receta de recall de 06; después comparación (tabla, ROC, Grad-CAM) y, opcional, ensemble+TTA. Después de 06.

## `best_model/` — pipeline actual

El mejor sistema individual + el análisis de recall que lo convierte en el sistema final de 05/06.

| Notebook | Qué hace |
|---|---|
| `02_train_resnet152_nocrop.ipynb` | Baseline letterbox |
| `02.1_train_resnet152_nocrop.ipynb` | Ajuste Optuna + letterbox (mejor ResNet-152 individual) |
| `05_improve_recall.ipynb` | Ensemble + TTA + umbral de recall (referencia FN 36 / FP 16) |
| `06_recall_focused_training_resnet152.ipynb` | Fine-tuning centrado en recall: FN &lt; FP con caída mínima de acc/F1/AUC |

## `other_experiments/` — el resto

Experimentos que no son el mejor pipeline (crop aleatorio, Optuna original, comparación de arquitecturas).

| Notebook | Qué hace |
|---|---|
| `02_train_resnet152.ipynb` | Baseline original (RandomResizedCrop) |
| `02.1_train_resnet152.ipynb` | Optuna reentrenado con crop |
| `03_optuna_resnet152.ipynb` | Búsqueda Optuna (AUC de validación, régimen reducido) |
| `07_train_model_zoo_recall_nocrop.ipynb` | Zoo (ResNet-50 / DenseNet-121 / EfficientNet-B0 / ConvNeXt-Tiny) **sin** Optuna, letterbox, receta de 06 → comparación (tabla / ROC / Grad-CAM) → ensemble opcional |

Flujo del notebook 07: **entrenar zoo → comparar (tabla / ROC / Grad-CAM) → (opcional) ensemble+TTA**.

Si 06 o 07 superan al sistema de 05 en validación (y se confirma en test), el README y las carpetas se actualizan: el notebook ganador pasa a `best_model/`.

## Sin fuga de datos

Entrenamiento solo con *train*. Validación elige época, hiperparámetros, miembros del ensemble y umbral. Test solo informa.
