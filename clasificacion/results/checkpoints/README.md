# Checkpoints (`artifacts/checkpoints/`)

Todos los `.pt` permanecen en **`artifacts/checkpoints/`** (ruta por defecto en `src/config.py` y notebooks). Clasificación conceptual:

## `current/` (producción / mejor sistema)

| Archivo | Origen | Rol |
|---------|--------|-----|
| `best_resnet152_disease_risk_optuna_nocrop.pt` | 02.1 nocrop | Mejor ResNet-152 individual (letterbox); warm-start de 06 |
| Miembros del ensemble de 05/06 | 02, 02.1 (crop + nocrop) | Ver `artifacts/recall/final_summary.json` |

Sistema **de referencia en test** (notebook 05, `FINAL: D: ensemble + TTA + umbral val`): FN **36**, FP **16**, recall **0.929**, AUC test **~0.971** (detalle en `results/current/metrics/final_summary_05_recall.json`).

## `baselines/`

| Archivo | Notebook |
|---------|----------|
| `best_resnet152_disease_risk.pt` | 02 crop |
| `best_resnet152_disease_risk_nocrop.pt` | 02 nocrop |
| `best_resnet152_disease_risk_optuna.pt` | 02.1 crop |

## `experiments/recall_focus/` (notebook 06)

| Archivo | Variante |
|---------|----------|
| `best_resnet152_recallfocus_control_nocrop.pt` | control (elegida en validación) |
| `best_resnet152_recallfocus_reweight_nocrop.pt` | re-ponderación |
| `best_resnet152_recallfocus_recallloss_nocrop.pt` | focal α recall |
| `best_resnet152_recallfocus_both_nocrop.pt` | ambas ideas |

## `experiments/model_zoo/` (notebook 07)

| Archivo | Arquitectura |
|---------|--------------|
| `recallfocus_resnet50_nocrop.pt` | ResNet-50 |
| `recallfocus_densenet121_nocrop.pt` | DenseNet-121 |
| `recallfocus_efficientnet_b0_nocrop.pt` | EfficientNet-B0 |
| `recallfocus_convnext_tiny_nocrop.pt` | ConvNeXt-Tiny |

## `archive/`

| Archivo | Notas |
|---------|--------|
| `archive/control_epoch1_aborted_by_log_encoding_error.pt` | Entrenamiento abortado; conservado por trazabilidad |

Recomendación Git: añadir `*.pt` y `artifacts/cache/` al `.gitignore` (ver raíz del proyecto).
