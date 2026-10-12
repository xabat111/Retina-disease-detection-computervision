# Resultados generados (`results/`)

Vista **organizada** de métricas, figuras y Grad-CAM. Los notebooks siguen escribiendo en `artifacts/figures/` y `artifacts/gradcam/` al re-ejecutarse; tras una corrida conviene volver a sincronizar con `python scripts/_organize_results_once.py` (idempotente) o copiar manualmente lo nuevo aquí.

## Estructura

| Carpeta | Contenido |
|---------|-----------|
| `current/` | Sistema de referencia del notebook **05** (ensemble + TTA + umbral en validación) |
| `baselines/` | Curvas y Grad-CAM de entrenamientos ResNet-152 con/sin letterbox (notebooks 02.x) |
| `experiments/` | Recall-focus (**06**), zoo de arquitecturas (**07**), gráficos Optuna (**03**) |
| `archive/` | Material auxiliar (p. ej. comprobación de letterbox) |

## Métricas JSON (copias de lectura)

- `current/metrics/final_summary_05_recall.json` ← `artifacts/recall/final_summary.json`
- `experiments/recall_focus/summary.json` ← `artifacts/recall_focus/summary.json`
- `experiments/model_zoo_recall/summary.json` ← `artifacts/model_zoo_recall/summary.json`
- `experiments/optuna/best_params.json` ← `artifacts/optuna/best_params.json`

Los directorios de predicciones cacheadas (`.npz`) permanecen en `artifacts/recall/predictions/`, `artifacts/recall_focus/predictions/` y `artifacts/model_zoo_recall/predictions/` porque los notebooks los referencian directamente.

## Checkpoints

Los pesos `.pt` **no** se mueven (evita romper rutas en `src/`). Índice conceptual: `results/checkpoints/README.md`.

## Mapa rápido (antiguo → nuevo) para figuras

| Antes | Ahora |
|-------|--------|
| `artifacts/figures/recall/` | `results/current/figures/recall_05/` |
| `artifacts/gradcam/recall/` | `results/current/gradcam/recall_05/` |
| `artifacts/figures/recall_focus/` | `results/experiments/recall_focus/figures/` |
| `artifacts/gradcam/recall_focus/` | `results/experiments/recall_focus/gradcam/` |
| `artifacts/figures/model_zoo_recall/` | `results/experiments/model_zoo_recall/figures/` |
| `artifacts/gradcam/model_zoo_recall/` | `results/experiments/model_zoo_recall/gradcam/` |
| `artifacts/gradcam/nocrop/` | `results/baselines/gradcam/nocrop/` |
| `artifacts/gradcam/baseline_*.png` (crop) | `results/baselines/gradcam/crop/` |
| `artifacts/gradcam/tuned_*.png` (crop) | `results/baselines/gradcam/crop/` |
| `artifacts/optuna/*.png` | `results/experiments/optuna/` |

`artifacts/gradcam/selected_images.json` sigue en su sitio (entrada común para Grad-CAM comparado).
