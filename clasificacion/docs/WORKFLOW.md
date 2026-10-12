# Flujo de trabajo — RFMiD `Disease_Risk`

Clasificación binaria de riesgo de enfermedad retiniana (RFMiD). El código vive en **`retina_clasificacion/`**; CSV e imágenes en **`Retina/`** (`DATA_ROOT`).

## Criterios de evaluación (este proyecto)

No se elige el “mejor” modelo solo por accuracy.

1. **Recall** alto y regla operativa **FN &lt; FP** en validación (con variantes bootstrap al 90 % para el umbral).
2. **Caída acotada** respecto al sistema anterior (p. ej. notebook 05): tolerancia **0.005** en accuracy, F1 y AUC en test al añadir recall-focus (06).
3. **AUC** (ROC) como métrica de ranking y selección de checkpoint en entrenamiento (`selection_metric=auc` o `val_loss` según notebook).
4. **Sin fuga:** train → ajuste; **validación** → época, Optuna, miembros del ensemble, umbral, TTA; **test** → informe final únicamente.

Sistema **de referencia** (notebook **05**, ensemble 4× ResNet-152 + TTA, umbral fijado en validación): en test, **FN 36 / FP 16**, recall **0.929**, especificidad **0.881**, accuracy **0.919**, F1 **0.948**, AUC **~0.971** (`artifacts/recall/final_summary.json`, copia en `results/current/metrics/final_summary_05_recall.json`).

Notebook **06** añade fine-tuning recall-focus: en test del ensemble elegido, **FN 28 / FP 22** (mejor recall, pero **FN &lt; FP no se cumple en test**; sí en validación según `artifacts/recall_focus/summary.json`).

## Orden de ejecución de notebooks

| Orden | Notebook | Carpeta | Qué hace |
|------:|----------|---------|----------|
| 1 | `01_build_datasets.ipynb` | `notebooks/01_data/` | Construye DataFrames train/val/test (`image_path`, `disease_risk`). No entrena. |
| 2 | `02_train_resnet152_nocrop.ipynb` | `best_model/` | Baseline ResNet-152 **letterbox** (sin RandomResizedCrop). |
| 3 | `02.1_train_resnet152_nocrop.ipynb` | `best_model/` | Reentrena con hiperparámetros Optuna (03) en letterbox → `best_resnet152_disease_risk_optuna_nocrop.pt`. |
| 4 | `05_improve_recall.ipynb` | `best_model/` | Caché TTA, **ensemble** de checkpoints existentes, umbrales (recall objetivo, FN&lt;FP bootstrap), IC bootstrap, Grad-CAM final. **Sistema de referencia.** |
| 5 | `06_recall_focused_training_resnet152.ipynb` | `best_model/` | Fine-tuning 2×2 (re-ponderación por época + focal α recall); combina con ensemble+TTA de 05; umbral solo en val. |
| 6 | `07_train_model_zoo_recall_nocrop.ipynb` | `other_experiments/` | Zoo (ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny) con receta de 06; tabla/ROC/Grad-CAM; ensemble+TTA opcional. |

### `other_experiments/` (baselines e historia)

| Notebook | Qué hace |
|----------|----------|
| `02_train_resnet152.ipynb` | Baseline con **RandomResizedCrop**. |
| `02.1_train_resnet152.ipynb` | Optuna + crop → checkpoints `*_optuna.pt` / baseline crop. |
| `03_optuna_resnet152.ipynb` | Búsqueda Optuna (AUC val, régimen reducido) → `artifacts/optuna/best_params.json`. |

Cada notebook resuelve `PROJECT_ROOT` subiendo directorios hasta encontrar `src/config.py` + `requirements.txt`.

## Técnicas usadas (mapa código ↔ notebook)

| Técnica | Dónde | Módulos / notas |
|---------|--------|------------------|
| **Letterbox (nocrop)** | 02.x nocrop, 05–07 | `src/data.py` (`LetterboxResize`), caché opcional `src/cached_datasets.py` |
| **Focal Loss** | Todos los entrenamientos | `src/losses.py`, hiperparámetros en Optuna |
| **Optuna (TPE)** | 03, hiperparámetros en 02.1 | `src/tuning.py`, `artifacts/optuna/` |
| **TTA** | 05, 06, 07 | `src/tta_inference.py` (identity, hflip, ±10°) |
| **Ensemble** | 05, 06, 07 | Promedio de probabilidades TTA; `src/ensemble_evaluation.py`, `src/recall_optimization.py` |
| **Umbral Youden** | Comparaciones individuales, baselines | `src/evaluate.py` |
| **Umbral recall / FN&lt;FP + bootstrap** | 05, 06 | `src/recall_optimization.py`, `src/recall_report.py` |
| **Re-ponderación hard examples** | 06 | `src/hard_example_training.py` |
| **Grad-CAM** | 02.x, 05, 06, 07 | `src/explain.py`, `src/gradcam_compare.py` |
| **Bootstrap IC 95 %** | 05, 06 | 1000 remuestreos, semilla 42 (`src/recall_config.py`) |

## Dónde están los resultados

| Tipo | Ubicación |
|------|-----------|
| **Checkpoints `.pt`** | `artifacts/checkpoints/` (índice: `results/checkpoints/README.md`) |
| **Métricas JSON (autoritativo)** | `artifacts/recall/`, `artifacts/recall_focus/`, `artifacts/model_zoo_recall/` |
| **Métricas JSON (copia legible)** | `results/current/metrics/`, `results/experiments/*/summary.json` |
| **Figuras y Grad-CAM organizadas** | `results/current/`, `results/baselines/`, `results/experiments/` |
| **Salida al re-ejecutar notebooks** | `artifacts/figures/`, `artifacts/gradcam/` (ver README en esas carpetas) |
| **Predicciones cacheadas (.npz)** | `artifacts/recall/predictions/`, `artifacts/recall_focus/predictions/`, `artifacts/model_zoo_recall/predictions/` |
| **Logs de entrenamiento / ejecución** | `artifacts/logs/` |
| **Scripts auxiliares históricos** | `archive/scripts/` (antes `artifacts/logs/_*.py`) |

Tras reorganizar figuras: `python scripts/_organize_results_once.py` (idempotente).

## Scripts CLI útiles

| Script | Uso |
|--------|-----|
| `scripts/check_notebooks.py` | Sintaxis + celda de setup de cada notebook |
| `scripts/precompute_recall_predictions.py` | Precalcular TTA para 05 |
| `scripts/train_recall_focus.py` | Entrenar variantes 06 en segundo plano |
| `scripts/train_zoo_recall.py` / `run_zoo_recall_comparison.py` | Entrenamiento/comparación 07 |

## Referencias

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701  
*Uso:* búsqueda de hiperparámetros (notebook 03, `src/tuning.py`).

Dietterich, T. G. (2000). Ensemble methods in machine learning. *Multiple Classifier Systems*, 1–15. https://doi.org/10.1007/3-540-45014-9_1  
*Uso:* justificación del ensemble por promedio de modelos (05–07).

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.  
*Uso:* intervalos de confianza al 95 % para métricas con umbral fijo (`src/recall_optimization.py`).

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision*, 2980–2988. https://doi.org/10.1109/ICCV.2017.324  
*Uso:* `BinaryFocalLoss` en entrenamiento y variantes recall (α, γ).

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*, 618–626. https://doi.org/10.1109/ICCV.2017.74  
*Uso:* mapas de explicabilidad (`src/explain.py`, `src/gradcam_compare.py`).

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60. https://doi.org/10.1186/s40544-019-0197-0  
*Uso:* marco de augmentations; TTA como inferencia con vistas adicionales (`src/tta_inference.py`).

Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training region-based object detectors with online hard example mining. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 761–769. https://doi.org/10.1109/CVPR.2016.89  
*Uso:* inspiración para re-ponderación por dificultad (`src/hard_example_training.py`).

Wang, T.-C., Liu, Y.-Y., Tsai, Y.-H., Yeh, Y.-R., & Wang, Y.-C. F. (2019). Test-time augmentation for deep learning-based cell segmentation. *Proceedings of the IEEE International Symposium on Biomedical Imaging*, 1544–1548. https://doi.org/10.1109/ISBI.2019.8759152  
*Uso:* práctica de TTA en imágenes biomédicas (promedio de predicciones por vista).

Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1), 32–35. https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3  
*Uso:* umbral que maximiza sensibilidad + especificidad − 1 en validación (`select_threshold_youden`).
