# RFMiD — clasificación binaria `Disease_Risk`

Proyecto de entrenamiento y evaluación de modelos (ResNet-152 y zoo) sobre el dataset **RFMiD**, con foco en **recall**, regla **FN &lt; FP** (con intervalos bootstrap en validación) y **AUC**, sin usar test para elegir hiperparámetros ni umbrales.

- **Código y artifacts:** `retina_clasificacion/` (este directorio)
- **Datos e imágenes RFMiD:** carpeta padre [`Retina/`](../) (`DATA_ROOT`)

Documentación de flujo, notebooks y resultados: **[`docs/WORKFLOW.md`](docs/WORKFLOW.md)**.

Estructura resumida:

```
retina_clasificacion/
├── docs/WORKFLOW.md      # Flujo, criterios, referencias
├── notebooks/
│   ├── 01_data/          # Construcción de DataFrames
│   ├── best_model/       # Pipeline principal (02.x, 05, 06)
│   └── other_experiments/
├── src/                  # Entrenamiento, TTA, ensemble, métricas
├── scripts/              # Entrenamiento por CLI, comprobaciones
├── artifacts/            # Checkpoints, cachés, logs, salida en vivo
├── results/              # Figuras y métricas organizadas
└── archive/scripts/      # Helpers puntuales de notebooks (histórico)
```

Ejecutar comprobación ligera de notebooks (sin entrenar):

```bash
python scripts/check_notebooks.py
```
