# Retina disease detection

This repository contains two complementary RFMiD research projects:

- [`clasificacion/`](clasificacion/) — binary disease-risk classification (`Disease_Risk`), with recall-focused training, ensembles, threshold selection, and evaluation. See its [README](clasificacion/README.md) and [workflow guide](clasificacion/docs/WORKFLOW.md).
- [`multilabel/`](multilabel/) — multilabel fundus classification across ODIR-7, Cluster-C, and Sup-K12 taxonomies, including ViT models, a Streamlit demo, Grad-CAM, and paper results. See its [README](multilabel/README.md).

[Watch the multilabel app demo](Demo.mp4) (MP4).

## Folder guide

### `clasificacion/`

| Folder | Contents |
|---|---|
| `src/` | Data handling, model training, TTA, ensembles, metrics, and explainability |
| `notebooks/` | Dataset preparation, training, tuning, and experiments |
| `scripts/` | Command-line training and validation helpers |
| `artifacts/` | Local checkpoints, caches, logs, and generated outputs |
| `results/` | Reported metrics, figures, and experiment summaries |
| `docs/` | Workflow and project documentation |
| `archive/` | Historical helper scripts |

### `multilabel/`

| Folder | Contents |
|---|---|
| `src/` | Multilabel models, taxonomies, calibration, and training pipelines |
| `app/` | Streamlit application, model loading, predictions, uncertainty, and Grad-CAM |
| `configs/` | Training configurations and the deployed ODIR-7 ensemble manifest |
| `data/` | Taxonomy definitions and label tables |
| `notebooks/` | Data download, taxonomy construction, training, and comparisons |
| `predictions/` | Reference predictions used to reproduce evaluation tables |
| `results/` | Final comparisons, model metrics, and selection metadata |
| `paper/` | Paper source and bibliography (local figures are excluded from Git) |
| `scripts/` | Training, evaluation, comparison, and Grad-CAM entry points |
| `tests/` | Automated tests |

## Data and model files

RFMiD images and large model checkpoints are not stored in Git. Each project README describes its data setup, model requirements, commands, and licensing notes. The multilabel checkpoints are ignored by Git and must be obtained separately to run the full app.
