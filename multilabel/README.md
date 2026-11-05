# Retina multilabel (RFMiD)

Multilabel fundus screening on **RFMiD** with ViT-L/14 (DINOv2 / RETFound) and a 3-model odir7 ensemble.
**Not a medical device** — research code only.

| Taxonomy | VAL-selected ensemble | F2 [95% CI] | AUROC |
|---|---|---|---|
| odir7 (deployed) | v2_518 + v2 + v3fundus | **0.835** [0.817, 0.852] | **0.954** |
| cluster_c | v2_518 + v3fundus | 0.775 | 0.940 |
| sup_k12 | focal seed2 + 2× inception | 0.687 | 0.937 |

Full tables: [`results/COMPARATIVA_FINAL.md`](results/COMPARATIVA_FINAL.md). Paper PDF: [`paper/paper_v3_long_20261102.pdf`](paper/paper_v3_long_20261102.pdf).

## Layout

```
README.md  LICENSE  requirements.txt  pyproject.toml  .gitignore  .gitattributes
src/                 retina_multilabel/, dual_tax/, paper_train/
notebooks/           00 download · 01 taxonomies · 02–06 train
configs/             YAMLs for the deployed odir7 ensemble
scripts/             01…05 entry points, train_vit_residual
data/taxonomies/     ODIR-7 / Cluster-C / Sup-K12 maps + label CSVs
predictions/reference/  canonical VAL/TEST probability dumps (paper tables)
results/             COMPARATIVA_FINAL.md, final_selection.json, CSVs
paper/               tex + bib + figures + PDF
app/                 Streamlit demo
tests/
```

## Evaluation protocol (paper tables)

This is exactly what [`scripts/03_evaluate_ensembles.py`](scripts/03_evaluate_ensembles.py) implements and what `results/COMPARATIVA_FINAL.md` / the v3 paper report. **It is not** the older specificity-constrained protocol used in earlier single-model experiments outside this repo.

| Item | Rule used for the paper / `COMPARATIVA_FINAL` |
|---|---|
| Split | Thresholds and model/ensemble **selection on VAL only**; metrics on **TEST** (n = 640) |
| Operating points | **`fbeta2` (headline):** per-label threshold maximises per-label F₂ on VAL (grid 0.01…0.99). **`recall80`:** per-label threshold so VAL recall ≥ 0.80. Both appear as separate blocks in the tables. |
| Labels | odir7: 7 · cluster_c: 10 · **sup_k12: 13 disease labels** (`sano` is present in the CSVs but **excluded** from macro averages) |
| Macro recall / precision | Mean of per-label recall / precision |
| **Macro F₂ (reported)** | \(F_2 = 5\,\bar P\,\bar R / (4\,\bar P + \bar R)\) from those **macro** \(\bar P,\bar R\) — **not** the mean of per-label F₂ |
| Macro F1 | Mean of per-label F1 |
| AUROC / mAP | Mean over labels with defined scores (threshold-free) |
| Ensembles | Equal-weight mean of probabilities; candidates = top-6 singles by VAL F₂; every 2–3 combo; pick by VAL F₂; drop TTA/no-TTA duplicates of the same run |
| CIs | Percentile bootstrap, B = 2000 resamples of the 640 TEST images; thresholds fixed |

Training still uses a **recall floor of 0.80** when fitting some checkpoints (`--recall-floor`); that is a training/selection aid inside a run, not the table-level specificity ≥ 0.80 rule.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Three ways to use this repo

### 1) Reproduce the metrics and tables (no GPU, no images)

Uses the shipped probability dumps under `predictions/reference/` (~38 unique models).

```bash
python scripts/03_evaluate_ensembles.py --pred-dir predictions/reference
python scripts/05_compare_with_reference.py --pred-dir predictions/reference   # delta F2 ≈ 0
```

Outputs land in `results/`. Headline numbers must match the table at the top of this README and `results/final_selection.json`.

### 2) Rebuild taxonomies and retrain

```bash
# Data (images are NOT in git) — CC BY 4.0; verify the licence yourself:
# https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid
export RETINA_DATA_ROOT=/path/to/RFMiD    # Training_Set / Evaluation_Set / Test_Set
# or: notebooks/00_download_data.ipynb  (Kaggle API via your kaggle.json — no embedded token)

python scripts/01_build_taxonomies.py                 # verifies shipped maps
# notebooks/01_build_taxonomies.ipynb               # maps + per-split counts

python scripts/02_train.py configs/vit_residual_attn_odir7_v2_518.yaml --dry-run
python scripts/02_train.py configs/vit_residual_attn_odir7_v2_518.yaml
python scripts/02_train.py configs/vit_residual_attn_odir7_v2.yaml
python scripts/02_train.py configs/vit_residual_attn_odir7_v3fundus.yaml   # needs HF access to RETFound
```

Colab / local training notebooks (cell 1 clones + `pip install -e .` on Colab):

| Notebook | Purpose |
|---|---|
| [`notebooks/00_download_data.ipynb`](notebooks/00_download_data.ipynb) | Download RFMiD (Kaggle API or manual) |
| [`notebooks/01_build_taxonomies.ipynb`](notebooks/01_build_taxonomies.ipynb) | Verify ODIR-7 / Cluster-C / Sup-K12 |
| [`notebooks/02_train_dual_taxonomy.ipynb`](notebooks/02_train_dual_taxonomy.ipynb) | BFPC + DINOv2 + residual-attn |
| [`notebooks/03_train_vit_fundus_odir7.ipynb`](notebooks/03_train_vit_fundus_odir7.ipynb) | RETFound-init ViTs on odir7 |
| [`notebooks/04_train_vit_fundus_cluster_c.ipynb`](notebooks/04_train_vit_fundus_cluster_c.ipynb) | RETFound-init ViTs on cluster_c |
| [`notebooks/05_train_vit_fundus_supk12.ipynb`](notebooks/05_train_vit_fundus_supk12.ipynb) | RETFound-init ViT-518 on sup_k12 |
| [`notebooks/06_train_paper_compare_supk12.ipynb`](notebooks/06_train_paper_compare_supk12.ipynb) | AE / encoder / BFPC / Inception / ViT |

Secrets come from Colab userdata / env — **no tokens are hardcoded**.

### 3) App and Grad-CAM (needs the checkpoints and the RFMiD test images)

The app works (verified end to end: ensemble loading, prediction and Grad-CAM on test images). It needs two things that are **not** in git:

1. **Checkpoints** (~1.16 GB each; GitHub file limit 100 MB). Download them from the Release / Hugging Face links (TODO) and place them as:

```
checkpoints/final_20261030/manifest.json
checkpoints/final_20261030/vit_residual_attn_odir7_v2_518/checkpoint.pt   (+ test_predictions.npz)
checkpoints/final_20261030/vit_residual_attn_odir7_v2/checkpoint.pt       (+ test_predictions.npz)
checkpoints/final_20261030/vit_residual_attn_odir7_v3fundus/checkpoint.pt (+ test_predictions.npz)
```

   The `test_predictions.npz` of each member is also in `predictions/reference/odir7/<run>/`.

2. **RFMiD test images**, downloaded with `notebooks/00_download_data.ipynb`. The app reads them from `../Test_Set/Test_Set/Test` (relative to the repo, see `app/settings.py`).

```bash
pip install -e ".[app]"
streamlit run app/app.py --server.port 8502
python scripts/04_gradcam.py
```

**Released weights for the other taxonomies.** Besides the three ODIR-7 members, the Release includes Cluster-C (`vit_residual_attn_cluster_c_v2`, `vit_dinov2_cluster_c_v3fundus`, `vit_residual_attn_cluster_c_v3fundus_518`) and Sup-K12 (`inception_sup_k12_sano_hires448_asl`, `inception_sup_k12_sano_hires448_asl_seed1`). The checkpoint of `vit_residual_attn_cluster_c_v2_518` was lost; its predictions are kept in `predictions/reference/`, so every reported metric is still reproducible, and the model can be retrained with `notebooks/04_train_vit_fundus_cluster_c.ipynb` and its config.
| Checkpoint | img | Init | Config |
|---|---|---|---|
| `vit_residual_attn_odir7_v2_518` | 518 | DINOv2 | [`configs/vit_residual_attn_odir7_v2_518.yaml`](configs/vit_residual_attn_odir7_v2_518.yaml) |
| `vit_residual_attn_odir7_v2` | 392 | DINOv2 | [`configs/vit_residual_attn_odir7_v2.yaml`](configs/vit_residual_attn_odir7_v2.yaml) |
| `vit_residual_attn_odir7_v3fundus` | 392 | RETFound (CC BY-NC) | [`configs/vit_residual_attn_odir7_v3fundus.yaml`](configs/vit_residual_attn_odir7_v3fundus.yaml) |

Optional Git LFS tracking for local clones is declared in [`.gitattributes`](.gitattributes); the recommended distribution channel remains HF or a Release.

## Paper

```bash
cd paper && latexmk -pdf paper_v3_long_20261102.tex
```

## Licences

| Component | Licence |
|---|---|
| **Code in this repository** | [MIT](LICENSE) |
| RFMiD images / labels | **CC BY 4.0** (Pachade et al., 2021) — **verify** on the DataPort page before redistributing; images are not shipped here |
| DINOv2 weights | Apache 2.0 (Meta AI) |
| **RETFound-DINOv2** (`v3fundus` member) | **CC BY-NC 4.0** (non-commercial). The deployed 3-model ensemble inherits this restriction |

```bibtex
@misc{arin2026retina,
  author = {Arin, Xabat},
  title  = {Multilabel retinal disease screening on RFMiD with DINOv2/RETFound ViT-L ensembles},
  year   = {2026},
  note   = {TODO: add repository URL}
}
```

Please also cite RFMiD (Pachade et al., 2021), DINOv2 (Oquab et al., 2023) and RETFound (Zhou et al., Nature 2023).
