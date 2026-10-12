import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.config import default_config
from src.data import build_split_dataframe
from src.recall_config import PREDICTION_CACHE_DIR
from src.recall_optimization import compute_operating_point_metrics

config = default_config()
val_labels = build_split_dataframe(config.val_labels_csv, config.val_images_dir)["disease_risk"].to_numpy(float)
test_labels = build_split_dataframe(config.test_labels_csv, config.test_images_dir)["disease_risk"].to_numpy(float)
names = ["ajustado_nocrop", "baseline_nocrop", "ajustado_original", "baseline_original"]


def averaged(split):
    members = []
    for name in names:
        with np.load(PREDICTION_CACHE_DIR / f"{name}_{split}.npz") as cached:
            members.append(np.mean([cached[v] for v in cached.files], axis=0))
    return np.mean(members, axis=0)


val_p, test_p = averaged("val"), averaged("test")
print("AUC val/test", compute_operating_point_metrics(val_p, val_labels, 0.5)["auc"],
      compute_operating_point_metrics(test_p, test_labels, 0.5)["auc"])
print("thr | val FN FP acc | test FN FP acc")
for threshold in np.arange(0.05, 0.50, 0.025):
    v = compute_operating_point_metrics(val_p, val_labels, threshold)
    t = compute_operating_point_metrics(test_p, test_labels, threshold)
    (vtn, vfp), (vfn, vtp) = v["confusion_matrix"]
    (ttn, tfp), (tfn, ttp) = t["confusion_matrix"]
    print(f"{threshold:.3f} | {vfn:3d} {vfp:3d} {v['accuracy']:.3f} f1 {v['f1']:.3f}| {tfn:3d} {tfp:3d} {t['accuracy']:.3f} f1 {t['f1']:.3f}")
