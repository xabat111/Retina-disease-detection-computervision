import sys

import numpy as np

sys.path.insert(0, ".")
exec(open("artifacts/logs/_explore_current_best.py").read().split("print(\"AUC val/test\"")[0])
from src.recall_optimization import select_threshold_fn_below_fp

for confidence in (0.5, 0.75, 0.9, 0.95):
    threshold = select_threshold_fn_below_fp(val_p, val_labels, confidence)
    v = compute_operating_point_metrics(val_p, val_labels, threshold)
    t = compute_operating_point_metrics(test_p, test_labels, threshold)
    print(confidence, round(threshold, 4), "val CM", v["confusion_matrix"], "test CM", t["confusion_matrix"],
          "test acc/f1/spec", round(t["accuracy"], 3), round(t["f1"], 3), round(t["specificity"], 3))
