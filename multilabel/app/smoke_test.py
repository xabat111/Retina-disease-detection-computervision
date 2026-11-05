"""Smoke test of the final app: analyze_image (UI pipeline) on TEST images vs stored ensemble npz decisions."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from app import settings
from app.analysis import analyze_image
from app.model_loader import load_deployed_model
from app.preprocessing import decode_image
from app.sample_images import load_sample_catalog
from app.gradcam import render_overlay

t0 = time.time(); dep = load_deployed_model(); print("load s", round(time.time() - t0, 1), flush=True)
print(dep.test_metrics.round(3).to_string(), "unreliable:", dep.unreliable_labels, flush=True)
cat = load_sample_catalog(); print("catalog", len(cat.identifiers), flush=True)
lab = pd.read_csv(settings.TEST_LABELS_CSV)
ref = np.mean([np.load(settings.CHECKPOINT_DIR / m / "test_predictions.npz")["probabilities"] for m in
               ["vit_residual_attn_odir7_v2_518", "vit_residual_attn_odir7_v2", "vit_residual_attn_odir7_v3fundus"]], 0)
rows = []
for i in [int(a) for a in sys.argv[1:]] or [0, 110, 300]:
    iid = int(lab.ID.iloc[i]); t = time.time()
    res = analyze_image(dep, decode_image(cat.image_path(iid).read_bytes()), 2)
    clean = res.pass_probabilities[0]
    det = np.array([a.detected for a in res.assessments])
    ov = render_overlay(res.image_uint8, res.gradcam_maps[1], 0.45, True)
    rows.append(dict(test_index=i, id=iid, max_abs_diff=round(float(np.abs(clean - ref[i]).max()), 5),
                     same_decisions=bool((det == (ref[i] >= dep.thresholds)).all()),
                     pred=res.detected_labels, truth=cat.positive_labels(iid), overall=res.overall_uncertainty,
                     gradcam=list(res.gradcam_maps[0].shape), overlay=list(ov.shape), sec=round(time.time() - t, 1)))
    print(rows[-1], flush=True)
ok = all(r["max_abs_diff"] < 1e-2 and r["same_decisions"] for r in rows)
(ROOT / "app" / "smoke_test_result.json").write_text(json.dumps(dict(ok=ok, rows=rows), indent=2))
print("SMOKE OK" if ok else "SMOKE FAIL")
