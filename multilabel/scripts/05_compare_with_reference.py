"""Compare a user's prediction directory against the paper reference numbers.

Protocol (same as the paper):
  - fit per-label thresholds on VAL (fbeta2)
  - report macro metrics on TEST
  - for ensembles: mean of member probabilities

Reference headline numbers are read from results/final_selection.json
when present (not hardcoded). Expected layout for --pred-dir:
  <pred-dir>/<taxonomy>/<model_name>/{test,val}_predictions*.npz
or a flat tree of paired test_/val_predictions*.npz (same as 01).

Example (delta ~ 0 on the shipped reference):
  python evaluation/compare_with_reference.py --pred-dir predictions/reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_selection(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing reference file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "results" / "final_selection.json",
    )
    ap.add_argument("--op", default="fbeta2", choices=["fbeta2", "recall80"])
    ap.add_argument(
        "--ensembles",
        action="store_true",
        default=True,
        help="Also score the VAL-selected ensemble members from final_selection.json",
    )
    args = ap.parse_args()
    pred_dir = args.pred_dir if args.pred_dir.is_absolute() else ROOT / args.pred_dir
    ref = load_selection(args.reference if args.reference.is_absolute() else ROOT / args.reference)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cmp03", ROOT / "scripts" / "03_evaluate_ensembles.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    models = mod.collect_models(pred_dir)
    if not models:
        raise SystemExit(f"No models found under {pred_dir}")

    rows = []
    for tax, info in ref.items():
        # VAL-selected ensemble name encodes members
        ens_name = info["val_selected_ensemble"]
        ref_row = info["ensemble_row"]
        if ens_name.startswith("ENS[") and ens_name.endswith("]"):
            members = [x.strip() for x in ens_name[4:-1].split(" + ")]
        else:
            members = [ens_name]
        missing = [m for m in members if m not in models]
        if missing:
            rows.append(
                dict(
                    taxonomy=tax,
                    model=ens_name,
                    status=f"MISSING members: {missing}",
                    ref_f2=ref_row["f2"],
                    user_f2=None,
                    delta_f2=None,
                    ref_auroc=ref_row["auroc"],
                    user_auroc=None,
                    delta_auroc=None,
                )
            )
            continue
        ms = [models[m] for m in members]
        e = dict(
            tax=tax,
            pt=np.mean([x["pt"] for x in ms], 0),
            pv=np.mean([x["pv"] for x in ms], 0),
            yt=ms[0]["yt"],
            yv=ms[0]["yv"],
            labels=ms[0]["labels"],
            src="user",
        )
        k = mod.keep_idx(e["labels"])
        thr_fn = mod.thr_fbeta if args.op == "fbeta2" else mod.thr_recall
        t = thr_fn(e["pv"][:, k], e["yv"][:, k])
        r = mod.point(e["pt"][:, k], e["yt"][:, k], t)
        rows.append(
            dict(
                taxonomy=tax,
                model=ens_name,
                status="ok",
                ref_f2=float(ref_row["f2"]),
                user_f2=float(r["f2"]),
                delta_f2=float(r["f2"] - ref_row["f2"]),
                ref_auroc=float(ref_row["auroc"]),
                user_auroc=float(r["auroc"]),
                delta_auroc=float(r["auroc"] - ref_row["auroc"]),
                thresholds=json.dumps([round(float(x), 3) for x in t]),
            )
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    # summary
    ok = df[df.status == "ok"]
    if len(ok):
        print("\nMax |delta F2|:", float(ok.delta_f2.abs().max()))
        print("Max |delta AUROC|:", float(ok.delta_auroc.abs().max()))
        if float(ok.delta_f2.abs().max()) < 1e-6 and float(ok.delta_auroc.abs().max()) < 1e-6:
            print("PASS: matches reference within 1e-6")
        else:
            print("NOTE: non-zero delta (different preds or threshold grid drift)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
