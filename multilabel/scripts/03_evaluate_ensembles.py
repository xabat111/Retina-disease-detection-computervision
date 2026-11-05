"""Unified TEST comparison + paired bootstrap + ensembles.

Thresholds fitted on VAL only; metrics reported on TEST (n=640).
Default prediction root: predictions/reference (canonical paper dumps).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TAX = {7: "odir7", 10: "cluster_c", 14: "sup_k12"}


def load(f):
    d = np.load(f, allow_pickle=True)
    return (
        d["probabilities"].astype(float),
        d["targets"].astype(int),
        [str(x) for x in d["label_names"]],
    )


def name_of(f: str, pred_dir: Path) -> str:
    p = Path(f)
    suf = p.name.replace("test_predictions", "").replace(".npz", "")
    # canonical layout: predictions/reference/<tax>/<model>/file.npz
    try:
        rel = p.resolve().relative_to(pred_dir.resolve())
        parts = rel.parts
        if len(parts) >= 2:
            return parts[-2] + suf
    except ValueError:
        pass
    return p.parent.name + suf


def thr_fbeta(p, y, beta=2.0):
    ts = np.linspace(0.01, 0.99, 99)
    out = []
    for j in range(p.shape[1]):
        best, bt = -1, 0.5
        for t in ts:
            pr = p[:, j] >= t
            tp = (pr & (y[:, j] == 1)).sum()
            fp = (pr & (y[:, j] == 0)).sum()
            fn = ((~pr) & (y[:, j] == 1)).sum()
            f = (1 + beta**2) * tp / max((1 + beta**2) * tp + beta**2 * fn + fp, 1)
            if f > best:
                best, bt = f, t
        out.append(bt)
    return np.array(out)


def thr_recall(p, y, floor=0.8):
    out = []
    for j in range(p.shape[1]):
        pos = np.sort(p[y[:, j] == 1, j])[::-1]
        k = int(np.ceil(floor * len(pos))) - 1 if len(pos) else 0
        out.append(pos[max(k, 0)] if len(pos) else 0.5)
    return np.array(out)


def point(p, y, t):
    pr = p >= t
    tp = (pr & (y == 1)).sum(0)
    fp = (pr & (y == 0)).sum(0)
    fn = ((~pr) & (y == 1)).sum(0)
    R = tp / np.maximum(tp + fn, 1)
    P = tp / np.maximum(tp + fp, 1)
    F1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    Rm, Pm = R.mean(), P.mean()
    au = np.mean(
        [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    )
    ap = np.mean(
        [average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if y[:, j].sum() > 0]
    )
    return dict(recall=Rm, precision=Pm, f1=F1.mean(), f2=5 * Pm * Rm / max(4 * Pm + Rm, 1e-9), auroc=au, map=ap)


def keep_idx(labels):
    return [i for i, l in enumerate(labels) if l != "sano"]


def collect_models(pred_dir: Path):
    models, seen = {}, {}
    pattern = str(pred_dir / "**" / "test_predictions*.npz")
    for f in sorted(glob.glob(pattern, recursive=True)):
        v = f.replace("test_predictions", "val_predictions")
        if not os.path.exists(v) or "final_comparison" in f:
            continue
        if "partial" in f.replace("\\", "/"):
            continue
        pt, yt, ln = load(f)
        pv, yv, _ = load(v)
        h = hashlib.md5(pt.round(6).tobytes()).hexdigest()
        n = name_of(f, pred_dir)
        if h in seen:
            continue
        seen[h] = n
        if n in models:
            n = n + "@" + Path(f).parent.name
        models[n] = dict(
            tax=TAX[pt.shape[1]],
            pt=pt,
            yt=yt,
            pv=pv,
            yv=yv,
            labels=ln,
            src="reference",
            path=f,
        )
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=ROOT / "predictions" / "reference",
        help="Directory of VAL/TEST npz files (default: predictions/reference)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results",
        help="Output directory for CSVs / pickle",
    )
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()
    pred_dir = args.pred_dir if args.pred_dir.is_absolute() else ROOT / args.pred_dir
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    os.chdir(ROOT)
    models = collect_models(pred_dir)
    print(len(models), "unique models from", pred_dir)
    if not models:
        raise SystemExit(
            f"No prediction pairs found under {pred_dir}. "
            "Expected .../<model>/test_predictions*.npz + matching val_predictions*.npz"
        )

    rng = np.random.default_rng(args.seed)
    B = args.bootstrap
    IDX = rng.integers(0, 640, size=(B, 640))
    W = np.zeros((B, 640))
    np.add.at(W, (np.repeat(np.arange(B), 640), IDX.ravel()), 1)

    def boot(p, y, t):
        pr = (p >= t).astype(float)
        y = y.astype(float)
        tp = W @ (pr * y)
        fp = W @ (pr * (1 - y))
        fn = W @ ((1 - pr) * y)
        R = tp / np.maximum(tp + fn, 1)
        P = tp / np.maximum(tp + fp, 1)
        F1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        Rm, Pm = R.mean(1), P.mean(1)
        aus, aps = [], []
        for j in range(p.shape[1]):
            o = np.argsort(p[:, j])
            yj = y[o, j]
            Wj = W[:, o]
            wn = Wj * (1 - yj)
            wp = Wj * yj
            cneg = np.cumsum(wn, 1) - wn
            aus.append((wp * cneg).sum(1) / np.maximum(wp.sum(1) * wn.sum(1), 1))
            od = o[::-1]
            Wd = W[:, od]
            yd = y[od, j]
            ctp = np.cumsum(Wd * yd, 1)
            call = np.cumsum(Wd, 1)
            aps.append((Wd * yd * ctp / np.maximum(call, 1)).sum(1) / np.maximum((Wd * yd).sum(1), 1))
        return dict(
            recall=Rm,
            precision=Pm,
            f1=F1.mean(1),
            f2=5 * Pm * Rm / np.maximum(4 * Pm + Rm, 1e-9),
            auroc=np.mean(aus, 0),
            map=np.mean(aps, 0),
        )

    def evaluate(name, m, extra=None):
        k = keep_idx(m["labels"])
        pv, yv, pt, yt = m["pv"][:, k], m["yv"][:, k], m["pt"][:, k], m["yt"][:, k]
        rows = []
        for op, fn in [("fbeta2", thr_fbeta), ("recall80", thr_recall)]:
            t = fn(pv, yv)
            r = point(pt, yt, t)
            rv = point(pv, yv, t)
            b = boot(pt, yt, t)
            row = dict(
                taxonomy=m["tax"],
                model=name,
                source=m["src"],
                op=op,
                n_labels=len(k),
                val_f2=rv["f2"],
                val_auroc=rv["auroc"],
            )
            for key in r:
                row[key] = r[key]
                row[key + "_lo"] = np.percentile(b[key], 2.5)
                row[key + "_hi"] = np.percentile(b[key], 97.5)
            row.update(extra or {})
            row["thresholds"] = json.dumps([round(float(x), 3) for x in t])
            rows.append(row)
            m.setdefault("boot", {})[op] = b
            m.setdefault("thr", {})[op] = t
        return rows

    rows = []
    for n, m in models.items():
        rows += evaluate(n, m, dict(kind="single", path=m["path"]))

    df0 = pd.DataFrame(rows)
    ens_rows = []
    for tax in TAX.values():
        c = df0[(df0.taxonomy == tax) & (df0.op == "fbeta2") & (~df0.model.str.contains("partial"))].sort_values(
            "val_f2", ascending=False
        )
        cand = list(c.model.head(6))
        for r in (2, 3):
            for combo in itertools.combinations(cand, r):
                ms = [models[x] for x in combo]
                if not all(
                    np.array_equal(ms[0]["yt"], x["yt"]) and np.array_equal(ms[0]["yv"], x["yv"]) for x in ms
                ):
                    continue
                e = dict(
                    tax=tax,
                    pt=np.mean([x["pt"] for x in ms], 0),
                    pv=np.mean([x["pv"] for x in ms], 0),
                    yt=ms[0]["yt"],
                    yv=ms[0]["yv"],
                    labels=ms[0]["labels"],
                    src="ensemble",
                )
                en = "ENS[" + " + ".join(combo) + "]"
                models[en] = e
                ens_rows += evaluate(en, e, dict(kind=f"ensemble{r}", path=""))
    df = pd.DataFrame(rows + ens_rows)

    diffs = []
    for (tax, op), g in df.groupby(["taxonomy", "op"]):
        ref = g.sort_values("f2", ascending=False).iloc[0].model
        rb = models[ref]["boot"][op]
        for mname in g.model:
            b = models[mname]["boot"][op]
            d = {k: rb[k] - b[k] for k in ("f2", "f1", "auroc", "map", "recall", "precision")}
            diffs.append(
                dict(
                    taxonomy=tax,
                    op=op,
                    reference=ref,
                    model=mname,
                    **{f"d_{k}": np.mean(v) for k, v in d.items()},
                    **{f"d_{k}_lo": np.percentile(v, 2.5) for k, v in d.items()},
                    **{f"d_{k}_hi": np.percentile(v, 97.5) for k, v in d.items()},
                    p_ref_better_f2=float((d["f2"] <= 0).mean()),
                )
            )
    dd = pd.DataFrame(diffs)
    df.to_csv(out / "all_models_test_bootstrap.csv", index=False)
    dd.to_csv(out / "paired_diff_vs_best.csv", index=False)
    pickle.dump(
        {
            k: {kk: v[kk] for kk in ("tax", "thr", "labels", "src") if kk in v} | {"path": v.get("path", "")}
            for k, v in models.items()
        },
        open(out / "models_meta.pkl", "wb"),
    )
    for tax in TAX.values():
        for op in ("fbeta2", "recall80"):
            g = df[(df.taxonomy == tax) & (df.op == op)].sort_values("f2", ascending=False)
            print(f"\n## {tax} {op} singles={sum(g.kind=='single')} ens={sum(g.kind!='single')}")
            best_val_ens = g[g.kind != "single"].sort_values("val_f2", ascending=False).head(1)
            best_val_single = g[g.kind == "single"].sort_values("val_f2", ascending=False).head(1)
            print("VAL-selected single:", best_val_single[["model", "f2", "auroc"]].values.tolist())
            print("VAL-selected ensemble:", best_val_ens[["model", "f2", "auroc"]].values.tolist())
            print(
                g[g.kind == "single"]
                .head(8)[["model", "recall", "precision", "f1", "f2", "f2_lo", "f2_hi", "auroc", "map", "val_f2"]]
                .round(3)
                .to_string()
            )
            print(g[g.kind != "single"].head(3)[["model", "f2", "f2_lo", "f2_hi", "auroc", "val_f2"]].round(3).to_string())

    # stash for 02_summary_paired (optional import helpers)
    meta = {"models": models, "keep_idx": keep_idx, "thr_fbeta": thr_fbeta, "thr_recall": thr_recall, "boot": boot, "W": W}
    pickle.dump(meta, open(out / "_runtime_models.pkl", "wb"))
    print("wrote", out)


if __name__ == "__main__":
    main()
