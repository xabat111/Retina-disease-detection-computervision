"""Grad-CAM (ViT token Grad-CAM, last block) for one checkpoint per taxonomy.

Checkpoints and RFMiD images are NOT in git. Set:
  RETINA_DATA_ROOT  -> RFMiD root (Training_Set / Evaluation_Set / Test_Set)
  RETINA_CKPT_DIR   -> final checkpoint directory (default: checkpoints/final_20261030)
                       (default: checkpoints/ or $RETINA_CKPT_DIR)
Or pass --ckpt-dir / --data-root / --img-dir.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def resolve_paths(args):
    data_root = Path(
        args.data_root
        or os.environ.get("RETINA_DATA_ROOT")
        or ROOT.parents[1]
    )
    ckpt_dir = Path(
        args.ckpt_dir
        or os.environ.get("RETINA_CKPT_DIR")
        or (ROOT / "checkpoints" / "final_20261030")
    )
    img_dir = Path(args.img_dir) if args.img_dir else None
    if img_dir is None and data_root:
        # RFMiD layout variants
        for cand in [
            data_root / "Test_Set" / "Test_Set" / "Test",
            data_root / "Test_Set" / "Test",
        ]:
            if cand.is_dir():
                img_dir = cand
                break
    return data_root, ckpt_dir, img_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="Directory containing per-run checkpoint files "
                         "(e.g. checkpoints/final_20261030 or a HF download root)")
    ap.add_argument("--img-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gradcam")
    ap.add_argument("--labels-csv", type=Path,
                    default=ROOT / "data" / "taxonomies" / "odir7" / "labels" / "clustered_test.csv")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "predictions" / "reference")
    args = ap.parse_args()

    data_root, ckpt_dir, img_dir = resolve_paths(args)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    missing = []
    if img_dir is None or not Path(img_dir).is_dir():
        missing.append(
            "Test images not found. Set RETINA_DATA_ROOT to the RFMiD root "
            "or pass --img-dir pointing at the Test PNG folder."
        )
    # Prefer final ensemble members; fall back to paper singles under ckpt_dir
    MODELS = {
        "odir7": dict(
            run="vit_residual_attn_odir7_v2_518",
            kind="res",
            img=518,
            npz_tax="odir7",
            npz_model="vit_residual_attn_odir7_v2_518",
        ),
        "cluster_c": dict(
            run="vit_residual_attn_cluster_c_v3fundus",
            kind="res",
            img=392,
            npz_tax="cluster_c",
            npz_model="vit_residual_attn_cluster_c_v3fundus",
        ),
        "sup_k12": dict(
            run="vit_dinov2_sup_k12_sano_hires518_asl_fundus",
            kind="lin",
            img=518,
            npz_tax="sup_k12",
            npz_model="vit_dinov2_sup_k12_sano_hires518_asl_fundus_notta",
        ),
    }
    for tax, m in MODELS.items():
        # search checkpoint
        cands = [
            ckpt_dir / m["run"] / f"{m['run']}.pt",
            ckpt_dir / m["run"] / "checkpoint.pt",
            ckpt_dir / "final_20261030" / m["run"] / f"{m['run']}.pt",
            ckpt_dir / "final_20261030" / m["run"] / "checkpoint.pt",
            ROOT / "checkpoints" / m["run"] / f"{m['run']}.pt",
            ROOT / "checkpoints" / m["run"] / "checkpoint.pt",
        ]
        ck = next((c for c in cands if c.is_file()), None)
        m["ck"] = ck
        if ck is None:
            missing.append(f"Missing checkpoint for {m['run']} under {ckpt_dir} (set RETINA_CKPT_DIR).")
    if missing:
        raise SystemExit("Grad-CAM cannot run:\n- " + "\n- ".join(missing) +
                         "\nSee models/README.md and REPRODUCE.md.")

    sys.path[:0] = [str(ROOT / "src")]
    import retina_multilabel.model as rm

    _orig = rm._create_timm_backbone

    def _nopre(name, img_size=None, *, dynamic_img_size=False):
        import timm

        kw = {"pretrained": False, "num_classes": 0}
        if img_size is not None:
            kw["img_size"] = img_size
        if dynamic_img_size:
            kw["dynamic_img_size"] = True
        return timm.create_model(name, **kw)

    rm._create_timm_backbone = _nopre
    from dual_tax.vit_residual_attn import build_vit_residual_attn
    from paper_train.models.vit_adapter import build_vit_cluster_model

    torch.set_num_threads(max(1, torch.get_num_threads()))
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    IMG_DIR = Path(img_dir)
    pred_dir = args.pred_dir if args.pred_dir.is_absolute() else ROOT / args.pred_dir

    def load_img(path, size):
        pil = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        arr = np.array(pil, dtype=np.uint8)
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255.0
        return arr, (x - MEAN) / STD

    def build(m, n):
        ck = torch.load(m["ck"], map_location="cpu", weights_only=False)
        if m["kind"] == "res":
            model = build_vit_residual_attn(n, img_size=m["img"])
        else:
            model = build_vit_cluster_model(n, backbone_choice="dinov2", img_size=m["img"])
        sd = ck.get("model_state_dict", ck.get("model", ck))
        mi, un = model.load_state_dict(sd, strict=False)
        print(m["run"], "missing", len(mi), mi[:3], "unexpected", len(un), un[:3], flush=True)
        assert not un and not mi, "state_dict mismatch"
        for p in model.parameters():
            p.requires_grad_(False)
        return model.eval()

    def gradcam(model, x, cls):
        store = {}
        blk = model.backbone.blocks[-1].norm1

        def _hook(mod, i, o):
            o.retain_grad()
            store["a"] = o

        h = blk.register_forward_hook(_hook)
        x = x.clone().requires_grad_(True)
        logits = model(x)
        h.remove()
        logits[0, cls].backward()
        a, g = store["a"][0], store["a"].grad[0]
        npre = getattr(model.backbone, "num_prefix_tokens", 1)
        a, g = a[npre:], g[npre:]
        w = g.mean(0)
        cam = torch.relu((a * w).sum(-1))
        s = int(round(cam.numel() ** 0.5))
        cam = cam.reshape(s, s).detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return torch.sigmoid(logits[0]).detach().numpy(), cam

    lab = pd.read_csv(args.labels_csv)
    # odir7 reference preds for picking confident TPs
    zpath = pred_dir / "odir7" / "vit_residual_attn_odir7_v2_518" / "test_predictions.npz"
    if not zpath.exists():
        raise SystemExit(f"Missing reference predictions: {zpath}")
    z = np.load(zpath)
    names = [str(s) for s in z["label_names"]]
    P = z["probabilities"]
    Y = z["targets"]
    assert np.array_equal(Y, lab[names].to_numpy()), "npz/csv order mismatch"
    ids = []
    for j, nm in enumerate(["D", "G", "C", "A", "M", "O", "N"]):
        jj = names.index(nm)
        cand = np.flatnonzero(Y[:, jj] == 1)
        cand = sorted(cand, key=lambda i: (-int(Y[i].sum() == 1), -P[i, jj]))
        pick = next(i for i in cand if i not in [k for k, _ in ids])
        ids.append((pick, nm))

    rows = {}
    meta = []
    for tax, m in MODELS.items():
        npz_name = 'test_predictions.npz'
        zz = np.load(pred_dir / m["npz_tax"] / m["npz_model"] / npz_name)
        ln = [str(s) for s in zz["label_names"]]
        model = build(m, len(ln))
        rows[tax] = []
        for i, nm in ids:
            iid = int(lab["ID"].iloc[i])
            t0 = time.time()
            arr, x = load_img(IMG_DIR / f"{iid}.png", m["img"])
            with torch.no_grad():
                pr = torch.sigmoid(model(x)[0]).numpy()
            cand = [k for k in range(len(ln)) if ln[k].lower() not in ("sano", "n", "normal")]
            truth = [ln[k] for k in range(len(ln)) if zz["targets"][i, k] == 1]
            cls = max(cand, key=lambda k: pr[k]) if nm != "N" else int(np.argmax(pr))
            pr2, cam = gradcam(model, x, cls)
            err = float(np.abs(pr - zz["probabilities"][i]).max())
            rows[tax].append(dict(arr=arr, cam=cam, cls=ln[cls], p=float(pr[cls]), truth=truth))
            meta.append(
                dict(
                    taxonomy=tax,
                    run=m["run"],
                    test_index=int(i),
                    image_id=iid,
                    odir7_label=nm,
                    cam_class=ln[cls],
                    prob=round(float(pr[cls]), 4),
                    truth=";".join(truth),
                    max_abs_diff_vs_saved_npz=round(err, 5),
                    sec=round(time.time() - t0, 1),
                )
            )
            print(meta[-1], flush=True)
        del model
    pd.DataFrame(meta).to_csv(out / "gradcam_meta.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taxes = list(MODELS)
    n = len(ids)
    fig, ax = plt.subplots(n, 1 + len(taxes), figsize=(3.1 * (1 + len(taxes)), 3.1 * n))
    for r, (i, nm) in enumerate(ids):
        base = rows["odir7"][r]["arr"]
        ax[r, 0].imshow(base)
        ax[r, 0].set_title(
            f'ID {int(lab.ID.iloc[i])} | ODIR-7 GT: {",".join(rows["odir7"][r]["truth"])}', fontsize=8
        )
        for c, tax in enumerate(taxes, 1):
            d = rows[tax][r]
            s = d["arr"].shape[0]
            cam = (
                np.array(
                    Image.fromarray((d["cam"] * 255).astype(np.uint8)).resize((s, s), Image.Resampling.BILINEAR)
                )
                / 255.0
            )
            ax[r, c].imshow(d["arr"])
            ax[r, c].imshow(cam, cmap="jet", alpha=0.4)
            ax[r, c].set_title(f'{tax}: {d["cls"]} p={d["p"]:.2f}\nGT: {",".join(d["truth"])[:40]}', fontsize=7)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    fig.savefig(out / "gradcam_comparison.png", dpi=110)
    print("saved", out / "gradcam_comparison.png")


if __name__ == "__main__":
    main()
