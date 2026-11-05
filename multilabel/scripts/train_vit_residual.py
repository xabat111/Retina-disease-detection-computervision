#!/usr/bin/env python3
"""Train DINOv2 + residual token attention on odir7 / cluster_c."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dual_tax.cli import train_argparser, v2_kwargs
from dual_tax.taxonomy import resolve_taxonomy
from dual_tax.vit_residual_attn import build_vit_residual_attn
from paper_train.train_pipeline import TrainConfig, run_training

def main() -> None:
    p = train_argparser(__doc__)
    p.add_argument("--img-size", type=int, default=392)
    p.add_argument("--unfreeze-blocks", type=int, default=0)
    args = p.parse_args()
    cluster_map, labels_dir = resolve_taxonomy(args.taxonomy)
    run_name = args.run_name or f"vit_residual_attn_{args.taxonomy}"
    run_dir = ROOT / "outputs" / "runs" / run_name
    num_classes = len(json.loads(cluster_map.read_text(encoding="utf-8"))["cluster_names_ordered"])
    model = build_vit_residual_attn(
        num_classes, img_size=args.img_size, freeze_backbone=True, n_unfrozen_blocks=args.unfreeze_blocks,
    )
    cfg = TrainConfig(
        model_name="vit_residual_attn", run_dir=run_dir, labels_dir=labels_dir, cluster_map=cluster_map,
        img_size=args.img_size, epochs=args.epochs, paper_epochs=args.paper_epochs, batch_size=args.batch_size,
        lr=args.lr, label_smoothing=0.0, use_paper_a_preprocess=False, dual_view=False,
        num_workers=args.num_workers, seed=args.seed, early_stopping_patience=args.patience,
        max_train_samples=args.max_train_samples, max_eval_samples=args.max_eval_samples,
        pos_weight_cap=args.pos_weight_cap, recall_floor=args.recall_floor,
        extra={"taxonomy": args.taxonomy, "justification": "BFPC RAM on ViT tokens", "unfreeze_blocks": args.unfreeze_blocks},
        **v2_kwargs(args),
    )
    note = (f"DINOv2 last-{args.unfreeze_blocks} blocks + residual token attn" if args.unfreeze_blocks > 0
            else "DINOv2 frozen + residual token attn")
    run_training(model, cfg, build_note=f"{note}; val_loss; pos_weight; thr@recall_floor")

if __name__ == "__main__":
    main()
