"""Argparse shared by dual-taxonomy train scripts (avoids clash with paper_compare --taxonomy)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PAPER_EPOCHS = 100
DEFAULT_QUICK_EPOCHS = 8


def train_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--taxonomy", required=True, choices=["odir7", "cluster_c"])
    p.add_argument("--epochs", type=int, default=DEFAULT_QUICK_EPOCHS)
    p.add_argument("--paper-epochs", type=int, default=PAPER_EPOCHS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Subsample val/test (smoke only; omit for paper runs)",
    )
    p.add_argument("--pos-weight-cap", type=float, default=20.0)
    p.add_argument("--recall-floor", type=float, default=0.80)
    # v2 options (defaults = v1 behaviour)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Separate lr for backbone.* params (ViT partial fine-tune). None = --lr",
    )
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument(
        "--pretrained",
        choices=["default", "fundus"],
        default=None,
        action=_EnvAction,
        env_var="VIT_PRETRAINED",
        help="DINOv2 backbone init: default (timm LVD-142M) or fundus (RETFound-DINOv2, needs HF_TOKEN "
        "or --fundus-weights). Sets $VIT_PRETRAINED; ignored by non-DINOv2 models.",
    )
    p.add_argument(
        "--fundus-weights",
        default=None,
        action=_EnvAction,
        env_var="VIT_FUNDUS_WEIGHTS",
        help="Local RETFound_dinov2_meh.pth (sets $VIT_FUNDUS_WEIGHTS)",
    )
    return p


class _EnvAction(argparse.Action):
    """Store the value and mirror it into an environment variable read by retina_multilabel.model."""

    def __init__(self, *args, env_var: str, **kwargs):
        self.env_var = env_var
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        os.environ[self.env_var] = str(values)


def v2_kwargs(args) -> dict:
    """TrainConfig kwargs shared by all train scripts for the v2 options."""
    return {
        "weight_decay": args.weight_decay,
        "backbone_lr": args.backbone_lr,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "amp": args.amp,
        "grad_clip": args.grad_clip,
    }
