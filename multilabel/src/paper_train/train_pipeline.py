"""Shared train/eval pipeline for clustered multilabel CSVs (ODIR-8 or latent).

Reads ``clustered_{train,val,test}.csv`` + RFMiD images via
``retina_multilabel.data.locate_split_files``. Supports paper-A IAM preprocess,
BCEWithLogits (+ optional label smoothing), Adam, and AP/AUROC/F1 macro metrics.
All-zero classes (e.g. H on RFMiD) are masked from loss and metrics.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retina_multilabel.data import locate_split_files  # noqa: E402
from retina_multilabel.evaluate import (  # noqa: E402
    choose_threshold_f1_with_recall_floor,
    macro_average_precision,
    macro_auroc,
    multilabel_metric_matrix,
)
from retina_multilabel.utils import get_device, set_seed  # noqa: E402

from .preprocess_paper_a import weighted_enhancement  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Paper A: 100 epochs; default here is a reduced CPU-friendly schedule.
PAPER_EPOCHS = 100
DEFAULT_QUICK_EPOCHS = 8
INCEPTION_AUX_WEIGHT = 0.4


@dataclass
class TrainConfig:
    model_name: str
    run_dir: Path
    labels_dir: Path
    cluster_map: Path
    img_size: int = 224
    epochs: int = DEFAULT_QUICK_EPOCHS
    paper_epochs: int = PAPER_EPOCHS
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    use_paper_a_preprocess: bool = False
    dual_view: bool = False
    num_workers: int = 0
    seed: int = 42
    early_stopping_patience: int = 5
    threshold: float = 0.5
    pretrained: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None  # subsample val/test (smoke only; full for paper runs)
    # Recall-oriented training / calibration (optional)
    pos_weight_cap: float | None = None  # if set, BCE uses capped n_neg/n_pos
    recall_floor: float | None = None  # if set, calibrate val thresholds with F1@recall≥floor
    # v2 training options (defaults reproduce v1 behaviour exactly)
    backbone_lr: float | None = None  # separate lr for params named "backbone.*" (None = cfg.lr)
    scheduler: str = "none"  # "none" | "cosine" (per-epoch cosine with linear warmup)
    warmup_epochs: int = 0
    min_lr_ratio: float = 0.01  # cosine floor as a fraction of the group's base lr
    amp: bool = False  # torch.autocast fp16 on CUDA (no-op on CPU)
    grad_clip: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_cluster_map(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def active_class_mask(targets: np.ndarray) -> np.ndarray:
    """True for classes with ≥1 positive in the array (skip all-zero like H)."""
    return targets.sum(axis=0) > 0


def smooth_multilabel_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def _attach_paths(labels: pd.DataFrame, split: str) -> pd.DataFrame:
    files = locate_split_files(split)
    image_index = {
        p.stem: p
        for p in files.image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    }
    paths = []
    missing = []
    for identifier in labels["ID"]:
        key = str(identifier)
        path = image_index.get(key)
        if path is None:
            missing.append(key)
            paths.append(None)
        else:
            paths.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images for split={split}: {missing[:5]}..."
        )
    out = labels.copy()
    out.insert(1, "image_path", paths)
    return out


def load_clustered_tables(
    labels_dir: Path, label_names: list[str]
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        csv_path = labels_dir / f"clustered_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        raw = pd.read_csv(csv_path)
        missing_cols = [c for c in label_names if c not in raw.columns]
        if missing_cols:
            raise KeyError(f"{csv_path} missing columns {missing_cols}")
        keep = ["ID", *label_names]
        if "Disease_Risk" in raw.columns:
            keep.append("Disease_Risk")
        tables[split] = _attach_paths(raw[keep], split)
    return tables


def _preprocess_one(
    path: str, img_size: int, use_paper_a_preprocess: bool
) -> torch.Tensor:
    """Load → resize → optional IAM (at target size) → uint8 CHW.

    IAM is applied after resize for CPU feasibility; paper applies enhancement on
    the working resolution used for training (here ``img_size``).
    """
    with Image.open(path) as im:
        pil = im.convert("RGB")
    # Downscale first (RFMiD originals are large); IAM Gaussian blur is O(HW).
    pil = pil.resize((img_size, img_size), Image.Resampling.BICUBIC)
    if use_paper_a_preprocess:
        pil = weighted_enhancement(pil)
    arr = np.array(pil, dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def cache_split_images(
    table: pd.DataFrame,
    img_size: int,
    *,
    use_paper_a_preprocess: bool,
    cache_path: Path | None = None,
    description: str = "images",
) -> torch.Tensor:
    """Preprocess once (disk cache if path given) → uint8 (N,3,H,W)."""
    paths = table["image_path"].tolist()
    if cache_path is not None and cache_path.is_file():
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            isinstance(blob, dict)
            and blob.get("n") == len(paths)
            and blob.get("img_size") == img_size
            and blob.get("iam") == use_paper_a_preprocess
        ):
            print(f"Loaded image cache {cache_path} ({len(paths)} {description})")
            return blob["images"]

    tensors = [
        _preprocess_one(p, img_size, use_paper_a_preprocess)
        for p in tqdm(paths, desc=f"Caching {description}")
    ]
    images = torch.stack(tensors)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "images": images,
                "n": len(paths),
                "img_size": img_size,
                "iam": use_paper_a_preprocess,
            },
            cache_path,
        )
        print(f"Wrote image cache {cache_path}")
    return images


class CachedClusteredDataset(Dataset):
    """In-memory uint8 images + labels; aug + ImageNet normalize at getitem."""

    def __init__(
        self,
        images: torch.Tensor,
        targets: np.ndarray,
        *,
        train: bool = False,
        dual_view: bool = False,
    ) -> None:
        self.images = images
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.dual_view = dual_view and train
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if train:
            self.aug: T.Compose | None = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(p=0.2),
                    T.RandomRotation(15, fill=0),
                ]
            )
        else:
            self.aug = None

    def __len__(self) -> int:
        return len(self.images)

    def _to_model_tensor(self, image_u8: torch.Tensor) -> torch.Tensor:
        x = image_u8
        if self.aug is not None:
            x = self.aug(x)
        return self.normalize(x.float() / 255.0)

    def __getitem__(self, index: int):
        y = self.targets[index]
        img = self.images[index]
        if self.dual_view:
            return self._to_model_tensor(img), self._to_model_tensor(img), y
        return self._to_model_tensor(img), y


# Back-compat alias used by older smoke tests
ClusteredImageDataset = CachedClusteredDataset


def make_loaders(
    tables: dict[str, pd.DataFrame],
    label_names: list[str],
    cfg: TrainConfig,
) -> dict[str, DataLoader]:
    # Shared across runs with same taxonomy/size/preprocess (not under run_dir)
    taxonomy_tag = cfg.labels_dir.parent.name
    iam_tag = "iam" if cfg.use_paper_a_preprocess else "plain"
    cache_root = cfg.run_dir.parents[1] / "image_cache" / f"{taxonomy_tag}_{cfg.img_size}_{iam_tag}"
    loaders: dict[str, DataLoader] = {}
    for split, table in tables.items():
        is_train = split == "train"
        # Include row count so subsampled smoke runs don't collide with full caches
        cache_path = cache_root / f"{split}_n{len(table)}.pt"
        images = cache_split_images(
            table,
            cfg.img_size,
            use_paper_a_preprocess=cfg.use_paper_a_preprocess,
            cache_path=cache_path,
            description=f"{split}",
        )
        targets = table[label_names].to_numpy(dtype=np.float32)
        ds = CachedClusteredDataset(
            images,
            targets,
            train=is_train,
            dual_view=cfg.dual_view,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=is_train,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loaders


def _unpack_batch(batch, dual_view: bool, device: torch.device):
    if dual_view:
        x1, x2, y = batch
        return x1.to(device), x2.to(device), y.to(device)
    x, y = batch
    return x.to(device), None, y.to(device)


def _model_forward(model: nn.Module, x: torch.Tensor, x2: torch.Tensor | None):
    if x2 is not None:
        return model(x, x2)
    out = model(x)
    return out


def _main_logits(out) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    return out


def _bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_mask: torch.Tensor | None,
    label_smoothing: float,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    y = smooth_multilabel_targets(targets, label_smoothing)
    pw = pos_weight
    if class_mask is not None:
        logits = logits[:, class_mask]
        y = y[:, class_mask]
        if pw is not None:
            pw = pw[class_mask]
    return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def compute_pos_weight(targets: np.ndarray, cap: float) -> torch.Tensor:
    """Per-class n_neg/n_pos capped (boosts recall on rare labels)."""
    n = targets.shape[0]
    pos = targets.sum(axis=0).astype(np.float64)
    neg = n - pos
    raw = np.ones_like(pos)
    valid = pos > 0
    raw[valid] = neg[valid] / pos[valid]
    raw = np.clip(raw, 1.0, float(cap))
    raw[~valid] = 1.0
    return torch.tensor(raw, dtype=torch.float32)


def build_param_groups(
    model: nn.Module, lr: float, backbone_lr: float | None, weight_decay: float
) -> list[dict[str, Any]]:
    """Two groups: ``backbone.*`` params at ``backbone_lr`` (if given), the rest at ``lr``."""
    if backbone_lr is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr, "weight_decay": weight_decay, "name": "all"}]
    bb, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if name.startswith("backbone.") else head).append(p)
    groups: list[dict[str, Any]] = []
    if bb:
        groups.append({"params": bb, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"})
    if head:
        groups.append({"params": head, "lr": lr, "weight_decay": weight_decay, "name": "head"})
    return groups


def lr_scale_for_epoch(epoch: int, total: int, *, warmup: int, min_ratio: float, kind: str) -> float:
    """Multiplier applied to each group's base lr at the start of ``epoch`` (1-based)."""
    if kind == "none":
        return 1.0
    if kind != "cosine":
        raise ValueError(f"unknown scheduler {kind!r}")
    if warmup > 0 and epoch <= warmup:
        return epoch / float(warmup)
    span = max(1, total - warmup)
    t = min(1.0, max(0.0, (epoch - warmup - 1) / float(span)))
    import math

    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    dual_view: bool,
    class_mask: torch.Tensor | None,
    label_smoothing: float,
    pos_weight: torch.Tensor | None = None,
    log_every: int = 50,
    scaler: "torch.amp.GradScaler | None" = None,
    amp: bool = False,
    grad_clip: float | None = None,
) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    n_batches = len(loader)
    use_amp = bool(amp and device.type == "cuda")
    for bi, batch in enumerate(loader, start=1):
        x, x2, y = _unpack_batch(batch, dual_view, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = _model_forward(model, x, x2)
        # Loss in fp32 for numerical stability with pos_weight
        if isinstance(out, tuple):
            main, aux = out
            loss = _bce_loss(
                main.float(), y, class_mask=class_mask, label_smoothing=label_smoothing, pos_weight=pos_weight
            ) + INCEPTION_AUX_WEIGHT * _bce_loss(
                aux.float(), y, class_mask=class_mask, label_smoothing=label_smoothing, pos_weight=pos_weight
            )
        else:
            loss = _bce_loss(
                out.float(), y, class_mask=class_mask, label_smoothing=label_smoothing, pos_weight=pos_weight
            )
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], grad_clip
                )
            optimizer.step()
        loss_sum += loss.item() * len(y)
        n += len(y)
        if log_every > 0 and (bi % log_every == 0 or bi == n_batches):
            print(
                f"  batch {bi}/{n_batches} loss={loss_sum / max(n, 1):.4f}",
                flush=True,
            )
    return loss_sum / max(n, 1)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    dual_view: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    for batch in loader:
        x, x2, y = _unpack_batch(batch, dual_view, device)
        out = _model_forward(model, x, x2)
        logits_list.append(_main_logits(out).float().cpu())
        targets_list.append(y.cpu())
    return torch.cat(logits_list).numpy(), torch.cat(targets_list).numpy()


def compute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    active_mask: np.ndarray,
    *,
    threshold: float = 0.5,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    # Metrics only on active classes
    t = targets[:, active_mask]
    p = probabilities[:, active_mask]
    names = [n for n, a in zip(label_names, active_mask) if a]
    if thresholds is not None:
        thr = list(thresholds)
        if len(thr) != t.shape[1]:
            raise ValueError(f"thresholds len {len(thr)} != active classes {t.shape[1]}")
    else:
        thr = [threshold] * t.shape[1]
    matrix = multilabel_metric_matrix(t, p, thr)
    # columns: auroc, average_precision, recall, precision, f1, specificity
    macro = matrix[-2]
    micro = matrix[-1]
    per_label = []
    for i, name in enumerate(names):
        row = matrix[i]
        per_label.append(
            {
                "label": name,
                "auroc": float(row[0]),
                "average_precision": float(row[1]),
                "recall": float(row[2]),
                "precision": float(row[3]),
                "f1": float(row[4]),
                "specificity": float(row[5]),
                "threshold": float(thr[i]),
            }
        )
    return {
        "ap_macro": float(macro_average_precision(t, p)),
        "auroc_macro": float(macro_auroc(t, p)),
        "recall_macro": float(macro[2]),
        "precision_macro": float(macro[3]),
        "f1_macro": float(macro[4]),
        "recall_micro": float(micro[2]),
        "precision_micro": float(micro[3]),
        "f1_micro": float(micro[4]),
        "active_labels": names,
        "skipped_labels": [n for n, a in zip(label_names, active_mask) if not a],
        "per_label": per_label,
        "threshold": threshold if thresholds is None else "per_label",
        "thresholds": thr,
    }


def run_training(
    model: nn.Module,
    cfg: TrainConfig,
    *,
    build_note: str = "",
) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    cluster = load_cluster_map(cfg.cluster_map)
    label_names: list[str] = list(cluster["cluster_names_ordered"])
    tables = load_clustered_tables(cfg.labels_dir, label_names)
    if cfg.max_train_samples is not None and cfg.max_train_samples < len(tables["train"]):
        tables["train"] = (
            tables["train"]
            .sample(n=cfg.max_train_samples, random_state=cfg.seed)
            .reset_index(drop=True)
        )
        print(f"Subsampled train to {len(tables['train'])} rows (seed={cfg.seed})")
    if cfg.max_eval_samples is not None:
        for split in ("val", "test"):
            n = len(tables[split])
            if cfg.max_eval_samples < n:
                tables[split] = (
                    tables[split]
                    .sample(n=cfg.max_eval_samples, random_state=cfg.seed)
                    .reset_index(drop=True)
                )
        print(
            f"Subsampled val/test to <={cfg.max_eval_samples} (smoke); "
            f"val={len(tables['val'])} test={len(tables['test'])}"
        )
    loaders = make_loaders(tables, label_names, cfg)

    # Mask classes with zero positives across train+val+test
    all_y = np.vstack(
        [tables[s][label_names].to_numpy(dtype=np.float32) for s in ("train", "val", "test")]
    )
    active_np = active_class_mask(all_y)
    class_mask = torch.tensor(active_np, dtype=torch.bool, device=device)
    skipped = [n for n, a in zip(label_names, active_np) if not a]
    if skipped:
        print(f"Masking all-zero classes from loss/metrics: {skipped}")

    pos_weight_t: torch.Tensor | None = None
    if cfg.pos_weight_cap is not None:
        y_train = tables["train"][label_names].to_numpy(dtype=np.float32)
        pos_weight_t = compute_pos_weight(y_train, cfg.pos_weight_cap).to(device)
        print(f"pos_weight cap={cfg.pos_weight_cap}: {pos_weight_t.detach().cpu().tolist()}")

    model = model.to(device)
    groups = build_param_groups(model, cfg.lr, cfg.backbone_lr, cfg.weight_decay)
    optimizer = torch.optim.Adam(groups)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(
        "param groups: "
        + ", ".join(f"{g['name']}={sum(p.numel() for p in g['params']):,}@{g['lr']:g}" for g in groups)
        + f" | trainable={n_trainable:,} | scheduler={cfg.scheduler} warmup={cfg.warmup_epochs} "
        f"amp={cfg.amp and device.type == 'cuda'}",
        flush=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))

    history_rows: list[dict[str, float]] = []
    # Model selection / early-stop: val BCE loss only (not mAP).
    best_val_loss = float("inf")
    best_val_f1 = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    print(
        f"[{cfg.model_name}] device={device} epochs={cfg.epochs} "
        f"(paper_epochs hook={cfg.paper_epochs}) bs={cfg.batch_size} "
        f"img={cfg.img_size} lr={cfg.lr} labels={len(label_names)} "
        f"active={int(active_np.sum())} smoothing={cfg.label_smoothing} "
        f"select=val_loss",
        flush=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        scale = lr_scale_for_epoch(
            epoch,
            cfg.epochs,
            warmup=cfg.warmup_epochs,
            min_ratio=cfg.min_lr_ratio,
            kind=cfg.scheduler,
        )
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * scale
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            dual_view=cfg.dual_view,
            class_mask=class_mask,
            label_smoothing=cfg.label_smoothing,
            pos_weight=pos_weight_t,
            scaler=scaler,
            amp=cfg.amp,
            grad_clip=cfg.grad_clip,
        )
        val_logits, val_targets = predict_logits(
            model, loaders["val"], device, dual_view=False
        )
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_metrics = compute_metrics(
            val_targets, val_probs, label_names, active_np, threshold=cfg.threshold
        )
        # val loss without smoothing for monitoring
        with torch.no_grad():
            vt = torch.as_tensor(val_targets, device=device)
            vl = torch.as_tensor(val_logits, device=device)
            val_loss = _bce_loss(
                vl, vt, class_mask=class_mask, label_smoothing=0.0
            ).item()

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[-1]["lr"]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ap_macro": val_metrics["ap_macro"],
            "val_auroc_macro": val_metrics["auroc_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
        }
        history_rows.append(row)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | F1 {val_metrics['f1_macro']:.4f} | "
            f"R {val_metrics['recall_macro']:.4f} | P {val_metrics['precision_macro']:.4f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = float(val_metrics["f1_macro"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= cfg.early_stopping_patience:
            print(f"Early stop: no val_loss improvement since epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    # Re-create in case another process moved/archived the run dir mid-training.
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "label_names": label_names,
            "active_mask": active_np.tolist(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_f1_macro": best_val_f1,
            "selection_metric": "val_loss",
            "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}},
        },
        ckpt_path,
    )

    # Final TEST evaluation (+ optional recall-floor thresholds from VAL)
    test_logits, test_targets = predict_logits(
        model, loaders["test"], device, dual_view=False
    )
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    # Best-checkpoint VAL predictions: saved so thresholds / ensembles can be
    # recalibrated offline without ever touching TEST labels.
    val_logits, val_targets = predict_logits(model, loaders["val"], device, dual_view=False)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    np.savez_compressed(
        cfg.run_dir / "val_predictions.npz",
        logits=val_logits,
        probabilities=val_probs,
        targets=val_targets,
        label_names=np.array(label_names),
        active_mask=active_np,
    )
    calibrated_thr: list[float] | None = None
    if cfg.recall_floor is not None:
        calibrated_thr = []
        for i, active in enumerate(active_np):
            if not active:
                continue
            calibrated_thr.append(
                choose_threshold_f1_with_recall_floor(
                    val_targets[:, i], val_probs[:, i], float(cfg.recall_floor)
                )
            )
        print(f"Calibrated thresholds (recall_floor={cfg.recall_floor}): {calibrated_thr}")

    test_metrics = compute_metrics(
        test_targets,
        test_probs,
        label_names,
        active_np,
        threshold=cfg.threshold,
        thresholds=calibrated_thr,
    )

    history = pd.DataFrame(history_rows)
    history_path = cfg.run_dir / "history.csv"
    history.to_csv(history_path, index=False)

    per_label_df = pd.DataFrame(test_metrics["per_label"])
    per_label_path = cfg.run_dir / "test_metrics_per_label.csv"
    per_label_df.to_csv(per_label_path, index=False)

    summary_metrics = {
        "ap_macro": test_metrics["ap_macro"],
        "auroc_macro": test_metrics["auroc_macro"],
        "recall_macro": test_metrics["recall_macro"],
        "precision_macro": test_metrics["precision_macro"],
        "f1_macro": test_metrics["f1_macro"],
        "recall_micro": test_metrics["recall_micro"],
        "precision_micro": test_metrics["precision_micro"],
        "f1_micro": test_metrics["f1_micro"],
        "active_labels": test_metrics["active_labels"],
        "skipped_labels": test_metrics["skipped_labels"],
        "threshold": test_metrics["threshold"],
        "thresholds": test_metrics["thresholds"],
        "recall_floor": cfg.recall_floor,
        "pos_weight_cap": cfg.pos_weight_cap,
    }
    metrics_path = cfg.run_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")

    # Flat CSV for compare script
    summary_csv = cfg.run_dir / "test_metrics_summary.csv"
    pd.DataFrame(
        [
            {
                "model": cfg.model_name,
                "recall_macro": summary_metrics["recall_macro"],
                "precision_macro": summary_metrics["precision_macro"],
                "f1_macro": summary_metrics["f1_macro"],
                "f1_micro": summary_metrics["f1_micro"],
                "auroc_macro": summary_metrics["auroc_macro"],
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "epochs_run": len(history_rows),
            }
        ]
    ).to_csv(summary_csv, index=False)

    np.savez_compressed(
        cfg.run_dir / "test_predictions.npz",
        logits=test_logits,
        probabilities=test_probs,
        targets=test_targets,
        label_names=np.array(label_names),
        active_mask=active_np,
    )

    run_summary = {
        "model": cfg.model_name,
        "status": "completed",
        "device": str(device),
        "seconds": time.time() - t0,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1_macro": best_val_f1,
        "selection_metric": "val_loss",
        "epochs_requested": cfg.epochs,
        "paper_epochs": cfg.paper_epochs,
        "schedule_note": (
            f"Reduced schedule ({cfg.epochs} ep) for comparable TEST metrics; "
            f"re-run with --epochs {cfg.paper_epochs} for full paper protocol."
            if cfg.epochs < cfg.paper_epochs
            else "Full paper epoch budget."
        ),
        "img_size": cfg.img_size,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "backbone_lr": cfg.backbone_lr,
        "scheduler": cfg.scheduler,
        "warmup_epochs": cfg.warmup_epochs,
        "amp": bool(cfg.amp and device.type == "cuda"),
        "weight_decay": cfg.weight_decay,
        "trainable_params": int(n_trainable),
        "label_smoothing": cfg.label_smoothing,
        "use_paper_a_preprocess": cfg.use_paper_a_preprocess,
        "dual_view": cfg.dual_view,
        "n_train": len(tables["train"]),
        "n_val": len(tables["val"]),
        "n_test": len(tables["test"]),
        "label_names": label_names,
        "skipped_labels": skipped,
        "cluster_map": str(cfg.cluster_map),
        "labels_dir": str(cfg.labels_dir),
        "checkpoint": str(ckpt_path),
        "history": str(history_path),
        "test_metrics": summary_metrics,
        "build_note": build_note,
        "extra": cfg.extra,
    }
    summary_path = cfg.run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps({"test_metrics": summary_metrics, "run_dir": str(cfg.run_dir)}, indent=2))
    return run_summary


def common_argparser(description: str):
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--cluster-map",
        type=Path,
        default=None,
        help="cluster_map.json (default: ODIR-8 taxonomy)",
    )
    p.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory with clustered_{train,val,test}.csv",
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_QUICK_EPOCHS)
    p.add_argument("--paper-epochs", type=int, default=PAPER_EPOCHS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--taxonomy",
        choices=["odir8", "latent"],
        default="odir8",
        help="Default label scheme when --cluster-map/--labels-dir omitted (primary: odir8)",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional subsample of train set (smoke / CPU speed)",
    )
    p.add_argument(
        "--pos-weight-cap",
        type=float,
        default=None,
        help="If set, BCE pos_weight=min(n_neg/n_pos, cap) per label (recall boost)",
    )
    p.add_argument(
        "--recall-floor",
        type=float,
        default=None,
        help="If set, calibrate TEST thresholds on VAL: max F1 s.t. recall>=floor",
    )
    return p


def resolve_label_paths(
    exp_root: Path,
    cluster_map: Path | None,
    labels_dir: Path | None,
    *,
    taxonomy: str = "odir8",
) -> tuple[Path, Path]:
    """Resolve cluster_map + labels_dir. Default taxonomy is ODIR-8 (paper primary).

    Latent clustering writes map under ``outputs/clustering/`` and CSVs under
    ``outputs/labels/`` (not necessarily a sibling ``labels/`` folder).
    """
    odir_map = exp_root / "outputs" / "odir8_taxonomy" / "cluster_map.json"
    odir_labels = exp_root / "outputs" / "odir8_taxonomy" / "labels"
    latent_map = exp_root / "outputs" / "clustering" / "cluster_map.json"
    # Historical layout from 01_run_latent_clustering.py
    latent_label_candidates = [
        exp_root / "outputs" / "labels",
        exp_root / "outputs" / "clustering" / "labels",
    ]
    latent_labels = next(
        (p for p in latent_label_candidates if (p / "clustered_train.csv").is_file()),
        latent_label_candidates[0],
    )
    latent_ready = latent_map.is_file() and (latent_labels / "clustered_train.csv").is_file()
    odir_ready = odir_map.is_file() and (odir_labels / "clustered_train.csv").is_file()

    if cluster_map is None and labels_dir is None:
        if taxonomy == "latent":
            if not latent_ready:
                raise SystemExit("Latent taxonomy requested but incomplete under outputs/clustering + outputs/labels.")
            return latent_map, latent_labels
        if odir_ready:
            return odir_map, odir_labels
        if latent_ready:
            return latent_map, latent_labels
        raise SystemExit("No ready taxonomy (need cluster_map.json + clustered_train.csv).")

    if cluster_map is None:
        assert labels_dir is not None
        if labels_dir.resolve() == odir_labels.resolve():
            cluster_map = odir_map
        elif any(labels_dir.resolve() == c.resolve() for c in latent_label_candidates):
            cluster_map = latent_map
        else:
            candidate = labels_dir.parent / "cluster_map.json"
            cluster_map = candidate if candidate.is_file() else odir_map

    if labels_dir is None:
        if cluster_map.resolve() == latent_map.resolve() and latent_ready:
            labels_dir = latent_labels
        else:
            sibling = cluster_map.parent / "labels"
            if (sibling / "clustered_train.csv").is_file():
                labels_dir = sibling
            elif odir_ready:
                labels_dir = odir_labels
            else:
                raise SystemExit(f"No labels directory for {cluster_map}")

    if not cluster_map.is_file():
        raise SystemExit(f"Missing cluster map: {cluster_map}")
    if not (labels_dir / "clustered_train.csv").is_file():
        raise SystemExit(f"Missing clustered_train.csv under {labels_dir}")
    return cluster_map, labels_dir
