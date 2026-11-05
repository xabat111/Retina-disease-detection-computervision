"""Approach C — ResNet50 trained multilabel on ~40 RFMiD disease columns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

from ..encoders import EncoderBundle
from ..rfmid_columns import disease_columns_present
from .protocol import index_images, resolve_device


class SupervisedResNetEncoder(nn.Module):
    """ResNet50 backbone + multilabel head; `.embed(x)` returns penultimate features."""

    def __init__(self, n_labels: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)
        self.embed_dim = int(backbone.fc.in_features)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))
        self.head = nn.Linear(self.embed_dim, n_labels)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return z.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class _EmbedOnly(nn.Module):
    def __init__(self, model: SupervisedResNetEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.embed(x)


class _MultilabelImageDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        labels: torch.Tensor,
        image_index: dict[str, Path],
        transform,
    ) -> None:
        self.ids = ids
        self.labels = labels
        self.image_index = image_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        path = self.image_index[self.ids[idx]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.labels[idx]


@dataclass
class SupervisedTrainConfig:
    img_size: int = 224
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-4
    pretrained: bool = True
    max_train_images: int | None = None
    device: str = "auto"
    num_workers: int = 0
    freeze_backbone_epochs: int = 0


def supervised_transform(img_size: int, train: bool = False) -> T.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    ops: list = [T.Resize((img_size, img_size))]
    if train:
        ops.extend([T.RandomHorizontalFlip(), T.RandomRotation(10)])
    ops.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return T.Compose(ops)


def train_supervised_encoder(
    labels_csv: Path,
    image_dir: Path,
    output_dir: Path,
    cfg: SupervisedTrainConfig | None = None,
) -> Path:
    """Train ResNet50 multilabel on fine-grained RFMiD columns (BCEWithLogitsLoss)."""
    cfg = cfg or SupervisedTrainConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)

    df = pd.read_csv(labels_csv)
    cols = disease_columns_present(list(df.columns))
    if not cols:
        raise RuntimeError(f"No disease columns in {labels_csv}")
    id_col = "ID" if "ID" in df.columns else df.columns[0]
    images = index_images(image_dir)

    keep_mask = df[id_col].astype(str).isin(images.keys())
    df = df.loc[keep_mask].reset_index(drop=True)
    if cfg.max_train_images is not None:
        df = df.iloc[: cfg.max_train_images].reset_index(drop=True)

    ids = df[id_col].astype(str).tolist()
    y = torch.tensor(df[cols].astype(float).values, dtype=torch.float32)

    transform = supervised_transform(cfg.img_size, train=True)
    loader = DataLoader(
        _MultilabelImageDataset(ids, y, images, transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(ids) >= cfg.batch_size,
    )

    model = SupervisedResNetEncoder(n_labels=len(cols), pretrained=cfg.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_backbone_epochs and epoch <= cfg.freeze_backbone_epochs:
            for p in model.features.parameters():
                p.requires_grad_(False)
            params = model.head.parameters()
        else:
            for p in model.features.parameters():
                p.requires_grad_(True)
            params = model.parameters()
        opt = torch.optim.Adam(params, lr=cfg.lr)

        model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Sup epoch {epoch}/{cfg.epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        mean_loss = running / max(n_batches, 1)
        history.append({"epoch": epoch, "bce": mean_loss})
        print(f"epoch {epoch}: bce={mean_loss:.6f}")

    ckpt_path = output_dir / "supervised_encoder.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_columns": cols,
            "config": asdict(cfg),
            "embed_dim": model.embed_dim,
            "img_size": cfg.img_size,
            "n_labels": len(cols),
        },
        ckpt_path,
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "checkpoint": str(ckpt_path),
        "n_images": len(ids),
        "n_labels": len(cols),
        "final_bce": history[-1]["bce"] if history else None,
        "epochs": cfg.epochs,
        "img_size": cfg.img_size,
        "device": str(device),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return ckpt_path


def load_supervised_encoder_bundle(checkpoint: Path, device: str = "auto") -> EncoderBundle:
    """Load trained supervised encoder; embeddings = penultimate ResNet features."""
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    n_labels = int(ckpt.get("n_labels", len(ckpt["label_columns"])))
    img_size = int(ckpt.get("img_size", ckpt.get("config", {}).get("img_size", 224)))
    pretrained = bool(ckpt.get("config", {}).get("pretrained", True))
    model = SupervisedResNetEncoder(n_labels=n_labels, pretrained=False)
    # pretrained flag only matters at construction for weight download; we load state
    _ = pretrained
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return EncoderBundle(
        name="supervised_resnet50_multilabel",
        model=_EmbedOnly(model),
        transform=supervised_transform(img_size, train=False),
        embed_dim=int(ckpt.get("embed_dim", model.embed_dim)),
        img_size=img_size,
    )
