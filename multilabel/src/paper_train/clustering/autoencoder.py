"""Approach B — convolutional autoencoder trained from scratch on RFMiD images."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

from ..encoders import EncoderBundle
from .protocol import index_images, resolve_device


class ConvAutoencoder(nn.Module):
    """Lightweight conv AE; bottleneck vector is the embedding for clustering."""

    def __init__(self, latent_dim: int = 128, img_size: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        # 128 → 64 → 32 → 16 → 8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        flat = 256 * (img_size // 16) * (img_size // 16)
        self.fc_enc = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._flat = flat
        self._spatial = img_size // 16

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(-1, 256, self._spatial, self._spatial)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class _EncoderOnly(nn.Module):
    """Wraps ConvAutoencoder.encode for the shared clustering protocol."""

    def __init__(self, ae: ConvAutoencoder) -> None:
        super().__init__()
        self.ae = ae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(x)


class _UnlabeledImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


@dataclass
class AETrainConfig:
    img_size: int = 128
    latent_dim: int = 128
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    max_train_images: int | None = None
    device: str = "auto"
    num_workers: int = 0


def ae_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # [0,1] for Sigmoid reconstruction
        ]
    )


def train_autoencoder(
    image_dir: Path,
    output_dir: Path,
    cfg: AETrainConfig | None = None,
) -> Path:
    """Train AE from scratch on unlabeled train images. Saves checkpoint + history."""
    cfg = cfg or AETrainConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)

    images = list(index_images(image_dir).values())
    if cfg.max_train_images is not None:
        images = images[: cfg.max_train_images]
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    transform = ae_transform(cfg.img_size)
    loader = DataLoader(
        _UnlabeledImageDataset(images, transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(images) >= cfg.batch_size,
    )

    model = ConvAutoencoder(latent_dim=cfg.latent_dim, img_size=cfg.img_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    history: list[dict] = []
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        running = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"AE epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        mean_loss = running / max(n_batches, 1)
        history.append({"epoch": epoch, "mse": mean_loss})
        print(f"epoch {epoch}: mse={mean_loss:.6f}")

    ckpt_path = output_dir / "autoencoder.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "latent_dim": cfg.latent_dim,
            "img_size": cfg.img_size,
        },
        ckpt_path,
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "checkpoint": str(ckpt_path),
        "n_images": len(images),
        "final_mse": history[-1]["mse"] if history else None,
        "epochs": cfg.epochs,
        "latent_dim": cfg.latent_dim,
        "img_size": cfg.img_size,
        "device": str(device),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return ckpt_path


def load_ae_encoder_bundle(checkpoint: Path, device: str = "auto") -> EncoderBundle:
    """Load trained AE and return EncoderBundle using bottleneck embeddings."""
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    img_size = int(ckpt.get("img_size", cfg.get("img_size", 128)))
    latent_dim = int(ckpt.get("latent_dim", cfg.get("latent_dim", 128)))
    ae = ConvAutoencoder(latent_dim=latent_dim, img_size=img_size)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    enc = _EncoderOnly(ae)
    return EncoderBundle(
        name="conv_autoencoder",
        model=enc,
        transform=ae_transform(img_size),
        embed_dim=latent_dim,
        img_size=img_size,
    )
