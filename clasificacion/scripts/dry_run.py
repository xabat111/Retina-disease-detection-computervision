"""Dry-run: imports + 1 train batch + val metrics (no full training)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cache local del proyecto (evita conflictos con ~/.cache si no es un directorio)
torch_home = PROJECT_ROOT / "artifacts" / "torch_cache"
torch_home.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(torch_home)

from src.config import default_config
from src.data import create_dataloaders
from src.evaluate import evaluate_loader
from src.losses import BinaryFocalLoss
from src.model import build_resnet152, count_trainable_parameters, describe_freeze_policy
from src.train import create_optimizer, get_device, set_seed


def main() -> None:
    config = default_config()
    config.batch_size = 2
    config.num_workers = 0
    set_seed(config.seed)
    device = get_device()

    print("=== Dry-run (1 train batch + val eval) ===")
    print(f"epochs default={default_config().epochs}")
    print(f"device={device}")

    try:
        model = build_resnet152(config).to(device)
        print("pretrained weights: OK")
    except Exception as exc:  # noqa: BLE001 — dry-run fallback
        print(f"pretrained failed: {type(exc).__name__}: {exc}")
        config.pretrained = False
        model = build_resnet152(config).to(device)
        print("using pretrained=False for dry-run")

    trainable, total = count_trainable_parameters(model)
    print(f"trainable/total={trainable:,}/{total:,}")
    print(describe_freeze_policy(model))

    train_loader, val_loader, _test_loader, train_df, val_df, test_df = create_dataloaders(
        config
    )
    print(f"splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("train balance:", train_df["disease_risk"].value_counts().to_dict())

    sample_path = Path(train_df["image_path"].iloc[0])
    print(f"sample image exists={sample_path.is_file()} path={sample_path}")

    criterion = BinaryFocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    optimizer = create_optimizer(model, config)

    model.train()
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"train_loss (1 batch)={loss.item():.4f}")

    val_metrics = evaluate_loader(model, val_loader, device, criterion=criterion)
    print(
        f"Epoch 01/{config.epochs} | "
        f"train_loss={loss.item():.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"AUC={val_metrics['auc']:.4f} | "
        f"F1={val_metrics['f1']:.4f} | "
        f"Acc={val_metrics['accuracy']:.4f}"
    )
    print("DRY-RUN OK")


if __name__ == "__main__":
    main()
