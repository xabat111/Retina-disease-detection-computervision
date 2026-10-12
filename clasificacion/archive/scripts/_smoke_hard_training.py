import sys
import time
from dataclasses import replace
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from src.cached_datasets import build_cached_letterbox_splits
from src.hard_example_training import train_with_hard_examples
from src.recall_focus_config import VARIANTS, WARM_START_CHECKPOINT, build_variant_train_config
from src.tuning import InMemoryImageDataset

variant = VARIANTS["both"]
config = replace(
    build_variant_train_config(variant),
    epochs=2,
    checkpoint_dir=Path("artifacts/smoke_test/checkpoints"),
    best_checkpoint_name="smoke_both.pt",
)
splits = build_cached_letterbox_splits(config)
small_train = InMemoryImageDataset(splits.train_dataset.images[:64], splits.train_dataset.labels[:64], splits.train_dataset.transform)
small_val = InMemoryImageDataset(splits.val_dataset.images[:64], splits.val_dataset.labels[:64], splits.val_dataset.transform)
val_loader = DataLoader(small_val, batch_size=32)
history = []
start = time.time()
smoke_hard = replace(variant.hard_config, uniform_warmup_epochs=1)
train_with_hard_examples(small_train, val_loader, config, smoke_hard, WARM_START_CHECKPOINT, history)
print("tiempo", time.time() - start)
for row in history:
    print(row)
