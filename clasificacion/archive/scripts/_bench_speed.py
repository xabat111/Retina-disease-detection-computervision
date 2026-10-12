import sys
import time
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from src.cached_datasets import build_cached_letterbox_splits
from src.config import default_config
from src.data import PREPROCESSING_LETTERBOX
from src.losses import BinaryFocalLoss
from src.models_zoo import build_model_from_config
from src.train import create_optimizer

base = replace(default_config(), preprocessing=PREPROCESSING_LETTERBOX, pretrained=False)
splits = build_cached_letterbox_splits(base)
loader = DataLoader(splits.train_dataset, batch_size=16, shuffle=True, num_workers=0)

# Coste de preparar batches (augmentation) sin modelo
start = time.time()
for index, _ in enumerate(loader):
    if index == 20:
        break
print(f"datos: {(time.time() - start) / 20:.2f} s/batch (batch 16)")

specs = [("resnet152 layer3+4", replace(base, unfreeze_layer3=True))]
specs.append(("resnet152 layer4", replace(base, unfreeze_layer3=False)))
for architecture, blocks in (("resnet50", 2), ("densenet121", 2), ("efficientnet_b0", 2), ("convnext_tiny", 2)):
    specs.append((f"{architecture} blocks={blocks}", replace(base, backbone_name=architecture, unfrozen_blocks=blocks)))

criterion = BinaryFocalLoss()
batch = next(iter(loader))
for name, config in specs:
    model = build_model_from_config(config)
    optimizer = create_optimizer(model, config)
    model.train()
    images, labels = batch
    for step in range(5):
        if step == 1:
            start = time.time()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    train_step = (time.time() - start) / 4
    model.eval()
    with torch.no_grad():
        start = time.time()
        model(torch.cat([images, images]))
        eval_step = (time.time() - start) / 32
    epoch_minutes = (120 * train_step + 640 * eval_step) / 60
    print(f"{name}: train {train_step:.2f} s/batch16, eval {eval_step * 1000:.0f} ms/img "
          f"-> ~{epoch_minutes:.1f} min/epoch (compute) ", flush=True)
