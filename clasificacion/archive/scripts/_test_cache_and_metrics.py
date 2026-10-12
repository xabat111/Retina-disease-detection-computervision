import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, ".")
from src.cached_datasets import build_cached_letterbox_splits
from src.config import default_config
from src.data import PREPROCESSING_LETTERBOX, create_dataloaders
from src.recall_selection import compute_recall_selection_metrics
from src.hard_example_training import HardExampleConfig, HardExampleTracker

config = replace(default_config(), preprocessing=PREPROCESSING_LETTERBOX)

start = time.time()
splits = build_cached_letterbox_splits(config)
print("cache construida/cargada en", round(time.time() - start), "s")

# 1) Equivalencia con el pipeline original (val determinista y train con la misma semilla)
original_train, original_val, *_ = create_dataloaders(config)[:3]
original_train_dataset = original_train.dataset
original_val_dataset = original_val.dataset
for index in (0, 5, 100):
    a, _ = splits.val_dataset[index]
    b, _ = original_val_dataset[index]
    print("val", index, torch.equal(a, b), float((a - b).abs().max()))
    torch.manual_seed(7)
    a, _ = splits.train_dataset[index]
    torch.manual_seed(7)
    b, _ = original_train_dataset[index]
    print("train", index, torch.equal(a, b), float((a - b).abs().max()))

# 2) Métricas de selección con un caso sintético
rng = np.random.default_rng(0)
labels = np.r_[np.ones(500), np.zeros(140)]
probabilities = np.r_[rng.beta(5, 2, 500), rng.beta(2, 5, 140)]
print(compute_recall_selection_metrics(probabilities, labels))
perfect = labels * 0.9 + 0.05
print("perfecto", compute_recall_selection_metrics(perfect, labels))

# 3) Tracker
tracker = HardExampleTracker(np.array([1, 1, 0, 0]), HardExampleConfig())
tracker.update(np.array([0, 0, 1, 2, 2]), np.array([0.1, 0.3, 0.9, 0.8, 0.6]))
print(tracker.smoothed_error, tracker.sampling_weights(), tracker.describe())
tracker.update(np.array([0, 1, 2, 3]), np.array([0.2, 0.9, 0.1, 0.3]))
print(tracker.smoothed_error, tracker.sampling_weights())
