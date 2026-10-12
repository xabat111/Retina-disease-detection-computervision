"""Fine-tuning centrado en recall: re-muestreo por época de los ejemplos que siguen fallando.

Dos ideas, ambas configurables y desactivables (con `weighting_strength=0` se obtiene un
fine-tuning normal, útil como control):

1. Re-ponderación por época (estilo boosting / OHEM). Durante cada época se guarda, GRATIS, lo que
   el modelo predijo para cada imagen en el propio forward de entrenamiento. Con ello se mantiene
   un "error suavizado" por imagen (media móvil exponencial) y la época siguiente se muestrea con
   un `WeightedRandomSampler` que repite más las imágenes que siguen mal clasificadas,
   y más aún los falsos negativos (positivos predichos como sanos).
   La Focal Loss ya enfatiza ejemplos difíciles DENTRO de cada batch; esto añade énfasis a nivel
   de época en las que SIGUEN siendo difíciles.
   Contra el ruido de etiquetas (enfermedades raras, imágenes ambiguas): el error se suaviza con
   una media móvil y el peso de muestreo tiene un tope, así que ninguna imagen domina la época.

2. Criterio centrado en recall: la loss es la `BinaryFocalLoss` de siempre pero con `focal_alpha`
   configurable (alpha pesa a los POSITIVOS y 1 - alpha a los negativos) y el checkpoint se elige
   con un criterio de validación orientado a recall (`src/recall_selection.py`).

Sin fuga de datos: el entrenamiento solo ve train; validación solo elige la época y el checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.config import TrainConfig
from src.evaluate import evaluate_loader
from src.losses import BinaryFocalLoss
from src.model import count_trainable_parameters, save_model
from src.models_zoo import build_model_from_config
from src.train import (
    create_optimizer,
    create_scheduler,
    get_device,
    resolve_selection_metric,
    set_seed,
    step_scheduler,
)
from src.tuning import InMemoryImageDataset


@dataclass(frozen=True)
class HardExampleConfig:
    """Parámetros de la re-ponderación por época.

    peso_i = min(max_sample_weight, 1 + weighting_strength * error_suavizado_i * factor_clase_i)
    donde factor_clase = positive_boost para positivos (coste de un FN) y 1 para negativos, y
    error = probabilidad que el modelo asignó a la clase equivocada (0 = acierta con seguridad).
    """

    weighting_strength: float = 3.0
    positive_boost: float = 2.0
    max_sample_weight: float = 6.0
    # Peso de la historia en la media móvil del error: 0 = solo la última época
    ema_momentum: float = 0.5
    # Épocas iniciales con muestreo uniforme (solo para registrar errores)
    uniform_warmup_epochs: int = 1

    @property
    def is_active(self) -> bool:
        return self.weighting_strength > 0.0


class IndexedDataset(Dataset):
    """Añade el índice de la muestra al par (imagen, etiqueta) para poder registrar sus errores."""

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image, label = self.base_dataset[index]
        return image, label, index


class HardExampleTracker:
    """Error suavizado por imagen y pesos de muestreo derivados de él."""

    def __init__(self, labels: np.ndarray, hard_config: HardExampleConfig) -> None:
        self.labels = labels.astype(int)
        self.hard_config = hard_config
        self.smoothed_error = np.full(len(labels), np.nan)

    def update(self, indices: np.ndarray, positive_probabilities: np.ndarray) -> None:
        """Incorpora lo observado en una época (una imagen puede aparecer varias veces o ninguna)."""
        sample_labels = self.labels[indices]
        wrong_class_probability = np.where(
            sample_labels == 1, 1.0 - positive_probabilities, positive_probabilities
        )
        n_samples = len(self.labels)
        times_seen = np.bincount(indices, minlength=n_samples)
        epoch_mean_error = np.bincount(indices, weights=wrong_class_probability, minlength=n_samples)
        was_seen = times_seen > 0
        epoch_mean_error[was_seen] /= times_seen[was_seen]

        is_first_observation = was_seen & np.isnan(self.smoothed_error)
        has_history = was_seen & ~np.isnan(self.smoothed_error)
        momentum = self.hard_config.ema_momentum
        self.smoothed_error[is_first_observation] = epoch_mean_error[is_first_observation]
        self.smoothed_error[has_history] = (
            momentum * self.smoothed_error[has_history]
            + (1.0 - momentum) * epoch_mean_error[has_history]
        )

    def sampling_weights(self) -> np.ndarray:
        """Pesos para `WeightedRandomSampler`; todo 1.0 si aún no hay errores registrados."""
        error = np.nan_to_num(self.smoothed_error, nan=0.0)
        class_factor = np.where(self.labels == 1, self.hard_config.positive_boost, 1.0)
        weights = 1.0 + self.hard_config.weighting_strength * error * class_factor
        return np.minimum(weights, self.hard_config.max_sample_weight)

    def describe(self) -> dict[str, float]:
        """Resumen para el historial: cuántas imágenes siguen mal clasificadas (error > 0.5)."""
        error = np.nan_to_num(self.smoothed_error, nan=0.0)
        weights = self.sampling_weights()
        positives = self.labels == 1
        return {
            "n_hard_positives": int(np.sum(positives & (error > 0.5))),
            "n_hard_negatives": int(np.sum(~positives & (error > 0.5))),
            "mean_positive_weight": float(weights[positives].mean()),
            "mean_negative_weight": float(weights[~positives].mean()),
            "capped_fraction": float(np.mean(weights >= self.hard_config.max_sample_weight - 1e-9)),
        }


def _train_one_epoch_tracking_errors(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Una época de entrenamiento; devuelve (loss media, índices vistos, P(positivo) en el forward).

    Las probabilidades salen del mismo forward que calcula la loss (con augmentation y en modo
    train), por lo que registrar los errores no añade coste.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    seen_indices: list[np.ndarray] = []
    seen_probabilities: list[np.ndarray] = []

    for images, labels, indices in tqdm(train_loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        seen_indices.append(indices.numpy())
        seen_probabilities.append(torch.sigmoid(logits.detach().view(-1)).cpu().numpy())

    return (
        running_loss / max(num_batches, 1),
        np.concatenate(seen_indices),
        np.concatenate(seen_probabilities).astype(np.float64),
    )


def _build_epoch_loader(
    indexed_dataset: IndexedDataset,
    sampling_weights: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """DataLoader que muestrea con reemplazo (misma longitud de época que el normal)."""
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sampling_weights, dtype=torch.double),
        num_samples=len(indexed_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(indexed_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def _load_warm_start(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


def train_with_hard_examples(
    train_dataset: InMemoryImageDataset,
    val_loader: DataLoader,
    config: TrainConfig,
    hard_config: HardExampleConfig,
    warm_start_checkpoint: Path | None = None,
    history: list[dict] | None = None,
) -> nn.Module:
    """Fine-tuning con re-ponderación por época y checkpoint por criterio de validación.

    El mejor checkpoint (según `config.selection_metric`, ver `src.train.SELECTION_METRICS`) se
    guarda en `config.checkpoint_dir / config.best_checkpoint_name`; si ya existe se lanza un
    error para no sobrescribir nada. Con `warm_start_checkpoint` se parte de unos pesos ya
    entrenados (se evalúan primero y se anotan como época 0 en `history`; no cuentan como
    candidatos a mejor checkpoint).
    """
    selection_key, higher_is_better = resolve_selection_metric(config.selection_metric)
    checkpoint_path = config.checkpoint_dir / config.best_checkpoint_name
    if checkpoint_path.exists():
        raise FileExistsError(f"No se sobrescribe un checkpoint existente: {checkpoint_path}")

    set_seed(config.seed)
    device = get_device()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config).to(device)
    if warm_start_checkpoint is not None:
        _load_warm_start(model, warm_start_checkpoint, device)
    trainable, total = count_trainable_parameters(model)
    print(f"Dispositivo: {device} | parámetros entrenables: {trainable:,} / {total:,}")
    print(
        f"Épocas: {config.epochs} | lr={config.learning_rate:.2e} | scheduler={config.lr_scheduler} | "
        f"alpha={config.focal_alpha} | gamma={config.focal_gamma} | selección={config.selection_metric}"
    )
    print(f"Re-ponderación: {hard_config}")

    criterion = BinaryFocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    indexed_train_dataset = IndexedDataset(train_dataset)
    train_labels = np.asarray(train_dataset.labels, dtype=float)
    tracker = HardExampleTracker(train_labels, hard_config)

    if warm_start_checkpoint is not None and history is not None:
        initial_metrics = evaluate_loader(model, val_loader, device, criterion=criterion)
        history.append(_history_row(0, float("nan"), initial_metrics, optimizer, 0.0, tracker))
        print(f"Época 0 (pesos de partida): {_format_epoch_metrics(initial_metrics)}")

    best_value = -float("inf") if higher_is_better else float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        use_weights = hard_config.is_active and epoch > hard_config.uniform_warmup_epochs
        sampling_weights = tracker.sampling_weights() if use_weights else np.ones(len(train_labels))
        train_loader = _build_epoch_loader(
            indexed_train_dataset, sampling_weights, config.batch_size, seed=config.seed + epoch
        )

        train_loss, seen_indices, seen_probabilities = _train_one_epoch_tracking_errors(
            model, train_loader, criterion, optimizer, device
        )
        tracker.update(seen_indices, seen_probabilities)
        val_metrics = evaluate_loader(model, val_loader, device, criterion=criterion)

        selection_value = float(val_metrics[selection_key])
        if history is not None:
            history.append(
                _history_row(epoch, train_loss, val_metrics, optimizer, time.time() - epoch_start_time, tracker)
            )
        step_scheduler(scheduler, float(val_metrics["auc"]))
        print(
            f"Epoch {epoch:02d}/{config.epochs} | train_loss={train_loss:.4f} | "
            f"{_format_epoch_metrics(val_metrics)} | "
            f"{time.time() - epoch_start_time:.0f}s"
        )

        is_improvement = (
            selection_value > best_value if higher_is_better else selection_value < best_value
        )
        if is_improvement:
            best_value = selection_value
            epochs_without_improvement = 0
            save_model(
                model,
                checkpoint_path,
                metadata={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "selection_metric": config.selection_metric,
                    "selection_value": best_value,
                    "val_metrics": val_metrics,
                    "backbone": config.backbone_name,
                    "preprocessing": config.preprocessing,
                    "warm_start_checkpoint": str(warm_start_checkpoint) if warm_start_checkpoint else None,
                    "hard_example_config": asdict(hard_config),
                    "focal_alpha": config.focal_alpha,
                    "focal_gamma": config.focal_gamma,
                },
            )
            print(f"  -> nuevo mejor ({config.selection_metric}={best_value:.4f}) guardado en {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping en época {epoch} (paciencia={config.early_stopping_patience})")
                break

    print(f"Mejor {config.selection_metric} de validación: {best_value:.4f}")
    return model


def _history_row(
    epoch: int,
    train_loss: float,
    val_metrics: dict,
    optimizer: torch.optim.Optimizer,
    epoch_seconds: float,
    tracker: HardExampleTracker,
) -> dict:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": float(val_metrics["loss"]),
        "val_auc": float(val_metrics["auc"]),
        "val_f1": float(val_metrics["f1"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_best_f2": float(val_metrics["best_f2"]),
        "val_recall_at_spec_85": float(val_metrics["recall_at_spec_85"]),
        "val_spec_at_fn_lt_fp": float(val_metrics["spec_at_fn_lt_fp"]),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch_seconds": epoch_seconds,
        **tracker.describe(),
    }


def _format_epoch_metrics(val_metrics: dict) -> str:
    return (
        f"val_loss={val_metrics['loss']:.4f} | AUC={val_metrics['auc']:.4f} | "
        f"F2*={val_metrics['best_f2']:.4f} | rec@spec85={val_metrics['recall_at_spec_85']:.4f} | "
        f"spec@FN<FP={val_metrics['spec_at_fn_lt_fp']:.4f}"
    )
