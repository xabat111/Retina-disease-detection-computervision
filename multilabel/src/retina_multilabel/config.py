"""Configuración central del proyecto: todos los hiperparámetros viven aquí."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

SMOKE_TEST_ENV_VARIABLE = "RETINA_SMOKE_TEST"

PosWeightMode = Literal["capped", "uncapped", "soft_cap", "class_balanced", "iterative"]
IterativeInitMode = Literal["soft_cap", "uncapped", "capped"]


def _default_num_workers() -> int:
    """Pocos workers: el preprocesado es ligero y Kaggle ofrece ~4 CPUs."""
    return min(4, os.cpu_count() or 1)


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 224
    # Por debajo de este número de positivos (en val o test) las métricas de
    # un grupo no son fiables y se marcan como tal.
    min_positives_for_reliable_metrics: int = 20
    num_workers: int = field(default_factory=_default_num_workers)
    # Solo para la prueba rápida: nº máximo de filas por split (train, val, test).
    max_rows_per_split: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class ModelConfig:
    # "auto": RETFound si hay token de Hugging Face y se descarga bien; si no, ViT-L público.
    # "public" | "dinov2" | "retfound" | "smoke" (ViT diminuto sin pesos, solo para pruebas).
    backbone_choice: str = "auto"
    public_model_name: str = "vit_large_patch16_224.augreg_in21k_ft_in1k"
    dinov2_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    # Si se fija (p. ej. 392), timm interpola los embeddings de posición.
    img_size: int | None = None
    retfound_repo_id: str = "YukunZhou/RETFound_mae_natureCFP"
    retfound_filename: str = "RETFound_mae_natureCFP.pth"
    hf_token_secret_name: str = "HF_TOKEN"
    # DINOv2 backbone init: "default" (timm LVD-142M weights) or "fundus" (RETFound-DINOv2,
    # YukunZhou/RETFound_dinov2_meh, gated: needs HF_TOKEN or a local file). If left at "default",
    # the VIT_PRETRAINED / VIT_FUNDUS_WEIGHTS environment variables are honoured.
    vit_pretrained: str = "default"
    vit_fundus_weights: str | None = None


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    max_epochs: int = 40
    early_stopping_patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Tope del peso de positivos de la BCE para evitar gradientes enormes en clases raras.
    max_pos_weight: float = 20.0
    # Modo de pos_weight. Default "capped" = BaselineV2 (no romper defaults).
    # Ablation: uncapped | soft_cap | class_balanced | iterative (ver EXPERIMENT_CLASS_WEIGHTS.md).
    pos_weight_mode: PosWeightMode = "capped"
    max_pos_weight_uncapped: float = 100.0
    cb_beta: float = 0.9999
    # Iterative adapt (solo si pos_weight_mode="iterative"):
    iterative_init: IterativeInitMode = "soft_cap"
    iterative_init_cap: float = 40.0
    adapt_every_n_epochs: int = 2
    adapt_warmup_epochs: int = 2
    adapt_eta: float = 0.15
    adapt_momentum: float = 0.7
    adapt_anchor: float = 0.05
    adapt_w_max: float = 50.0
    # Aumentos ligeros (solo en train): volteo horizontal, rotación pequeña y color.
    rotation_degrees: float = 10.0
    color_jitter_strength: float = 0.2


@dataclass(frozen=True)
class EvaluationConfig:
    # Criterio para elegir el umbral por grupo en validación: "f1" o "f2".
    threshold_criterion: str = "f1"
    # Cribado sano-vs-enfermo: se prioriza la sensibilidad (F2).
    screening_threshold_criterion: str = "f2"
    # Segundo punto de operación del cribado: mayor umbral cuya sensibilidad en val sea >= este
    # valor (comparable con el modelo binario previo, que tenía sensibilidad ~0.93).
    screening_target_recall: float = 0.93
    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95


@dataclass(frozen=True)
class ExplainConfig:
    cases_per_outcome: int = 2  # verdaderos positivos y falsos negativos por grupo
    rollout_layers: int = 12  # últimas capas que se componen en el attention rollout
    # Limita los mapas al campo de visión (los ViT generan "tokens artefacto" en el fondo negro).
    mask_background: bool = True
    use_integrated_gradients: bool = True
    integrated_gradients_steps: int = 16


@dataclass(frozen=True)
class FinetuneConfig:
    """Ajuste fino parcial (notebooks 05-06). No cambia el baseline de solo cabeza."""

    n_unfrozen_blocks: int = 4
    head_learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-5
    llrd_decay: float = 0.75
    # Focal Loss: gamma modula ejemplos fáciles; alpha_pos pondera positivos
    # (los negativos reciben 1 - alpha_pos). No hay pos_weight aparte.
    focal_gamma: float = 2.0
    focal_alpha_pos: float = 0.25
    max_epochs: int = 25
    early_stopping_patience: int = 6
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = True
    lr_scheduler: str = "cosine"
    # Checkpoint + early-stop: "val_loss" (default, BaselineV2) o "val_mAP".
    selection_metric: str = "val_loss"


@dataclass(frozen=True)
class OptunaConfig:
    """Búsqueda corta en GPU. El objetivo es minimizar val_loss; train/val no miran el test."""

    timeout_seconds: float = 9000.0  # 2.5 h de búsqueda; ~0.5-1 h extra para el reentrenamiento
    min_trials: int = 4
    max_trials: int = 24
    min_epochs_per_trial: int = 2
    max_epochs_per_trial: int = 6
    n_startup_trials: int = 3
    n_warmup_steps: int = 1
    probe_unfrozen_blocks: int = 4
    probe_batch_size: int = 16
    sqlite_filename: str = "optuna_finetune.db"
    best_params_filename: str = "best_params.json"
    # Fracción del timeout reservada mentalmente para el notebook 06 (no se gasta aquí).
    retrain_time_fraction: float = 0.25


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    smoke_test: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)


def build_smoke_test_config() -> ProjectConfig:
    """Configuración mínima para comprobar el código de punta a punta en CPU."""
    base = ProjectConfig(smoke_test=True)
    return replace(
        base,
        data=replace(base.data, num_workers=0, max_rows_per_split=(96, 64, 96)),
        model=replace(base.model, backbone_choice="smoke"),
        train=replace(base.train, batch_size=16, max_epochs=2, early_stopping_patience=2),
        evaluation=replace(base.evaluation, bootstrap_resamples=20),
        explain=replace(
            base.explain, cases_per_outcome=1, rollout_layers=2, integrated_gradients_steps=4
        ),
        finetune=replace(
            base.finetune,
            n_unfrozen_blocks=2,
            max_epochs=2,
            early_stopping_patience=2,
            use_gradient_checkpointing=False,
        ),
        optuna=replace(
            base.optuna,
            timeout_seconds=90.0,
            min_trials=2,
            max_trials=2,
            min_epochs_per_trial=1,
            max_epochs_per_trial=1,
            n_startup_trials=1,
            n_warmup_steps=0,
            probe_unfrozen_blocks=2,
            probe_batch_size=16,
        ),
    )


def get_config() -> ProjectConfig:
    """Devuelve la configuración por defecto o la de prueba rápida si se pide por entorno."""
    if os.environ.get(SMOKE_TEST_ENV_VARIABLE) == "1":
        return build_smoke_test_config()
    return ProjectConfig()


def build_baseline_v2_config() -> ProjectConfig:
    """Receta BaselineV2: DINOv2 ViT-L/14 @392, BCE+pos_weight, fixed-LR unfreeze.

    Coincide con el checkpoint `fixed_lr_unfreeze_2_checkpoint.pt` (salvo umbrales de
    evaluación: el entrenamiento usaba f1; el ranking TEST usa auprc_recall).
    """
    base = ProjectConfig()
    return replace(
        base,
        data=replace(base.data, image_size=392, num_workers=min(4, os.cpu_count() or 1)),
        model=replace(
            base.model,
            backbone_choice="dinov2",
            dinov2_model_name="vit_large_patch14_dinov2.lvd142m",
            img_size=392,
        ),
        train=replace(
            base.train,
            batch_size=16,
            max_epochs=100,
            early_stopping_patience=12,
            learning_rate=2e-4,
            weight_decay=0.01,
        ),
        evaluation=replace(base.evaluation, threshold_criterion="auprc_recall"),
        finetune=replace(
            base.finetune,
            n_unfrozen_blocks=2,
            head_learning_rate=2e-4,
            backbone_learning_rate=1e-5,
            llrd_decay=1.0,
            max_epochs=100,
            early_stopping_patience=12,
            use_gradient_checkpointing=True,
            lr_scheduler="cosine",
        ),
    )
