"""Modelo: backbone ViT-Large congelado (timm público o RETFound) + cabeza lineal de 10 salidas."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn

from .config import ModelConfig, ProjectConfig
from .labels import LABEL_NAMES

BACKBONE_KINDS = ("public", "dinov2", "retfound", "smoke")
# Con RETFound (MAE) el nuevo LayerNorm tras el pooling no existe en el checkpoint.
_RETFOUND_ALLOWED_MISSING_PREFIX = "fc_norm."
CHECKPOINT_FILE_NAME = "head_checkpoint.pt"
FINETUNE_CHECKPOINT_FILE_NAME = "finetune_checkpoint.pt"
BASELINE_V2_CHECKPOINT_FILE_NAME = "baseline_v2_head_checkpoint.pt"


@dataclass
class BackboneBundle:
    """Backbone congelado y la información necesaria para usarlo correctamente."""

    backbone: nn.Module
    kind: str  # "public" | "retfound" | "smoke"
    description: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    num_features: int
    uses_cls_token: bool  # True: el vector pooled sale del token [CLS]; False: media de parches


def _freeze(backbone: nn.Module) -> nn.Module:
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone.eval()


def _create_timm_backbone(
    model_name: str,
    img_size: int | None = None,
    *,
    dynamic_img_size: bool = False,
) -> nn.Module:
    kwargs: dict[str, Any] = {"pretrained": True, "num_classes": 0}
    if img_size is not None:
        kwargs["img_size"] = img_size
    if dynamic_img_size:
        # DINOv2 defaults to 518; allow 224/392 paper-compare caches without assert.
        kwargs["dynamic_img_size"] = True
    return timm.create_model(model_name, **kwargs)


def _build_public_backbone(model_config: ModelConfig) -> BackboneBundle:
    backbone = _create_timm_backbone(model_config.public_model_name, model_config.img_size)
    pretrained_cfg = backbone.pretrained_cfg
    return BackboneBundle(
        backbone=_freeze(backbone),
        kind="public",
        description=f"ViT-L público timm ({model_config.public_model_name})",
        mean=tuple(pretrained_cfg.get("mean", IMAGENET_DEFAULT_MEAN)),
        std=tuple(pretrained_cfg.get("std", IMAGENET_DEFAULT_STD)),
        num_features=backbone.num_features,
        uses_cls_token=True,
    )


VIT_FUNDUS_REPO = "YukunZhou/RETFound_dinov2_meh"
VIT_FUNDUS_FILE = "RETFound_dinov2_meh.pth"
VIT_PRETRAINED_CHOICES = ("default", "fundus")


def clean_retfound_dinov2_state_dict(checkpoint: dict) -> dict:
    """Strip wrappers/prefixes of a RETFound-DINOv2 checkpoint and map SwiGLU names to timm's MLP."""
    state_dict = checkpoint
    for key in ("teacher", "model", "state_dict"):
        if isinstance(state_dict, dict) and key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break
    clean = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "backbone.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        key = key.replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2.")
        clean[key] = value
    return clean


def load_retfound_dinov2_weights(
    backbone: nn.Module, path: str | Path | None = None, min_fraction: float = 0.9
) -> dict[str, Any]:
    """Load RETFound-DINOv2 (fundus) weights into a timm DINOv2 ViT-L, resampling pos_embed if needed.

    ``path`` defaults to $VIT_FUNDUS_WEIGHTS; otherwise the gated checkpoint is downloaded from
    Hugging Face with $HF_TOKEN. Raises if fewer than ``min_fraction`` of the keys match.
    """
    from timm.layers import resample_abs_pos_embed

    path = path or os.environ.get("VIT_FUNDUS_WEIGHTS")
    if not path:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(
                f"--pretrained fundus needs HF_TOKEN (gated {VIT_FUNDUS_REPO}) or VIT_FUNDUS_WEIGHTS"
            )
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(VIT_FUNDUS_REPO, VIT_FUNDUS_FILE, token=token)
    clean = clean_retfound_dinov2_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    own = backbone.state_dict()
    if "pos_embed" in clean and "pos_embed" in own and clean["pos_embed"].shape != own["pos_embed"].shape:
        prefix_tokens = 0 if getattr(backbone, "no_embed_class", False) else getattr(backbone, "num_prefix_tokens", 1)
        clean["pos_embed"] = resample_abs_pos_embed(
            clean["pos_embed"], new_size=list(backbone.patch_embed.grid_size), num_prefix_tokens=prefix_tokens
        )
    shape_skipped = [k for k, v in clean.items() if k in own and own[k].shape != v.shape]
    for key in shape_skipped:
        clean.pop(key)
    result = backbone.load_state_dict(clean, strict=False)
    loaded = len([k for k in clean if k in own])
    print(
        f"[fundus] {path}: loaded={loaded}/{len(own)} missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)} shape_skipped={shape_skipped}",
        flush=True,
    )
    if loaded < min_fraction * len(own):
        raise RuntimeError(f"RETFound-DINOv2 checkpoint matched only {loaded}/{len(own)} keys")
    return {"loaded": loaded, "total": len(own), "missing": result.missing_keys, "unexpected": result.unexpected_keys}


def resolve_vit_pretrained(model_config: ModelConfig) -> str:
    """Config value wins; "default" falls back to $VIT_PRETRAINED (used by the Colab notebooks)."""
    choice = (getattr(model_config, "vit_pretrained", "default") or "default").strip().lower()
    if choice == "default":
        choice = os.environ.get("VIT_PRETRAINED", "default").strip().lower() or "default"
    if choice not in VIT_PRETRAINED_CHOICES:
        raise ValueError(f"vit_pretrained must be one of {VIT_PRETRAINED_CHOICES}, got {choice!r}")
    return choice


def _build_dinov2_backbone(model_config: ModelConfig) -> BackboneBundle:
    backbone = _create_timm_backbone(
        model_config.dinov2_model_name,
        model_config.img_size,
        dynamic_img_size=True,
    )
    pretrained_cfg = backbone.pretrained_cfg
    description = f"ViT-L DINOv2 timm ({model_config.dinov2_model_name})"
    if resolve_vit_pretrained(model_config) == "fundus":
        load_retfound_dinov2_weights(backbone, getattr(model_config, "vit_fundus_weights", None))
        description += " + RETFound-DINOv2 fundus init"
    return BackboneBundle(
        backbone=_freeze(backbone),
        kind="dinov2",
        description=description,
        mean=tuple(pretrained_cfg.get("mean", IMAGENET_DEFAULT_MEAN)),
        std=tuple(pretrained_cfg.get("std", IMAGENET_DEFAULT_STD)),
        num_features=backbone.num_features,
        uses_cls_token=True,
    )


# The Colab fundus notebooks wrap this function at runtime unless it is already marked; the
# native implementation handles VIT_PRETRAINED=fundus itself, so the runtime patch becomes a no-op.
_build_dinov2_backbone._fundus_patched = True  # type: ignore[attr-defined]


def _create_retfound_architecture() -> nn.Module:
    """ViT-L/16 con pooling medio de parches + LayerNorm, como en el ajuste fino de RETFound."""
    return timm.create_model(
        "vit_large_patch16_224", pretrained=False, num_classes=0, global_pool="avg", fc_norm=True
    )


def load_retfound_state_dict(backbone: nn.Module, checkpoint_path: str | Path) -> None:
    """Carga los pesos de RETFound en el backbone y comprueba que el resultado es coherente."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model_keys = set(backbone.state_dict())
    compatible = {k: v for k, v in state_dict.items() if k in model_keys}
    result = backbone.load_state_dict(compatible, strict=False)
    unexpected_missing = [
        key for key in result.missing_keys if not key.startswith(_RETFOUND_ALLOWED_MISSING_PREFIX)
    ]
    if unexpected_missing:
        raise RuntimeError(f"El checkpoint de RETFound no encaja con el modelo: {unexpected_missing[:5]}")


def _build_retfound_backbone(model_config: ModelConfig, hf_token: str | None) -> BackboneBundle:
    if not hf_token:
        raise RuntimeError("RETFound requiere un token de Hugging Face (Kaggle Secret HF_TOKEN).")
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=model_config.retfound_repo_id,
        filename=model_config.retfound_filename,
        token=hf_token,
    )
    backbone = _create_retfound_architecture()
    load_retfound_state_dict(backbone, checkpoint_path)
    return BackboneBundle(
        backbone=_freeze(backbone),
        kind="retfound",
        description=f"RETFound MAE (fondo de ojo en color) - {model_config.retfound_repo_id}",
        mean=tuple(IMAGENET_DEFAULT_MEAN),
        std=tuple(IMAGENET_DEFAULT_STD),
        num_features=backbone.num_features,
        uses_cls_token=False,
    )


def _build_smoke_backbone() -> BackboneBundle:
    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    return BackboneBundle(
        backbone=_freeze(backbone),
        kind="smoke",
        description="ViT-Tiny aleatorio (solo prueba rápida)",
        mean=tuple(IMAGENET_DEFAULT_MEAN),
        std=tuple(IMAGENET_DEFAULT_STD),
        num_features=backbone.num_features,
        uses_cls_token=True,
    )


def build_backbone(
    model_config: ModelConfig, hf_token: str | None = None, kind: str | None = None
) -> BackboneBundle:
    """Crea el backbone congelado.

    `kind` fuerza un tipo concreto (p. ej. el guardado en el checkpoint). Si es None se usa
    `model_config.backbone_choice`; con "auto" se prueba RETFound si hay token y, si falla
    o no hay token, se usa el ViT-L público.
    """
    kind = kind or model_config.backbone_choice
    if kind == "smoke":
        return _build_smoke_backbone()
    if kind == "public":
        return _build_public_backbone(model_config)
    if kind == "dinov2":
        return _build_dinov2_backbone(model_config)
    if kind == "retfound":
        return _build_retfound_backbone(model_config, hf_token)
    if kind != "auto":
        raise ValueError(f"backbone_choice desconocido: {kind}. Opciones: auto, {BACKBONE_KINDS}")

    if not hf_token:
        print("Sin token de Hugging Face: se usa el ViT-L público (RETFound omitido).")
        return _build_public_backbone(model_config)
    try:
        return _build_retfound_backbone(model_config, hf_token)
    except Exception as error:  # sin acceso al repositorio con acceso restringido, red, etc.
        print(f"RETFound no disponible ({type(error).__name__}); se usa el ViT-L público.")
        return _build_public_backbone(model_config)


class FrozenBackboneClassifier(nn.Module):
    """Backbone congelado (siempre en eval) + cabeza lineal: lo único entrenable."""

    def __init__(self, bundle: BackboneBundle, num_classes: int = len(LABEL_NAMES)) -> None:
        super().__init__()
        self.bundle = bundle
        self.backbone = bundle.backbone
        self.head = nn.Linear(bundle.num_features, num_classes)
        self.register_buffer("mean", torch.tensor(bundle.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(bundle.std).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "FrozenBackboneClassifier":
        super().train(mode)
        self.backbone.eval()  # sin dropout/drop-path en el backbone aunque entrene la cabeza
        return self

    def normalize(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """uint8 (B,3,H,W) -> float normalizado con la media/desviación de los pesos usados."""
        return (images_uint8.float() / 255.0 - self.mean) / self.std

    def extract_features(self, normalized_images: torch.Tensor) -> torch.Tensor:
        """Vector pooled del backbone sin gradiente (con precisión mixta en GPU)."""
        use_amp = normalized_images.device.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return self.backbone(normalized_images).float()

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        """Logits completos con grafo de gradiente hasta la entrada (para explicabilidad)."""
        return self.head(self.backbone(normalized_images))


def extract_features_batched(
    model: FrozenBackboneClassifier,
    images_uint8: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Características de todas las imágenes (CPU, float32). Válido porque el backbone no cambia."""
    model.eval()
    features = []
    for start in range(0, len(images_uint8), batch_size):
        batch = images_uint8[start : start + batch_size].to(device)
        features.append(model.extract_features(model.normalize(batch)).cpu())
    return torch.cat(features)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(parámetros entrenables, parámetros totales)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_head_checkpoint(
    path: Path,
    model: FrozenBackboneClassifier,
    thresholds: list[float],
    screening_threshold: float,
    config: ProjectConfig,
) -> None:
    """Guarda solo la cabeza (unos pocos KB) y lo necesario para reconstruir y evaluar."""
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "label_names": LABEL_NAMES,
            "backbone_kind": model.bundle.kind,
            "backbone_description": model.bundle.description,
            "head_type": getattr(model, "checkpoint_head_type", "linear"),
            "thresholds": thresholds,
            "screening_threshold": screening_threshold,
            "config": asdict(config),
        },
        path,
    )


def load_head_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_classifier(
    checkpoint: dict[str, Any],
    model_config: ModelConfig,
    device: torch.device,
    hf_token: str | None = None,
) -> FrozenBackboneClassifier:
    """Reconstruye backbone + cabeza exactamente como se entrenó (mismo tipo de backbone)."""
    bundle = build_backbone(model_config, hf_token=hf_token, kind=checkpoint["backbone_kind"])
    model = FrozenBackboneClassifier(bundle, num_classes=len(checkpoint["label_names"]))
    model.head.load_state_dict(checkpoint["head_state_dict"])
    return model.to(device).eval()


def predict_image_probabilities(
    model: FrozenBackboneClassifier,
    images_uint8: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Probabilidades sigmoide (N, C) del modelo completo sobre imágenes uint8 sin aumentos."""
    features = extract_features_batched(model, images_uint8, batch_size, device)
    with torch.no_grad():
        return torch.sigmoid(model.head(features.to(device))).cpu().numpy()


def unfreeze_last_vit_blocks(backbone: nn.Module, n_blocks: int) -> None:
    """Congela todo el backbone y descongela los últimos `n_blocks` + el LayerNorm final.

    La cabeza no vive en el backbone: se deja entrenable en el clasificador. Con 0 bloques
    esto equivale al baseline (solo cabeza).
    """
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    blocks = list(backbone.blocks)
    n_blocks = min(max(0, n_blocks), len(blocks))
    if n_blocks == 0:
        return
    for block in blocks[-n_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    final_norm = getattr(backbone, "norm", None)
    if isinstance(final_norm, nn.Module):
        for parameter in final_norm.parameters():
            parameter.requires_grad = True
    extra_norm = getattr(backbone, "fc_norm", None)
    if isinstance(extra_norm, nn.Module):
        for parameter in extra_norm.parameters():
            parameter.requires_grad = True


def _force_eval(module: nn.Module | None) -> None:
    if isinstance(module, nn.Module):
        module.eval()


class FinetuneClassifier(FrozenBackboneClassifier):
    """Mismo ViT + cabeza, pero los últimos bloques y el norm sí se entrenan.

    No altera el baseline: los notebooks 01-04 siguen usando `FrozenBackboneClassifier`.
    """

    def __init__(
        self,
        bundle: BackboneBundle,
        n_unfrozen_blocks: int,
        num_classes: int = len(LABEL_NAMES),
    ) -> None:
        super().__init__(bundle, num_classes)
        self.n_unfrozen_blocks = min(max(0, n_unfrozen_blocks), len(list(self.backbone.blocks)))
        unfreeze_last_vit_blocks(self.backbone, self.n_unfrozen_blocks)

    def train(self, mode: bool = True) -> "FinetuneClassifier":
        nn.Module.train(self, mode)
        if mode:
            n_frozen = len(list(self.backbone.blocks)) - self.n_unfrozen_blocks
            for index, block in enumerate(self.backbone.blocks):
                if index < n_frozen:
                    block.eval()
            _force_eval(getattr(self.backbone, "patch_embed", None))
            _force_eval(getattr(self.backbone, "pos_drop", None))
            _force_eval(getattr(self.backbone, "patch_drop", None))
            _force_eval(getattr(self.backbone, "norm_pre", None))
        return self

    def enable_gradient_checkpointing(self, enabled: bool) -> None:
        setter = getattr(self.backbone, "set_grad_checkpointing", None)
        if callable(setter):
            setter(enabled and self.n_unfrozen_blocks > 0)


def adapt_finetune_state_dict(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Ajusta pesos al modelo actual (p. ej. pos_embed DINOv2 518→392).

    Checkpoints entrenados con timm antiguo guardan ``pos_embed`` nativo (1370);
    ``create_model(..., img_size=392)`` espera 785. Sin esto, ``load_state_dict``
    falla aunque ``strict=False`` (size mismatch).
    """
    model_state = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key not in model_state:
            adapted[key] = value
            continue
        target = model_state[key]
        if value.shape == target.shape:
            adapted[key] = value
            continue
        if key.endswith("pos_embed") and value.ndim == 3 and target.ndim == 3:
            old_tokens = int(value.shape[1] - 1)
            new_tokens = int(target.shape[1] - 1)
            old_grid = int(round(old_tokens**0.5))
            new_grid = int(round(new_tokens**0.5))
            if old_grid * old_grid == old_tokens and new_grid * new_grid == new_tokens:
                from timm.layers import resample_abs_pos_embed

                adapted[key] = resample_abs_pos_embed(
                    value,
                    new_size=(new_grid, new_grid),
                    old_size=(old_grid, old_grid),
                    num_prefix_tokens=1,
                )
                print(
                    f"Adaptado {key}: {tuple(value.shape)} -> {tuple(adapted[key].shape)} "
                    f"(grid {old_grid}->{new_grid})"
                )
                continue
        print(f"Omitiendo {key}: checkpoint {tuple(value.shape)} vs modelo {tuple(target.shape)}")
    return adapted


def save_finetune_checkpoint(
    path: Path,
    model: FinetuneClassifier,
    thresholds: list[float],
    screening_threshold: float,
    config: ProjectConfig,
    best_params: dict[str, Any] | None = None,
    label_names: list[str] | None = None,
) -> None:
    """Guarda el estado completo (bloques descongelados incluidos) y los umbrales de val."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_unfrozen_blocks": model.n_unfrozen_blocks,
            "label_names": list(label_names) if label_names is not None else list(LABEL_NAMES),
            "backbone_kind": model.bundle.kind,
            "backbone_description": model.bundle.description,
            "thresholds": thresholds,
            "screening_threshold": screening_threshold,
            "best_params": best_params or {},
            "config": asdict(config),
        },
        path,
    )


def restore_finetune_classifier(
    checkpoint: dict[str, Any],
    model_config: ModelConfig,
    device: torch.device,
    hf_token: str | None = None,
) -> FinetuneClassifier:
    bundle = build_backbone(model_config, hf_token=hf_token, kind=checkpoint["backbone_kind"])
    head_type = checkpoint.get("head_type", "linear")
    if head_type == "linear":
        classifier_type = FinetuneClassifier
    elif head_type == "residual_attention_pool":
        from .attention_pool import ResidualAttentionFinetuneClassifier

        classifier_type = ResidualAttentionFinetuneClassifier
    else:
        raise ValueError(f"Tipo de cabeza desconocido en checkpoint: {head_type!r}")
    model = classifier_type(
        bundle,
        n_unfrozen_blocks=int(checkpoint["n_unfrozen_blocks"]),
        num_classes=len(checkpoint["label_names"]),
        **({"dropout": float(checkpoint.get("best_params", {}).get("head_dropout", 0.2))}
           if head_type == "residual_attention_pool" else {}),
    )
    state = adapt_finetune_state_dict(model, checkpoint["model_state_dict"])
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()
