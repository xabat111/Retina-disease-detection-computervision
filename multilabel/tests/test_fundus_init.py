"""RETFound-DINOv2 (fundus) init: checkpoint cleaning, pos_embed resampling and config/env resolution."""

import timm
import torch

from retina_multilabel.config import ModelConfig
from retina_multilabel.model import (
    _build_dinov2_backbone,
    clean_retfound_dinov2_state_dict,
    load_retfound_dinov2_weights,
    resolve_vit_pretrained,
)


def _tiny(img_size):
    return timm.create_model(
        "vit_small_patch14_dinov2", pretrained=False, num_classes=0, img_size=img_size, dynamic_img_size=True
    )


def test_clean_state_dict_strips_wrappers_and_maps_swiglu():
    ckpt = {"teacher": {"backbone.blocks.0.mlp.w12.weight": 1, "module.cls_token": 2}}
    assert clean_retfound_dinov2_state_dict(ckpt) == {"blocks.0.mlp.fc1.weight": 1, "cls_token": 2}


def test_load_resamples_pos_embed_and_loads(tmp_path):
    source = _tiny(224)
    torch.nn.init.constant_(source.cls_token, 0.5)
    path = tmp_path / "fake_retfound.pth"
    torch.save({"teacher": {f"backbone.{k}": v for k, v in source.state_dict().items()}}, path)
    target = _tiny(126)
    info = load_retfound_dinov2_weights(target, path)
    assert info["loaded"] == info["total"]
    assert torch.allclose(target.cls_token, torch.full_like(target.cls_token, 0.5))


def test_resolve_vit_pretrained(monkeypatch):
    monkeypatch.delenv("VIT_PRETRAINED", raising=False)
    assert resolve_vit_pretrained(ModelConfig()) == "default"
    monkeypatch.setenv("VIT_PRETRAINED", "fundus")
    assert resolve_vit_pretrained(ModelConfig()) == "fundus"
    assert resolve_vit_pretrained(ModelConfig(vit_pretrained="fundus")) == "fundus"


def test_native_builder_is_marked_so_notebook_patch_is_noop():
    assert getattr(_build_dinov2_backbone, "_fundus_patched", False)
