import numpy as np
import pytest
import timm
import torch

from retina_multilabel.config import ExplainConfig, build_smoke_test_config
from retina_multilabel.explain import (
    ViTExplainer,
    is_integrated_gradients_available,
    overlay_heatmap,
    select_xai_cases,
)
from retina_multilabel.labels import LABEL_NAMES
from retina_multilabel.losses import build_loss
from retina_multilabel.model import (
    FrozenBackboneClassifier,
    build_backbone,
    count_parameters,
    extract_features_batched,
    load_retfound_state_dict,
)

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_classifier() -> FrozenBackboneClassifier:
    bundle = build_backbone(build_smoke_test_config().model)
    return FrozenBackboneClassifier(bundle).eval()


def test_only_the_head_is_trainable(tiny_classifier):
    trainable, total = count_parameters(tiny_classifier)
    head_parameters = sum(p.numel() for p in tiny_classifier.head.parameters())
    assert trainable == head_parameters < total
    assert tiny_classifier.head.out_features == len(LABEL_NAMES)


def test_head_forward_backward_updates_only_head(tiny_classifier):
    images = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
    features = extract_features_batched(tiny_classifier, images, 2, DEVICE)
    assert features.shape == (4, tiny_classifier.bundle.num_features)

    tiny_classifier.train()
    assert not tiny_classifier.backbone.training  # el backbone se queda en eval
    loss = build_loss(torch.ones(len(LABEL_NAMES)))(
        tiny_classifier.head(features), torch.randint(0, 2, (4, len(LABEL_NAMES))).float()
    )
    loss.backward()
    assert tiny_classifier.head.weight.grad is not None
    assert all(p.grad is None for p in tiny_classifier.backbone.parameters())
    tiny_classifier.eval()


def test_explanations_have_patch_grid_shape(tiny_classifier):
    explainer = ViTExplainer(tiny_classifier, ExplainConfig(rollout_layers=2, integrated_gradients_steps=2))
    image = tiny_classifier.normalize(torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8))

    rollout = explainer.attention_rollout(image)
    cam = explainer.grad_cam(image, class_index=3)
    assert rollout.shape == cam.shape == (14, 14)
    assert 0 <= float(rollout.min()) and float(rollout.max()) <= 1.0001
    assert float(cam.max()) <= 1.0001
    assert float(rollout.max()) > 0.99 and float(cam.max()) > 0.99  # mapas no degenerados (no todo ceros)

    fundus = np.zeros((224, 224, 3), dtype=np.uint8)
    fundus[60:160, 60:160] = 200  # campo de visión sintético sobre fondo negro
    overlay = overlay_heatmap(fundus, cam)
    assert overlay.shape == (224, 224, 3)
    assert np.allclose(overlay[:20, :20], 0)  # el fondo queda sin mapa de color
    if is_integrated_gradients_available():
        assert explainer.integrated_gradients(image, class_index=3).shape == (224, 224)


def test_select_xai_cases_returns_confident_hits_and_worst_misses():
    targets = np.zeros((6, 2))
    targets[:5, 0] = 1
    probabilities = np.zeros((6, 2))
    probabilities[:5, 0] = [0.9, 0.8, 0.2, 0.1, 0.7]
    cases = select_xai_cases(targets, probabilities, [0.5, 0.5], 0, cases_per_outcome=2)
    assert cases == [(0, "TP"), (1, "TP"), (3, "FN"), (2, "FN")]


def test_retfound_loader_accepts_checkpoint_without_fc_norm(tmp_path):
    architecture = dict(
        num_classes=0, global_pool="avg", fc_norm=True, pretrained=False
    )
    source = timm.create_model("vit_tiny_patch16_224", **architecture)
    state = {k: v for k, v in source.state_dict().items() if not k.startswith("fc_norm.")}
    state["head.weight"] = torch.zeros(3, 192)  # cabeza ajustada que debe ignorarse
    torch.save({"model": state}, tmp_path / "retfound.pth")

    target = timm.create_model("vit_tiny_patch16_224", **architecture)
    load_retfound_state_dict(target, tmp_path / "retfound.pth")
    assert torch.equal(target.blocks[0].attn.qkv.weight, source.blocks[0].attn.qkv.weight)


def test_retfound_loader_rejects_incompatible_checkpoint(tmp_path):
    torch.save({"model": {"unrelated.weight": torch.zeros(1)}}, tmp_path / "bad.pth")
    target = timm.create_model("vit_tiny_patch16_224", num_classes=0, pretrained=False)
    with pytest.raises(RuntimeError):
        load_retfound_state_dict(target, tmp_path / "bad.pth")
