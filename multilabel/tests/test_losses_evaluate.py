import numpy as np
import pytest
import torch
import torch.nn.functional as F

from retina_multilabel.evaluate import (
    METRIC_NAMES,
    build_metrics_table,
    choose_threshold,
    choose_threshold_for_recall,
    confusion_matrices,
    evaluate_disease_screening,
    hamming_loss,
    labels_that_worsened,
    macro_auroc,
    macro_average_precision,
    multilabel_metric_matrix,
    subset_accuracy,
)
from retina_multilabel.labels import HEALTHY_INDEX, LABEL_NAMES
from retina_multilabel.losses import (
    MultilabelFocalLoss,
    adapt_pos_weights,
    build_focal_loss,
    build_loss,
    class_balanced_pos_weights,
    compute_pos_weights,
    initial_pos_weights_for_mode,
    raw_neg_pos_ratio,
    soft_cap_weights,
)

N_CLASSES = len(LABEL_NAMES)


def _synthetic_predictions(n_rows: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    targets = (rng.random((n_rows, N_CLASSES)) < 0.25).astype(np.float32)
    targets[:, HEALTHY_INDEX] = 1 - targets[:, :HEALTHY_INDEX].max(axis=1)
    probabilities = np.clip(0.3 * targets + 0.7 * rng.random(targets.shape), 0, 1)
    return targets, probabilities


def test_focal_loss_shapes_gradients_and_gamma_zero_matches_scaled_bce():
    logits = torch.randn(6, N_CLASSES, requires_grad=True)
    targets = torch.randint(0, 2, (6, N_CLASSES)).float()
    loss = build_focal_loss(gamma=2.0, alpha_pos=0.25)(logits, targets)
    assert loss.shape == ()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()

    logits_bce = torch.randn(6, N_CLASSES)
    targets_bce = torch.randint(0, 2, (6, N_CLASSES)).float()
    bce = F.binary_cross_entropy_with_logits(logits_bce, targets_bce, reduction="mean")
    focal_gamma_zero = MultilabelFocalLoss(gamma=0.0, alpha_pos=0.5)(logits_bce, targets_bce)
    # gamma=0 → modulador 1; alpha_pos=0.5 → α_t constante 0.5, así que 0.5 × BCE.
    assert torch.allclose(focal_gamma_zero, 0.5 * bce, rtol=1e-5, atol=1e-6)


def test_pos_weights_are_clipped_and_computed_from_given_targets():
    targets = np.zeros((100, 3))
    targets[:50, 0] = 1  # frecuente -> peso 1
    targets[:2, 1] = 1  # raro -> 49, recortado a 20
    weights = compute_pos_weights(targets, max_pos_weight=20.0)
    assert weights.tolist() == [1.0, 20.0, 20.0]  # sin positivos: se recorta al máximo
    assert build_loss(weights).pos_weight.shape == (3,)


def test_uncapped_and_soft_cap_leave_rare_class_above_hard_twenty():
    targets = np.zeros((100, 2))
    targets[:2, 0] = 1  # raw = 98/2 = 49
    targets[:50, 1] = 1
    raw = raw_neg_pos_ratio(targets)
    assert raw[0] > 20
    uncapped = compute_pos_weights(targets, 20.0, mode="uncapped", max_pos_weight_uncapped=100.0)
    soft = soft_cap_weights(raw, 20.0)
    assert float(uncapped[0]) == pytest.approx(49.0)
    assert 15.0 < soft[0] < 20.0  # cerca del techo, sin hard-clip exacto a 20
    iterative = initial_pos_weights_for_mode(
        targets, 20.0, "iterative", iterative_init="soft_cap", iterative_init_cap=40.0
    )
    assert float(iterative[0]) > 20.0


def test_adapt_pos_weights_upweights_low_ap_class_with_damping():
    current = np.array([5.0, 5.0, 5.0])
    # class 2 lags badly
    val_ap = np.array([0.80, 0.75, 0.20])
    val_rec = np.array([0.80, 0.70, 0.10])
    adapted = adapt_pos_weights(
        current, val_ap, val_recall=val_rec, base=current, eta=0.2, momentum=0.5, w_max=50.0
    )
    assert adapted[2] > current[2]
    assert adapted[0] <= current[0] + 0.01
    assert adapted.max() <= 50.0


def test_class_balanced_weights_finite_and_above_one():
    targets = np.zeros((200, 3))
    targets[:100, 0] = 1
    targets[:10, 1] = 1
    targets[:3, 2] = 1
    w = class_balanced_pos_weights(targets)
    assert w.shape == (3,)
    assert np.all(w >= 1.0)
    assert np.all(np.isfinite(w))
    assert w[2] > w[0]


def test_choose_threshold_uses_scores_and_handles_degenerate_classes():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert 0.3 < choose_threshold(y_true, y_score) <= 0.7
    assert choose_threshold(np.zeros(6), y_score) == 0.5


def test_metric_matrix_shape_and_values():
    targets, probabilities = _synthetic_predictions()
    matrix = multilabel_metric_matrix(targets, probabilities, [0.5] * N_CLASSES)
    assert matrix.shape == (N_CLASSES + 2, len(METRIC_NAMES))
    assert np.nanmin(matrix) >= 0 and np.nanmax(matrix) <= 1
    assert 0.6 < macro_auroc(targets, probabilities) < 1.0


def test_metrics_table_and_screening_have_expected_structure():
    targets, probabilities = _synthetic_predictions()
    table = build_metrics_table(targets, probabilities, [0.5] * N_CLASSES, 20, 0.95, 0, 20)
    assert list(table.index) == LABEL_NAMES + ["macro", "micro"]
    assert (table["auroc_ic_low"] <= table["auroc_ic_high"]).all()

    screening = evaluate_disease_screening(targets, probabilities, 0.5, 20, 0.95, 0)
    assert list(screening.index) == METRIC_NAMES
    assert screening.loc["auroc", "valor"] > 0.6


def test_confusion_matrices_sum_to_number_of_rows():
    targets, probabilities = _synthetic_predictions()
    matrices = confusion_matrices(targets, probabilities, [0.5] * N_CLASSES)
    assert matrices.shape == (N_CLASSES, 2, 2)
    assert (matrices.sum(axis=(1, 2)) == len(targets)).all()


def test_threshold_for_target_recall_keeps_at_least_that_recall():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95])
    threshold = choose_threshold_for_recall(y_true, y_score, 0.8)
    recall = ((y_score >= threshold) & (y_true == 1)).sum() / y_true.sum()
    assert recall >= 0.8 and threshold == 0.5


def test_macro_ap_and_secondary_scores():
    targets, probabilities = _synthetic_predictions()
    assert 0.4 < macro_average_precision(targets, probabilities) < 1.0
    predictions = (probabilities >= 0.5).astype(int)
    zeros = np.zeros_like(predictions)
    assert 0.0 <= subset_accuracy(targets, predictions) <= 1.0
    assert hamming_loss(targets, zeros) == pytest.approx(float(targets.mean()), rel=1e-5)
    assert subset_accuracy(targets, zeros) == float((targets.sum(axis=1) == 0).mean())


def test_labels_that_worsened_lists_only_declines():
    targets, probabilities = _synthetic_predictions()
    baseline = build_metrics_table(targets, probabilities, [0.5] * N_CLASSES, 10, 0.95, 0, 20)
    worse_probabilities = np.clip(probabilities - 0.3, 0, 1)
    worse_probabilities[:, 0] = np.clip(probabilities[:, 0] + 0.2, 0, 1)
    tuned = build_metrics_table(targets, worse_probabilities, [0.5] * N_CLASSES, 10, 0.95, 0, 20)
    declined = labels_that_worsened(baseline, tuned, "average_precision")
    assert list(declined.columns)[:1] == ["etiqueta"]
    assert (declined["delta"] < 0).all()
