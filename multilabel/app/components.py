"""Bloques de interfaz: KPIs, tablas, galería Grad-CAM y explicación de la incertidumbre."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from . import settings
from .analysis import AnalysisResult, compare_to_reference
from .gradcam import render_overlay
from .model_loader import DeployedModel
from .plots import build_probability_figure, display_name
from .styles import badge_html
from .uncertainty import (
    UNCERTAINTY_HIGH,
    UNCERTAINTY_LOW,
    UNCERTAINTY_MEDIUM,
    ClassAssessment,
    strip_uncertainty_assessments,
)

HEALTHY_LABEL = settings.HEALTHY_LABEL
GRADCAM_COLUMNS = 3


# --------------------------------------------------------------------------- helpers de formato


def notice(message_html: str, kind: str = "info") -> None:
    css_class = {"info": "notice", "warning": "notice notice-warning", "error": "notice notice-error"}[kind]
    st.markdown(f'<div class="{css_class}">{message_html}</div>', unsafe_allow_html=True)


def _kpi_card(eyebrow: str, value_html: str, note: str) -> None:
    st.markdown(
        f'<div class="kpi"><div class="eyebrow">{eyebrow}</div>'
        f'<div class="kpi-value">{value_html}</div>'
        f'<div class="kpi-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-state"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- KPIs


def _uncertainty_chip_html(assessment: ClassAssessment) -> str:
    """Chip de incertidumbre; los no detectados dudosos van atenuados y etiquetados."""
    name = html.escape(display_name(assessment.label))
    if assessment.detected:
        return (
            f'<span class="certainty-chip">{badge_html(assessment.uncertainty)}'
            f'<span class="certainty-chip-label">{name}</span></span>'
        )
    return (
        f'<span class="certainty-chip certainty-chip-muted">'
        f'{badge_html(assessment.uncertainty)}'
        f'<span class="certainty-chip-label">{name}</span>'
        f'<span class="certainty-chip-meta">not detected \u00b7 {html.escape(assessment.uncertainty)}</span>'
        f"</span>"
    )


def _render_uncertainty_chips(assessments: list[ClassAssessment]) -> None:
    """Detectadas + no detectadas dudosas (Medium/High); sin números TTA aquí."""
    strip = strip_uncertainty_assessments(assessments)
    if not strip:
        st.markdown(
            '<div class="certainty-strip"><div class="eyebrow">Class uncertainty</div>'
            '<div class="kpi-note">No detections; all non-detections are Low uncertainty</div></div>',
            unsafe_allow_html=True,
        )
        return
    chips = "".join(_uncertainty_chip_html(assessment) for assessment in strip)
    st.markdown(
        f'<div class="certainty-strip"><div class="eyebrow">Class uncertainty</div>'
        f'<div class="certainty-chips">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_kpis(result: AnalysisResult, deployed: DeployedModel, num_passes: int) -> None:
    detected = [a for a in result.assessments if a.detected]
    strongest = max(detected, key=lambda a: a.probability, default=None)
    _render_uncertainty_chips(result.assessments)
    columns = st.columns(3)
    with columns[0]:
        names = ", ".join(display_name(a.label) for a in detected) or "None"
        _kpi_card("Detected classes", f"{len(detected)}", names)
    with columns[1]:
        if strongest is None:
            _kpi_card("Strongest detection", "\u2014", "No class reached its threshold")
        else:
            _kpi_card(
                "Strongest detection",
                f"{strongest.probability:.2f}",
                f"{display_name(strongest.label)} (threshold {strongest.threshold:.2f})",
            )
    with columns[2]:
        _kpi_card(
            "Analysis",
            f"{result.seconds:.0f} s",
            f"1 clean pass (prediction) + {num_passes} TTA passes (uncertainty only) \u00b7 {deployed.device.type.upper()}",
        )
    st.caption(
        "Per-class uncertainty (TTA): **Low** (green) = model is sure; **High** (red) = model doubts. "
        "Numbers in the detail section. "
        "The prediction (probabilities, thresholds, detections, Grad-CAM) uses the **original image only** "
        "(one clean forward pass). TTA augmented passes estimate stability; they do **not** change which "
        "classes are detected, and they are not a calibrated probability of being right."
    )


def render_reference_match(result: AnalysisResult, reference_labels: list[str]) -> None:
    """Exact match / extra / missing frente a las etiquetas ODIR-7 reference de una muestra de test."""
    exact, extra, missing = compare_to_reference(result.detected_labels, reference_labels)
    reference_text = ", ".join(display_name(label) for label in reference_labels) or "none"
    if exact:
        notice(
            "<b>Exact match with ODIR-7 reference.</b> The predicted set equals the reference labels "
            f"({html.escape(reference_text)}).",
            "info",
        )
    else:
        extra_text = ", ".join(display_name(label) for label in extra) or "none"
        missing_text = ", ".join(display_name(label) for label in missing) or "none"
        notice(
            "<b>Not an exact match with ODIR-7 reference.</b> "
            f"Reference: {html.escape(reference_text)}. "
            f"Extra (predicted, not in ODIR-7 reference): {html.escape(extra_text)}. "
            f"Missing (in ODIR-7 reference, not predicted): {html.escape(missing_text)}.",
            "warning",
        )
    st.caption(
        "Exact match = the full 7-label vector is identical (subset accuracy on this image): "
        "no extra class and no missing class."
    )


def render_warnings(result: AnalysisResult, deployed: DeployedModel) -> None:
    if result.healthy_conflict_resolved:
        kept = (
            "Healthy"
            if result.healthy_side_kept == HEALTHY_LABEL
            else "the pathology class(es)"
        )
        notice(
            "<b>Healthy/disease conflict resolved:</b> both Healthy and at least one "
            "pathology exceeded their thresholds. Training labels make them mutually "
            "exclusive, so only the higher-probability side was kept "
            f"({html.escape(kept)}). Detection counts and overall uncertainty use this "
            "resolved verdict; ranking scores are unchanged.",
            "info",
        )
    for assessment in result.assessments:
        if not (assessment.detected and assessment.unreliable_class):
            continue
        positives = int(deployed.test_metrics.loc[assessment.label, "n_positivos"])
        notice(
            f"<b>{html.escape(display_name(assessment.label))} is unreliable:</b> only {positives} "
            "positive images exist in the test set, so its threshold and metrics are not trustworthy. "
            "Ignore this detection unless confirmed otherwise.",
            "error",
        )


# --------------------------------------------------------------------------- deteccion


def render_probability_chart(result: AnalysisResult) -> None:
    st.plotly_chart(build_probability_figure(result.assessments), width="stretch")


def _sorted_assessments(result: AnalysisResult):
    """Detectadas primero y, dentro de cada grupo, de mayor a menor probabilidad."""
    return sorted(result.assessments, key=lambda a: (not a.detected, -a.probability))


def build_detail_table(result: AnalysisResult, deployed: DeployedModel) -> pd.DataFrame:
    rows = []
    for assessment in _sorted_assessments(result):
        metrics = deployed.test_metrics.loc[assessment.label]
        rows.append(
            {
                "Class": display_name(assessment.label) + (" \u26a0" if assessment.unreliable_class else ""),
                "Detected": "Yes" if assessment.detected else "No",
                "Probability": assessment.probability,
                "Threshold": assessment.threshold,
                "Test precision": float(metrics["precision"]),
                "Test recall": float(metrics["recall"]),
                "Test AUROC": float(metrics["auroc"]),
                "Test positives": int(metrics["n_positivos"]),
            }
        )
    return pd.DataFrame(rows)


def render_detail_table(result: AnalysisResult, deployed: DeployedModel) -> None:
    st.dataframe(
        build_detail_table(result, deployed),
        hide_index=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Threshold": st.column_config.NumberColumn(format="%.3f"),
            "Test precision": st.column_config.NumberColumn(format="%.2f"),
            "Test recall": st.column_config.NumberColumn(format="%.2f"),
            "Test AUROC": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "Test columns come from the held-out test split at these thresholds (precision/recall are "
        "population statistics, not a per-image guarantee). \u26a0 = too few positives to trust."
    )


# --------------------------------------------------------------------------- Grad-CAM


def default_gradcam_classes(result: AnalysisResult) -> list[str]:
    """Clases detectadas; si no hay ninguna, la mas cercana a su umbral (probabilidad/umbral)."""
    detected = [a.label for a in result.assessments if a.detected]
    if detected:
        return detected
    closest = max(result.assessments, key=lambda a: a.probability / a.threshold)
    return [closest.label]


def render_gradcam_gallery(
    result: AnalysisResult,
    deployed: DeployedModel,
    selected_labels: list[str],
    overlay_alpha: float,
    mask_background: bool,
) -> None:
    if not selected_labels:
        empty_state("No class selected", "Pick at least one class above to see its Grad-CAM heatmap.")
        return
    for row_start in range(0, len(selected_labels), GRADCAM_COLUMNS):
        columns = st.columns(GRADCAM_COLUMNS)
        for column, label in zip(columns, selected_labels[row_start : row_start + GRADCAM_COLUMNS]):
            class_index = deployed.label_names.index(label)
            assessment = result.assessments[class_index]
            overlay = render_overlay(
                result.image_uint8, result.gradcam_maps[class_index], overlay_alpha, mask_background
            )
            with column:
                st.image(overlay, width="stretch")
                verdict = "detected" if assessment.detected else "not detected"
                st.markdown(
                    f"**{html.escape(display_name(label))}** \u00b7 {verdict}<br>"
                    f'<span class="caption-muted">p = {assessment.probability:.2f} \u00b7 threshold '
                    f"{assessment.threshold:.2f} \u00b7 uncertainty {assessment.uncertainty}</span>",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------- incertidumbre


def build_uncertainty_table(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for assessment in _sorted_assessments(result):
        rows.append(
            {
                "Class": display_name(assessment.label),
                "Verdict": "Detected" if assessment.detected else "Not detected",
                "Mean \u00b1 std": f"{assessment.mean:.3f} \u00b1 {assessment.std:.3f}",
                "~95 % range": f"{assessment.interval_low:.2f} \u2013 {assessment.interval_high:.2f}",
                "Verdict agreement": assessment.agreement,
                "Margin (\u03c3)": assessment.margin_in_sigmas,
                "Entropy (bits)": assessment.entropy_bits,
                "Disagreement MI (bits)": assessment.mutual_information_bits,
                "Uncertainty": assessment.uncertainty,
                "Why": assessment.uncertainty_reason,
            }
        )
    return pd.DataFrame(rows)


def render_uncertainty_section(result: AnalysisResult, num_passes: int) -> None:
    highlighted = strip_uncertainty_assessments(result.assessments)
    for assessment in highlighted:
        verdict = "detected" if assessment.detected else "not detected"
        st.markdown(
            f"{badge_html(assessment.uncertainty)} &nbsp; **{html.escape(display_name(assessment.label))}** "
            f"({verdict}) \u2014 "
            f"p = {assessment.probability:.2f}, {assessment.mean:.2f} \u00b1 {assessment.std:.2f} across "
            f"{num_passes + 1} passes; the verdict held in {assessment.agreement:.0%} of them. "
            f"{html.escape(assessment.uncertainty_reason)}",
            unsafe_allow_html=True,
        )
    if not highlighted:
        st.markdown(
            "No class reached its threshold and every non-detection is Low uncertainty; "
            "the table below shows the full TTA detail."
        )
    elif not any(a.detected for a in highlighted):
        st.caption(
            "No class reached its threshold; the summaries above are the dubious non-detections."
        )

    st.dataframe(
        build_uncertainty_table(result),
        hide_index=True,
        column_config={
            "Verdict agreement": st.column_config.ProgressColumn(
                "Verdict agreement", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "Margin (\u03c3)": st.column_config.NumberColumn(format="%.1f"),
            "Entropy (bits)": st.column_config.NumberColumn(format="%.2f"),
            "Disagreement MI (bits)": st.column_config.NumberColumn(format="%.3f"),
            "Why": st.column_config.TextColumn(width="large"),
        },
    )
    render_uncertainty_explainer(num_passes)


def render_uncertainty_explainer(num_passes: int) -> None:
    with st.expander("How to read this (method, criteria and limits)"):
        st.markdown(
            f"""
**The verdict uses the original image only.** Detected classes, probabilities vs thresholds
and Grad-CAM come from **one clean forward pass**. The {num_passes} TTA passes below are
**uncertainty only**: they never change which classes are detected.

**Method: test-time augmentation (TTA), not MC Dropout.** The model has no dropout layers
(3 DINOv2 ViT-L residual-attention members, no dropout at inference), so MC Dropout would inject
noise the network never saw and would not mean anything. Instead the image is perturbed
{num_passes} times with the *same* augmentations used in training (horizontal flip, rotation
\u2264 10\u00b0, brightness/contrast/saturation \u00b10.2) and compared with the untouched image.

**What each column means**
- **Mean \u00b1 std / ~95 % range:** spread of the probability across passes (mean \u00b1 1.96 std, clipped to 0\u20131).
- **Verdict agreement:** share of passes that give the same detected/not-detected decision as the untouched image.
- **Margin (\u03c3):** distance from the probability to the class threshold, in standard deviations of the passes.
- **Entropy:** binary entropy of the mean probability (0 = decided, 1 = coin flip).
- **Disagreement MI:** entropy of the mean minus mean entropy of each pass; large only when passes disagree.

**Uncertainty label** — score = verdict agreement% \u2212 std\u00d7100 (high score = stable)
- {badge_html(UNCERTAINTY_LOW)} Low uncertainty when score \u2265 {settings.HIGH_MIN_CERTAINTY_SCORE:.0f} (model stable)
- {badge_html(UNCERTAINTY_MEDIUM)} Medium when {settings.MEDIUM_MIN_CERTAINTY_SCORE:.0f} \u2264 score &lt; {settings.HIGH_MIN_CERTAINTY_SCORE:.0f}
- {badge_html(UNCERTAINTY_HIGH)} High uncertainty when score &lt; {settings.MEDIUM_MIN_CERTAINTY_SCORE:.0f}, and always for unreliable classes (too few test positives).
- Overall uncertainty = the highest (worst) label among detected classes.
- The top strip and summaries above the table list **every detection** plus **non-detections whose uncertainty is Medium or High**. Stable Low non-detections are omitted there; the table still lists all 7 classes.
- No Normal/disease exclusivity rule is applied: each class is thresholded independently (same as the ensemble evaluation).

**Honest limits.** This is stability under perturbations, **not a calibrated probability** of being right.
A stable prediction can still be wrong (systematic bias, out-of-distribution images, label noise), and
the cut-offs above are conventions of this app, not statistically derived values. The model was not
recalibrated on a held-out set, so read probabilities as scores, not as risk percentages.
""",
            unsafe_allow_html=True,
        )
