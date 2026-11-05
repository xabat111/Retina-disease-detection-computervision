"""Figuras Plotly: probabilidad por clase frente a su umbral, con rango de incertidumbre."""

from __future__ import annotations

import plotly.graph_objects as go

from .settings import CLASS_DISPLAY_NAMES
from .uncertainty import ClassAssessment

COLOR_DETECTED = "#3FB8AF"
COLOR_NOT_DETECTED = "#8B98A5"
COLOR_THRESHOLD = "#F2B84B"


def display_name(label: str) -> str:
    return CLASS_DISPLAY_NAMES.get(label, label)


def build_probability_figure(assessments: list[ClassAssessment]) -> go.Figure:
    """Barras horizontales (pase limpio) + barra de error (rango ~95 % de los pases) + umbral.

    Un gráfico = un mensaje: qué clases superan su propio umbral. El umbral difiere por
    clase (se calibró en validación), por eso se dibuja un marcador por barra y no una línea.
    """
    ordered = sorted(assessments, key=lambda a: a.probability)
    names = [display_name(a.label) + (" ⚠" if a.unreliable_class else "") for a in ordered]
    colors = [COLOR_DETECTED if a.detected else COLOR_NOT_DETECTED for a in ordered]

    figure = go.Figure()
    figure.add_bar(
        y=names,
        x=[a.probability for a in ordered],
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            symmetric=False,
            array=[max(a.interval_high - a.probability, 0.0) for a in ordered],
            arrayminus=[max(a.probability - a.interval_low, 0.0) for a in ordered],
            color="rgba(230,237,243,0.75)",
            thickness=1.5,
        ),
        hovertemplate="%{y}<br>probability %{x:.3f}<extra></extra>",
        name="Probability (untouched image)",
    )
    figure.add_scatter(
        y=names,
        x=[a.threshold for a in ordered],
        mode="markers",
        marker=dict(symbol="diamond", size=11, color=COLOR_THRESHOLD, line=dict(width=1, color="#0B1014")),
        hovertemplate="%{y}<br>threshold %{x:.3f}<extra></extra>",
        name="Decision threshold (VAL Fβ2)",
    )
    detected_count = sum(a.detected for a in ordered)
    figure.update_layout(
        title=dict(text=f"{detected_count} class(es) above their own threshold", font=dict(size=16)),
        xaxis=dict(title="Probability (0–1) · whiskers: ~95 % range under perturbations", range=[0, 1.02],
                   gridcolor="rgba(128,128,128,0.25)"),
        yaxis=dict(title=None, automargin=True),
        height=420,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.22),
        font=dict(family="IBM Plex Sans, system-ui, sans-serif"),
    )
    return figure
