"""Incertidumbre por aumentos en test (TTA) con las perturbaciones de entrenamiento.

Por qué TTA y no MC Dropout
---------------------------
MC Dropout exige que la red se haya entrenado con dropout activo: solo entonces las
máscaras aleatorias en inferencia aproximan una posterior bayesiana (Gal & Ghahramani, 2016).
Este modelo (DINOv2 ViT-L de timm con `drop_rate = proj_drop = drop_path = 0` y cabeza
`nn.Linear` sin dropout, ajustado con BCE) NO tiene ninguna capa de dropout. Activar
dropout artificial en inferencia injertaría ruido que el modelo nunca vio y daría una
incertidumbre sin interpretación.

En su lugar se mide la **sensibilidad de la predicción a perturbaciones que el modelo SÍ
vio al entrenar** (volteo horizontal, rotación <= 10 grados, brillo/contraste/saturación
+-0.2: `data.build_train_augmentation`). Es un TTA estocástico: N pases aumentados + 1 pase
limpio. Qué significa la medida (y qué no):
  * Mide estabilidad ante nuisances de captura/orientación (parte "aleatoria" de entrada).
  * NO es una posterior calibrada ni cubre la incertidumbre epistémica (otros pesos
    posibles, imágenes fuera de distribución). Un modelo puede ser estable y equivocarse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import settings

# Niveles de *incertidumbre* (no certeza): Low = modelo estable, High = duda.
UNCERTAINTY_HIGH = "High"
UNCERTAINTY_MEDIUM = "Medium"
UNCERTAINTY_LOW = "Low"
# De peor a mejor (más incertidumbre → menos): High es lo peor.
UNCERTAINTY_ORDER = (UNCERTAINTY_HIGH, UNCERTAINTY_MEDIUM, UNCERTAINTY_LOW)
_EPSILON = 1e-6


# --------------------------------------------------------------------------- pases estocásticos


def _augment(image, generator):
    """Perturbaciones tipo entrenamiento: volteo horizontal, rotacion <= 10 grados, brillo/contraste/saturacion +-0.2."""
    from PIL import Image, ImageEnhance
    r = lambda lo, hi: float(lo + (hi - lo) * generator.random())
    out = image
    if generator.random() < 0.5:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    out = out.rotate(r(-10, 10), resample=Image.Resampling.BILINEAR)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = enhancer(out).enhance(r(0.8, 1.2))
    return out


def run_tta_passes(
    ensemble,
    image,
    num_augmented_passes: int,
    seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Probabilidades del ensemble (num_augmented_passes + 1, C); fila 0 = imagen original (la prediccion)."""
    generator = np.random.default_rng(seed)
    views = [image] + [_augment(image, generator) for _ in range(num_augmented_passes)]
    probabilities = []
    for index, view in enumerate(views):
        probabilities.append(np.asarray(ensemble.predict(view)["prob"], dtype=np.float64))
        if progress_callback is not None:
            progress_callback(index + 1, len(views))
    return np.stack(probabilities)


# --------------------------------------------------------------------------- métricas por clase


def binary_entropy_bits(probabilities: np.ndarray) -> np.ndarray:
    """Entropía de Bernoulli en bits: 0 = seguro, 1 = moneda al aire."""
    clipped = np.clip(probabilities, _EPSILON, 1 - _EPSILON)
    return -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))


@dataclass(frozen=True)
class ClassAssessment:
    """Veredicto y fiabilidad de una clase para una imagen."""

    label: str
    probability: float  # pase limpio: la que se compara con el umbral
    threshold: float
    detected: bool
    mean: float  # media de todos los pases
    std: float
    interval_low: float
    interval_high: float
    agreement: float  # fracción de pases con el mismo veredicto que el pase limpio
    margin_in_sigmas: float  # distancia al umbral / std de los pases
    entropy_bits: float  # incertidumbre total de la media
    mutual_information_bits: float  # parte debida al desacuerdo entre pases (tipo BALD)
    uncertainty: str
    uncertainty_reason: str
    unreliable_class: bool


def certainty_score(agreement: float, std: float) -> float:
    """Combina acuerdo de veredicto y dispersión de probabilidad en un solo score.

    `agreement` es fracción (0–1); `std` es la desviación típica de la probabilidad
    entre pases TTA (fracción, p. ej. 0.14 = 14 puntos). Convención de la app:
    penalizar el acuerdo por la amplitud del std, no por el margen en sigmas.
    Score alto = predicción estable; la etiqueta de *incertidumbre* lo invierte en UI.
    """
    return agreement * 100.0 - std * 100.0


def grade_uncertainty(
    agreement: float, std: float, unreliable_class: bool
) -> tuple[str, str]:
    """Etiqueta Low/Medium/High de *incertidumbre* a partir del score acuerdo−std.

    Mismo score que antes (estabilidad); los nombres se leen como incertidumbre:
    score alto → Low (verde), score bajo → High (rojo). Unreliable → High.
    """
    if unreliable_class:
        return UNCERTAINTY_HIGH, "Class has too few test positives for its metrics to be trusted."
    score = certainty_score(agreement, std)
    if score >= settings.HIGH_MIN_CERTAINTY_SCORE:
        return (
            UNCERTAINTY_LOW,
            f"Score {score:.0f} (agreement {agreement:.0%} − std×100 = {std * 100:.0f}).",
        )
    if score >= settings.MEDIUM_MIN_CERTAINTY_SCORE:
        return (
            UNCERTAINTY_MEDIUM,
            f"Score {score:.0f} (agreement {agreement:.0%} − std×100 = {std * 100:.0f}).",
        )
    return (
        UNCERTAINTY_HIGH,
        f"Score {score:.0f} (agreement {agreement:.0%} − std×100 = {std * 100:.0f}).",
    )


def assess_predictions(
    pass_probabilities: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
    unreliable_labels: set[str],
) -> list[ClassAssessment]:
    """Resume los pases (T+1, C) en un `ClassAssessment` por clase.

    El veredicto `detected` aplica la misma exclusividad sano/enfermedad que
    `evaluate.assign_predictions`. El acuerdo TTA se mide contra ese veredicto.
    """
    clean = pass_probabilities[0]
    mean = pass_probabilities.mean(axis=0)
    std = pass_probabilities.std(axis=0, ddof=1)
    # Umbral por etiqueta y, si sano y una enfermedad disparan a la vez, un solo lado
    # (la misma regla que en evaluate.assign_predictions).
    # odir7: sin regla de exclusividad Normal/enfermedad (igual que la evaluacion del ensemble)
    resolved = (pass_probabilities >= thresholds).astype(int)
    detected = resolved[0].astype(bool)
    agreement = (resolved == detected).mean(axis=0)
    margin_in_sigmas = np.abs(clean - thresholds) / np.maximum(std, _EPSILON)

    total_entropy = binary_entropy_bits(mean)
    expected_entropy = binary_entropy_bits(pass_probabilities).mean(axis=0)
    mutual_information = np.maximum(total_entropy - expected_entropy, 0.0)

    assessments = []
    for index, label in enumerate(label_names):
        is_unreliable = label in unreliable_labels
        uncertainty, reason = grade_uncertainty(
            float(agreement[index]), float(std[index]), is_unreliable
        )
        half_width = settings.INTERVAL_Z_SCORE * std[index]
        assessments.append(
            ClassAssessment(
                label=label,
                probability=float(clean[index]),
                threshold=float(thresholds[index]),
                detected=bool(detected[index]),
                mean=float(mean[index]),
                std=float(std[index]),
                interval_low=float(max(mean[index] - half_width, 0.0)),
                interval_high=float(min(mean[index] + half_width, 1.0)),
                agreement=float(agreement[index]),
                margin_in_sigmas=float(margin_in_sigmas[index]),
                entropy_bits=float(total_entropy[index]),
                mutual_information_bits=float(mutual_information[index]),
                uncertainty=uncertainty,
                uncertainty_reason=reason,
                unreliable_class=is_unreliable,
            )
        )
    return assessments


def overall_uncertainty(assessments: list[ClassAssessment]) -> tuple[str, str]:
    """Incertidumbre global conservadora: la peor (más alta) entre las detectadas.

    Si no se detecta nada, la peor entre las clases fiables. Las no fiables no rebajan
    un informe donde no se han detectado.
    """
    detected = [a for a in assessments if a.detected]
    considered = detected or [a for a in assessments if not a.unreliable_class]
    worst = min(considered, key=lambda a: UNCERTAINTY_ORDER.index(a.uncertainty))
    scope = "detected classes" if detected else "all reliable classes (nothing detected)"
    return worst.uncertainty, f"Highest uncertainty among {scope}."


def is_uncertain_non_detection(assessment: ClassAssessment) -> bool:
    """Negativo dudoso: no detectado e incertidumbre TTA Medium o High.

    Usa los mismos umbrales de `grade_uncertainty`. Si la clase es `unreliable`, se
    re-evalúa sin ese flag: "pocos positivos en test" no basta para meter un negativo
    estable en la franja; solo entra si el score TTA no es Low uncertainty.
    """
    if assessment.detected:
        return False
    if assessment.unreliable_class:
        tta_uncertainty, _ = grade_uncertainty(
            assessment.agreement, assessment.std, unreliable_class=False
        )
        return tta_uncertainty != UNCERTAINTY_LOW
    return assessment.uncertainty != UNCERTAINTY_LOW


def strip_uncertainty_assessments(assessments: list[ClassAssessment]) -> list[ClassAssessment]:
    """Clases para la franja superior: todas las detectadas + no detectadas dudosas.

    Omite los negativos Low (score estable) para no saturar con las 10 clases.
    Orden: detectadas por probabilidad; luego dudosas High antes que Medium.
    """
    detected = sorted(
        (a for a in assessments if a.detected),
        key=lambda a: -a.probability,
    )
    dubious = sorted(
        (a for a in assessments if is_uncertain_non_detection(a)),
        key=lambda a: (UNCERTAINTY_ORDER.index(a.uncertainty), -a.probability),
    )
    return detected + dubious
