"""Orquesta un analisis: prediccion del ensemble (pase limpio) + TTA (incertidumbre) + Grad-CAM."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image

from . import settings
from .gradcam import compute_gradcam_maps
from .model_loader import DeployedModel
from .preprocessing import preprocess_fundus_image
from .uncertainty import ClassAssessment, assess_predictions, overall_uncertainty, run_tta_passes


@dataclass
class AnalysisResult:
    image_uint8: torch.Tensor
    assessments: list[ClassAssessment]
    pass_probabilities: np.ndarray
    gradcam_maps: dict[int, np.ndarray]
    overall_uncertainty: str
    overall_uncertainty_note: str
    seconds: float
    healthy_conflict_resolved: bool = False
    healthy_side_kept: str | None = None

    @property
    def detected_indices(self) -> list[int]:
        return [i for i, a in enumerate(self.assessments) if a.detected]

    @property
    def detected_labels(self) -> list[str]:
        return [self.assessments[i].label for i in self.detected_indices]


def compare_to_reference(detected_labels: list[str], reference_labels: list[str]) -> tuple[bool, list[str], list[str]]:
    predicted, reference = set(detected_labels), set(reference_labels)
    extra = [l for l in settings.LABEL_NAMES if l in predicted - reference]
    missing = [l for l in settings.LABEL_NAMES if l in reference - predicted]
    return predicted == reference, extra, missing


def _deterministic_seed(image: Image.Image) -> int:
    return int(hashlib.sha256(np.asarray(image).tobytes()).hexdigest()[:8], 16)


def analyze_image(deployed: DeployedModel, image: Image.Image, num_augmented_passes: int,
                  progress_callback: Callable[[int, int], None] | None = None) -> AnalysisResult:
    started = time.perf_counter()
    image = image.convert("RGB")
    pass_probabilities = run_tta_passes(deployed.ensemble, image, num_augmented_passes,
                                        seed=_deterministic_seed(image), progress_callback=progress_callback)
    assessments = assess_predictions(pass_probabilities, deployed.thresholds, deployed.label_names,
                                     deployed.unreliable_labels)
    from app.ensemble import preprocess
    _, size, model = deployed.ensemble.members[settings.GRADCAM_MEMBER]
    image_uint8 = preprocess_fundus_image(image, size)
    gradcam_maps = compute_gradcam_maps(model, preprocess(image, size)[1], list(range(len(deployed.label_names))))
    uncertainty, note = overall_uncertainty(assessments)
    return AnalysisResult(image_uint8=image_uint8, assessments=assessments, pass_probabilities=pass_probabilities,
                          gradcam_maps=gradcam_maps, overall_uncertainty=uncertainty, overall_uncertainty_note=note,
                          seconds=time.perf_counter() - started)
