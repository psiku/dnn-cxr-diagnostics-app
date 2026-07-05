"""Shared helpers for API-facing prediction payloads."""

from __future__ import annotations

import numpy as np

from src.core.pathologies import PATHOLOGIES
from src.core.utils.image import encode_array_to_base64
from src.domain.prediction.models import PathologyScore, TriageAssessment


def build_pathology_predictions(
    probs: np.ndarray,
    pathologies: tuple[str, ...] | list[str] = PATHOLOGIES,
) -> list[PathologyScore]:
    """Map model probabilities to ordered pathology prediction models."""
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    return [
        PathologyScore(pathology=pathologies[i], probability=float(probs[i]))
        for i in range(len(pathologies))
    ]


def build_prediction_result(
    probs: np.ndarray,
    weighted_cam: np.ndarray,
    pathologies: tuple[str, ...] | list[str] = PATHOLOGIES,
) -> dict[str, object]:
    """Build the shared /predict response payload."""
    return {
        "predictions": build_pathology_predictions(probs, pathologies=pathologies),
        "base_64_heatmap": encode_array_to_base64(weighted_cam),
    }


def build_triage_result(
    probs: np.ndarray,
    weighted_cam: np.ndarray,
    triage_assessment: TriageAssessment,
    pathologies: tuple[str, ...] | list[str] = PATHOLOGIES,
) -> dict[str, object]:
    """Build the /triage response payload from prediction and triage outputs."""
    return {
        **build_prediction_result(probs, weighted_cam, pathologies=pathologies),
        **triage_assessment.model_dump(),
    }
