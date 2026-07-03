"""Unit tests for XRayTriageService.

The predictor is mocked so these tests do not require model weights or the
thresholds JSON file on disk.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.models.predictor import ChestXRayPredictor
from src.services.inference_service import XRayTriageService


NUM_CLASSES = len(XRayTriageService.PATHOLOGIES)


def make_predictor(
    probs: np.ndarray,
    weighted_cam: np.ndarray | None = None,
) -> MagicMock:
    """Build a stub ChestXRayPredictor whose `predict` returns fixed values."""
    if weighted_cam is None:
        weighted_cam = np.zeros((4, 4), dtype=np.float32)

    predictor = MagicMock(spec=ChestXRayPredictor)
    predictor.predict.return_value = {
        "probs": np.asarray(probs, dtype=np.float32),
        "weighted_cam": weighted_cam,
    }
    return predictor


def make_service(
    thresholds: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    weighted_cam: np.ndarray | None = None,
) -> XRayTriageService:
    """Build a service wired to a stub predictor."""
    if probs is None:
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    predictor = make_predictor(probs, weighted_cam=weighted_cam)
    return XRayTriageService(predictor, thresholds=thresholds)


def _uniform_thresholds(value: float = 0.5) -> np.ndarray:
    return np.full(NUM_CLASSES, value, dtype=np.float32)


def test_default_thresholds_used_when_none_provided():
    # Setup
    service = make_service(thresholds=None)

    # Assert
    assert service.thresholds.shape == (NUM_CLASSES,)
    assert np.allclose(service.thresholds, XRayTriageService.DEFAULT_THRESHOLD)


def test_custom_thresholds_are_kept_as_float32():
    # Setup
    custom = np.linspace(0.1, 0.9, NUM_CLASSES)

    service = make_service(thresholds=custom)

    # Assert
    assert service.thresholds.dtype == np.float32
    assert np.allclose(service.thresholds, custom.astype(np.float32))


def test_wrong_threshold_length_raises_value_error():
    # Setup and Assert
    with pytest.raises(ValueError, match=str(NUM_CLASSES)):
        make_service(thresholds=np.array([0.5, 0.5]))


def test_determine_triage_level_low():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.3))
    service.MEDIUM_MARGIN = 0.15
    probs = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    # Action
    result = service._determine_triage_level(probs)

    # Assert
    assert result == "low"


def test_determine_triage_level_medium():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    service.MEDIUM_MARGIN = 0.15
    probs = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    # Action
    result = service._determine_triage_level(probs)

    # Assert
    assert result == "medium"


def test_determine_triage_level_high():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    service.CRITICAL_MARGIN = 0.5
    service.CRITICAL_FINDING_COUNT = 15
    probs = np.array([0.2] * NUM_CLASSES)

    # Action
    result = service._determine_triage_level(probs)

    # Assert
    assert result == "high"


def test_determine_triage_level_critical_margin_exceeded():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    service.CRITICAL_MARGIN = 0.1
    service.CRITICAL_FINDING_COUNT = 15
    probs = np.array([0.2] * NUM_CLASSES)

    # Action
    result = service._determine_triage_level(probs)

    # Assert
    assert result == "critical"


def test_determine_triage_level_critical_finding_count_exceeded():
    service = make_service(thresholds=_uniform_thresholds(0.1))
    service.CRITICAL_MARGIN = 0.5
    service.CRITICAL_FINDING_COUNT = 3
    probs = np.array([0.2] * NUM_CLASSES)

    # Action
    result = service._determine_triage_level(probs)

    # Assert
    assert result == "critical"


def test_get_high_risk_findings_none_found():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    probs = np.array([0.05] * NUM_CLASSES)

    # Action
    result = service._get_high_risk_findings(probs)

    # Assert
    assert result == []


def test_get_high_risk_findings_one_found():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    probs = np.array([0.2] + [0.05] * (NUM_CLASSES - 1), dtype=np.float32)

    # Action
    result = service._get_high_risk_findings(probs)

    # Assert - Infiltration is the first pathology in the list
    assert result == [{'pathology': service.PATHOLOGIES[0], 'probability': pytest.approx(0.2)}]


def test_get_high_risk_findings_multiple_found():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    probs = np.array([0.2] * NUM_CLASSES, dtype=np.float32)

    # Action
    result = service._get_high_risk_findings(probs)

    # Assert - All pathologies are found
    assert result == [{'pathology': service.PATHOLOGIES[0], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[1], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[2], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[3], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[4], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[5], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[6], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[7], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[8], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[9], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[10], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[11], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[12], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[13], 'probability': pytest.approx(0.2)}]


def test_get_high_risk_findings_sorted_by_probability():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    probs = np.array([0.2, 0.1, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)

    # Action
    result = service._get_high_risk_findings(probs)

    # Assert
    assert result == [{'pathology': service.PATHOLOGIES[2], 'probability': pytest.approx(0.3)}, {'pathology': service.PATHOLOGIES[0], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[1], 'probability': pytest.approx(0.1)}]


def test_predict_and_triage_low():
    # Setup
    probs = np.full(NUM_CLASSES, 0.05, dtype=np.float32)
    service = make_service(
        thresholds=_uniform_thresholds(0.3),  # All probs below threshold
        probs=probs,
    )
    # Action
    result = service.predict_and_triage("any-string-the-mock-ignores")
    # Assert
    assert result["triage_level"] == "low"
    assert result["high_risk_findings"] == []


def test_predict_and_triage_few_examples():
    # Setup
    probs = np.asarray([0.2, 0.1, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    service = make_service(
        thresholds=_uniform_thresholds(0.1),  # few probs above threshold
        probs=probs,
    )
    service.CRITICAL_FINDING_COUNT = 4
    service.CRITICAL_MARGIN = 0.5
    # Action
    result = service.predict_and_triage("any-string-the-mock-ignores")
    # Assert
    assert result["triage_level"] == "high"
    assert result["high_risk_findings"] == [{'pathology': service.PATHOLOGIES[2], 'probability': pytest.approx(0.3)}, {'pathology': service.PATHOLOGIES[0], 'probability': pytest.approx(0.2)}, {'pathology': service.PATHOLOGIES[1], 'probability': pytest.approx(0.1)}]