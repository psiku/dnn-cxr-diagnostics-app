"""Unit tests for XRayTriageService.

These tests exercise triage logic only and do not require model weights or
the thresholds JSON file on disk.
"""

import numpy as np
import pytest

from src.domain.prediction.models import PathologyScore
from src.services.triage_service import XRayTriageService


NUM_CLASSES = len(XRayTriageService.PATHOLOGIES)


def make_service(thresholds: np.ndarray | None = None) -> XRayTriageService:
    return XRayTriageService(thresholds=thresholds)


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
    probs = np.array(
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ]
    )

    # Action
    result = service._determine_triage_level(probs)

    # Assert
    assert result == "low"


def test_determine_triage_level_medium():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    service.MEDIUM_MARGIN = 0.15
    probs = np.array(
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ]
    )

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
    assert len(result) == 1
    assert result[0].pathology == service.PATHOLOGIES[0]
    assert result[0].probability == pytest.approx(0.2)


def test_get_high_risk_findings_multiple_found():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    probs = np.array([0.2] * NUM_CLASSES, dtype=np.float32)

    # Action
    result = service._get_high_risk_findings(probs)

    # Assert - All pathologies are found
    assert len(result) == NUM_CLASSES
    assert all(finding.probability == pytest.approx(0.2) for finding in result)


def test_get_high_risk_findings_sorted_by_probability():
    # Setup
    service = make_service(thresholds=_uniform_thresholds(0.1))
    probs = np.array(
        [
            0.2,
            0.1,
            0.3,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ],
        dtype=np.float32,
    )

    # Action
    result = service._get_high_risk_findings(probs)

    # Assert
    assert [finding.model_dump() for finding in result] == [
        {"pathology": service.PATHOLOGIES[2], "probability": pytest.approx(0.3)},
        {"pathology": service.PATHOLOGIES[0], "probability": pytest.approx(0.2)},
        {"pathology": service.PATHOLOGIES[1], "probability": pytest.approx(0.1)},
    ]


def test_assess_low():
    # Setup
    probs = np.full(NUM_CLASSES, 0.05, dtype=np.float32)
    service = make_service(thresholds=_uniform_thresholds(0.3))

    # Action
    result = service.assess(probs)

    # Assert
    assert result.triage_level == "low"
    assert result.high_risk_findings == []


def test_assess_few_examples():
    # Setup
    probs = np.asarray(
        [
            0.2,
            0.1,
            0.3,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ],
        dtype=np.float32,
    )
    service = make_service(thresholds=_uniform_thresholds(0.1))
    service.CRITICAL_FINDING_COUNT = 4
    service.CRITICAL_MARGIN = 0.5

    # Action
    result = service.assess(probs)

    # Assert
    assert result.triage_level == "high"
    assert [finding.model_dump() for finding in result.high_risk_findings] == [
        {"pathology": service.PATHOLOGIES[2], "probability": pytest.approx(0.3)},
        {"pathology": service.PATHOLOGIES[0], "probability": pytest.approx(0.2)},
        {"pathology": service.PATHOLOGIES[1], "probability": pytest.approx(0.1)},
    ]
