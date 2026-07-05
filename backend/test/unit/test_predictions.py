import numpy as np
import pytest

from src.core.pathologies import PATHOLOGIES
from src.domain.prediction.models import PathologyScore, TriageAssessment
from src.services.prediction_format import (
    build_pathology_predictions,
    build_prediction_result,
    build_triage_result,
)


def test_build_pathology_predictions():
    probs = np.array([0.2, 0.8], dtype=np.float32)

    predictions = build_pathology_predictions(probs, pathologies=PATHOLOGIES[:2])

    assert len(predictions) == 2
    assert predictions[0].pathology == PATHOLOGIES[0]
    assert predictions[0].probability == pytest.approx(0.2)
    assert predictions[1].pathology == PATHOLOGIES[1]
    assert predictions[1].probability == pytest.approx(0.8)


def test_build_prediction_result(monkeypatch):
    def fake_encode_array_to_base64(weighted_cam):
        assert weighted_cam.shape == (2, 2)
        return "encoded-heatmap"

    monkeypatch.setattr(
        "src.services.prediction_format.encode_array_to_base64",
        fake_encode_array_to_base64,
    )

    probs = np.array([0.2, 0.8], dtype=np.float32)
    weighted_cam = np.zeros((2, 2), dtype=np.float32)

    result = build_prediction_result(probs, weighted_cam, pathologies=PATHOLOGIES[:2])

    assert result["base_64_heatmap"] == "encoded-heatmap"
    assert result["predictions"][0].pathology == PATHOLOGIES[0]
    assert result["predictions"][0].probability == pytest.approx(0.2)
    assert result["predictions"][1].pathology == PATHOLOGIES[1]
    assert result["predictions"][1].probability == pytest.approx(0.8)


def test_build_triage_result(monkeypatch):
    def fake_encode_array_to_base64(weighted_cam):
        return "encoded-heatmap"

    monkeypatch.setattr(
        "src.services.prediction_format.encode_array_to_base64",
        fake_encode_array_to_base64,
    )

    probs = np.array([0.2, 0.8], dtype=np.float32)
    weighted_cam = np.zeros((2, 2), dtype=np.float32)
    triage_assessment = TriageAssessment(
        triage_level="high",
        high_risk_findings=[
            PathologyScore(pathology=PATHOLOGIES[1], probability=0.8)
        ],
    )

    result = build_triage_result(
        probs,
        weighted_cam,
        triage_assessment,
        pathologies=PATHOLOGIES[:2],
    )

    assert result["base_64_heatmap"] == "encoded-heatmap"
    assert result["triage_level"] == "high"
    assert result["high_risk_findings"] == [
        finding.model_dump() for finding in triage_assessment.high_risk_findings
    ]
    assert result["predictions"][0].pathology == PATHOLOGIES[0]
    assert result["predictions"][0].probability == pytest.approx(0.2)
    assert result["predictions"][1].pathology == PATHOLOGIES[1]
    assert result["predictions"][1].probability == pytest.approx(0.8)
