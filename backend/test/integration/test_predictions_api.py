"""Integration tests for prediction API error mapping and responses."""

from io import BytesIO
from unittest.mock import MagicMock

import base64
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app
from src.api.dependencies import get_prediction_service, get_triage_service
from src.core.exceptions import (
    EmptySegmentationMaskError,
    InvalidImageError,
    SegmentationNotConfiguredError,
)
from src.core.pathologies import PATHOLOGIES
from src.domain.prediction.models import PathologyScore, PredictionOutput, TriageAssessment
from src.services.prediction_format import build_prediction_result


def _make_base64_image() -> str:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(100, 120, 140)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def mock_prediction_service():
    service = MagicMock()
    service.use_mask = False
    service.use_mask_channel = False
    return service


@pytest.fixture
def mock_triage_service():
    return MagicMock()


@pytest.fixture
def client(mock_prediction_service, mock_triage_service):
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    app.dependency_overrides[get_triage_service] = lambda: mock_triage_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_predict_returns_422_for_invalid_image(client, mock_prediction_service):
    mock_prediction_service.predict_for_api.side_effect = InvalidImageError(
        "Invalid base64 image payload"
    )

    response = client.post("/predict", json={"base_64_image": "not-an-image"})

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid base64 image payload"


def test_predict_returns_503_when_segmentation_not_configured(
    client, mock_prediction_service
):
    mock_prediction_service.predict_for_api.side_effect = SegmentationNotConfiguredError(
        "Segmentation is not configured. Set segmentation_weights_path in config."
    )

    response = client.post(
        "/predict",
        json={"base_64_image": _make_base64_image()},
    )

    assert response.status_code == 503
    assert "Segmentation is not configured" in response.json()["detail"]


def test_predict_returns_400_for_empty_segmentation_mask(client, mock_prediction_service):
    mock_prediction_service.predict_for_api.side_effect = EmptySegmentationMaskError(
        "Segmentation produced an empty thoracic mask"
    )

    response = client.post(
        "/predict",
        json={"base_64_image": _make_base64_image()},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Segmentation produced an empty thoracic mask"


def test_predict_returns_prediction_response(client, mock_prediction_service):
    probs = np.zeros(len(PATHOLOGIES), dtype=np.float32)
    weighted_cam = np.zeros((16, 16), dtype=np.float32)
    mock_prediction_service.predict_for_api.return_value = build_prediction_result(
        probs,
        weighted_cam,
    )

    response = client.post("/predict", json={"base_64_image": _make_base64_image()})

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == len(PATHOLOGIES)
    assert body["base_64_heatmap"]


def test_triage_returns_typed_response(client, mock_prediction_service, mock_triage_service):
    probs = np.zeros(len(PATHOLOGIES), dtype=np.float32)
    weighted_cam = np.zeros((16, 16), dtype=np.float32)
    mock_prediction_service.predict.return_value = PredictionOutput(
        probs=probs,
        weighted_cam=weighted_cam,
    )
    mock_triage_service.assess.return_value = TriageAssessment(
        triage_level="low",
        high_risk_findings=[
            PathologyScore(pathology=PATHOLOGIES[0], probability=0.12)
        ],
    )

    response = client.post("/triage", json={"base_64_image": _make_base64_image()})

    assert response.status_code == 200
    body = response.json()
    assert body["triage_level"] == "low"
    assert body["high_risk_findings"][0]["pathology"] == PATHOLOGIES[0]
    assert body["predictions"]
