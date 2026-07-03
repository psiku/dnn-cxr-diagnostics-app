import numpy as np
import pytest

from src.api.routes.predictions import _build_prediction_models, _build_predictions_and_heatmap
from src.core.pathologies import PATHOLOGIES


def test_build_predictions_and_heatmap(monkeypatch):
    def fake_encode_array_to_base64(weighted_cam):
        assert weighted_cam.shape == (2, 2)
        return "encoded-heatmap"

    monkeypatch.setattr(
        "src.api.routes.predictions.encode_array_to_base64",
        fake_encode_array_to_base64,
    )

    probs = np.array([0.2, 0.8], dtype=np.float32)
    weighted_cam = np.zeros((2, 2), dtype=np.float32)

    predictions, heatmap = _build_predictions_and_heatmap(probs, weighted_cam)

    assert heatmap == "encoded-heatmap"
    assert len(predictions) == 2
    assert [item.pathology for item in predictions] == list(PATHOLOGIES[:2])
    assert predictions[0].probability == pytest.approx(0.2, rel=1e-6, abs=1e-6)
    assert predictions[1].probability == pytest.approx(0.8, rel=1e-6, abs=1e-6)


def test_build_prediction_models():
    items = [
        {"pathology": PATHOLOGIES[0], "probability": 0.15},
        {"pathology": PATHOLOGIES[1], "probability": 0.85},
    ]

    predictions = _build_prediction_models(items)

    assert len(predictions) == 2
    assert [item.pathology for item in predictions] == list(PATHOLOGIES[:2])
    assert predictions[0].probability == pytest.approx(0.15, rel=1e-6, abs=1e-6)
    assert predictions[1].probability == pytest.approx(0.85, rel=1e-6, abs=1e-6)
    assert all(hasattr(item, "model_dump") for item in predictions)