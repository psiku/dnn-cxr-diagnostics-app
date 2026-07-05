"""Unit tests for PredictionService orchestration."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import base64
import numpy as np
import pytest
import torch
from PIL import Image

from src.core.exceptions import SegmentationNotConfiguredError
from src.core.models.predictor import ChestXRayPredictor
from src.domain.prediction.models import PredictionOutput
from src.services.prediction_service import PredictionService


def _make_base64_image(size: tuple[int, int] = (120, 90)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, color=(123, 45, 67)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _make_predictor(num_classes: int = 3) -> MagicMock:
    predictor = MagicMock(spec=ChestXRayPredictor)
    predictor.device = "cpu"
    predictor.grayscale = False
    predictor.predict.return_value = {
        "probs": np.zeros(num_classes, dtype=np.float32),
        "weighted_cam": np.zeros((90, 120), dtype=np.float32),
    }
    predictor.predict_tensor.return_value = {
        "probs": np.zeros(num_classes, dtype=np.float32),
        "weighted_cam": np.zeros((224, 224), dtype=np.float32),
    }
    return predictor


def test_prediction_service_without_segmentation_uses_full_image_path():
    predictor = _make_predictor()
    service = PredictionService(predictor)
    image_base64 = _make_base64_image()

    result = service.predict(image_base64, use_mask=False)

    predictor.predict.assert_called_once_with(image_base64)
    predictor.predict_tensor.assert_not_called()
    assert isinstance(result, PredictionOutput)
    assert result.probs.shape == (3,)


def test_prediction_service_use_mask_without_segmentation_raises():
    predictor = _make_predictor()
    service = PredictionService(predictor)

    with pytest.raises(SegmentationNotConfiguredError, match="Segmentation is not configured"):
        service.predict(_make_base64_image(), use_mask=True)

    predictor.predict.assert_not_called()
    predictor.predict_tensor.assert_not_called()


@patch("src.services.prediction_service.paste_cam_to_full_image")
@patch("src.services.prediction_service.convert_array_to_tensor")
def test_prediction_service_use_mask_orchestrates_segmentation_and_predictor(
    mock_convert_array,
    mock_paste_cam,
):
    predictor = _make_predictor()
    segmentation_service = MagicMock()
    service = PredictionService(predictor, segmentation_service=segmentation_service)
    image_base64 = _make_base64_image()

    segmentation_service.segment.return_value = MagicMock(
        cropped_gray=np.zeros((40, 50), dtype=np.uint8),
        bbox=(10, 20, 60, 70),
        original_size=(120, 90),
    )
    mock_convert_array.return_value = (torch.zeros(1, 3, 224, 224), (50, 40))
    mock_paste_cam.return_value = np.ones((90, 120), dtype=np.float32)

    result = service.predict(image_base64, use_mask=True)

    segmentation_service.segment.assert_called_once_with(image_base64)
    predictor.predict_tensor.assert_called_once()
    mock_paste_cam.assert_called_once()
    assert result.weighted_cam.shape == (90, 120)


def test_prediction_service_segment_delegates_when_configured():
    predictor = _make_predictor()
    segmentation_service = MagicMock()
    service = PredictionService(predictor, segmentation_service=segmentation_service)

    service.segment("payload")

    segmentation_service.segment.assert_called_once_with("payload")


def test_prediction_service_segment_raises_when_not_configured():
    service = PredictionService(_make_predictor())

    with pytest.raises(SegmentationNotConfiguredError, match="Segmentation is not configured"):
        service.segment("payload")
