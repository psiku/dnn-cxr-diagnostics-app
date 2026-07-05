"""Unit tests for ChestXRayPredictor.

The underlying ChestXRayClassifier is mocked, so these tests do not download
pretrained weights and do not run a real ResNet forward pass.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.core.models.classifier import ChestXRayClassifier
from src.core.models.predictor import ChestXRayPredictor


def _make_mock_model(
    num_classes: int = 5,
    num_transition_channels: int = 4,
    feature_hw: tuple[int, int] = (7, 7),
    logits: torch.Tensor | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics ChestXRayClassifier just enough for the predictor."""
    model = MagicMock(spec=ChestXRayClassifier)
    model.num_classes = num_classes
    model.training = False

    h, w = feature_hw
    if logits is None:
        logits = torch.zeros(1, num_classes)

    model.return_value = {
        "logits": logits,
        "transition_maps": torch.ones(1, num_transition_channels, h, w),
        "pooled_features": torch.zeros(1, num_transition_channels),
    }

    # `nn.Linear.weight` has shape [out_features, in_features] = [C, D]
    model.prediction = MagicMock()
    model.prediction.weight = torch.ones(num_classes, num_transition_channels)

    # Identity normalisation keeps the heatmap shape predictable.
    model._normalize_map = lambda cam, eps=1e-8: cam

    return model


def _make_predictor(
    num_classes: int = 5,
    device: str = "cpu",
) -> ChestXRayPredictor:
    return ChestXRayPredictor(
        _make_mock_model(num_classes=num_classes),
        device=device,
    )


def _make_base64_image(
    size: tuple[int, int] = (100, 80),
    color: tuple[int, int, int] = (123, 45, 67),
) -> str:
    """Encode an in-memory PNG to base64."""
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_init_stores_model_and_device():
    # Setup
    model = _make_mock_model(num_classes=3)

    # Action
    predictor = ChestXRayPredictor(model, device="cpu")

    # Assert
    assert predictor.model is model
    assert predictor.device == "cpu"


def test_init_default_device_is_cpu():
    # Setup
    model = _make_mock_model()

    # Action
    predictor = ChestXRayPredictor(model)

    # Assert
    assert predictor.device == "cpu"


def test_init_moves_model_to_device_and_sets_eval():
    # Setup
    model = _make_mock_model()

    # Action
    ChestXRayPredictor(model, device="cpu")

    # Assert
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once_with()


def test_init_accepts_custom_device():
    # Setup
    model = _make_mock_model()

    # Action
    predictor = ChestXRayPredictor(model, device="cuda:0")

    # Assert
    assert predictor.device == "cuda:0"
    model.to.assert_called_once_with("cuda:0")


def test_predict_returns_dict_with_expected_keys():
    # Setup
    predictor = _make_predictor(num_classes=3)
    image_base64 = _make_base64_image()

    # Action
    result = predictor.predict(image_base64)

    # Assert
    assert isinstance(result, dict)
    assert set(result) == {"probs", "weighted_cam"}


@pytest.mark.parametrize("num_classes", [1, 3, 14, 20])
def test_predict_probs_shape_matches_num_classes(num_classes):
    # Setup
    predictor = _make_predictor(num_classes=num_classes)
    image_base64 = _make_base64_image()

    # Action
    result = predictor.predict(image_base64)

    # Assert
    assert isinstance(result["probs"], np.ndarray)
    assert result["probs"].shape == (num_classes,)


def test_predict_applies_sigmoid_to_logits():
    # Setup
    logits = torch.tensor([[-2.0, 0.0, 2.0]])
    model = _make_mock_model(num_classes=3, logits=logits)
    predictor = ChestXRayPredictor(model)
    image_base64 = _make_base64_image()

    # Action
    result = predictor.predict(image_base64)

    # Assert
    expected = torch.sigmoid(logits).squeeze(0).numpy()
    np.testing.assert_allclose(result["probs"], expected, rtol=1e-5)


def test_predict_weighted_cam_is_2d_float_numpy():
    # Setup
    predictor = _make_predictor(num_classes=3)
    image_base64 = _make_base64_image(size=(64, 64))

    # Action
    result = predictor.predict(image_base64)

    # Assert
    cam = result["weighted_cam"]
    assert isinstance(cam, np.ndarray)
    assert cam.ndim == 2
    assert np.issubdtype(cam.dtype, np.floating)


def test_predict_weighted_cam_size_follows_original_image_size():
    # Setup
    predictor = _make_predictor(num_classes=3)
    image_base64 = _make_base64_image(size=(100, 80))

    # Action
    result = predictor.predict(image_base64)

    # Assert
    assert result["weighted_cam"].shape == (100, 80)


def test_predict_runs_a_single_forward_pass():
    # Setup
    model = _make_mock_model(num_classes=3)
    predictor = ChestXRayPredictor(model)
    image_base64 = _make_base64_image()

    # Action
    predictor.predict(image_base64)

    # Assert
    assert model.call_count == 1


@patch("src.core.models.predictor.convert_base64_to_tensor")
def test_predict_forwards_image_and_device_to_converter(mock_convert):
    # Setup
    mock_convert.return_value = (
        torch.zeros(1, 3, 224, 224),
        (40, 60),
    )
    predictor = _make_predictor(num_classes=3, device="cpu")

    # Action
    predictor.predict("base64-payload")

    # Assert
    mock_convert.assert_called_once_with(
        "base64-payload",
        "cpu",
        grayscale=False,
    )


def test_compute_weighted_cam_shape_matches_original_size():
    # Setup
    predictor = _make_predictor(num_classes=3)

    # Action
    cam = predictor._compute_weighted_cam(
        transition_maps=torch.ones(1, 4, 7, 7),
        probs=torch.tensor([0.1, 0.2, 0.3]),
        display_size=(50, 60),
    )

    # Assert
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (50, 60)


def test_compute_weighted_cam_calls_model_normalize_map():
    # Setup
    model = _make_mock_model(num_classes=3, num_transition_channels=4)
    normalize = MagicMock(side_effect=lambda cam, eps=1e-8: cam)
    model._normalize_map = normalize
    predictor = ChestXRayPredictor(model)

    # Action
    predictor._compute_weighted_cam(
        transition_maps=torch.ones(1, 4, 7, 7),
        probs=torch.tensor([0.1, 0.2, 0.3]),
        display_size=(32, 32),
    )

    # Assert
    normalize.assert_called_once()


def test_compute_weighted_cam_is_nonnegative_after_relu():
    """Even with negative probs, the CAM should be clamped to >= 0 by ReLU."""
    # Setup
    predictor = _make_predictor(num_classes=3)

    # Action
    cam = predictor._compute_weighted_cam(
        transition_maps=torch.randn(1, 4, 7, 7),
        probs=torch.tensor([-1.0, -1.0, -1.0]),
        display_size=(32, 32),
    )

    # Assert
    assert np.all(cam >= 0)


def test_compute_weighted_cam_zero_probs_yields_zero_cam():
    """With probs all zero, the weighted combination is zero everywhere."""
    # Setup
    predictor = _make_predictor(num_classes=3)

    # Action
    cam = predictor._compute_weighted_cam(
        transition_maps=torch.ones(1, 4, 7, 7),
        probs=torch.zeros(3),
        display_size=(32, 32),
    )

    # Assert
    np.testing.assert_array_equal(cam, np.zeros((32, 32)))
