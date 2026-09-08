import json

import numpy as np
import pytest
import torch
import yaml

from src.core.config import load_config
from src.core.models.classifier import ChestXRayClassifier
from src.core.models.loader import load_model, load_thresholds

EXPECTED_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
PATHOLOGIES = ["A", "B", "C", "D", "E"]


def _write_npy(path):
    np.save(str(path), np.array(EXPECTED_THRESHOLDS, dtype=np.float32))


def _write_json_list(path):
    path.write_text(json.dumps(EXPECTED_THRESHOLDS))


def _write_json_dict(path):
    path.write_text(json.dumps(dict(zip(PATHOLOGIES, EXPECTED_THRESHOLDS))))


def _write_yaml(path):
    path.write_text(yaml.safe_dump(EXPECTED_THRESHOLDS))


def _write_pt(path):
    torch.save(torch.tensor(EXPECTED_THRESHOLDS), str(path))


def test_load_model(tmp_path):
    # Setup
    model_path = tmp_path / "dummy_model.pth"
    config_file = tmp_path / "config.yaml"

    model_params = {
        "num_classes": 5,
        "backbone_name": "resnet18",
        "pretrained": False,
        "grayscale": True,
        "backbone_trainable_layers": [],
        "in_features": 512,
        "transition_dim": 128,
        "use_transition": True,
        "pooling": "avg",
        "lse_r": 5.0,
        "dropout": 0.5,
    }

    dummy_model = ChestXRayClassifier(**model_params)
    torch.save(dummy_model.state_dict(), model_path)

    config_data = {
        "model_path": str(model_path).replace("\\", "/"),
        "model_cfg": model_params,
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Action
    config = load_config(str(config_file))
    model = load_model(config)

    # Assert
    assert model is not None
    assert isinstance(model, ChestXRayClassifier)
    assert not model.training
    assert model.use_mask_channel is False
    assert model.input_channels == 1


def test_classifier_use_mask_channel_accepts_four_channel_rgb_input():
    model = ChestXRayClassifier(
        num_classes=3,
        backbone_name="resnet18",
        pretrained=False,
        grayscale=False,
        use_mask_channel=True,
        backbone_trainable_layers=[],
        in_features=512,
        transition_dim=64,
        use_transition=True,
        pooling="avg",
        lse_r=5.0,
        dropout=0.0,
    )
    model.eval()

    assert model.input_channels == 4
    out = model(torch.zeros(1, 4, 224, 224))
    assert out["logits"].shape == (1, 3)

    with pytest.raises(ValueError, match="Expected 4 input channels"):
        model(torch.zeros(1, 3, 224, 224))


@pytest.mark.parametrize(
    "filename, writer",
    [
        ("thresholds.npy", _write_npy),
        ("thresholds.json", _write_json_list),
        ("thresholds.yaml", _write_yaml),
        ("thresholds.pt", _write_pt),
    ],
)
def test_load_thresholds_supported_formats(tmp_path, filename, writer):
    # Setup
    thresholds_path = tmp_path / filename
    writer(thresholds_path)

    # Action
    loaded = load_thresholds(str(thresholds_path))

    # Assert
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (len(EXPECTED_THRESHOLDS),)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, EXPECTED_THRESHOLDS)


def test_load_thresholds_json_dict_reorders_by_pathologies(tmp_path):
    # Setup
    thresholds_path = tmp_path / "thresholds.json"
    _write_json_dict(thresholds_path)
    shuffled_order = ["C", "A", "E", "B", "D"]
    expected = [EXPECTED_THRESHOLDS[PATHOLOGIES.index(name)] for name in shuffled_order]

    # Action
    loaded = load_thresholds(str(thresholds_path), pathologies=shuffled_order)

    # Assert
    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, expected)


def test_load_thresholds_missing_pathology_raises(tmp_path):
    # Setup
    thresholds_path = tmp_path / "thresholds.json"
    _write_json_dict(thresholds_path)

    # Assert
    with pytest.raises(KeyError, match="Z"):
        load_thresholds(str(thresholds_path), pathologies=["A", "Z"])


def test_load_thresholds_returns_none_when_path_empty():
    assert load_thresholds(None) is None
    assert load_thresholds("") is None


def test_load_thresholds_unsupported_extension_raises(tmp_path):
    # Setup
    thresholds_path = tmp_path / "thresholds.txt"
    thresholds_path.write_text("0.1 0.2 0.3 0.4 0.5")

    # Assert
    with pytest.raises(ValueError, match="Unsupported thresholds file format"):
        load_thresholds(str(thresholds_path))
