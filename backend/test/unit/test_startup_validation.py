import numpy as np
import pytest

from src.core.config import Config, ModelConfig, validate_mask_flags, validate_runtime_alignment


def _model_cfg(**overrides) -> ModelConfig:
    base = {
        "num_classes": 3,
        "backbone_name": "resnet18",
        "pretrained": False,
        "grayscale": False,
        "backbone_trainable_layers": [],
        "transition_dim": 64,
        "use_transition": True,
        "pooling": "avg",
        "lse_r": 10.0,
        "dropout": 0.0,
    }
    base.update(overrides)
    return ModelConfig(**base)


def test_validate_runtime_alignment_accepts_matching_config():
    validate_runtime_alignment(
        num_classes=3,
        pathologies=("A", "B", "C"),
        thresholds=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )


def test_validate_runtime_alignment_allows_missing_thresholds():
    validate_runtime_alignment(
        num_classes=2,
        pathologies=("A", "B"),
        thresholds=None,
    )


def test_validate_runtime_alignment_rejects_num_classes_mismatch():
    with pytest.raises(ValueError, match="model_cfg.num_classes"):
        validate_runtime_alignment(
            num_classes=5,
            pathologies=("A", "B", "C"),
            thresholds=None,
        )


def test_validate_runtime_alignment_rejects_threshold_count_mismatch():
    with pytest.raises(ValueError, match="thresholds.json"):
        validate_runtime_alignment(
            num_classes=3,
            pathologies=("A", "B", "C"),
            thresholds=np.array([0.1, 0.2], dtype=np.float32),
            thresholds_path="config/thresholds.json",
        )


def test_validate_mask_flags_accepts_defaults():
    config = Config(
        model_path="/tmp/model.ckpt",
        model_cfg=_model_cfg(),
    )
    validate_mask_flags(config)


def test_validate_mask_flags_rejects_mask_channel_without_use_mask():
    config = Config(
        model_path="/tmp/model.ckpt",
        model_cfg=_model_cfg(use_mask_channel=True),
        use_mask=False,
        segmentation_weights_path="/tmp/seg.pt",
    )
    with pytest.raises(ValueError, match="use_mask_channel=true requires use_mask=true"):
        validate_mask_flags(config)


def test_validate_mask_flags_rejects_use_mask_without_weights():
    config = Config(
        model_path="/tmp/model.ckpt",
        model_cfg=_model_cfg(),
        use_mask=True,
    )
    with pytest.raises(ValueError, match="segmentation_weights_path"):
        validate_mask_flags(config)


def test_validate_mask_flags_accepts_use_mask_with_weights():
    config = Config(
        model_path="/tmp/model.ckpt",
        model_cfg=_model_cfg(use_mask_channel=True),
        use_mask=True,
        segmentation_weights_path="/tmp/seg.pt",
    )
    validate_mask_flags(config)
