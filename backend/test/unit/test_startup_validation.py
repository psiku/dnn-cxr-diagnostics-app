import numpy as np
import pytest

from src.core.config import validate_runtime_alignment


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
