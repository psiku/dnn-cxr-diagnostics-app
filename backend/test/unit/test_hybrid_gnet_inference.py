"""Unit tests for shared HybridGNet inference helpers."""

import numpy as np

from src.hybrid_gnet.inference import (
    CONTOUR_SPLITS,
    HEART_MASK_VALUE,
    LUNG_MASK_VALUE,
    SEGMENTATION_SIZE,
    dense_mask_from_contours,
    split_contours,
)


def test_split_contours_uses_shared_splits():
    total_points = sum(CONTOUR_SPLITS)
    points = np.arange(total_points * 2, dtype=np.int32).reshape(-1, 2)

    rl, ll, heart = split_contours(points)

    assert rl.shape == (CONTOUR_SPLITS[0], 2)
    assert ll.shape == (CONTOUR_SPLITS[1], 2)
    assert heart.shape == (CONTOUR_SPLITS[2], 2)
    assert np.array_equal(rl, points[: CONTOUR_SPLITS[0]])
    assert np.array_equal(ll, points[CONTOUR_SPLITS[0] : sum(CONTOUR_SPLITS[:2])])
    assert np.array_equal(heart, points[sum(CONTOUR_SPLITS[:2]) :])


def test_dense_mask_from_contours_draws_lungs_and_heart():
    rl = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.int32)
    ll = np.array([[30, 10], [40, 10], [40, 20], [30, 20]], dtype=np.int32)
    heart = np.array([[20, 25], [30, 25], [30, 35], [20, 35]], dtype=np.int32)

    mask = dense_mask_from_contours(rl, ll, heart, size=64)

    assert mask.shape == (64, 64)
    assert mask[15, 15] == LUNG_MASK_VALUE
    assert mask[15, 35] == LUNG_MASK_VALUE
    assert mask[30, 25] == HEART_MASK_VALUE
    assert mask[0, 0] == 0
    assert SEGMENTATION_SIZE == 1024
