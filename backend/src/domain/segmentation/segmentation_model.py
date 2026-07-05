"""Domain models for HybridGNet segmentation output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationResult:
    dense_mask: np.ndarray  # (1024, 1024) uint8 — lungs=128, heart=255
    lung_mask: np.ndarray  # binary, lungs only
    thoracic_mask: np.ndarray  # binary, lungs + heart
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in 1024 space
    cropped_gray: np.ndarray  # cropped thoracic region, uint8
    original_size: tuple[int, int]  # (W, H) of decoded image
    contours: dict[str, np.ndarray]  # RL, LL, H point arrays
