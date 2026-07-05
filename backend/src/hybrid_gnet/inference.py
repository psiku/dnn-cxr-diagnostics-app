"""Shared HybridGNet inference helpers used by the API and model loader."""

from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp
import torch

from src.hybrid_gnet.model import Hybrid
from src.hybrid_gnet.utils import genMatrixesLungsHeart, scipy_to_torch_sparse

SEGMENTATION_SIZE = 1024
CONTOUR_SPLITS = (44, 50, 26)
RL_CONTOUR_POINTS, LL_CONTOUR_POINTS, HEART_CONTOUR_POINTS = CONTOUR_SPLITS

LUNG_MASK_VALUE = 128
HEART_MASK_VALUE = 255


def split_contours(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split flattened model output into right lung, left lung, and heart contours."""
    points = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    rl_end = RL_CONTOUR_POINTS
    ll_end = rl_end + LL_CONTOUR_POINTS
    return points[:rl_end], points[rl_end:ll_end], points[ll_end:]


def dense_mask_from_contours(
    rl: np.ndarray,
    ll: np.ndarray,
    heart: np.ndarray,
    size: int = SEGMENTATION_SIZE,
) -> np.ndarray:
    """Rasterize organ contours into a dense segmentation mask."""
    mask = np.zeros((size, size), dtype=np.uint8)

    for contour, value in (
        (rl, LUNG_MASK_VALUE),
        (ll, LUNG_MASK_VALUE),
        (heart, HEART_MASK_VALUE),
    ):
        pts = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(mask, [pts], -1, int(value), thickness=-1)

    return mask


def build_hybrid_model(device: str | torch.device) -> Hybrid:
    """Construct a HybridGNet model with the lungs+heart graph configuration."""
    torch_device = torch.device(device)

    A, AD, D, U = genMatrixesLungsHeart()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config: dict = {
        "n_nodes": [N1, N1, N1, N2, N2, N2],
        "latents": 64,
        "inputsize": SEGMENTATION_SIZE,
        "skip_features": 32,
        "filters": [2, 32, 32, 32, 16, 16, 16],
    }

    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    A_t, D_t, U_t = (
        [scipy_to_torch_sparse(x).to(torch_device) for x in matrices]
        for matrices in (A_, D_, U_)
    )

    return Hybrid(config.copy(), D_t, U_t, A_t).to(torch_device)
