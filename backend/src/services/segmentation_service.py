"""Application service: HybridGNet lung/heart segmentation."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from src.core.models.segmentation_loader import SegmentationArtifacts
from src.core.utils.image import decode_base64_to_image
from src.core.exceptions import EmptySegmentationMaskError
from src.domain.segmentation.segmentation_model import SegmentationResult
from src.hybrid_gnet.inference import (
    HEART_MASK_VALUE,
    LUNG_MASK_VALUE,
    SEGMENTATION_SIZE,
    dense_mask_from_contours,
    split_contours,
)


class SegmentationService:
    def __init__(self, artifacts: SegmentationArtifacts) -> None:
        self._model = artifacts.model
        self._device = artifacts.device

    @torch.no_grad()
    def segment(self, image_base64: str) -> SegmentationResult:
        pil = decode_base64_to_image(image_base64).convert("L")
        original_size = pil.size  # (width, height)

        gray = np.array(pil, dtype=np.uint8)
        gray_1024 = (
            cv2.resize(
                gray,
                (SEGMENTATION_SIZE, SEGMENTATION_SIZE),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            / 255.0
        )

        tensor = (
            torch.from_numpy(gray_1024)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
            .float()
        )

        output = self._model(tensor)
        if isinstance(output, tuple):
            output = output[0]

        points = (
            output.cpu().numpy().reshape(-1, 2) * SEGMENTATION_SIZE
        ).round().astype(int)
        rl, ll, heart = split_contours(points)

        dense_mask = dense_mask_from_contours(rl, ll, heart)
        lung_mask = (dense_mask == LUNG_MASK_VALUE).astype(np.uint8)
        thoracic_mask = (dense_mask > 0).astype(np.uint8)
        bbox = self._bbox_from_mask(thoracic_mask)

        x0, y0, x1, y1 = bbox
        cropped = (gray_1024[y0:y1, x0:x1] * 255).astype(np.uint8)

        return SegmentationResult(
            dense_mask=dense_mask,
            lung_mask=lung_mask,
            thoracic_mask=thoracic_mask,
            bbox=bbox,
            cropped_gray=cropped,
            original_size=original_size,
            contours={"RL": rl, "LL": ll, "H": heart},
        )

    def _bbox_from_mask(
        self,
        mask: np.ndarray,
        padding: int = 16,
    ) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            raise EmptySegmentationMaskError(
                "Segmentation produced an empty thoracic mask"
            )

        x0 = max(0, int(xs.min()) - padding)
        x1 = min(SEGMENTATION_SIZE, int(xs.max()) + padding + 1)
        y0 = max(0, int(ys.min()) - padding)
        y1 = min(SEGMENTATION_SIZE, int(ys.max()) + padding + 1)
        return x0, y0, x1, y1
