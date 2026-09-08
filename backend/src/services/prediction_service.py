"""Orchestrates classifier inference with optional segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.exceptions import SegmentationNotConfiguredError
from src.core.models.predictor import ChestXRayPredictor
from src.core.utils.image import (
    convert_array_to_tensor,
    convert_crop_and_mask_to_tensor,
    paste_cam_to_full_image,
)
from src.domain.prediction.models import PredictionOutput
from src.hybrid_gnet.inference import SEGMENTATION_SIZE
from src.services.prediction_format import build_prediction_result

if TYPE_CHECKING:
    from src.services.segmentation_service import SegmentationService


class PredictionService:
    """Combines ChestXRayPredictor and optional SegmentationService."""

    MODEL_INPUT_SIZE = 224

    def __init__(
        self,
        predictor: ChestXRayPredictor,
        segmentation_service: SegmentationService | None = None,
        *,
        use_mask: bool = False,
        use_mask_channel: bool = False,
    ) -> None:
        self._predictor = predictor
        self._segmentation = segmentation_service
        self._use_mask = use_mask
        self._use_mask_channel = use_mask_channel

    @property
    def segmentation_available(self) -> bool:
        return self._segmentation is not None

    @property
    def use_mask(self) -> bool:
        return self._use_mask

    @property
    def use_mask_channel(self) -> bool:
        return self._use_mask_channel

    def predict(self, image_base64: str) -> PredictionOutput:
        """
        Run prediction on a full image or on a thoracic crop when ``use_mask``
        is enabled in config.
        """
        if not self._use_mask:
            return self._to_prediction_output(self._predictor.predict(image_base64))

        if self._segmentation is None:
            raise SegmentationNotConfiguredError(
                "Segmentation is not configured. Set segmentation_weights_path in config."
            )

        seg = self._segmentation.segment(image_base64)
        if self._use_mask_channel:
            x0, y0, x1, y1 = seg.bbox
            mask_crop = seg.thoracic_mask[y0:y1, x0:x1]
            image_tensor, _crop_size = convert_crop_and_mask_to_tensor(
                seg.cropped_gray,
                mask_crop,
                self._predictor.device,
                grayscale=self._predictor.grayscale,
            )
        else:
            image_tensor, _crop_size = convert_array_to_tensor(
                seg.cropped_gray,
                self._predictor.device,
                grayscale=self._predictor.grayscale,
            )

        result = self._to_prediction_output(
            self._predictor.predict_tensor(
                image_tensor,
                display_size=(self.MODEL_INPUT_SIZE, self.MODEL_INPUT_SIZE),
            )
        )
        weighted_cam = paste_cam_to_full_image(
            result.weighted_cam,
            bbox=seg.bbox,
            seg_size=SEGMENTATION_SIZE,
            full_size=seg.original_size,
        )
        return PredictionOutput(probs=result.probs, weighted_cam=weighted_cam)

    def predict_for_api(self, image_base64: str) -> dict[str, object]:
        """Run prediction and return the payload expected by ``PredictionResponse``."""
        result = self.predict(image_base64)
        return build_prediction_result(result.probs, result.weighted_cam)

    def segment(self, image_base64: str):
        """Run segmentation only. Raises if segmentation is not configured."""
        if self._segmentation is None:
            raise SegmentationNotConfiguredError(
                "Segmentation is not configured. Set segmentation_weights_path in config."
            )
        return self._segmentation.segment(image_base64)

    @staticmethod
    def _to_prediction_output(result: dict[str, object]) -> PredictionOutput:
        return PredictionOutput(
            probs=np.asarray(result["probs"], dtype=np.float32),
            weighted_cam=np.asarray(result["weighted_cam"], dtype=np.float32),
        )
