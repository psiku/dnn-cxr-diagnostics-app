"""Classifier inference — no segmentation orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.core.models.classifier import ChestXRayClassifier
from src.core.utils.image import convert_base64_to_tensor


class ChestXRayPredictor:
    """Wrapper around ChestXRayClassifier for tensor-level inference."""

    def __init__(
        self,
        model: ChestXRayClassifier,
        device: str = "cpu",
        *,
        grayscale: bool = False,
    ):
        """
        Initialize predictor.

        Args:
            model: Loaded ChestXRayClassifier model
            device: Device to run inference on
            grayscale: Whether the classifier expects a single-channel input
        """
        self.model = model
        self.device = device
        self.grayscale = grayscale
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_base64: str) -> dict[str, Any]:
        """Run prediction on a full base64-encoded image."""
        image_tensor, original_size = convert_base64_to_tensor(
            image_base64,
            self.device,
            grayscale=self.grayscale,
        )
        return self.predict_tensor(image_tensor, display_size=original_size)

    @torch.no_grad()
    def predict_tensor(
        self,
        image_tensor: torch.Tensor,
        display_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Run prediction on a preprocessed image tensor."""
        output = self.model(image_tensor)
        logits = output["logits"]
        probs = torch.sigmoid(logits).squeeze(0)
        transition_maps = output["transition_maps"]

        weighted_cam = self._compute_weighted_cam(
            transition_maps=transition_maps,
            probs=probs,
            display_size=display_size,
        )

        return {
            "probs": probs.detach().cpu().numpy(),
            "weighted_cam": weighted_cam,
        }

    @torch.no_grad()
    def _compute_weighted_cam(
        self,
        transition_maps: torch.Tensor,
        probs: torch.Tensor,
        display_size: tuple[int, int],
    ) -> np.ndarray:
        combined_weights = probs @ self.model.prediction.weight  # [D]

        cam = torch.einsum("d,bdhw->bhw", combined_weights, transition_maps)
        cam = F.relu(cam)
        cam = self.model._normalize_map(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=display_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return cam[0].detach().cpu().numpy()
