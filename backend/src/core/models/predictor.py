"""High-level prediction interface."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any
from src.core.models.classifier import ChestXRayClassifier
from src.core.utils.image import convert_base64_to_tensor


class ChestXRayPredictor:
    """Wrapper around ChestXRayClassifier for inference."""

    def __init__(self, model: ChestXRayClassifier, device: str = "cpu"):
        """
        Initialize predictor.

        Args:
            model: Loaded ChestXRayClassifier model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_base64: str, use_mask: bool = False) -> dict[str, Any]:
        """
        Run prediction on an image.

        Args:
            image_base64: Base64 encoded image string
            use_mask: Whether to apply segmentation mask (not yet implemented)

        Returns:
            Dictionary with predictions, probabilities, and heatmap
        """
        if use_mask:
            raise NotImplementedError("Mask-based prediction not yet implemented")

        # Convert image
        image_tensor, original_size = convert_base64_to_tensor(image_base64, self.device)

        # Forward pass
        output = self.model(image_tensor)
        logits = output["logits"]
        probs = torch.sigmoid(logits).squeeze(0)
        transition_maps = output["transition_maps"]

        # Compute weighted CAM
        weighted_cam = self._compute_weighted_cam(
            transition_maps=transition_maps,
            probs=probs,
            image_tensor=image_tensor,
            original_size=original_size,
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
        image_tensor: torch.Tensor,
        original_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Compute weighted Class Activation Map.

        Args:
            transition_maps: [B, D, h, w] from model
            probs: [num_classes] sigmoid probabilities
            image_tensor: [B, 3, H, W] input image

        Returns:
            [H, W] normalized numpy heatmap
        """
        # Weight each channel by its corresponding class probability
        combined_weights = probs @ self.model.prediction.weight  # [D]

        # Compute CAM
        cam = torch.einsum("d,bdhw->bhw", combined_weights, transition_maps)
        cam = F.relu(cam)
        cam = self.model._normalize_map(cam)

        # Resize to input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return cam[0].detach().cpu().numpy()
