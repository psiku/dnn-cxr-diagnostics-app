"""HybridGNet segmentation model loading utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.hybrid_gnet.inference import build_hybrid_model
from src.hybrid_gnet.model import Hybrid


@dataclass
class SegmentationArtifacts:
    model: Hybrid
    device: str


def load_segmentation_model(
    weights_path: str, device: str = "cpu"
) -> SegmentationArtifacts:
    """
    Load a HybridGNet segmentation model from a checkpoint.

    Args:
        weights_path: Path to the ``.pt`` weights file.
        device: Device to load the model to (``"cpu"`` or ``"cuda"``).

    Returns:
        Loaded model in eval mode wrapped with its runtime device.
    """
    torch_device = torch.device(device)
    hybrid = build_hybrid_model(torch_device)
    hybrid.load_state_dict(
        torch.load(weights_path, map_location=torch_device, weights_only=False)
    )
    hybrid.eval()

    return SegmentationArtifacts(model=hybrid, device=device)
