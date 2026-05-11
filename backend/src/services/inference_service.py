"""Inference service with triage logic."""
from typing import Any
import numpy as np
from src.core.models.predictor import ChestXRayPredictor


class XRayTriageService:
    """Business logic for X-ray triage and reporting."""

    # Pathology names in the order produced by the trained model.
    PATHOLOGIES = [
        "Infiltration",
        "Effusion",
        "Atelectasis",
        "Nodule",
        "Mass",
        "Pneumothorax",
        "Consolidation",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Emphysema",
        "Edema",
        "Fibrosis",
        "Pneumonia",
        "Hernia",
    ]

    # Fallback threshold used when the checkpoint does not provide per-class values
    DEFAULT_THRESHOLD = 0.5

    # How far above a class threshold escalates that class to "critical"
    CRITICAL_MARGIN = 0.20

    # How far below a class threshold still counts as "medium" risk
    MEDIUM_MARGIN = 0.15

    # Number of simultaneously positive findings that escalate to "critical"
    CRITICAL_FINDING_COUNT = 3

    def __init__(
        self,
        predictor: ChestXRayPredictor,
        thresholds: np.ndarray | None = None,
    ):
        """
        Initialize service with a predictor and optional per-class thresholds.

        Args:
            predictor: Loaded ChestXRayPredictor.
            thresholds: Optional per-pathology probability thresholds, one value
                per class in the order of :attr:`PATHOLOGIES`. When ``None`` a
                uniform fallback threshold is used.
        """
        self.predictor = predictor
        self.thresholds = self._resolve_thresholds(thresholds)

    @classmethod
    def _resolve_thresholds(cls, thresholds: np.ndarray | None) -> np.ndarray:
        num_classes = len(cls.PATHOLOGIES)

        if thresholds is None:
            return np.full(num_classes, cls.DEFAULT_THRESHOLD, dtype=np.float32)

        thresholds = np.asarray(thresholds, dtype=np.float32).reshape(-1)
        if thresholds.shape[0] != num_classes:
            raise ValueError(
                f"Expected {num_classes} thresholds, got {thresholds.shape[0]}"
            )
        return thresholds

    def predict_and_triage(self, image_base64: str) -> dict[str, Any]:
        """
        Run prediction and perform triage.

        Returns:
            Dictionary with predictions, triage level, and recommendations
        """
        result = self.predictor.predict(image_base64)
        probs = result["probs"]
        weighted_cam = result["weighted_cam"]

        predictions = [
            {
                "pathology": self.PATHOLOGIES[i],
                "probability": float(probs[i]),
            }
            for i in range(len(self.PATHOLOGIES))
        ]

        triage_level = self._determine_triage_level(probs)
        high_risk_findings = self._get_high_risk_findings(probs)

        return {
            "predictions": predictions,
            "weighted_cam": weighted_cam,
            "triage_level": triage_level,
            "high_risk_findings": high_risk_findings,
        }

    def _determine_triage_level(self, probs: np.ndarray) -> str:
        """
        Determine triage priority by comparing each probability with its
        per-class threshold.

        Returns:
            "critical", "high", "medium", or "low"
        """
        excess = np.asarray(probs, dtype=np.float32) - self.thresholds
        max_excess = float(np.max(excess))
        positive_count = int(np.sum(excess >= 0.0))

        if max_excess >= self.CRITICAL_MARGIN or positive_count >= self.CRITICAL_FINDING_COUNT:
            return "critical"
        if max_excess >= 0.0:
            return "high"
        if max_excess >= -self.MEDIUM_MARGIN:
            return "medium"
        return "low"

    def _get_high_risk_findings(self, probs: np.ndarray) -> list[dict[str, Any]]:
        """Return classes whose probability meets or exceeds their threshold."""
        findings = []
        for i, prob in enumerate(probs):
            if prob >= self.thresholds[i]:
                findings.append({
                    "pathology": self.PATHOLOGIES[i],
                    "probability": float(prob),
                })
        return sorted(findings, key=lambda x: x["probability"], reverse=True)
