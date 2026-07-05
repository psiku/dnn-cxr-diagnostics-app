"""Triage logic based on model probability outputs."""

from __future__ import annotations

import numpy as np

from src.core.pathologies import PATHOLOGIES
from src.domain.prediction.models import PathologyScore, TriageAssessment, TriageLevel


class XRayTriageService:
    """Business logic for X-ray triage assessment from prediction probabilities."""

    PATHOLOGIES = PATHOLOGIES

    DEFAULT_THRESHOLD = 0.5
    CRITICAL_MARGIN = 0.20
    MEDIUM_MARGIN = 0.15
    CRITICAL_FINDING_COUNT = 3

    def __init__(self, thresholds: np.ndarray | None = None):
        """
        Initialize service with optional per-pathology probability thresholds.

        Args:
            thresholds: Optional per-pathology probability thresholds, one value
                per class in the order of :attr:`PATHOLOGIES`. When ``None`` a
                uniform fallback threshold is used.
        """
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

    def assess(self, probs: np.ndarray) -> TriageAssessment:
        """
        Assess triage priority from model probabilities.

        Returns:
            Structured triage level and high-risk findings.
        """
        return TriageAssessment(
            triage_level=self._determine_triage_level(probs),
            high_risk_findings=self._get_high_risk_findings(probs),
        )

    def _determine_triage_level(self, probs: np.ndarray) -> TriageLevel:
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

    def _get_high_risk_findings(self, probs: np.ndarray) -> list[PathologyScore]:
        """Return classes whose probability meets or exceeds their threshold."""
        findings: list[PathologyScore] = []
        for i, prob in enumerate(probs):
            if prob >= self.thresholds[i]:
                findings.append(
                    PathologyScore(
                        pathology=self.PATHOLOGIES[i],
                        probability=float(prob),
                    )
                )
        return sorted(findings, key=lambda finding: finding.probability, reverse=True)
