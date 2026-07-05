"""Typed models for classifier and triage pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

TriageLevel = Literal["critical", "high", "medium", "low"]


@dataclass(frozen=True)
class PredictionOutput:
    """Raw classifier output before API formatting."""

    probs: np.ndarray
    weighted_cam: np.ndarray


class PathologyScore(BaseModel):
    """Single pathology probability."""

    pathology: str
    probability: float


class TriageAssessment(BaseModel):
    """Triage rules applied to model probabilities."""

    triage_level: TriageLevel
    high_risk_findings: list[PathologyScore] = Field(default_factory=list)
