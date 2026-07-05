"""Response schemas."""

from typing import Literal

from pydantic import BaseModel

from src.domain.prediction.models import PathologyScore, TriageLevel

PathologyPrediction = PathologyScore


class HealthResponse(BaseModel):
    """Readiness probe response when inference services are loaded."""

    status: Literal["ready"]
    device: str
    segmentation_enabled: bool


class PredictionResponse(BaseModel):
    """Standard prediction endpoint response."""

    predictions: list[PathologyPrediction]
    base_64_heatmap: str


class TriageResponse(BaseModel):
    """Triage endpoint response with triage level and high-risk findings."""

    predictions: list[PathologyPrediction]
    base_64_heatmap: str
    triage_level: TriageLevel
    high_risk_findings: list[PathologyPrediction]


class PathologiesResponse(BaseModel):
    """Ordered pathology names for UI dropdowns and labeling."""

    pathologies: list[str]
