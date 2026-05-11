"""Response schemas."""
from pydantic import BaseModel


class PathologyPrediction(BaseModel):
    """Single pathology prediction."""
    pathology: str
    probability: float


class PredictionResponse(BaseModel):
    """Standard prediction endpoint response."""
    predictions: list[PathologyPrediction]
    base_64_heatmap: str


class TriageResponse(BaseModel):
    """Triage endpoint response with triage level and high-risk findings."""
    predictions: list[PathologyPrediction]
    base_64_heatmap: str
    triage_level: str  # "critical", "high", "medium", "low"
    high_risk_findings: list[PathologyPrediction]
