"""FastAPI dependency injection."""
from typing import Annotated
from fastapi import Depends
from src.core.models.predictor import ChestXRayPredictor
from src.services.inference_service import XRayTriageService


async def get_predictor() -> ChestXRayPredictor:
    """Get or create predictor instance from app state."""
    # This will be populated by the app lifespan context
    from fastapi import Request
    from fastapi.requests import Request as FastAPIRequest

    # Placeholder - actual implementation will use app state
    pass


async def get_triage_service(
    predictor: Annotated[ChestXRayPredictor, Depends(get_predictor)]
) -> XRayTriageService:
    """Create triage service instance."""
    return XRayTriageService(predictor)
