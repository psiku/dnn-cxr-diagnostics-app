"""FastAPI dependency injection — read services from app.state (set in lifespan)."""

from typing import Annotated

from fastapi import Depends, Request

from src.services.description_service import DescriptionService
from src.services.triage_service import XRayTriageService
from src.services.labeling_service import LabelingService
from src.services.prediction_service import PredictionService


async def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


async def get_triage_service(request: Request) -> XRayTriageService:
    return request.app.state.triage_service


async def get_description_service(request: Request) -> DescriptionService:
    return request.app.state.description_service


async def get_labeling_service(request: Request) -> LabelingService:
    return request.app.state.labeling_service


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
TriageServiceDep = Annotated[XRayTriageService, Depends(get_triage_service)]
DescriptionServiceDep = Annotated[DescriptionService, Depends(get_description_service)]
LabelingServiceDep = Annotated[LabelingService, Depends(get_labeling_service)]
