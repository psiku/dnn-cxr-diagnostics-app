"""Prediction endpoints."""

import logging
import time

from fastapi import APIRouter, status

from src.api.dependencies import PredictionServiceDep, TriageServiceDep
from src.api.schemas import ImageRequest, PredictionResponse, TriageResponse
from src.services.prediction_format import build_triage_result

router = APIRouter(tags=["Predictions"])
logger = logging.getLogger(__name__)


@router.post(
    "/predict", status_code=status.HTTP_200_OK, response_model=PredictionResponse
)
async def predict(
    request: ImageRequest,
    prediction_service: PredictionServiceDep,
):
    """
    Run prediction on an X-ray image.

    Returns:
        PredictionResponse with predictions and heatmap
    """
    started = time.perf_counter()
    result = prediction_service.predict_for_api(request.base_64_image)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "POST /predict completed in %.1f ms (use_mask=%s, use_mask_channel=%s)",
        elapsed_ms,
        prediction_service.use_mask,
        prediction_service.use_mask_channel,
    )
    return PredictionResponse(**result)


@router.post("/triage", status_code=status.HTTP_200_OK, response_model=TriageResponse)
async def triage(
    request: ImageRequest,
    prediction_service: PredictionServiceDep,
    triage_service: TriageServiceDep,
):
    """
    Run prediction and triage assessment on an X-ray image.

    Returns:
        TriageResponse with predictions, triage level, and high-risk findings
    """
    started = time.perf_counter()
    prediction = prediction_service.predict(request.base_64_image)
    triage_assessment = triage_service.assess(prediction.probs)
    result = build_triage_result(
        prediction.probs,
        prediction.weighted_cam,
        triage_assessment,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "POST /triage completed in %.1f ms (use_mask=%s, use_mask_channel=%s, "
        "triage_level=%s)",
        elapsed_ms,
        prediction_service.use_mask,
        prediction_service.use_mask_channel,
        triage_assessment.triage_level,
    )
    return TriageResponse(**result)
