"""Prediction endpoints."""
from fastapi import APIRouter, Request, status
from src.api.schemas import ImageRequest, PredictionResponse, TriageResponse, PathologyPrediction
from src.core.utils.image import encode_array_to_base64
from src.services.inference_service import XRayTriageService

router = APIRouter(tags=["Predictions"])


@router.post("/predict", status_code=status.HTTP_200_OK, response_model=PredictionResponse)
async def predict(fastapi_request: Request, request: ImageRequest, use_mask: bool = False):
    """
    Run prediction on an X-ray image.

    Args:
        request: ImageRequest with base64-encoded image
        use_mask: Whether to use segmentation mask (not yet supported)

    Returns:
        PredictionResponse with predictions and heatmap
    """
    predictor = fastapi_request.app.state.predictor

    result = predictor.predict(request.base_64_image, use_mask=use_mask)
    probs = result["probs"]
    weighted_cam = result["weighted_cam"]

    predictions = [
        PathologyPrediction(pathology=XRayTriageService.PATHOLOGIES[i], probability=float(probs[i]))
        for i in range(len(probs))
    ]

    base_64_heatmap = encode_array_to_base64(weighted_cam)

    return PredictionResponse(
        predictions=predictions,
        base_64_heatmap=base_64_heatmap
    )


@router.post("/triage", status_code=status.HTTP_200_OK, response_model=TriageResponse)
async def triage(fastapi_request: Request, request: ImageRequest):
    """
    Run prediction and triage assessment on an X-ray image.

    Returns:
        TriageResponse with predictions, triage level, and high-risk findings
    """
    triage_service = fastapi_request.app.state.triage_service

    result = triage_service.predict_and_triage(request.base_64_image)

    base_64_heatmap = encode_array_to_base64(result["weighted_cam"])

    return TriageResponse(
        predictions=[PathologyPrediction(**p) for p in result["predictions"]],
        base_64_heatmap=base_64_heatmap,
        triage_level=result["triage_level"],
        high_risk_findings=[PathologyPrediction(**f) for f in result["high_risk_findings"]],
    )
