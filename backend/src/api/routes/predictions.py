"""Prediction endpoints."""
from fastapi import APIRouter, status

from src.api.dependencies import PredictorDep, TriageServiceDep
from src.api.schemas import ImageRequest, PredictionResponse, PathologyPrediction, TriageResponse
from src.core.pathologies import PATHOLOGIES
from src.core.utils.image import encode_array_to_base64

router = APIRouter(tags=["Predictions"])


def _build_predictions_and_heatmap(probs, weighted_cam):
    predictions = [
        PathologyPrediction(pathology=PATHOLOGIES[i], probability=float(probs[i]))
        for i in range(len(probs))
    ]
    base_64_heatmap = encode_array_to_base64(weighted_cam)
    return predictions, base_64_heatmap


def _build_prediction_models(items):
    return [PathologyPrediction(**item) for item in items]


@router.post("/predict", status_code=status.HTTP_200_OK, response_model=PredictionResponse)
async def predict(request: ImageRequest, predictor: PredictorDep, use_mask: bool = False):
    """
    Run prediction on an X-ray image.

    Args:
        request: ImageRequest with base64-encoded image
        use_mask: Whether to use segmentation mask (not yet supported)

    Returns:
        PredictionResponse with predictions and heatmap
    """
    result = predictor.predict(request.base_64_image, use_mask=use_mask)
    predictions, base_64_heatmap = _build_predictions_and_heatmap(
        result["probs"],
        result["weighted_cam"],
    )

    return PredictionResponse(
        predictions=predictions,
        base_64_heatmap=base_64_heatmap
    )


@router.post("/triage", status_code=status.HTTP_200_OK, response_model=TriageResponse)
async def triage(request: ImageRequest, triage_service: TriageServiceDep):
    """
    Run prediction and triage assessment on an X-ray image.

    Returns:
        TriageResponse with predictions, triage level, and high-risk findings
    """
    result = triage_service.predict_and_triage(request.base_64_image)

    predictions = _build_prediction_models(result["predictions"])
    high_risk_findings = _build_prediction_models(result["high_risk_findings"])
    base_64_heatmap = encode_array_to_base64(result["weighted_cam"])

    return TriageResponse(
        predictions=predictions,
        base_64_heatmap=base_64_heatmap,
        triage_level=result["triage_level"],
        high_risk_findings=high_risk_findings,
    )
