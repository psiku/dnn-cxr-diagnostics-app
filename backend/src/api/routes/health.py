"""Health and readiness routes."""

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    response_model=HealthResponse,
)
def health_check(request: Request) -> HealthResponse:
    """
    Readiness probe.

    Returns 503 until classifier and triage services are loaded at startup.
    """
    prediction_service = getattr(request.app.state, "prediction_service", None)
    triage_service = getattr(request.app.state, "triage_service", None)
    device = getattr(request.app.state, "device", None)
    segmentation_service = getattr(request.app.state, "segmentation_service", None)

    if prediction_service is None or triage_service is None or device is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference services are not ready",
        )

    return HealthResponse(
        status="ready",
        device=device,
        segmentation_enabled=segmentation_service is not None,
    )
