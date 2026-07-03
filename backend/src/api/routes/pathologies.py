"""Health and metadata routes."""
from fastapi import APIRouter, status

from src.api.schemas import PathologiesResponse
from src.core.pathologies import PATHOLOGIES

router = APIRouter(tags=["Pathologies"])


@router.get("/pathologies", status_code=status.HTTP_200_OK, response_model=PathologiesResponse)
def list_pathologies():
    """Ordered pathology class names (same order as model output)."""
    return PathologiesResponse(pathologies=list(PATHOLOGIES))
