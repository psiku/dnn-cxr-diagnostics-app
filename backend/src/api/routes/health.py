"""Health and metadata routes."""
from fastapi import APIRouter, status

from src.api.schemas import PathologiesResponse
from src.core.pathologies import PATHOLOGIES

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
