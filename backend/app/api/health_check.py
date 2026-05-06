from fastapi import APIRouter, status

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Sprawdzenie stanu aplikacji",
)
def health_check():
    return {"status": "ok", "message": "System działa poprawnie"}