"""Map domain/service exceptions to HTTP responses."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.exceptions import (
    EmptySegmentationMaskError,
    InvalidImageError,
    SegmentationNotConfiguredError,
)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidImageError)
    async def invalid_image_handler(
        _request: Request, exc: InvalidImageError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc) or "Invalid base64 image payload"},
        )

    @app.exception_handler(SegmentationNotConfiguredError)
    async def segmentation_not_configured_handler(
        _request: Request, exc: SegmentationNotConfiguredError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc) or "Segmentation is not configured"},
        )

    @app.exception_handler(EmptySegmentationMaskError)
    async def empty_segmentation_mask_handler(
        _request: Request, exc: EmptySegmentationMaskError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc) or "Segmentation produced an empty mask"},
        )
