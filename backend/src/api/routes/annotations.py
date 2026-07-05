"""Endpoints for clinical descriptions (with PDF export) and image labeling."""

from __future__ import annotations

import uuid
from datetime import date

from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from src.api.dependencies import DescriptionServiceDep, LabelingServiceDep
from src.core.annotations_config import MAX_ANNOTATION_UPLOAD_BYTES
from src.domain.annotations.description_model import Description
from src.domain.annotations.labeling_model import LabeledImage

router = APIRouter(prefix="/annotations", tags=["Annotations"])


async def _read_upload(file: UploadFile | None, *, limit: int) -> bytes | None:
    if file is None:
        return None
    data = await file.read()
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {limit} bytes",
        )
    return data


def _parse_optional_date(label: str, raw: str | None) -> date | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{label} must be an ISO date (YYYY-MM-DD)",
        ) from e


@router.post("/descriptions", status_code=status.HTTP_201_CREATED)
async def create_description(
    description_service: DescriptionServiceDep,
    technical_description: str = Form("", description="Technical / findings text"),
    conclusions: str = Form("", description="Conclusions"),
    description: str = Form("", description="Legacy combined notes"),
    first_name: str = Form("", description="Patient given name"),
    surname: str = Form("", description="Patient surname"),
    date_of_birth: str | None = Form(None, description="YYYY-MM-DD"),
    study_name: str = Form("", description="Study title"),
    projection: str | None = Form(None, description="AP, PA, or L"),
    study_date: str | None = Form(None, description="YYYY-MM-DD"),
    heatmap_overlay_alpha: float = Form(
        0.35, description="0–1 blend for PDF composite"
    ),
    image_id: str | None = Form(
        None, description="Optional UUID; generated if omitted"
    ),
    original: UploadFile | None = File(None),
    heatmap: UploadFile | None = File(None),
):
    """Store structured report fields plus optional original and heatmap images on disk."""
    uid: uuid.UUID | None = None
    if image_id is not None and image_id.strip() != "":
        try:
            uid = uuid.UUID(image_id)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="image_id must be a valid UUID",
            ) from e

    date_of_birth = _parse_optional_date("date_of_birth", date_of_birth)
    study_date = _parse_optional_date("study_date", study_date)

    kwargs = dict(
        technical_description=technical_description,
        conclusions=conclusions,
        description=description,
        first_name=first_name,
        surname=surname,
        date_of_birth=date_of_birth,
        study_name=study_name,
        projection=projection or "",
        study_date=study_date,
        heatmap_overlay_alpha=heatmap_overlay_alpha,
    )

    try:
        desc = (
            Description(image_id=uid, **kwargs)
            if uid is not None
            else Description(**kwargs)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e

    attachments: dict[str, bytes] = {}
    filename_for: dict[str, str] = {}

    orig_bytes = await _read_upload(original, limit=MAX_ANNOTATION_UPLOAD_BYTES)
    if orig_bytes:
        attachments["original"] = orig_bytes
        if original and original.filename:
            filename_for["original"] = original.filename

    heat_bytes = await _read_upload(heatmap, limit=MAX_ANNOTATION_UPLOAD_BYTES)
    if heat_bytes:
        attachments["heatmap"] = heat_bytes
        if heatmap and heatmap.filename:
            filename_for["heatmap"] = heatmap.filename

    description_service.save_with_attachments(
        desc,
        attachments,
        filename_for_role=filename_for or None,
    )
    return {"image_id": str(desc.image_id)}


@router.get("/descriptions/{image_id}/pdf")
async def download_description_pdf(
    description_service: DescriptionServiceDep,
    image_id: uuid.UUID,
    figures: Literal["none", "original", "composite", "both"] = Query(
        "both",
        description=(
            "Which images to attach: none (text only); original; composite (blended study+heatmap); "
            "both (original page then overlay page when files allow)."
        ),
    ),
):
    """PDF report; choose which figure pages to include."""
    try:
        pdf_bytes = description_service.export_pdf(image_id, figures=figures)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Description not found"
        )
    filename = f"description-{image_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/labels", status_code=status.HTTP_201_CREATED)
async def save_labeled_image(
    labeling_service: LabelingServiceDep,
    payload: str = Form(..., description="JSON for LabeledImage"),
    image: UploadFile = File(...),
):
    """Store the uploaded image and annotations.json produced from the labeling UI."""
    try:
        labeled = LabeledImage.model_validate_json(payload)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid labeling payload: {e}",
        ) from e

    image_bytes = await _read_upload(image, limit=MAX_ANNOTATION_UPLOAD_BYTES)
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file is empty",
        )

    name = image.filename or "upload.png"
    labeling_service.save(labeled, image_bytes, name)
    return {"image_id": str(labeled.image_id)}
