import json
import uuid
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app
from src.api.dependencies import get_description_service, get_labeling_service
from src.domain.annotations.description_model import Description
from src.domain.annotations.labeling_model import LabeledImage
from src.repositories.description_repository import DescriptionRepository
from src.repositories.labeling_repository import LabelingRepository
from src.services.description_pdf import DescriptionPdfBuilder
from src.services.description_service import DescriptionService
from src.services.labeling_service import LabelingService


@pytest.fixture
def annotations_services(tmp_path: Path):
    root = tmp_path / "annotations"
    description_service = DescriptionService(
        DescriptionRepository(root),
        DescriptionPdfBuilder(),
    )
    labeling_service = LabelingService(LabelingRepository(root))
    return description_service, labeling_service, root


@pytest.fixture
def client(annotations_services):
    description_service, labeling_service, _ = annotations_services

    app.dependency_overrides[get_description_service] = lambda: description_service
    app.dependency_overrides[get_labeling_service] = lambda: labeling_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _png_bytes(
    color: tuple[int, int, int] = (123, 45, 67), size: tuple[int, int] = (8, 8)
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_create_description_saves_description_and_attachments(
    client, annotations_services
):
    _, _, root = annotations_services

    response = client.post(
        "/annotations/descriptions",
        data={
            "technical_description": "Finding noted.",
            "conclusions": "No acute disease.",
            "first_name": "Jan",
            "surname": "Kowalski",
            "study_name": "Chest PA",
            "projection": "PA",
            "heatmap_overlay_alpha": "0.35",
        },
        files={
            "original": ("study.png", b"original-bytes", "image/png"),
            "heatmap": ("heatmap.png", b"heatmap-bytes", "image/png"),
        },
    )

    assert response.status_code == 201
    image_id = uuid.UUID(response.json()["image_id"])

    repo = DescriptionRepository(root)
    manifest = repo.load_manifest(image_id)
    attachments = repo.attachment_paths(image_id)

    assert manifest.description.technical_description == "Finding noted."
    assert manifest.description.conclusions == "No acute disease."
    assert manifest.description.first_name == "Jan"
    assert manifest.description.surname == "Kowalski"
    assert manifest.description.study_name == "Chest PA"
    assert manifest.description.projection == "PA"
    assert set(attachments) == {"original", "heatmap"}
    assert attachments["original"].read_bytes() == b"original-bytes"
    assert attachments["heatmap"].read_bytes() == b"heatmap-bytes"


def test_download_description_pdf_returns_pdf(client, annotations_services):
    description_service, _, _ = annotations_services
    image_id = uuid.uuid4()
    description = Description(
        image_id=image_id,
        technical_description="Finding noted.",
        conclusions="Stable.",
    )
    original_bytes = _png_bytes((10, 20, 30))
    heatmap_bytes = _png_bytes((200, 40, 50))
    description_service.save_with_attachments(
        description,
        {"original": original_bytes, "heatmap": heatmap_bytes},
        filename_for_role={"original": "study.png", "heatmap": "heatmap.png"},
    )

    response = client.get(f"/annotations/descriptions/{image_id}/pdf?figures=both")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert response.headers["content-disposition"].startswith("attachment;")


def test_save_labeled_image_persists_label_and_image(client, annotations_services):
    _, _, root = annotations_services
    image_id = uuid.uuid4()
    payload = {
        "image_id": str(image_id),
        "bounding_boxes": [
            {"label": "opacity", "x": 10, "y": 10, "width": 20, "height": 20}
        ],
        "timestamp": "2026-01-01T12:00:00",
    }

    response = client.post(
        "/annotations/labels",
        data={"payload": json.dumps(payload)},
        files={"image": ("upload.png", b"image-bytes", "image/png")},
    )

    assert response.status_code == 201
    returned_image_id = uuid.UUID(response.json()["image_id"])
    assert returned_image_id == image_id

    repo = LabelingRepository(root)
    labeled = repo.load_labeled_image(image_id)
    image_path = repo.image_path(image_id)

    assert labeled.image_id == image_id
    assert len(labeled.bounding_boxes) == 1
    assert labeled.bounding_boxes[0].label == "opacity"
    assert image_path is not None
    assert image_path.read_bytes() == b"image-bytes"


def test_save_labeled_image_rejects_invalid_payload(client):
    response = client.post(
        "/annotations/labels",
        data={"payload": "not-json"},
        files={"image": ("upload.png", b"image-bytes", "image/png")},
    )

    assert response.status_code == 422
