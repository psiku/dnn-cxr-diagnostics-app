"""Application service: persist labeled images (bitmap + LabeledImage JSON)."""

from __future__ import annotations

import uuid
from pathlib import Path

from src.domain.annotations.labeling_model import LabeledImage
from src.repositories.labeling_repository import LabelingRepository


class LabelingService:
    def __init__(self, repository: LabelingRepository) -> None:
        self._repository = repository

    def save(
        self,
        labeled: LabeledImage,
        image_bytes: bytes,
        uploaded_filename: str,
    ) -> uuid.UUID:
        self._repository.save(labeled, image_bytes, uploaded_filename)
        return labeled.image_id

    def load(self, image_id: uuid.UUID) -> LabeledImage:
        return self._repository.load_labeled_image(image_id)

    def stored_image_path(self, image_id: uuid.UUID) -> Path | None:
        return self._repository.image_path(image_id)
