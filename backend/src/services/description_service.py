"""Application service: save descriptions with files and export PDF."""
from __future__ import annotations

import uuid

from src.domain.annotations.description_model import Description
from src.repositories.description_repository import DescriptionRepository
from src.services.description_pdf import DescriptionPdfBuilder, PdfFiguresMode


class DescriptionService:
    def __init__(
        self,
        repository: DescriptionRepository,
        pdf_builder: DescriptionPdfBuilder,
    ) -> None:
        self._repository = repository
        self._pdf_builder = pdf_builder

    def save_with_attachments(
        self,
        description: Description,
        attachments: dict[str, bytes],
        *,
        filename_for_role: dict[str, str] | None = None,
    ) -> uuid.UUID:
        self._repository.save(
            description,
            attachments,
            filename_for_role=filename_for_role,
        )
        return description.image_id

    def export_pdf(self, image_id: uuid.UUID, *, figures: PdfFiguresMode = "both") -> bytes:
        manifest = self._repository.load_manifest(image_id)
        paths = self._repository.attachment_paths(image_id)
        return self._pdf_builder.build(manifest, paths, figures=figures)
