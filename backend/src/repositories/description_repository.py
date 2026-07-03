"""Persist clinical descriptions and optional image attachments as JSON + files."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from src.domain.annotations.description_model import Description
from src.core.utils.image import get_image_suffix


class DescriptionManifest(BaseModel):
    """On-disk shape for meta.json (description fields + attachment filenames)."""

    description: Description = Field(..., description="Clinical text and ids.")
    attachment_filenames: dict[str, str] = Field(
        default_factory=dict,
        description="Logical role name -> filename within the description folder.",
    )


class DescriptionRepository:
    META_FILENAME = "meta.json"
    ALLOWED_ATTACHMENT_ROLES = frozenset({"original", "heatmap"})

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._descriptions = self._root / "descriptions"

    def description_dir(self, image_id: uuid.UUID) -> Path:
        return self._descriptions / str(image_id)

    def save(
        self,
        description: Description,
        attachments: dict[str, bytes],
        *,
        filename_for_role: dict[str, str] | None = None,
    ) -> Path:
        """
        Write meta.json and attachment files under descriptions/{image_id}/.

        attachments keys must be subset of ALLOWED_ATTACHMENT_ROLES.
        Files are stored as original.<ext> / heatmap.<ext> unless filename_for_role overrides basename.
        """
        bad = set(attachments) - self.ALLOWED_ATTACHMENT_ROLES
        if bad:
            raise ValueError(f"Unsupported attachment roles: {sorted(bad)}")

        folder = self.description_dir(description.image_id)
        folder.mkdir(parents=True, exist_ok=True)

        filename_for_role = filename_for_role or {}
        attachment_filenames: dict[str, str] = {}

        for role, data in attachments.items():
            ext = get_image_suffix(filename_for_role.get(role))
            basename = f"{role}{ext}"
            path = folder / basename
            path.write_bytes(data)
            attachment_filenames[role] = basename

        manifest = DescriptionManifest(
            description=description,
            attachment_filenames=attachment_filenames,
        )
        meta_path = folder / self.META_FILENAME
        meta_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        return folder

    def load_manifest(self, image_id: uuid.UUID) -> DescriptionManifest:
        folder = self.description_dir(image_id)
        meta_path = folder / self.META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(str(meta_path))
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        return DescriptionManifest.model_validate(raw)

    def attachment_paths(self, image_id: uuid.UUID) -> dict[str, Path]:
        manifest = self.load_manifest(image_id)
        folder = self.description_dir(image_id)
        out: dict[str, Path] = {}
        for role, fname in manifest.attachment_filenames.items():
            p = folder / fname
            if p.is_file():
                out[role] = p
        return out
