"""Persist labeled images: image file + meta.json (LabelingManifest)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from src.domain.annotations.labeling_model import LabeledImage
from src.core.utils.image import get_image_suffix


class LabelingManifest(BaseModel):
    """On-disk shape for meta.json (labeled image fields + attachment filenames)."""

    labeled: LabeledImage = Field(..., description="Bounding boxes and metadata.")
    attachment_filenames: dict[str, str] = Field(
        default_factory=dict,
        description="Logical role name -> filename within the label folder.",
    )


class LabelingRepository:
    META_FILENAME = "meta.json"
    LEGACY_ANNOTATIONS_FILENAME = "annotations.json"
    IMAGE_BASENAME = "image"
    IMAGE_ROLE = "image"

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._labels = self._root / "labels"

    def label_dir(self, image_id: uuid.UUID) -> Path:
        return self._labels / str(image_id)

    def save(
        self,
        labeled: LabeledImage,
        image_bytes: bytes,
        uploaded_filename: str,
    ) -> Path:
        folder = self.label_dir(labeled.image_id)
        folder.mkdir(parents=True, exist_ok=True)

        suffix = get_image_suffix(uploaded_filename)
        image_filename = f"{self.IMAGE_BASENAME}{suffix}"
        image_path = folder / image_filename
        image_path.write_bytes(image_bytes)

        manifest = LabelingManifest(
            labeled=labeled,
            attachment_filenames={self.IMAGE_ROLE: image_filename},
        )
        meta_path = folder / self.META_FILENAME
        meta_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        return folder

    def load_manifest(self, image_id: uuid.UUID) -> LabelingManifest:
        folder = self.label_dir(image_id)
        meta_path = folder / self.META_FILENAME
        if meta_path.is_file():
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            return LabelingManifest.model_validate(raw)

        legacy_path = folder / self.LEGACY_ANNOTATIONS_FILENAME
        if legacy_path.is_file():
            return _manifest_from_legacy(legacy_path)

        raise FileNotFoundError(str(meta_path))

    def load_labeled_image(self, image_id: uuid.UUID) -> LabeledImage:
        return self.load_manifest(image_id).labeled

    def attachment_paths(self, image_id: uuid.UUID) -> dict[str, Path]:
        manifest = self.load_manifest(image_id)
        folder = self.label_dir(image_id)
        out: dict[str, Path] = {}
        for role, fname in manifest.attachment_filenames.items():
            p = folder / fname
            if p.is_file():
                out[role] = p
        return out

    def image_path(self, image_id: uuid.UUID) -> Path | None:
        return self.attachment_paths(image_id).get(self.IMAGE_ROLE)


def _manifest_from_legacy(legacy_path: Path) -> LabelingManifest:
    """Read pre-meta.json folders that stored _storage inside annotations.json."""
    raw = json.loads(legacy_path.read_text(encoding="utf-8"))
    storage = raw.pop("_storage", None) or {}
    image_file = storage.get("image_file", "")
    labeled = LabeledImage.model_validate(raw)
    attachment_filenames = {LabelingRepository.IMAGE_ROLE: image_file} if image_file else {}
    return LabelingManifest(labeled=labeled, attachment_filenames=attachment_filenames)
