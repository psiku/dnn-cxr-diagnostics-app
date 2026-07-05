"""On-disk manifest for stored labeled images."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.annotations.labeling_model import LabeledImage


class LabelingManifest(BaseModel):
    """Shape of ``meta.json`` for a label folder on disk."""

    labeled: LabeledImage = Field(..., description="Bounding boxes and metadata.")
    attachment_filenames: dict[str, str] = Field(
        default_factory=dict,
        description="Logical role name -> filename within the label folder.",
    )
