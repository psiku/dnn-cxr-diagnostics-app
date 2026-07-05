"""On-disk manifest for stored clinical descriptions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.annotations.description_model import Description


class DescriptionManifest(BaseModel):
    """Shape of ``meta.json`` for a description folder on disk."""

    description: Description = Field(..., description="Clinical text and ids.")
    attachment_filenames: dict[str, str] = Field(
        default_factory=dict,
        description="Logical role name -> filename within the description folder.",
    )
