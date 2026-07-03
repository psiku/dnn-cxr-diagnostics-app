"""Description model for clinical reports tied to an image."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class Description(BaseModel):
    """Structured report fields plus optional legacy single `description` text."""

    image_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Correlation id for attachments and labeling.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Record creation time.",
    )

    first_name: str = ""
    surname: str = ""
    date_of_birth: date | None = None
    study_name: str = ""
    projection: str = ""
    study_date: date | None = None
    technical_description: str = ""
    conclusions: str = ""
    description: str = Field(
        default="",
        description="Legacy free-text field; prefer technical_description and conclusions.",
    )

    heatmap_overlay_alpha: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Blend factor used when composing heatmap over the study for PDF.",
    )

    @field_validator("first_name", "surname", "study_name", mode="before")
    @classmethod
    def strip_optional_str(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("technical_description", "conclusions", "description", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("projection", mode="before")
    @classmethod
    def validate_projection(cls, value: object) -> str:
        if value is None:
            return ""
        s = str(value).strip().upper()
        if s == "":
            return ""
        if s not in ("AP", "PA", "L"):
            raise ValueError("projection must be empty or one of: AP, PA, L")
        return s

    @field_validator("image_id", mode="before")
    @classmethod
    def validate_image_id(cls, value: object) -> uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value
        if isinstance(value, str):
            try:
                return uuid.UUID(value)
            except ValueError:
                raise ValueError("Image id must be a valid UUID.") from None
        raise ValueError("Image id must be a valid UUID.")

    @model_validator(mode="after")
    def at_least_one_report_section(self) -> Description:
        has_legacy = bool(self.description.strip())
        has_technical = bool(self.technical_description.strip())
        has_conclusions = bool(self.conclusions.strip())
        if not (has_legacy or has_technical or has_conclusions):
            raise ValueError(
                "Provide at least one of: technical description, conclusions, or legacy description text."
            )
        return self

    def __repr__(self) -> str:
        return (
            f"Description(image_id={self.image_id}, study_name={self.study_name!r}, "
            f"timestamp={self.timestamp})"
        )
