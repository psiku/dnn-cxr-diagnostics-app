"""Labeling model."""
import uuid
from datetime import datetime

from pydantic import BaseModel, Field, computed_field, field_validator

from .bounding_box_model import BoundingBox


class LabeledImage(BaseModel):
    image_id: uuid.UUID = Field(..., description="The id of the image.")
    bounding_boxes: list[BoundingBox] = Field(..., description="The bounding boxes of the image.")
    timestamp: datetime = Field(..., description="The timestamp of the image.")

    @field_validator("image_id", mode="before")
    @classmethod
    def validate_image_id(cls, value: object) -> uuid.UUID:
        """Coerce strings to UUID or reject invalid values."""
        if isinstance(value, uuid.UUID):
            return value
        if isinstance(value, str):
            try:
                return uuid.UUID(value)
            except ValueError:
                raise ValueError("Image id must be a valid UUID.") from None
        raise ValueError("Image id must be a valid UUID.")

    @field_validator("bounding_boxes")
    @classmethod
    def validate_bounding_boxes(cls, value: list[BoundingBox]) -> list[BoundingBox]:
        """Validate the bounding boxes."""
        if len(value) == 0:
            raise ValueError("Bounding boxes must be a non-empty list.")
        return value

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime:
        """Validate the timestamp."""
        if value is None:
            raise ValueError("Timestamp must be a non-empty datetime.")
        return value

    @computed_field
    @property
    def number_of_annotations(self) -> int:
        return len(self.bounding_boxes)

    def __repr__(self) -> str:
        return (
            f"LabeledImage(image_id={self.image_id}, bounding_boxes={self.bounding_boxes}, "
            f"timestamp={self.timestamp}, number_of_annotations={self.number_of_annotations})"
        )

    def __str__(self) -> str:
        return self.__repr__()
