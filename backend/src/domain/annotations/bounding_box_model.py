"""Bounding box model."""

from pydantic import Field, BaseModel
from pydantic import field_validator


class BoundingBox(BaseModel):
    """Axis-aligned box: ``x``/``y`` are the center; ``width``/``height`` are full extent."""

    label: str = Field(..., description="The label of the bounding box.")
    x: int = Field(ge=0, description="The x coordinate of the bounding box.")
    y: int = Field(ge=0, description="The y coordinate of the bounding box.")
    width: int = Field(ge=0, description="The width of the bounding box.")
    height: int = Field(ge=0, description="The height of the bounding box.")

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, value: object) -> str:
        """Validate the label."""
        if value is None:
            raise ValueError("Label must be a non-empty string.")
        if str(value).strip() == "":
            raise ValueError("Label must be a non-empty string.")
        return str(value).strip()

    @property
    def coordinates(self) -> tuple[int, int, int, int]:
        """Convert the bounding box to coordinates."""
        x_min = self.x - self.width / 2
        y_min = self.y - self.height / 2
        x_max = self.x + self.width / 2
        y_max = self.y + self.height / 2

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def __repr__(self) -> str:
        return f"BoundingBox(label={self.label}, x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def __str__(self) -> str:
        return self.__repr__()
