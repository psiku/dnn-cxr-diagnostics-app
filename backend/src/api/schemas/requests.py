"""Request schemas."""

from pydantic import BaseModel


class ImageRequest(BaseModel):
    """Base64 encoded image request."""

    base_64_image: str
