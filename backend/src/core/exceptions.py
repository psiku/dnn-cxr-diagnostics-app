"""Application-specific errors mapped to HTTP responses in the API layer."""


class InvalidImageError(ValueError):
    """Raised when a base64 payload cannot be decoded into a valid image."""


class SegmentationNotConfiguredError(RuntimeError):
    """Raised when mask-based inference is requested without segmentation weights."""


class EmptySegmentationMaskError(ValueError):
    """Raised when segmentation does not produce a usable thoracic region."""
