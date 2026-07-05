"""Paths and limits for on-disk annotation storage."""

import os
from pathlib import Path

MAX_ANNOTATION_UPLOAD_BYTES = 15 * 1024 * 1024


def annotations_data_root() -> Path:
    """Root directory for descriptions, PDFs, and labeled images (created on startup)."""
    env = os.environ.get("ANNOTATIONS_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    backend_root = Path(__file__).resolve().parent.parent.parent
    return backend_root / "data" / "annotations"
