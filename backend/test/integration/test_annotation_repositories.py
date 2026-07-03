"""Tests for annotation repositories and PDF builder. This is an integration test, because it works with the filesystem."""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from src.domain.annotations.bounding_box_model import BoundingBox
from src.domain.annotations.description_model import Description
from src.domain.annotations.labeling_model import LabeledImage
from src.repositories.description_repository import DescriptionRepository
from src.repositories.labeling_repository import LabelingRepository
from src.services.description_pdf import DescriptionPdfBuilder


def test_description_repository_roundtrip(tmp_path: Path):
    root = tmp_path / "ann"
    repo = DescriptionRepository(root)
    desc = Description(technical_description="Finding noted.", image_id=uuid.uuid4())
    repo.save(desc, {"original": b"\x89PNG\r\n\x1a\n", "heatmap": b"\x89PNG\r\n\x1a\n"})

    m = repo.load_manifest(desc.image_id)
    assert m.description.technical_description == "Finding noted."
    assert set(m.attachment_filenames) == {"original", "heatmap"}

    paths = repo.attachment_paths(desc.image_id)
    assert paths["original"].is_file()
    assert paths["heatmap"].is_file()


def test_labeling_repository_roundtrip(tmp_path: Path):
    root = tmp_path / "ann"
    repo = LabelingRepository(root)
    lid = uuid.uuid4()
    ts = datetime.now()
    bb = BoundingBox(label="opacity", x=10, y=10, width=20, height=20)
    labeled = LabeledImage(image_id=lid, bounding_boxes=[bb], timestamp=ts)
    repo.save(labeled, b"\x89PNG\r\n\x1a\n", "shot.png")

    loaded = repo.load_labeled_image(lid)
    assert loaded.image_id == lid
    assert len(loaded.bounding_boxes) == 1
    assert loaded.bounding_boxes[0].label == "opacity"

    img = repo.image_path(lid)
    assert img is not None and img.is_file()

    manifest = repo.load_manifest(lid)
    assert manifest.labeled.image_id == lid
    assert manifest.attachment_filenames == {"image": "image.png"}

    folder = repo.label_dir(lid)
    assert (folder / "meta.json").is_file()
    raw = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert "labeled" in raw
    assert raw["attachment_filenames"]["image"] == "image.png"


def test_labeling_repository_reads_legacy_annotations_json(tmp_path: Path):
    root = tmp_path / "ann"
    repo = LabelingRepository(root)
    lid = uuid.uuid4()
    folder = repo.label_dir(lid)
    folder.mkdir(parents=True)
    (folder / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    legacy = {
        "image_id": str(lid),
        "bounding_boxes": [
            {"label": "opacity", "x": 10, "y": 10, "width": 20, "height": 20}
        ],
        "timestamp": "2026-01-01T12:00:00",
        "_storage": {"image_file": "image.png"},
    }
    (folder / "annotations.json").write_text(json.dumps(legacy), encoding="utf-8")

    loaded = repo.load_labeled_image(lid)
    assert loaded.bounding_boxes[0].label == "opacity"
    assert repo.image_path(lid) is not None


def test_description_pdf_builder_smoke(tmp_path: Path):
    repo = DescriptionRepository(tmp_path / "ann")
    desc = Description(technical_description="Report body.", image_id=uuid.uuid4())
    repo.save(desc, {})
    manifest = repo.load_manifest(desc.image_id)
    paths = repo.attachment_paths(desc.image_id)

    pdf_bytes = DescriptionPdfBuilder().build(manifest, paths)
    assert pdf_bytes.startswith(b"%PDF")
