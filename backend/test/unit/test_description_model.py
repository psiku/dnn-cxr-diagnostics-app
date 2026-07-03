import uuid
from datetime import date, datetime

import pytest

from src.domain.annotations.description_model import Description


def test_description_defaults():
    d = Description(technical_description="Finding in right lower lobe")
    assert d.technical_description == "Finding in right lower lobe"
    assert isinstance(d.image_id, uuid.UUID)
    assert isinstance(d.timestamp, datetime)


def test_description_explicit_fields():
    image_id = uuid.uuid4()
    ts = datetime(2024, 1, 1, 12, 0, 0)
    d = Description(
        technical_description="Note",
        image_id=image_id,
        timestamp=ts,
    )
    assert d.image_id == image_id
    assert d.timestamp == ts


def test_description_image_id_from_string():
    uid = uuid.uuid4()
    d = Description(technical_description="x", image_id=str(uid))
    assert d.image_id == uid


@pytest.mark.parametrize("image_id", [None, "", "not-a-uuid"])
def test_description_invalid_image_id(image_id):
    with pytest.raises(ValueError):
        Description(technical_description="valid text", image_id=image_id)


def test_description_requires_some_report_text():
    with pytest.raises(ValueError):
        Description(
            technical_description="",
            conclusions="",
            description="",
        )


def test_description_whitespace_only_report():
    with pytest.raises(ValueError):
        Description(technical_description="   ", conclusions="   ")


def test_legacy_description_only():
    d = Description(description="Legacy body only")
    assert d.description == "Legacy body only"


def test_conclusions_only():
    d = Description(conclusions="Stable.")
    assert d.conclusions == "Stable."


def test_projection_validation():
    with pytest.raises(ValueError):
        Description(technical_description="x", projection="INVALID")


def test_full_metadata():
    d = Description(
        technical_description="Infiltrate noted.",
        conclusions="Follow-up advised.",
        first_name="Ann",
        surname="Smith",
        date_of_birth=date(1980, 5, 20),
        study_name="CXR-001",
        projection="PA",
        study_date=date(2026, 5, 10),
        heatmap_overlay_alpha=0.4,
    )
    assert d.projection == "PA"
    assert d.heatmap_overlay_alpha == 0.4
