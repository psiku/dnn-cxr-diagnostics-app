from src.domain.annotations.bounding_box_model import BoundingBox
import pytest


def test_bounding_box_model_init():
    """Test the initialization of the BoundingBox model."""
    # Setup
    bounding_box = BoundingBox(label="test", x=10, y=10, width=10, height=10)

    # Assert
    assert bounding_box.label == "test"
    assert bounding_box.x == 10
    assert bounding_box.y == 10
    assert bounding_box.width == 10
    assert bounding_box.height == 10


@pytest.mark.parametrize(
    "label, x, y, width, height",
    [
        ("test", -10, 10, 10, 10),
        ("test", 10, -10, 10, 10),
        ("test", 10, 10, -10, 10),
        ("test", 10, 10, 10, -10),
    ],
)
def test_bounding_box_model_negative_coordinates(label, x, y, width, height):
    """Test the initialization of the BoundingBox model with negative coordinates."""
    with pytest.raises(ValueError):
        BoundingBox(label=label, x=x, y=y, width=width, height=height)


def test_bounding_box_model_empty_label():
    """Test the initialization of the BoundingBox model with an empty label."""
    with pytest.raises(ValueError):
        BoundingBox(label="", x=10, y=10, width=10, height=10)


def test_bounding_box_model_whitespace_label():
    """Test the initialization of the BoundingBox model with whitespace-only label."""
    with pytest.raises(ValueError):
        BoundingBox(label="   ", x=10, y=10, width=10, height=10)


def test_bounding_box_model_white_characters_label():
    """Test the initialization of the BoundingBox model with a label containing only white characters."""
    with pytest.raises(ValueError):
        BoundingBox(label="\t ", x=10, y=10, width=10, height=10)


def test_bounding_box_label_is_none():
    """Test the initialization of the BoundingBox model with a None label."""
    with pytest.raises(ValueError):
        BoundingBox(label=None, x=10, y=10, width=10, height=10)


def test_bounding_box_model_x_is_negative():
    """Test the initialization of the BoundingBox model with a negative x coordinate."""
    with pytest.raises(ValueError):
        BoundingBox(label="test", x=-1, y=10, width=10, height=10)


def test_bounding_box_model_y_is_negative():
    """Test the initialization of the BoundingBox model with a negative y coordinate."""
    with pytest.raises(ValueError):
        BoundingBox(label="test", x=10, y=-1, width=10, height=10)


def test_bounding_box_model_width_is_negative():
    """Test the initialization of the BoundingBox model with a negative width."""
    with pytest.raises(ValueError):
        BoundingBox(label="test", x=10, y=10, width=-1, height=10)


def test_bounding_box_model_height_is_negative():
    """Test the initialization of the BoundingBox model with a negative height."""
    with pytest.raises(ValueError):
        BoundingBox(label="test", x=10, y=10, width=10, height=-1)


def test_bounding_box_model_to_coordinates():
    """Test the conversion of the BoundingBox model to coordinates."""
    # Setup
    bounding_box = BoundingBox(label="test", x=10, y=10, width=10, height=10)

    # Action
    coordinates = bounding_box.coordinates

    # Assert
    assert coordinates == (5, 5, 15, 15)


def test_bounding_box_model_to_coordinates_with_odd_dimensions():
    """Test the conversion of the BoundingBox model to coordinates with odd dimensions."""
    # Setup
    bounding_box = BoundingBox(label="test", x=10, y=10, width=11, height=11)

    # Action
    coordinates = bounding_box.coordinates

    # Assert
    assert coordinates == (4, 4, 15, 15)


def test_bounding_box_model_to_repr():
    """Test the representation of the BoundingBox model."""
    # Setup
    bounding_box = BoundingBox(label="test", x=10, y=10, width=10, height=10)

    # Action
    repr = bounding_box.__repr__()

    # Assert
    assert repr == "BoundingBox(label=test, x=10, y=10, width=10, height=10)"
