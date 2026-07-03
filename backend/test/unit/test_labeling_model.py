from src.domain.annotations.labeling_model import LabeledImage
import pytest
import uuid
from datetime import datetime
from src.domain.annotations.bounding_box_model import BoundingBox


def test_labeled_image_model_init():
    """Test the initialization of the LabeledImage model."""
    # Setup
    bounding_box = BoundingBox(label="test", x=10, y=10, width=10, height=10)
    image_id = uuid.uuid4()
    timestamp = datetime.now()
    labeled_image = LabeledImage(image_id=image_id, bounding_boxes=[bounding_box], timestamp=timestamp)

    # Assert
    assert labeled_image.image_id == image_id
    assert labeled_image.bounding_boxes == [bounding_box]
    assert labeled_image.timestamp == timestamp
    assert labeled_image.number_of_annotations == 1


def test_labeled_image_model_empty_bounding_boxes():
    """Test the initialization of the LabeledImage model with empty bounding boxes."""
    with pytest.raises(ValueError):
        LabeledImage(image_id=uuid.uuid4(), bounding_boxes=[], timestamp=datetime.now())


@pytest.mark.parametrize("image_id", [None, "", "invalid"])
def test_labeled_image_model_invalid_image_id(image_id):
    """Test the initialization of the LabeledImage model with an invalid image id."""

    with pytest.raises(ValueError):
        LabeledImage(image_id=image_id, bounding_boxes=[BoundingBox(label="test", x=10, y=10, width=10, height=10)], timestamp=datetime.now())