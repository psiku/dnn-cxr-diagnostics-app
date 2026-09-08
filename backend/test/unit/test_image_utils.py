import pytest
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from src.core.exceptions import InvalidImageError
from src.core.utils.image import (
    decode_base64_to_image,
    encode_image_to_base64,
    encode_array_to_base64,
    get_image_suffix,
    get_image_transform,
    convert_base64_to_tensor,
    convert_crop_and_mask_to_tensor,
    resize_to_original_size,
)


def _make_base64_image(size: tuple[int, int] = (100, 80), color=(123, 45, 67)) -> str:
    """Create an in-memory PNG and return its base64 string."""
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_encode_image_to_base64(tmp_path):
    # Setup
    image_file = tmp_path / "tmp_image.png"
    img = Image.new("RGB", (4, 4), color=(123, 45, 67))
    img.save(image_file, format="PNG")

    # Action
    encoded_string = encode_image_to_base64(str(image_file))

    # Assert
    expected_base64 = base64.b64encode(image_file.read_bytes()).decode("utf-8")
    assert encoded_string == expected_base64


def test_decode_base64_to_image_roundtrip():
    # Setup
    original = Image.new("RGB", (3, 5), color=(0, 0, 255))
    buffer = BytesIO()
    original.save(buffer, format="PNG")
    base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Action
    decoded = decode_base64_to_image(base64_string)

    # Assert
    assert isinstance(decoded, Image.Image)
    assert decoded.size == original.size
    assert decoded.convert("RGB").getpixel((0, 0)) == (0, 0, 255)


def test_decode_base64_to_image_rejects_invalid_payload():
    with pytest.raises(InvalidImageError, match="Invalid base64 image payload"):
        decode_base64_to_image("not-valid-base64-image")


def test_enocde_array_to_base64():
    # Setup
    array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # Action
    encoded_string = encode_array_to_base64(array)

    # Assert
    assert isinstance(encoded_string, str)


def test_get_image_transform():
    # Setup
    transform = get_image_transform()

    # Assert
    assert isinstance(transform, transforms.Compose)
    # must have transforms
    assert any(isinstance(t, transforms.Resize) for t in transform.transforms)
    assert any(isinstance(t, transforms.ToTensor) for t in transform.transforms)
    assert any(isinstance(t, transforms.Normalize) for t in transform.transforms)


def test_convert_base64_to_tensor_rgb():
    # Setup
    image_base64 = _make_base64_image(size=(100, 80))

    # Action
    tensor, original_size = convert_base64_to_tensor(
        image_base64,
        torch.device("cpu"),
        grayscale=False,
    )

    # Assert
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert original_size == (100, 80)


def test_convert_base64_to_tensor_grayscale():
    # Setup
    image_base64 = _make_base64_image(size=(100, 80))

    # Action
    tensor, original_size = convert_base64_to_tensor(
        image_base64,
        torch.device("cpu"),
        grayscale=True,
    )

    # Assert
    assert tensor.shape == (1, 1, 224, 224)
    assert original_size == (100, 80)


def test_convert_crop_and_mask_to_tensor_stacks_mask_channel():
    gray = np.full((40, 50), 128, dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[5:30, 5:40] = 255

    tensor, crop_size = convert_crop_and_mask_to_tensor(
        gray,
        mask,
        torch.device("cpu"),
        grayscale=False,
    )

    assert tensor.shape == (1, 4, 224, 224)
    assert crop_size == (50, 40)
    assert float(tensor[0, 3].min()) >= 0.0
    assert float(tensor[0, 3].max()) <= 1.0
    assert float(tensor[0, 3].max()) > 0.0


def test_resize_to_original_size():
    # Setup
    image_tensor = torch.randn(3, 224, 224)
    dense_mask = np.random.rand(400, 400)

    # Action
    resized_image, resized_heatmap = resize_to_original_size(image_tensor, dense_mask)

    # Assert
    assert resized_image.shape == (3, 400, 400)
    assert resized_heatmap.shape == (400, 400)


def test_image_suffix_supported():
    assert get_image_suffix("image.png") == ".png"
    assert get_image_suffix("image.jpg") == ".jpg"
    assert get_image_suffix("image.jpeg") == ".jpeg"
    assert get_image_suffix("image.webp") == ".webp"
    assert get_image_suffix("image.gif") == ".gif"


def test_image_suffix_unsupported():
    assert get_image_suffix("image.bmp") == ".png"
    assert get_image_suffix("image.tiff") == ".png"
    assert get_image_suffix("image") == ".png"


def test_image_suffix_none():
    assert get_image_suffix(None) == ".png"
