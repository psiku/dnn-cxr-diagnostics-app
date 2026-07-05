"""Image processing and encoding utilities."""

import base64
import binascii
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import torch
import numpy as np
from torchvision import transforms

from src.core.exceptions import InvalidImageError


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a Base64 string."""
    image_bytes = Path(image_path).read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def encode_array_to_base64(array) -> str:
    """
    Encode a 2D array (like a Grad-CAM heatmap) to a Base64 string.
    Normalizes it and applies a jet colormap.
    """
    arr = np.array(array)
    # Normalize array to 0-255
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    arr = (arr * 255).astype(np.uint8)

    # Convert to PIL Image
    try:
        import matplotlib.pyplot as plt

        colormap = plt.get_cmap("jet")
        heatmap = (colormap(arr)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(heatmap)
    except ImportError:
        img = Image.fromarray(arr)

    # Save to Buffer and encode
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode a Base64 string and return a PIL Image."""
    try:
        image_bytes = base64.b64decode(base64_string, validate=True)
        image = Image.open(BytesIO(image_bytes))
        image.load()
        return image
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Invalid base64 image payload") from exc


def get_image_transform(size: int = 224) -> transforms.Compose:
    """Get standard image preprocessing transform for model input."""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_grayscale_transform(size: int = 224) -> transforms.Compose:
    """Preprocessing for single-channel model input."""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )


def preprocess_pil_to_tensor(
    image_pil: Image.Image,
    size: int = 224,
    grayscale: bool = False,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize, tensorize, and normalize a PIL image for model input."""
    original_size = image_pil.size

    if grayscale:
        image_pil = image_pil.convert("L")
        transform = get_grayscale_transform(size)
    else:
        image_pil = image_pil.convert("RGB")
        transform = get_image_transform(size)

    image_tensor = transform(image_pil).unsqueeze(0)
    return image_tensor, original_size


def convert_array_to_tensor(
    gray_array: np.ndarray,
    device: torch.device,
    size: int = 224,
    grayscale: bool = True,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert a grayscale uint8 crop to a preprocessed model tensor."""
    image_pil = Image.fromarray(gray_array, mode="L")
    image_tensor, crop_size = preprocess_pil_to_tensor(
        image_pil,
        size=size,
        grayscale=grayscale,
    )
    return image_tensor.to(device), crop_size


def paste_cam_to_full_image(
    cam_crop: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    seg_size: int,
    full_size: tuple[int, int],
    region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Map a CAM computed on a cropped region back to the original image size.

    The heatmap is kept as the full bbox rectangle: resize CAM to crop size,
    paste into the segmentation canvas at ``bbox``, then scale to ``full_size``.
    An optional ``region_mask`` can clip the result to an organ contour.
    """
    import cv2

    x0, y0, x1, y1 = bbox
    crop_w, crop_h = x1 - x0, y1 - y0

    cam_on_crop = cv2.resize(
        cam_crop,
        (crop_w, crop_h),
        interpolation=cv2.INTER_LINEAR,
    )

    full_cam = np.zeros((seg_size, seg_size), dtype=np.float32)
    full_cam[y0:y1, x0:x1] = cam_on_crop

    if region_mask is not None:
        full_cam *= region_mask.astype(np.float32)

    orig_w, orig_h = full_size
    return cv2.resize(
        full_cam,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )


def convert_base64_to_tensor(
    image_base64: str,
    device: torch.device,
    size: int = 224,
    grayscale: bool = False,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert base64 image string to preprocessed tensor."""
    image_pil = decode_base64_to_image(image_base64)
    image_tensor, original_size = preprocess_pil_to_tensor(
        image_pil,
        size=size,
        grayscale=grayscale,
    )
    return image_tensor.to(device), original_size


def resize_to_original_size(
    image_tensor: torch.Tensor, dense_mask: np.ndarray
) -> tuple[torch.Tensor, np.ndarray]:
    """Resize image tensor and heatmap to original mask dimensions."""
    original_size = (dense_mask.shape[1], dense_mask.shape[0])  # (width, height)

    resized_image = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0),
        size=original_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    resized_heatmap = (
        torch.nn.functional.interpolate(
            torch.tensor(dense_mask).unsqueeze(0).unsqueeze(0).float(),
            size=original_size,
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )

    return resized_image, resized_heatmap


def get_image_suffix(filename: str | None) -> str:
    """Get the file suffix (extension) of an image file."""
    suffix = Path(filename).suffix.lower() if filename else ""
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"} else ".png"
