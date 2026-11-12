
"""Image loading and RGB pixel conversion helpers."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path):
    """
    Load an image and convert it to an RGB uint8 NumPy array.

    Parameters
    ----------
    path : str or Path
        Path to the input image.

    Returns
    -------
    np.ndarray
        RGB image with shape (height, width, 3), dtype uint8.
    """

    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Đường dẫn không phải file: {path}"
        )

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image_array = np.array(image, dtype=np.uint8, copy=True)

    except Exception as exc:
        raise ValueError(
            f"Không thể đọc ảnh '{path}': {exc}"
        ) from exc

    if (
        image_array.ndim != 3
        or image_array.shape[2] != 3
        or image_array.shape[0] == 0
        or image_array.shape[1] == 0
    ):
        raise ValueError(
            f"Ảnh không có kích thước RGB hợp lệ: "
            f"{image_array.shape}"
        )

    return image_array


def image_to_pixels(image_array):
    """
    Convert an RGB image into an Nx3 float32 pixel matrix.

    Parameters
    ----------
    image_array : np.ndarray
        RGB image with shape (height, width, 3).

    Returns
    -------
    tuple
        pixels : np.ndarray
            Pixel matrix with shape (height * width, 3).
        height : int
            Image height.
        width : int
            Image width.
    """

    arr = np.asarray(image_array)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            "Ảnh cần có dạng H x W x 3 (RGB)."
        )

    height, width, _ = arr.shape

    if height <= 0 or width <= 0:
        raise ValueError(
            "Chiều cao và chiều rộng ảnh phải lớn hơn 0."
        )

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            "Dữ liệu ảnh phải có kiểu số."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "Dữ liệu ảnh chứa NaN hoặc Infinity."
        )

    if np.any(arr < 0) or np.any(arr > 255):
        raise ValueError(
            "Giá trị RGB phải nằm trong khoảng [0, 255]."
        )

    pixels = arr.astype(
        np.float32
    ).reshape(-1, 3)

    return pixels, height, width


def pixels_to_image(pixels, height, width):
    """
    Reshape RGB pixels into a uint8 image.

    Parameters
    ----------
    pixels : np.ndarray
        RGB pixel matrix with shape (height * width, 3).
    height : int
        Target image height.
    width : int
        Target image width.

    Returns
    -------
    np.ndarray
        Reconstructed RGB image with dtype uint8.
    """

    if (
        isinstance(height, bool)
        or not isinstance(height, (int, np.integer))
        or isinstance(width, bool)
        or not isinstance(width, (int, np.integer))
    ):
        raise TypeError(
            "height và width phải là số nguyên."
        )

    if height <= 0 or width <= 0:
        raise ValueError(
            "height và width phải lớn hơn 0."
        )

    arr = np.asarray(pixels)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            "pixels phải có dạng Nx3 RGB."
        )

    expected_pixels = height * width

    if arr.shape[0] != expected_pixels:
        raise ValueError(
            f"Số pixel không khớp: nhận được {arr.shape[0]}, "
            f"mong đợi {expected_pixels}."
        )

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            "Dữ liệu pixel phải có kiểu số."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "Dữ liệu pixel chứa NaN hoặc Infinity."
        )

    image_array = np.clip(
        arr.reshape(height, width, 3),
        0,
        255
    ).astype(np.uint8)

    return image_array
