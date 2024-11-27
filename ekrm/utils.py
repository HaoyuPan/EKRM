import cupy as cp
import numpy as np


def color_to_int(img: cp.ndarray) -> cp.ndarray:
    if img.ndim < 3 or img.shape[-1] < 1:
        raise ValueError("Input image must have at least one color channel.")

    channel_count = img.shape[-1]
    color_int_converter = 256 ** cp.arange(channel_count - 1, -1, -1)
    colors_int = cp.sum(img * color_int_converter, axis=-1)
    return colors_int


def int_to_color(colors_int: cp.ndarray, channel_count: int = 3) -> cp.ndarray:
    if channel_count <= 0:
        raise ValueError("channel_count must be a positive integer.")
    forward = 256 ** cp.arange(channel_count - 1, -1, -1)
    backward = 1 / cp.atleast_2d(forward)
    if backward.shape != (1, channel_count):
        raise AssertionError(
            f"Backward converter shape mismatch: expected (1, {channel_count}), got {backward.shape}")
    colors_int_2d = cp.atleast_2d(colors_int).T
    colors = (colors_int_2d * backward).astype(cp.uint8)
    return colors


def get_unique_colors(img: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    assert len(img.shape) == 3, img.shape
    colors_int = color_to_int(img).reshape(-1)
    colors_int, count = cp.unique(colors_int, return_counts=True)
    assert len(count.shape) == 1, count.shape
    assert len(colors_int.shape) == 1, colors_int.shape
    colors = int_to_color(colors_int, channel_count=img.shape[-1])
    assert len(colors.shape) == 2, colors.shape
    return colors, count


def ensure_numpy(array: np.ndarray | cp.ndarray) -> np.ndarray:
    if isinstance(array, cp.ndarray):
        array = array.get()
    if isinstance(array, np.ndarray):
        return array
    raise TypeError(type(array))
