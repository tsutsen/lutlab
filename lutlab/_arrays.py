"""Small array helpers shared across the package (must stay import-light)."""
from __future__ import annotations

import sys

import numpy as np

__all__ = ["is_torch", "to_float", "quantize"]


def is_torch(x) -> bool:
    """True if `x` is a torch tensor. Never imports torch itself."""
    t = sys.modules.get("torch")
    return t is not None and isinstance(x, t.Tensor)


def to_float(img, dtype=np.float32) -> np.ndarray:
    """Image -> float array in [0, 1]. uint8 / uint16 are scaled; floats pass through."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return (a.astype(np.float64) / 255.0).astype(dtype)
    if a.dtype == np.uint16:
        return (a.astype(np.float64) / 65535.0).astype(dtype)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(dtype, copy=False)
    raise TypeError(f"cannot convert dtype {a.dtype} to float image; "
                    "expected uint8, uint16 or a floating dtype")


def quantize(img, bit_depth, dtype=np.float32) -> np.ndarray:
    """Round a ``[0, 1]`` float array onto the ``2**bit_depth`` level grid.

    Result values are exactly ``k / (2**bit_depth - 1)`` (the LUT/image ``bit_depth``
    representation).
    """
    bit_depth = int(bit_depth)
    if bit_depth < 1:
        raise ValueError(f"bit_depth must be >= 1, got {bit_depth}")
    top = (2 ** bit_depth) - 1
    a = np.asarray(img)
    return (np.clip(np.rint(a.astype(np.float64) * top), 0, top) / top).astype(dtype)
