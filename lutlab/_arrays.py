"""Small array helpers shared across the package (must stay import-light)."""
from __future__ import annotations

import sys

import numpy as np

__all__ = ["is_torch", "to_float", "quantize"]


def is_torch(x) -> bool:
    """True if `x` is a torch tensor. Never imports torch itself."""
    t = sys.modules.get("torch")
    return t is not None and isinstance(x, t.Tensor)


def to_float(img, dtype=np.float32, *, source_depth: int | None = None) -> np.ndarray:
    """Image -> float array in [0, 1]. uint8 / uint16 are scaled; floats pass through.

    ``source_depth`` says how many bits the integer container actually uses (for
    example a 12-bit image stored in uint16: values 0..4095). It is ignored for
    float images and defaults to the full container width (8 or 16).
    """
    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.floating):
        if source_depth is not None:
            raise ValueError("source_depth applies to integer images only")
        return a.astype(dtype, copy=False)
    if a.dtype in (np.uint8, np.uint16):
        bits = 8 if a.dtype == np.uint8 else 16
        depth = bits if source_depth is None else int(source_depth)
        if not 1 <= depth <= bits:
            raise ValueError(f"source_depth must be between 1 and {bits} for {a.dtype}, "
                             f"got {depth}")
        return (a.astype(np.float64) / float((2 ** depth) - 1)).astype(dtype)
    raise TypeError(f"cannot convert dtype {a.dtype} to float image; "
                    "expected uint8, uint16 or a floating dtype")


def quantize(img, bit_depth, dtype=np.float32) -> np.ndarray:
    """Round a ``[0, 1]`` float array onto the ``2**bit_depth`` level grid (1..32).

    Result values are exactly ``k / (2**bit_depth - 1)`` (the LUT/image ``bit_depth``
    representation). Above 16 bits the result is still stored in the requested float
    dtype, so the precision is that of the dtype (float32 by default).
    """
    bit_depth = int(bit_depth)
    if not 1 <= bit_depth <= 32:
        raise ValueError(f"bit_depth must be between 1 and 32, got {bit_depth}")
    top = (2 ** bit_depth) - 1
    a = np.asarray(img)
    return (np.clip(np.rint(a.astype(np.float64) * top), 0, top) / top).astype(dtype)
