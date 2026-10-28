""".npy: the raw grid only. Axes, domain, name and space are NOT stored.

The LUT's compact storage is saved as-is where it is unambiguous: uint8 for
8-bit LUTs, uint16 for 16-bit, float32 for every other depth (1..7, 9..15,
17..32) and for unquantised LUTs -- values exact, but the ``bit_depth`` tag is
lost on read (a bare uint8/uint16/float32 file cannot be told apart from an
8-bit, 16-bit or unquantised one). Use .json (which records ``bit_depth``) when
the tag matters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..transforms.lut import LUT

__all__ = ["read", "write"]


def read(path, *, space: str = "srgb", bit_depth: int | None = None) -> LUT:
    p = Path(path)
    arr = np.load(p)
    n = (8 if arr.dtype == np.uint8 else 16 if arr.dtype == np.uint16 else None) \
        if bit_depth is None else bit_depth
    if n is not None and arr.dtype.kind in "iu":
        arr = arr.astype(np.float32) / float((2 ** n) - 1)
    return LUT(arr, name=p.stem, space=space, bit_depth=n)


def write(lut: LUT, path) -> None:
    if not (lut.is_uniform and lut.has_default_domain):
        raise ValueError(".npy stores the grid only; this LUT has a custom domain or "
                         "non-uniform axes -- use .json")
    grid = lut.raw
    if grid.dtype.kind == "u" and lut.bit_depth != (8 if grid.dtype == np.uint8 else 16):
        grid = lut.volume                     # 1..7 / 9..15-bit: float32, exact
    np.save(Path(path), grid)
