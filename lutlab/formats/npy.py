""".npy: the raw grid only. Axes, domain, name and space are NOT stored."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..transforms.lut import LUT

__all__ = ["read", "write"]


def read(path, *, space: str = "srgb") -> LUT:
    p = Path(path)
    return LUT(np.load(p), name=p.stem, space=space)


def write(lut: LUT, path) -> None:
    if not (lut.is_uniform and lut.has_default_domain):
        raise ValueError(".npy stores the grid only; this LUT has a custom domain or "
                         "non-uniform axes -- use .json")
    np.save(Path(path), lut.vol)
