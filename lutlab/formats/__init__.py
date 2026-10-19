"""Reading and writing LUT files. One ``load`` / ``save`` pair, dispatched on extension.

Fidelity (see each module for the exact loss):

- ``.cube``   3D cubic, uniform grid. LOSSY: 6-decimal float formatting (~1e-6 max error).
- ``.json``   LOSSLESS: any n_in, per-axis sizes, rectilinear axes, domain, space.
- ``.npy``    LOSSLESS raw grid (no metadata).
- ``.png``    Hald CLUT (ImageMagick ``hald:N``). LOSSY: quantised to the LUT's
  ``bit_depth`` (8- or 16-bit PNG; max error 1/(2**bit_depth-1)) of a render onto
  the standard Hald pattern. Needs OpenCV.
"""
from __future__ import annotations

from pathlib import Path

from . import cube, hald, lutjson, npy
from ..transforms.lut import LUT

__all__ = ["load", "save", "formats", "cube", "hald", "lutjson", "npy"]

_BY_EXT = {".cube": cube, ".json": lutjson, ".npy": npy, ".png": hald}


def formats() -> list[str]:
    return sorted(_BY_EXT)


def _module(path, fmt):
    ext = ("." + fmt.lstrip(".")) if fmt else Path(path).suffix.lower()
    try:
        return _BY_EXT[ext]
    except KeyError:
        raise ValueError(f"unknown LUT format {ext!r}; supported: {', '.join(formats())}") from None


def load(path, *, format: str | None = None, **kw) -> LUT:
    """Load a LUT. ``format`` overrides the file extension."""
    return _module(path, format).read(path, **kw)


def save(lut: LUT, path, *, format: str | None = None, **kw) -> None:
    """Save a LUT. ``format`` overrides the file extension."""
    _module(path, format).write(lut, path, **kw)
