"""Reading and writing LUT files. One ``load`` / ``save`` pair, dispatched on extension.

Fidelity (see each module for the exact loss):

- ``.cube``   3D cubic, uniform grid. Text; decimal precision follows the LUT's
  ``bit_depth`` (8 -> 3 digits, ..., 16 -> 5, above 16 -> 9), so a quantised LUT
  round-trips within half a grid step; unquantised uses 6 decimals (~1e-6 max
  error). The ``bit_depth`` tag itself is not stored.
- ``.json``   LOSSLESS: the LUT's compact grid (uint8/uint16/float32) plus explicit
  ``bit_depth``; any n_in, per-axis sizes, rectilinear axes, domain, space.
- ``.npy``    Values without metadata: uint8 (8-bit), uint16 (16-bit) or float32
  (every other depth and unquantised). A bare uint8/uint16/float32 file cannot
  carry ``bit_depth``, so the tag is lost on read; use ``.json`` for it.
- ``.png``    Hald CLUT, in either pixel layout: the canonical one (ImageMagick
  ``hald:N`` -- level N gives an N**3 px side, N**2 values per axis, one pixel
  per node) or the lutlab variant (a ceil(sqrt(L**3)) grid of cell x cell
  pixel blocks). Choose with ``layout=``; the levels/cell size are the
  "sizes" of the pattern. LOSSY: the render is quantised to an 8- or 16-bit
  PNG, so only the 8/16-bit tag is recovered; custom domains and non-uniform
  grids are rejected because the pattern stores neither. Needs OpenCV
  (``pip install "lutlab[png]"``).
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
