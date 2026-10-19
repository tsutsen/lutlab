"""Reading and writing LUT files. One `load` / `save` pair, dispatched on extension.

    .cube   3D, equal axis sizes, uniform grid (lossy to ~1e-6 at 6 decimals)
    .json   lossless; any n_in, per-axis sizes, rectilinear axes, domain, space
    .npy    raw grid only
"""
from __future__ import annotations

from pathlib import Path

from . import cube, lutjson, npy
from ..transforms.lut import LUT

__all__ = ["load", "save", "formats", "cube", "lutjson", "npy"]

_BY_EXT = {".cube": cube, ".json": lutjson, ".npy": npy}


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
