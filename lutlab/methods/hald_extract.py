"""Extract a LUT from a Hald CLUT render or a before/after pair.

Thin adapters over the external `robust_hald` package, imported lazily so
``import lutlab`` never needs it. Install it with
``pip install -e /path/to/robust_hald``.
"""
from __future__ import annotations

import numpy as np

from .._deps import require
from ..transforms.lut import LUT
from . import register

__all__ = ["make_hald", "extract_from_hald", "extract_from_pair"]


def make_hald(levels: int = 17, cell: int = 24):
    """Hald CLUT image generator -> ``(image, cell_bin, colors)`` (robust_hald)."""
    return require("robust_hald.hald").make_hald(levels, cell)


def extract_from_hald(J: np.ndarray, levels: int = 17, cell: int = 24, radius: int = 5,
                      patch: int = 4, d: int | None = None, method: str = "grid_pcell",
                      refine: bool = False, I: np.ndarray | None = None, **ref_kw) -> LUT:
    """Read a ``(D, D, D, 3)`` LUT back from a (possibly misaligned) Hald render ``J``.

    ``levels`` / ``cell`` MUST match how ``J`` was rendered. ``method`` is
    ``'grid_pcell'`` (per-cell median read-back after grid-flow alignment) or
    ``'naive'`` (whole-image remap + bin average). ``refine`` adds a
    differentiable refinement. ``d`` is the output grid size (default ``levels``).
    """
    naive = require("robust_hald.naive")
    if d is None:
        d = levels
    ref_hald, cell_bin, _colors = make_hald(levels, cell)
    base = I if I is not None else ref_hald
    if method == "grid_pcell":
        vol = naive.hald_percell_extract(J, levels, cell, cell_bin, radius, patch=patch)
    elif method == "naive":
        vol = naive.naive_extract(base, J, d)
    else:
        raise ValueError(f"unknown method {method!r}; use 'grid_pcell' or 'naive'")
    if refine:
        vol = require("robust_hald.extract").refine_lut(vol, base, ref_hald, **ref_kw)
    return LUT(vol, name=f"hald_{method}_L{levels}")


def extract_from_pair(I: np.ndarray, J: np.ndarray, d: int = 17, method: str = "naive",
                      **kw) -> LUT:
    """Extract a LUT from a before/after pair.

    ``'naive'``: aligned pixels, per-content-bin average. ``'farneback'``:
    optical-flow warp of J onto I, then naive. ``'grid_remap'``: Hald grid-flow
    warp, then naive.
    """
    naive = require("robust_hald.naive")
    fns = {"naive": naive.naive_extract, "farneback": naive.flow_aligned_extract,
           "grid_remap": naive.grid_aligned_extract}
    if method not in fns:
        raise ValueError(f"unknown method {method!r}; options: {', '.join(fns)}")
    return LUT(fns[method](I, J, d, **kw), name=f"extract_{method}")


@register("hald_extract", requires=("robust_hald",))
def build_hald_extract(*, image, **cfg) -> LUT:
    """LUT read back from a Hald render (``image``)."""
    return extract_from_hald(image, **cfg)


@register("pair_extract", requires=("robust_hald",))
def build_pair_extract(*, image, target, **cfg) -> LUT:
    """LUT extracted from an aligned or flow-aligned before/after pair."""
    return extract_from_pair(image, target, **cfg)
