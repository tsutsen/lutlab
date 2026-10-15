"""LUT extraction (re-exported from `robust_hald`). One import point.

Two convenience entry points, plus every low-level routine re-exported so
nothing is blocked:

::

    hald_extract(J, ...)      -- the Art 3 pipeline: take a (misaligned) Hald
                                 render, align it with the colour-invariant
                                 grid-flow, read the LUT back per-cell, and
                                 optionally refine differentiably.
    extract(I, J, d, method)  -- aligned or flow-aligned before/after pair.

`robust_hald` is a namespace package found on `sys.path` (see `lutlab/__init__.py`);
importing it does not modify it -- we only re-export.
"""
from __future__ import annotations

import numpy as np

from robust_hald.naive import (naive_extract, farneback_flow,
                               flow_aligned_extract, grid_aligned_extract,
                               hald_percell_extract)
from robust_hald.correspond import structure_patches, fuzzy_correspond
from robust_hald.extract import cluster_extract, barycentric_extract, refine_lut
from robust_hald.flow import grid_flow
from .lut import LUT

__all__ = ["hald_extract", "extract", "naive_extract", "farneback_flow",
           "flow_aligned_extract", "grid_aligned_extract", "hald_percell_extract",
           "structure_patches", "fuzzy_correspond", "cluster_extract",
           "barycentric_extract", "refine_lut", "grid_flow"]


def hald_extract(J: np.ndarray, levels: int = 17, cell: int = 24,
                 radius: int = 5, patch: int = 4, d: int | None = None,
                 method: str = "grid_pcell", refine: bool = False,
                 I: np.ndarray | None = None, **ref_kw) -> LUT:
    """Extract a `(D, D, D, 3)` LUT from a (possibly misaligned) Hald render.

    Takes the Hald/CLUT as it appears after the misalignment (`J`) and reads
    the LUT back. Key arguments:

    - `levels` / `cell` MUST match how `J` was rendered.
    - `method` = 'grid_pcell' (per-cell median read-back after grid-flow
      alignment, the robust Art 3 M2b path) or 'naive' (whole-image remap +
      bin average).
    - `refine` runs the optional differentiable `refine_lut` on top.
    - `d` is the LUT grid size (default: `levels`).

    Returns a `lutlab.LUT`.
    """
    if d is None:
        d = levels
    from robust_hald.hald import make_hald
    ref_hald, cell_bin, _colors = make_hald(levels, cell)
    base = I if I is not None else ref_hald
    if method == "grid_pcell":
        vol = hald_percell_extract(J, levels, cell, cell_bin, radius, patch=patch)
    elif method == "naive":
        vol = naive_extract(base, J, d)
    else:
        raise ValueError(f"unknown method {method!r}; use 'grid_pcell' or 'naive'")
    if refine:
        vol = refine_lut(vol, base, ref_hald, **ref_kw)
    return LUT(vol, name=f"hald_{method}_L{levels}")


def extract(I: np.ndarray, J: np.ndarray, d: int = 17,
            method: str = "naive", **kw) -> LUT:
    """Extract a LUT from a before/after pair (aligned or flow-aligned).

    method : 'naive' | 'farneback' | 'grid_remap'
        'naive'      = aligned pixels, per-content-bin average.
        'farneback'  = optical-flow warp of J onto I, then naive.
        'grid_remap' = Hald grid-flow warp of J onto I, then naive.
    """
    if method == "naive":
        vol = naive_extract(I, J, d, **kw)
    elif method == "farneback":
        vol = flow_aligned_extract(I, J, d, **kw)
    elif method == "grid_remap":
        vol = grid_aligned_extract(I, J, d, **kw)
    else:
        raise ValueError(f"unknown method {method!r}")
    return LUT(vol, name=f"extract_{method}")
