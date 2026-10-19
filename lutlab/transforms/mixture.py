"""`LUTMixture`: a weighted blend of LUTs, ``y = sum_s w_s * LUT_s(x)``.

Covers basis-LUT methods (global weights predicted per image) and segmented
LUTs (per-pixel masks). Weights are passed as ``context=`` with shape
``(S,)`` or ``(..., S)`` (channel-last), or bound at construction.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Cost
from .lut import LUT

__all__ = ["LUTMixture"]


class LUTMixture:
    def __init__(self, luts: Sequence[LUT], name: str = "mixture", weights=None) -> None:
        luts = list(luts)
        if not luts:
            raise ValueError("LUTMixture needs at least one LUT")
        if len({l.n_in for l in luts}) != 1:
            raise ValueError("all LUTs in a mixture must have the same number of inputs")
        self.luts = luts
        self.name = name
        self.weights = weights
        self.n_in = luts[0].n_in

    @property
    def n_luts(self) -> int:
        return len(self.luts)

    def __call__(self, x, *, context=None, interp: str = "multilinear",
                 backend: str = "auto", device=None, check_finite=None, **_kw):
        from ..kernels.dispatch import apply
        from .._arrays import is_torch
        w = context if context is not None else self.weights
        if w is None:
            raise ValueError("LUTMixture needs weights: pass context=(..., n_luts) or bind weights=")
        if is_torch(x):
            import torch
            w = torch.as_tensor(w, device=x.device, dtype=x.dtype)
        else:
            w = np.asarray(w, dtype=np.float32)
        if w.shape[-1] != self.n_luts:
            raise ValueError(f"weights have {w.shape[-1]} channels for {self.n_luts} LUTs")
        out = None
        for s, lut in enumerate(self.luts):
            y = apply(lut, x, interp=interp, backend=backend, device=device,
                      check_finite=check_finite)
            term = y * w[..., s:s + 1]
            out = term if out is None else out + term
        return out

    def cost(self) -> Cost:
        return Cost(params=sum(l.volume.size for l in self.luts),
                    nbytes=sum(l.nbytes for l in self.luts),
                    reads_per_px=sum(2 ** l.n_in for l in self.luts),
                    meta={"n_luts": self.n_luts})

    def resample(self, grid_size: int = 17, *, weights=None, **_kw) -> LUT:
        """Collapse to one LUT using GLOBAL weights ``(n_luts,)`` (exact when grids match)."""
        w = weights if weights is not None else self.weights
        if w is None or np.asarray(w).ndim != 1:
            raise ValueError("resample() needs global weights of shape (n_luts,)")
        first = self.luts[0]
        for l in self.luts[1:]:
            if l.sizes != first.sizes or not _same_axes(l, first):
                raise ValueError("resample() needs all LUTs on the same grid")
        stack = np.stack([l.volume for l in self.luts])
        blended = np.tensordot(np.asarray(w, np.float32), stack, axes=1)
        return first.replace(volume=blended, name=self.name).resample(grid_size)


def _same_axes(a: LUT, b: LUT) -> bool:
    if a.is_uniform and b.is_uniform:
        return True
    if a.is_uniform != b.is_uniform:
        return False
    return all(np.array_equal(x, y) for x, y in zip(a.axes, b.axes))
