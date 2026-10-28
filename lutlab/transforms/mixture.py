"""`LUTMixture`: a weighted blend of LUTs, ``y = sum_s w_s * LUT_s(x)``.

Covers basis-LUT methods (global weights predicted per image) and segmented
LUTs (per-pixel masks). Weights are passed as ``context=`` with shape
``(S,)`` or ``(..., S)`` (channel-last), or bound at construction.

Global weights on a shared grid/domain are baked into one LUT before lookup.
Per-pixel weights use one-pass numpy/torch kernels; the numba and fused paths
keep the per-LUT loop.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._arrays import is_torch
from .._deps import have
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

    def _same_grid_domain(self) -> bool:
        first = self.luts[0]
        return all(l.sizes == first.sizes and l.domain == first.domain
                   and _same_axes(l, first) for l in self.luts[1:])

    def _one_pass(self, x, backend: str, device) -> bool:
        if backend == "torch":
            return True
        if backend == "auto":
            if is_torch(x) or (device is not None and str(device).split(":")[0] != "cpu"):
                return True
            # shortcut: numba keeps the multi-threaded per-LUT loop (a dedicated
            # S-loop kernel is a separate change); fused stays multi-pass too.
            return not (have("numba") and self.luts[0].is_uniform)
        if backend == "numpy":
            return not is_torch(x) and (device is None
                                        or str(device).split(":")[0] == "cpu")
        return False

    def __call__(self, x, *, context=None, interp: str = "multilinear",
                 backend: str = "auto", device=None, check_finite=None, **_kw):
        from ..kernels.dispatch import apply, apply_mixture
        w = context if context is not None else self.weights
        if w is None:
            raise ValueError("LUTMixture needs weights: pass context=(..., n_luts) or bind weights=")
        if is_torch(x):
            import torch
            w = torch.as_tensor(w, device=x.device, dtype=x.dtype)
        else:
            w = w.detach().cpu().numpy() if is_torch(w) else np.asarray(w, dtype=np.float32)
        if w.shape[-1] != self.n_luts:
            raise ValueError(f"weights have {w.shape[-1]} channels for {self.n_luts} LUTs")

        if self._same_grid_domain():
            first = self.luts[0]
            if w.ndim == 1 and not (is_torch(w) and w.requires_grad):
                # Global weights: blend the grids once, then do one lookup.
                weights = w.detach().cpu().numpy() if is_torch(w) else np.asarray(w, np.float32)
                blended = np.tensordot(np.asarray(weights, np.float32),
                                       np.stack([l.volume for l in self.luts]), axes=1)
                baked = LUT(blended, name=self.name, axes=first.axes,
                            domain=first.domain, space=first.space)
                return apply(baked, x, interp=interp, backend=backend, device=device,
                             check_finite=check_finite)
            if w.ndim > 1 and self._one_pass(x, backend, device):
                domain = None if first.has_default_domain else first.domain
                return apply_mixture(self.luts, x, w, interp=interp, backend=backend,
                                     device=device, check_finite=check_finite, domain=domain)

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

    def resample(self, grid_size: int | None = None, **kw) -> LUT:
        """A dense LUT for bound global weights; per-pixel weights cannot be frozen."""
        if self.weights is None and "weights" not in kw:
            raise ValueError("resample() needs global weights of shape (n_luts,)")
        return self.to_lut(grid_size, **kw)

    def to_lut(self, grid_size: int | None = None, *, weights=None, **_kw) -> LUT:
        """The mixture as one dense LUT, blended with GLOBAL weights ``(n_luts,)``
        (exact when the LUTs share a grid). ``grid_size`` resamples the result."""
        w = weights if weights is not None else self.weights
        if w is None or np.asarray(w).ndim != 1:
            raise ValueError("to_lut() needs global weights of shape (n_luts,)")
        first = self.luts[0]
        for l in self.luts[1:]:
            if l.sizes != first.sizes or not _same_axes(l, first):
                raise ValueError("to_lut() needs all LUTs on the same grid")
            if l.domain != first.domain:
                raise ValueError("to_lut() needs all LUTs on the same input domain")
        stack = np.stack([l.volume for l in self.luts])
        blended = np.tensordot(np.asarray(w, np.float32), stack, axes=1)
        return first.replace(volume=blended, name=self.name).resample(grid_size)


def _same_axes(a: LUT, b: LUT) -> bool:
    if a.is_uniform and b.is_uniform:
        return True
    if a.is_uniform != b.is_uniform:
        return False
    return all(np.array_equal(x, y) for x, y in zip(a.axes, b.axes))
