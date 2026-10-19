"""Things you do *to* or *about* a LUT: resample, inspect, compare to identity."""
from __future__ import annotations

import numpy as np

from .kernels.dispatch import interpolate
from .transforms.lut import LUT, identity

__all__ = ["resample", "delta_from_identity", "max_delta", "tone_curve"]


def _grid_points(sizes) -> np.ndarray:
    pos = [np.linspace(0.0, 1.0, s, dtype=np.float32) for s in sizes]
    return np.stack(np.meshgrid(*pos, indexing="ij"), -1).reshape(-1, len(sizes))


def resample(lut: LUT, size, *, name: str | None = None) -> LUT:
    """Resample onto a uniform grid by multilinear interpolation.

    ``size`` is an int (same on every axis) or one size per axis.
    """
    sizes = (int(size),) * lut.n_in if np.isscalar(size) else tuple(int(s) for s in size)
    if len(sizes) != lut.n_in:
        raise ValueError(f"size has {len(sizes)} entries for a {lut.n_in}-input LUT")
    out = interpolate(lut.vol, _grid_points(sizes), axes=lut.axes, scheme="multilinear")
    return LUT(out.reshape(sizes + (3,)), name=name or lut.name,
               domain=lut.domain, space=lut.space)


def delta_from_identity(lut: LUT) -> np.ndarray:
    """``lut.vol`` minus the identity on the same grid: what the LUT changes."""
    ref = identity(sizes=lut.sizes, axes=lut.axes)
    return lut.vol - ref.vol


def max_delta(lut: LUT) -> float:
    """Largest per-channel change the LUT makes at any grid point."""
    return float(np.abs(delta_from_identity(lut)).max())


def tone_curve(lut: LUT, n: int = 33) -> tuple[np.ndarray, np.ndarray]:
    """Output along the grey diagonal ``(t, ..., t)``: returns ``(t, curve[n, 3])``."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = np.repeat(t[:, None], lut.n_in, axis=1)
    return t, interpolate(lut.vol, x, axes=lut.axes, scheme="multilinear")
