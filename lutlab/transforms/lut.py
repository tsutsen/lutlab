"""The `LUT` type: an immutable regular or rectilinear grid of RGB outputs.

Layout
------
``vol`` has shape ``(D0, ..., Dn-1, 3)``:

* the first ``n`` axes are the INPUT dimensions (axis 0 = input channel 0),
* the last axis is output RGB,
* ``vol[i0, ..., in-1]`` is the output at the grid point whose coordinate on
  axis ``k`` is ``axes[k][ik]`` (``ik / (Dk - 1)`` when the grid is uniform).

Axes may have different sizes and, optionally, non-uniform sample positions
(AdaInt-style). A 4D LUT is simply ``n = 4``: the 4th input is a context
channel supplied through ``context=`` when the LUT is called.

``domain`` is the input range the LUT is defined over (``.cube``
``DOMAIN_MIN/MAX``); inputs are rescaled to the unit cube before lookup.
"""
from __future__ import annotations

import numpy as np

from .base import Cost

__all__ = ["LUT", "identity"]


def _normalise_axes(axes, sizes):
    if axes is None:
        return None
    if len(axes) != len(sizes):
        raise ValueError(f"axes has {len(axes)} entries for a {len(sizes)}-input LUT")
    out, uniform = [], True
    for k, (a, D) in enumerate(zip(axes, sizes)):
        a = np.array(a, dtype=np.float64)
        if a.shape != (D,):
            raise ValueError(f"axes[{k}] must have shape ({D},), got {a.shape}")
        if abs(a[0]) > 1e-6 or abs(a[-1] - 1.0) > 1e-6:
            raise ValueError(f"axes[{k}] must start at 0 and end at 1")
        if not np.all(np.diff(a) > 0):
            raise ValueError(f"axes[{k}] must be strictly increasing")
        a[0], a[-1] = 0.0, 1.0
        uniform &= bool(np.allclose(a, np.linspace(0.0, 1.0, D), atol=1e-6))
        out.append(a.astype(np.float32))
    if uniform:
        return None
    for a in out:
        a.flags.writeable = False
    return tuple(out)


def _normalise_domain(domain, n):
    if domain is None:
        return (0.0,) * n, (1.0,) * n
    lo, hi = domain
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (n,))
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), (n,))
    if not np.all(hi > lo):
        raise ValueError("domain max must be greater than domain min on every axis")
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


class LUT:
    """A colour lookup table. Immutable: use `replace()` to derive variants."""

    def __init__(self, vol, name: str = "lut", *, axes=None, domain=None,
                 space: str = "srgb") -> None:
        v = np.array(vol, dtype=np.float32, order="C")          # always a copy
        if v.ndim < 2 or v.shape[-1] != 3:
            raise ValueError(f"LUT volume must have shape (D0, ..., Dn-1, 3); got {v.shape}")
        sizes = tuple(int(s) for s in v.shape[:-1])
        if min(sizes) < 2:
            raise ValueError(f"every LUT axis needs >= 2 samples; got sizes {sizes}")
        if not np.isfinite(v).all():
            raise ValueError("LUT contains non-finite values")
        v.flags.writeable = False
        self._vol = v
        self._sizes = sizes
        self._axes = _normalise_axes(axes, sizes)
        self._dmin, self._dmax = _normalise_domain(domain, len(sizes))
        self.name = str(name)
        self.space = str(space).lower()

    # -- structure ------------------------------------------------------- #
    @property
    def vol(self) -> np.ndarray:
        return self._vol

    @property
    def n_in(self) -> int:
        return len(self._sizes)

    @property
    def sizes(self) -> tuple[int, ...]:
        return self._sizes

    @property
    def shape(self) -> tuple[int, ...]:
        return self._vol.shape

    @property
    def D(self) -> int:
        """Grid size when every axis has the same size (else ValueError)."""
        if len(set(self._sizes)) != 1:
            raise ValueError(f"LUT axes differ in size {self._sizes}; use .sizes")
        return self._sizes[0]

    @property
    def axes(self) -> tuple[np.ndarray, ...] | None:
        """Per-axis sample positions, or None for a uniform grid."""
        return self._axes

    @property
    def is_uniform(self) -> bool:
        return self._axes is None

    @property
    def domain(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return self._dmin, self._dmax

    @property
    def has_default_domain(self) -> bool:
        return all(v == 0.0 for v in self._dmin) and all(v == 1.0 for v in self._dmax)

    def __repr__(self) -> str:
        kb = self._vol.nbytes / 1024
        return (f"LUT(name={self.name!r}, n_in={self.n_in}, sizes={self._sizes}, "
                f"space={self.space!r}, {'uniform' if self.is_uniform else 'rectilinear'}, "
                f"{kb:.1f} KiB)")

    # -- ColorTransform protocol ----------------------------------------- #
    def __call__(self, x, *, context=None, interp: str = "multilinear",
                 backend: str = "auto", device=None, check_finite=None):
        """Apply the LUT. See `lutlab.kernels.apply` for the full contract."""
        from ..kernels.dispatch import apply
        return apply(self, x, context=context, interp=interp, backend=backend,
                     device=device, check_finite=check_finite)

    def cost(self, scheme: str = "multilinear") -> Cost:
        reads = {"nearest": 1, "tetrahedral": 4}.get(scheme, 2 ** self.n_in)
        return Cost(params=int(self._vol.size), nbytes=int(self._vol.nbytes),
                    reads_per_px=reads, meta={"sizes": list(self._sizes)})

    def bake(self, D: int | None = None, **_kw) -> "LUT":
        """Resample onto a uniform D-per-axis grid (self if already that)."""
        if D is None or (self.is_uniform and self._sizes == (D,) * self.n_in):
            return self
        from ..analysis import resample
        return resample(self, D)

    # -- derivation / comparison ----------------------------------------- #
    def replace(self, *, vol=None, name=None, axes="keep", domain="keep", space=None) -> "LUT":
        return LUT(self._vol if vol is None else vol,
                   self.name if name is None else name,
                   axes=self._axes if axes == "keep" else axes,
                   domain=self.domain if domain == "keep" else domain,
                   space=self.space if space is None else space)

    def allclose(self, other: "LUT", atol: float = 1e-6) -> bool:
        return self._sizes == other._sizes and bool(
            np.allclose(self._vol, other._vol, atol=atol, rtol=0))

    # -- files ------------------------------------------------------------ #
    def save(self, path, **kw) -> None:
        from ..formats import save
        save(self, path, **kw)

    @classmethod
    def load(cls, path, **kw) -> "LUT":
        from ..formats import load
        return load(path, **kw)


def identity(D: int = 17, n_in: int = 3, *, sizes=None, axes=None,
             name: str = "identity", space: str = "srgb") -> LUT:
    """The do-nothing LUT. Output = the first three input axes.

    ``n_in=1`` gives a grey ramp ``(t, t, t)``; ``n_in>=4`` ignores the extra
    (context) axes. ``n_in=2`` has no canonical identity and raises.
    """
    if sizes is None:
        sizes = (int(D),) * int(n_in)
    else:
        sizes = tuple(int(s) for s in sizes)
        n_in = len(sizes)
    if n_in == 2:
        raise ValueError("a 2-input LUT has no canonical identity")
    if axes is None:
        pos = [np.linspace(0.0, 1.0, s) for s in sizes]
    else:
        pos = [np.asarray(a, dtype=np.float64) for a in axes]
    grids = np.meshgrid(*pos, indexing="ij")
    chans = [grids[0]] * 3 if n_in == 1 else grids[:3]
    return LUT(np.stack(chans, axis=-1), name=name, axes=axes, space=space)
