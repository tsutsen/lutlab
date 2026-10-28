"""The `LUT` type: an immutable regular or rectilinear grid of RGB outputs.

Layout
------
``volume`` has shape ``(D0, ..., Dn-1, 3)``:

* the first ``n`` axes are the INPUT dimensions (axis 0 = input channel 0),
* the last axis is output RGB,
* ``volume[i0, ..., in-1]`` is the output at the grid point whose coordinate on
  axis ``k`` is ``axes[k][ik]`` (``ik / (Dk - 1)`` when the grid is uniform).

Axes may have different sizes and, optionally, non-uniform sample positions
(AdaInt-style). A 4D LUT is simply ``n = 4``: the 4th input is a context
channel supplied through ``context=`` when the LUT is called.

``domain`` is the input range the LUT is defined over (``.cube``
``DOMAIN_MIN/MAX``); inputs are rescaled to the unit cube before lookup.
"""
from __future__ import annotations

import warnings

import numpy as np

from .base import Cost
from .._arrays import quantize

__all__ = ["LUT"]


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


def _grid_points(sizes) -> np.ndarray:
    pos = [np.linspace(0.0, 1.0, s, dtype=np.float32) for s in sizes]
    return np.stack(np.meshgrid(*pos, indexing="ij"), -1).reshape(-1, len(sizes))


def _resample(lut: "LUT", size) -> "LUT":
    """Resample a LUT onto a uniform grid by multilinear interpolation."""
    from ..kernels.dispatch import interpolate
    sizes = (int(size),) * lut.n_in if np.isscalar(size) else tuple(int(s) for s in size)
    if len(sizes) != lut.n_in:
        raise ValueError(f"size has {len(sizes)} entries for a {lut.n_in}-input LUT")
    out = interpolate(lut.volume, _grid_points(sizes), axes=lut.axes, scheme="multilinear")
    return LUT(out.reshape(sizes + (3,)), name=lut.name,
               domain=lut.domain, space=lut.space, bit_depth=lut.bit_depth)


class LUT:
    """A colour lookup table. Immutable: use `replace()` to derive variants.

    ``bit_depth`` (optional, 1..32) quantises the values to ``2**bit_depth`` levels --
    a quality/size/speed lever; ``None`` (default) keeps full float precision. Up to
    16 bits the codes are stored in uint8/uint16; above 16 bits they are stored as
    float32 (32 bits is the cap, i.e. float32 precision). Convert later with
    :meth:`to_bit_depth`.
    """

    def __init__(self, volume, name: str = "lut", *, axes=None, domain=None,
                 space: str = "srgb", bit_depth: int | None = None) -> None:
        v = np.array(volume, dtype=np.float32, order="C")          # always a copy
        if v.ndim < 2 or v.shape[-1] != 3:
            raise ValueError(f"LUT volume must have shape (D0, ..., Dn-1, 3); got {v.shape}")
        sizes = tuple(int(s) for s in v.shape[:-1])
        if min(sizes) < 2:
            raise ValueError(f"every LUT axis needs >= 2 samples; got sizes {sizes}")
        if not np.isfinite(v).all():
            raise ValueError("LUT contains non-finite values")
        if bit_depth is not None:
            n = int(bit_depth)
            if not 1 <= n <= 32:
                raise ValueError(f"bit_depth must be between 1 and 32, got {n}")
            top = (2 ** n) - 1
            if n <= 16:
                v = (quantize(v, n) * top).astype(np.uint8 if n <= 8 else np.uint16)
                scale = float(top)
            else:
                v = quantize(v, n)                 # float32: no compact integer fits
                scale = 1.0
        else:
            scale = 1.0
        v.flags.writeable = False
        self._volume = v
        self._scale = scale
        self._bit_depth = None if bit_depth is None else int(bit_depth)
        self._sizes = sizes
        self._axes = _normalise_axes(axes, sizes)
        self._dmin, self._dmax = _normalise_domain(domain, len(sizes))
        self.name = str(name)
        self.space = str(space).lower()

    # -- structure ------------------------------------------------------- #
    @property
    def volume(self) -> np.ndarray:
        """Read-only float32 ``[0, 1]`` volume (upcast from the stored dtype)."""
        if self._volume.dtype == np.float32:
            return self._volume
        v = self._volume.astype(np.float32) / self._scale
        v.flags.writeable = False
        return v

    @property
    def n_in(self) -> int:
        return len(self._sizes)

    @property
    def sizes(self) -> tuple[int, ...]:
        return self._sizes

    @property
    def raw(self) -> np.ndarray:
        """Compact stored volume (read-only): uint8 (1..8-bit), uint16 (9..16-bit)
        or float32 (above 16 bits, or no quantisation). ``volume`` is the normalised
        float32 view."""
        return self._volume

    @property
    def raw_top(self) -> int | None:
        """Highest integer code in :attr:`raw` (``2**bit_depth - 1``), or ``None``
        when ``raw`` is float32. Divide an integer ``raw`` by ``raw_top`` to get
        ``volume``."""
        if self._bit_depth is None or self._volume.dtype.kind != "u":
            return None
        return (2 ** self._bit_depth) - 1

    @property
    def bit_depth(self) -> int | None:
        """Value quantisation (``2**bit_depth`` levels, 1..32), or ``None`` for full float."""
        return self._bit_depth

    @property
    def shape(self) -> tuple[int, ...]:
        return self._volume.shape

    @property
    def nbytes(self) -> int:
        """Bytes actually stored (compact for a quantised LUT)."""
        return int(self._volume.nbytes)

    @property
    def grid_size(self) -> int:
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
        kb = self.nbytes / 1024
        bd = "float" if self._bit_depth is None else f"{self._bit_depth}-bit"
        return (f"LUT(name={self.name!r}, n_in={self.n_in}, sizes={self._sizes}, {bd}, "
                f"space={self.space!r}, {'uniform' if self.is_uniform else 'rectilinear'}, "
                f"{kb:.1f} KiB)")

    # -- transform interface --------------------------------------------- #
    def __call__(self, x, *, context=None, interp: str = "multilinear",
                 backend: str = "auto", device=None, check_finite=None):
        """Apply the LUT. See `lutlab.kernels.apply` for the full contract."""
        from ..kernels.dispatch import apply
        return apply(self, x, context=context, interp=interp, backend=backend,
                     device=device, check_finite=check_finite)

    def cost(self) -> Cost:
        """Static cost; ``reads_per_px`` assumes multilinear (2**n_in corners)."""
        return Cost(params=int(self._volume.size), nbytes=self.nbytes,
                    reads_per_px=2 ** self.n_in, meta={"sizes": list(self._sizes)})

    def resample(self, grid_size: int | None = None, **_kw) -> "LUT":
        """A dense `LUT` resampled onto a uniform ``grid_size``-per-axis grid (self if already that).

        ``grid_size`` is an int (same on every axis) or one size per axis.
        """
        if grid_size is None or (self.is_uniform and self._sizes == (grid_size,) * self.n_in):
            return self
        return _resample(self, grid_size)

    def to_lut(self, grid_size: int | None = None, **_kw) -> "LUT":
        """The transform as one dense `LUT`: self, or resampled onto ``grid_size``."""
        return self.resample(grid_size)

    # -- derivation -------------------------------------------------------- #
    def to_bit_depth(self, n: int) -> "LUT":
        """A copy with values quantised to ``n`` bits (1..32).

        Up-converting beyond the current depth adds no precision (the values only occupy
        ``2**bit_depth`` levels) and warns; down-converting is a lossy round. Above 16
        bits the values are stored as float32.
        """
        n = int(n)
        if not 1 <= n <= 32:
            raise ValueError(f"bit_depth must be between 1 and 32, got {n}")
        if self._bit_depth is not None and n > self._bit_depth:
            warnings.warn(f"up-converting {self._bit_depth}-bit -> {n}-bit adds no "
                          f"precision (only {2 ** self._bit_depth} levels present)",
                          UserWarning)
        return LUT(quantize(self.volume, n), name=self.name, axes=self._axes,
                   domain=self.domain, space=self.space, bit_depth=n)

    def replace(self, *, volume=None, name=None, axes="keep", domain="keep", space=None,
                bit_depth=None) -> "LUT":
        return LUT(self.volume if volume is None else volume,
                   self.name if name is None else name,
                   axes=self._axes if axes == "keep" else axes,
                   domain=self.domain if domain == "keep" else domain,
                   space=self.space if space is None else space,
                   bit_depth=self._bit_depth if bit_depth is None else bit_depth)

    def allclose(self, other: "LUT", atol: float = 1e-6) -> bool:
        """True when sizes, axes, domain and values all match within ``atol``."""
        if self._sizes != other._sizes or self.domain != other.domain:
            return False
        if self.is_uniform != other.is_uniform:
            return False
        if self._axes is not None and not all(
                np.allclose(a, b, atol=atol, rtol=0)
                for a, b in zip(self._axes, other._axes)):
            return False
        return bool(np.allclose(self.volume, other.volume, atol=atol, rtol=0))

    @classmethod
    def identity(cls, grid_size: int = 17, n_in: int = 3, *, sizes=None, axes=None,
                 name: str = "identity", space: str = "srgb") -> "LUT":
        """The do-nothing LUT. Output = the first three input axes.

        ``n_in=1`` gives a grey ramp ``(t, t, t)``; ``n_in>=4`` ignores the extra
        (context) axes. ``n_in=2`` has no canonical identity and raises.
        """
        if sizes is None:
            sizes = (int(grid_size),) * int(n_in)
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

    # -- files ------------------------------------------------------------ #
    def save(self, path, **kw) -> None:
        from ..formats import save
        save(self, path, **kw)

    @classmethod
    def load(cls, path, **kw) -> "LUT":
        from ..formats import load
        return load(path, **kw)
