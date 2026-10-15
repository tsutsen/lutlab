"""The LUT type: a regular grid of colours + one apply path.

Canonical layout
----------------
`vol` has shape `(D,)*n + (3,)`:
  * the first `n` axes are the INPUT dimensions (axis 0 = input dim 0, ...),
  * the last axis is the 3 output RGB channels,
  * `vol[i0, i1, ...]` is the output for input `(i0/(D-1), i1/(D-1), ...)`.

For the standard 3D RGB LUT that is `(D, D, D, 3)` with axis 0 = R, axis 1 = G,
axis 2 = B -- the same layout used by the grid_sample GPU path and the fused
HIP kernel. Every format in `lutlab.formats` round-trips through this layout,
which is what keeps the axis-order bugs from ever biting again.

Change dimensionality by simply making `vol` have a different number of axes:
  * 2D LUT  -> (D, D, 3)          (bilinear)
  * 3D LUT  -> (D, D, D, 3)       (trilinear / tetrahedral)
  * 4D LUT  -> (D, D, D, D, 3)    (quadrilinear, e.g. +luminance)
"""
from __future__ import annotations

import numpy as np

from . import interp as _interp

__all__ = ["LUT", "identity"]


def _grid(D: int, n: int) -> np.ndarray:
    return np.stack(np.meshgrid(*[np.linspace(0, 1, D)] * n, indexing="ij"),
                    axis=-1).astype(np.float32)


def identity(D: int = 17, ndim: int = 3, name: str = "identity") -> LUT:
    """The do-nothing LUT: output == input, on a D x ... x D grid."""
    return LUT(_grid(D, ndim), name=name)


class LUT:
    """A colour lookup table. See the module docstring for the layout."""

    def __init__(self, vol: np.ndarray, name: str = "lut") -> None:
        vol = np.asarray(vol, dtype=np.float32)
        if vol.ndim < 3 or vol.shape[-1] != 3:
            raise ValueError(f"LUT volume must be (D,)*n + (3,), got {vol.shape}")
        D0 = vol.shape[0]
        if any(s != D0 for s in vol.shape[:-1]):
            raise ValueError(f"LUT grid axes must all be D; got {vol.shape}")
        self.vol = np.ascontiguousarray(vol)
        self.name = name
        self.D = D0
        self.ndim = vol.ndim - 1           # number of input dims (2, 3, 4, ...)

    # -- basics ---------------------------------------------------------- #
    @property
    def shape(self) -> tuple[int, ...]:
        return self.vol.shape

    def to_numpy(self) -> np.ndarray:
        return self.vol

    def __len__(self) -> int:
        return self.vol.size

    def __repr__(self) -> str:
        return (f"LUT(name={self.name!r}, {self.ndim}D, D={self.D}, "
                f"{self.storage_bytes()} B)")

    def storage_bytes(self, dtype: type = np.float32) -> int:
        """Size of the raw grid on disk/memory in bytes."""
        return int(self.vol.size * dtype.itemsize)

    # -- analysis (for showing what a LUT does) -------------------------- #
    def delta_from_identity(self) -> np.ndarray:
        """vol minus the identity LUT: what this LUT actually changes."""
        return self.vol - identity(self.D, self.ndim).vol

    def tone_curves(self) -> dict[str, np.ndarray]:
        """Per-channel 1D tone curve along the main diagonal (input dim0=dim1=..).

        For a 3D LUT this is vol[k, k, k, c] -- the output channel c when the
        input is the grey ramp (t, t, t). Handy for a compact LUT figure.
        """
        d = np.arange(self.D)
        idx = tuple([d] * self.ndim)
        return {c: self.vol[idx + (c,)] for c in range(3)}

    def max_delta(self) -> float:
        """Largest per-channel change the LUT makes anywhere (0..1 scale)."""
        return float(np.abs(self.delta_from_identity()).max())

    # -- apply ----------------------------------------------------------- #
    def apply(self, img: np.ndarray, interp: str = "multilinear",
              backend: str = "auto") -> np.ndarray:
        """Apply this LUT to `img` (shape (..., ndim), values in [0,1])."""
        import importlib
        # import by module (not `from . import apply`) because __init__ binds
        # the *function* lutlab.apply over the submodule of the same name.
        mod = importlib.import_module(".apply", __package__)
        return mod.apply(self, img, interp=interp, backend=backend)

    def sample(self, x: np.ndarray, interp: str = "multilinear",
               engine: str = "auto") -> np.ndarray:
        """Interpolate the grid at points `x` (no backend dispatch).

        See also: :meth:`LUT.apply` (apply to a whole image).
        """
        return _interp.interp(self.vol, x, mode=interp, engine=engine)

    # -- format round-trips (delegated to lutlab.formats) ---------------- #
    def resample(self, D: int, name: str | None = None) -> LUT:
        from .formats import resample
        return resample(self, D, name=name or self.name)

    def to_cube(self, path: str | None = None, title: str | None = None) -> str:
        from .formats import to_cube
        return to_cube(self, path, title=title or self.name)

    @classmethod
    def from_cube(cls, path: str) -> LUT:
        from .formats import from_cube
        return cls.from_numpy(*from_cube(path))

    def to_json(self, path: str | None = None) -> str:
        from .formats import to_json
        return to_json(self, path)

    @classmethod
    def from_json(cls, path: str) -> LUT:
        from .formats import from_json
        return cls.from_numpy(*from_json(path))

    # -- constructors ---------------------------------------------------- #
    @classmethod
    def from_numpy(cls, vol: np.ndarray, name: str = "lut") -> LUT:
        return cls(vol, name=name)


if __name__ == "__main__":
    lut = identity(17, 3)
    print(lut)
    print("max delta (identity):", lut.max_delta())
    tc = lut.tone_curves()
    print("tone curve R[:4]:", np.round(tc[0][:4], 3))
    r = lut.resample(33)
    print("resample 17->33:", r.shape, r.name)
    out = lut.apply(np.linspace(0, 1, 5).reshape(-1, 1, 1, 3).repeat(3, axis=2)
                    .reshape(5, 3), interp="trilinear", backend="numpy")
    print("identity apply == input:", np.allclose(out, np.linspace(0, 1, 5).reshape(-1,1).repeat(3,1)))
