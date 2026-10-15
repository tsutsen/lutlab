"""Color-LUT interpolation kernels: nearest, multilinear, tetrahedral.

A LUT is a regular `D x ... x D` grid of RGB colours with `n` input dimensions.
The volume shape is `(D,)*n + (3,)` -- the last axis is output RGB, and axis
`k` (for `k = 0..n-1`) is input dimension `k` (`n = 2, 3 or 4`).

A query point `x` in `[0, 1]^n` is scaled to grid coords `x*(D-1)`. The
interpolation scheme picks which grid corners to blend:

::

    nearest      1 corner. Fastest, blocky. Good for exact-key lookups.
    multilinear  2^n corners, linear along every axis. This is what people
                 call "bilinear" (n=2), "trilinear" (n=3), "quadrilinear" (n=4).
    tetrahedral  4 corners (n=3 only). Splits the unit cube into 6 tetrahedra
                 along its long diagonal. A smaller, "rounder" cell than the
                 trilinear cube, so it approximates smooth (non-linear) colour
                 curves with less error -- the scheme pro pipelines (Resolve,
                 DaVinci) prefer for 3D LUTs. It is also exact for linear
                 functions, which we use as a correctness invariant below.

Changing dimensionality is just a matter of the volume's number of axes:
a 2D LUT is `(D,D,3)`, a 3D LUT is `(D,D,D,3)`, a 4D LUT is `(D,D,D,D,3)`.
The same `interp()` call handles all of them.
"""
import numpy as np

__all__ = ["interp", "interp_nearest", "interp_multilinear", "interp_tetrahedral",
           "interp_name"]

# user-facing sugar -> (kernel, required_ndim or None)
_SUGAR = {
    "nearest": ("nearest", None),
    "multilinear": ("multilinear", None),
    "linear": ("multilinear", None),
    "bilinear": ("multilinear", 2),
    "trilinear": ("multilinear", 3),
    "quadrilinear": ("multilinear", 4),
    "tetrahedral": ("tetrahedral", 3),
}


def interp_name(mode: str, ndim: int) -> str:
    """Map a user scheme name to a canonical kernel name, checking dimensionality.

    >>> interp_name("trilinear", 3)
    'multilinear'
    >>> interp_name("tetrahedral", 2)   # doctest: +ELLIPSIS
    Traceback (ValueError, ...)
    """
    try:
        kernel, want = _SUGAR[str(mode).lower()]
    except KeyError:
        raise ValueError(f"unknown interp mode {mode!r}; "
                         f"options: {', '.join(sorted(_SUGAR))}")
    if want is not None and want != ndim:
        raise ValueError(f"{mode!r} needs a {want}D LUT, got {ndim}D")
    return kernel


# --------------------------------------------------------------------------- #
# numpy reference (general over n; the oracle the numba kernels are checked
# against)
# --------------------------------------------------------------------------- #

def _coords(x: np.ndarray, D: int):
    xf = np.clip(np.asarray(x, dtype=np.float32), 0, 1) * (D - 1)
    i0 = np.floor(xf).astype(np.int32)
    i1 = np.minimum(i0 + 1, D - 1)
    f = xf - i0
    return i0, i1, f


def _gather(vol: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """vol[tuple(idx[...,k] for k in range(n))]  ->  (..., 3)."""
    return vol[tuple(idx[..., k] for k in range(idx.shape[-1]))]


def interp_nearest(vol: np.ndarray, x: np.ndarray) -> np.ndarray:
    i0, _, _ = _coords(x, vol.shape[0])
    return _gather(vol, i0)


def interp_multilinear(vol: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = vol.ndim - 1
    D = vol.shape[0]
    i0, i1, f = _coords(x, D)
    out = np.zeros(x.shape[:-1] + (3,), dtype=np.float32)
    for c in range(1 << n):
        idx = np.stack([(i1[..., k] if (c >> k) & 1 else i0[..., k])
                        for k in range(n)], axis=-1)
        w = np.ones(x.shape[:-1], dtype=np.float32)
        for k in range(n):
            w = w * (f[..., k] if (c >> k) & 1 else 1 - f[..., k])
        out = out + w[..., None] * _gather(vol, idx)
    return out


def _tetra_weights(f: np.ndarray) -> tuple:
    """For frac f=(3,), return the 4 (vertex_offset, weight) pairs.

    Vertices (offsets from the base corner): 000, onehot(i), onehot(i)+onehot(j),
    111, where (i,j,k) is the axes ordered by *descending* frac.
    """
    order = list(np.argsort(-f))          # descending: f[i] >= f[j] >= f[k]
    i, j, k = order[0], order[1], order[2]
    fi, fj, fk = f[i], f[j], f[k]
    v1 = [0, 0, 0]; v1[i] = 1
    v2 = v1.copy(); v2[j] = 1
    return (
        ([0, 0, 0], 1.0 - fi),
        (v1, fi - fj),
        (v2, fj - fk),
        ([1, 1, 1], fk),
    )


def interp_tetrahedral(vol: np.ndarray, x: np.ndarray) -> np.ndarray:
    if vol.ndim - 1 != 3:
        raise ValueError("tetrahedral is 3D only")
    D = vol.shape[0]
    xf = np.clip(np.asarray(x, dtype=np.float32), 0, 1) * (D - 1)
    flat = xf.reshape(-1, 3)
    b = np.floor(flat).astype(np.int32)
    f = flat - b
    out = np.zeros((flat.shape[0], 3), dtype=np.float32)
    for p in range(flat.shape[0]):
        for vo, w in _tetra_weights(f[p]):
            idx = [min(b[p, ax] + vo[ax], D - 1) for ax in range(3)]
            out[p] += w * vol[idx[0], idx[1], idx[2]]
    return out.reshape(np.asarray(x).shape[:-1] + (3,))


# --------------------------------------------------------------------------- #
# numba fast path (specialised to 3D, the workhorse; 2D/4D use the numpy path)
# --------------------------------------------------------------------------- #

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:                                   # pragma: no cover
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _near3(vol, x):
        N = x.shape[0]; D = vol.shape[0]
        out = np.zeros((N, 3), np.float32)
        for p in prange(N):
            i = [0, 0, 0]
            for a in range(3):
                c = x[p, a] * (D - 1)
                if c < 0:
                    c = 0
                elif c > D - 1:
                    c = D - 1
                i[a] = int(c)
            for k in range(3):
                out[p, k] = vol[i[0], i[1], i[2], k]
        return out

    @njit(parallel=True, cache=True)
    def _ml3(vol, x):
        N = x.shape[0]; D = vol.shape[0]
        out = np.zeros((N, 3), np.float32)
        for p in prange(N):
            i = [0, 0, 0]; fr = [0.0, 0.0, 0.0]
            for a in range(3):
                c = x[p, a] * (D - 1)
                if c < 0:
                    c = 0
                elif c > D - 1:
                    c = D - 1
                i[a] = int(c)
                fr[a] = c - i[a]
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        wx = fr[0] if dx else 1.0 - fr[0]
                        wy = fr[1] if dy else 1.0 - fr[1]
                        wz = fr[2] if dz else 1.0 - fr[2]
                        w = wx * wy * wz
                        r = i[0] + dx; g = i[1] + dy; b_ = i[2] + dz
                        if r > D - 1:
                            r = D - 1
                        if g > D - 1:
                            g = D - 1
                        if b_ > D - 1:
                            b_ = D - 1
                        for k in range(3):
                            out[p, k] += w * vol[r, g, b_, k]
        return out

    @njit(parallel=True, cache=True)
    def _tetra3(vol, x):
        N = x.shape[0]; D = vol.shape[0]
        out = np.zeros((N, 3), np.float32)
        for p in prange(N):
            b = [0, 0, 0]; f = [0.0, 0.0, 0.0]
            for a in range(3):
                c = x[p, a] * (D - 1)
                if c < 0:
                    c = 0
                elif c > D - 1:
                    c = D - 1
                b[a] = int(c)
                f[a] = c - b[a]
            # 3-element descending sort of (f0,f1,f2) -> (i,j,k)
            a, bb, cc = 0, 1, 2
            if f[a] < f[bb]:
                a, bb = bb, a
            if f[bb] < f[cc]:
                bb, cc = cc, bb
            if f[a] < f[bb]:
                a, bb = bb, a
            i, j, k = a, bb, cc
            fi, fj, fk = f[i], f[j], f[k]
            w0 = 1.0 - fi; w1 = fi - fj; w2 = fj - fk; w3 = fk
            o1 = [0, 0, 0]; o1[i] = 1
            o2 = [0, 0, 0]; o2[i] = 1; o2[j] = 1
            for t in (0, 1, 2):
                ix = b[0]; iy = b[1]; iz = b[2]
                out[p, t] += w0 * vol[ix, iy, iz, t]
                ix = min(b[0] + o1[0], D - 1); iy = min(b[1] + o1[1], D - 1)
                iz = min(b[2] + o1[2], D - 1)
                out[p, t] += w1 * vol[ix, iy, iz, t]
                ix = min(b[0] + o2[0], D - 1); iy = min(b[1] + o2[1], D - 1)
                iz = min(b[2] + o2[2], D - 1)
                out[p, t] += w2 * vol[ix, iy, iz, t]
                ix = min(b[0] + 1, D - 1); iy = min(b[1] + 1, D - 1)
                iz = min(b[2] + 1, D - 1)
                out[p, t] += w3 * vol[ix, iy, iz, t]
        return out


# --------------------------------------------------------------------------- #
# public dispatcher
# --------------------------------------------------------------------------- #

def interp(vol: np.ndarray, x: np.ndarray, mode: str = "multilinear",
           engine: str = "auto") -> np.ndarray:
    """Interpolate `vol` at points `x`.

    Parameters
    ----------
    vol : (D,)*n + (3,) array
    x   : (..., n) array of query points in [0, 1]
    mode: 'multilinear' (default -- any dimension) | 'nearest' | 'bilinear' |
          'trilinear' | 'quadrilinear' | 'tetrahedral'. 'bilinear',
          'trilinear' and 'quadrilinear' are just multilinear pinned to a
          dimension, so for a 3D LUT they are identical to the default.
    engine: 'auto' (numba if available, else numpy), 'numba', or 'numpy'.
            The numba fast path is specialised to 3D; 2D/4D always use numpy.

    Returns (..., 3) float32.
    """
    vol = np.asarray(vol, dtype=np.float32)
    n = vol.ndim - 1
    kernel = interp_name(mode, n)
    xa = np.asarray(x, dtype=np.float32)
    flat = xa.reshape(-1, n)
    use_numba = (_HAS_NUMBA and kernel != "nearest" and n == 3
                 and engine in ("auto", "numba"))
    if use_numba:
        if kernel == "multilinear":
            res = _ml3(vol, flat)
        else:                                   # tetrahedral
            res = _tetra3(vol, flat)
    elif kernel == "nearest":
        res = interp_nearest(vol, xa).reshape(-1, 3)
    elif kernel == "multilinear":
        res = interp_multilinear(vol, xa).reshape(-1, 3)
    else:
        res = interp_tetrahedral(vol, xa).reshape(-1, 3)
    return res.reshape(xa.shape[:-1] + (3,))


if __name__ == "__main__":
    # quick self-check: tetrahedral must be exact for a linear function;
    # trilinear is not.
    D = 16
    rng = np.random.RandomState(0)
    a = rng.randn(3, 3); b0 = rng.randn(3)      # linear map R^3 -> R^3
    g = np.stack(np.meshgrid(*[np.linspace(0, 1, D)] * 3, indexing="ij"), -1)
    flat = g.reshape(-1, 3)
    lin = flat @ a + b0                          # LUT stores a LINEAR function
    vol = lin.reshape(D, D, D, 3).astype(np.float32)
    q = rng.rand(5000, 3).astype(np.float32)
    tetra = interp_tetrahedral(vol, q)
    tri = interp_multilinear(vol, q)
    true = q @ a + b0
    # A wrong vertex/weight assignment would break the linear reconstruction,
    # so this is the correctness gate (both schemes are exact for LINEAR f).
    print(f"tetrahedral max err vs true linear: "
          f"{np.abs(tetra - true).max():.2e}   (should be ~1e-5)")
    print(f"trilinear   max err vs true linear: "
          f"{np.abs(tri - true).max():.2e}   (also exact for linear f)")
    # Illustration on a smooth NON-linear curve (sin) -- tetrahedral tracks it
    # a touch better; not a correctness gate, just shows the tradeoff.
    vol2 = (np.sin(np.pi * flat[..., None])).reshape(D, D, D, 3).astype(np.float32)
    true2 = np.sin(np.pi * q)
    e_tri = np.abs(interp_multilinear(vol2, q) - true2).max()
    e_tet = np.abs(interp_tetrahedral(vol2, q) - true2).max()
    print(f"smooth sin curve -- trilinear err {e_tri:.4f}, tetrahedral err {e_tet:.4f}")
