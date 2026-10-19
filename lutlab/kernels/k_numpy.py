"""k_numpy -- the reference implementation (the oracle).

General over any number of inputs, uniform or rectilinear grids, and all
three schemes. Deliberately simple and vectorised; every other backend is
tested against this one.
"""
from __future__ import annotations

import numpy as np

from ._common import strides_of
from ._registry import Backend, Request, register

_CHUNK = 1 << 18        # pixels per block: bounds temporary memory


def _locate(x, sizes, axes):
    """x (N, n) in [0,1] -> base cell index i0 (N, n) and cell fraction f (N, n)."""
    n = len(sizes)
    x = np.clip(x, 0.0, 1.0)
    i0 = np.empty(x.shape, np.int64)
    f = np.empty(x.shape, np.float32)
    for k in range(n):
        D = sizes[k]
        if axes is None:
            xf = x[:, k] * np.float32(D - 1)
            i = np.minimum(np.floor(xf).astype(np.int64), D - 2)
            fr = xf - i.astype(np.float32)
        else:
            pos = axes[k]
            i = np.clip(np.searchsorted(pos, x[:, k], side="right") - 1, 0, D - 2)
            fr = (x[:, k] - pos[i]) / (pos[i + 1] - pos[i])
        i0[:, k] = i
        f[:, k] = fr
    return i0, f


def _nearest(vf, strides, i0, f):
    sel = i0 + (f >= 0.5)
    return vf[(sel * strides).sum(1)]


def _multilinear(vf, strides, i0, f):
    n = f.shape[1]
    base = (i0 * strides).sum(1)
    out = np.zeros((f.shape[0], 3), np.float32)
    for c in range(1 << n):
        off = 0
        w = None
        for k in range(n):
            if (c >> k) & 1:
                off += int(strides[k])
                wk = f[:, k]
            else:
                wk = 1.0 - f[:, k]
            w = wk if w is None else w * wk
        out += w[:, None] * vf[base + off]
    return out


def _tetrahedral(vf, strides, i0, f):
    # 6 tetrahedra along the cube diagonal: sort fractions descending, walk
    # 000 -> +axis(max) -> +axis(mid) -> 111.
    order = np.argsort(-f, axis=1, kind="stable")
    fs = np.take_along_axis(f, order, axis=1)
    w0, w1, w2, w3 = 1.0 - fs[:, 0], fs[:, 0] - fs[:, 1], fs[:, 1] - fs[:, 2], fs[:, 2]
    base = (i0 * strides).sum(1)
    o1 = strides[order[:, 0]]
    o2 = o1 + strides[order[:, 1]]
    o3 = int(strides.sum())
    return (w0[:, None] * vf[base] + w1[:, None] * vf[base + o1]
            + w2[:, None] * vf[base + o2] + w3[:, None] * vf[base + o3])


_KERNELS = {"nearest": _nearest, "multilinear": _multilinear, "tetrahedral": _tetrahedral}


def run(vol, x, axes, scheme):
    sizes = vol.shape[:-1]
    strides = strides_of(sizes)
    vf = vol.reshape(-1, 3)
    fn = _KERNELS[scheme]
    out = np.empty((x.shape[0], 3), np.float32)
    for s in range(0, x.shape[0], _CHUNK):
        i0, f = _locate(x[s:s + _CHUNK], sizes, axes)
        out[s:s + _CHUNK] = fn(vf, strides, i0, f)
    return out


def _unsupported(req: Request):
    if req.kind != "cpu":
        return "numpy runs on the CPU only (device='cpu')"
    return None


register(Backend(name="numpy", run=run, unsupported=_unsupported,
                 priority=0, reference=True))
