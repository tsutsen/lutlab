"""Numba kernels (uniform grids). Imported lazily by k_numba."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

_CH = 2048


@njit(cache=True)
def _loc(c, D):
    if c < 0.0:
        c = 0.0
    elif c > 1.0:
        c = 1.0
    xf = c * (D - 1)
    i = int(xf)
    if i > D - 2:
        i = D - 2
    return i, xf - i


@njit(parallel=True, cache=True)
def nearest(vf, sizes, strides, x, out):
    N = x.shape[0]
    n = x.shape[1]
    for p in prange(N):
        idx = 0
        for k in range(n):
            D = sizes[k]
            c = x[p, k]
            if c < 0.0:
                c = 0.0
            elif c > 1.0:
                c = 1.0
            i = int(c * (D - 1) + 0.5)
            if i > D - 1:
                i = D - 1
            idx += i * strides[k]
        out[p, 0] = vf[idx, 0]
        out[p, 1] = vf[idx, 1]
        out[p, 2] = vf[idx, 2]


@njit(parallel=True, cache=True)
def multilinear(vf, sizes, strides, x, out):
    N = x.shape[0]
    n = x.shape[1]
    nc = 1 << n
    nchunks = (N + _CH - 1) // _CH
    for ch in prange(nchunks):
        fr = np.empty(n, np.float64)
        lo = ch * _CH
        hi = min(N, lo + _CH)
        for p in range(lo, hi):
            base = 0
            for k in range(n):
                i, f = _loc(x[p, k], sizes[k])
                fr[k] = f
                base += i * strides[k]
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            for corner in range(nc):
                idx = base
                w = 1.0
                for k in range(n):
                    if (corner >> k) & 1:
                        idx += strides[k]
                        w *= fr[k]
                    else:
                        w *= 1.0 - fr[k]
                a0 += w * vf[idx, 0]
                a1 += w * vf[idx, 1]
                a2 += w * vf[idx, 2]
            out[p, 0] = a0
            out[p, 1] = a1
            out[p, 2] = a2


@njit(parallel=True, cache=True)
def tetrahedral(vf, sizes, strides, x, out):
    N = x.shape[0]
    for p in prange(N):
        i0, f0 = _loc(x[p, 0], sizes[0])
        i1, f1 = _loc(x[p, 1], sizes[1])
        i2, f2 = _loc(x[p, 2], sizes[2])
        base = i0 * strides[0] + i1 * strides[1] + i2 * strides[2]
        a = 0
        b = 1
        c = 2
        fa = f0
        fb = f1
        fc = f2
        if fa < fb:
            a, b = b, a
            fa, fb = fb, fa
        if fb < fc:
            b, c = c, b
            fb, fc = fc, fb
        if fa < fb:
            a, b = b, a
            fa, fb = fb, fa
        v1 = base + strides[a]
        v2 = v1 + strides[b]
        v3 = base + strides[0] + strides[1] + strides[2]
        w0 = 1.0 - fa
        w1 = fa - fb
        w2 = fb - fc
        w3 = fc
        for t in range(3):
            out[p, t] = (w0 * vf[base, t] + w1 * vf[v1, t]
                         + w2 * vf[v2, t] + w3 * vf[v3, t])
