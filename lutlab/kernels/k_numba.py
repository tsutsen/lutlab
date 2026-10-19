"""k_numba -- fused multi-threaded CPU kernels (uniform grids, any n_in)."""
from __future__ import annotations

import importlib.util

import numpy as np

from ._common import strides_of
from ._registry import Backend, Request, register


def _available():
    if importlib.util.find_spec("numba") is None:
        return "numba is not installed (pip install lutlab[fast])"
    return None


def _unsupported(req: Request):
    if req.kind != "cpu":
        return "numba runs on the CPU only"
    if not req.uniform:
        return "numba kernels need a uniform grid (use backend='numpy')"
    return None


def run(vol, x, axes, scheme):
    from . import _numba_jit as J
    out = np.empty((x.shape[0], 3), np.float32)
    if x.shape[0] == 0:
        return out
    sizes = np.asarray(vol.shape[:-1], np.int64)
    fn = {"nearest": J.nearest, "multilinear": J.multilinear,
          "tetrahedral": J.tetrahedral}[scheme]
    fn(vol.reshape(-1, 3), sizes, strides_of(sizes), x, out)
    return out


register(Backend(name="numba", run=run, unsupported=_unsupported,
                 available=_available, priority=20))
