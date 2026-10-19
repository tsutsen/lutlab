"""Interpolation kernels + backend registry + the one `apply()` entry point.

Backends (registry key -> module):

    numpy  -> k_numpy    reference oracle; any n_in, any grid, any scheme
    numba  -> k_numba    fused CPU kernels; uniform grids
    torch  -> k_torch    differentiable, any device; any n_in, any grid
    fused  -> k_fused    native HIP kernel; opt-in, inference only

Add a backend by building a `Backend` and calling `register()`.
"""
from ._registry import (Backend, BackendError, BackendInfo, Request, infos as backends,
                        register, select)
from .schemes import SCHEMES, canonical_scheme
from . import k_numpy, k_numba, k_torch, k_fused          # noqa: F401  (register backends)
from .dispatch import apply, interpolate

__all__ = ["apply", "interpolate", "backends", "register", "select", "Backend",
           "BackendError", "BackendInfo", "Request", "SCHEMES", "canonical_scheme"]
