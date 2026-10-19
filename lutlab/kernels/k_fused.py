"""k_fused -- single-pass HIP/CUDA kernel from the external `robust_hald` package.

Opt-in only (never chosen by ``backend="auto"``): it needs a GPU, the
`robust_hald` package and a compiler toolchain, supports only cubic uniform
3D trilinear lookups and has no backward pass.
"""
from __future__ import annotations

import importlib.util

from ._registry import Backend, Request, register


def _available():
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
    try:
        spec = importlib.util.find_spec("robust_hald")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return ("robust_hald is not importable "
                "(pip install -e /path/to/robust_hald)")
    import torch
    if not torch.cuda.is_available():
        return "no CUDA/ROCm device"
    return None


def _unsupported(req: Request):
    if req.kind != "cuda":                  # ROCm devices also report "cuda"
        return "fused kernel needs a CUDA/ROCm device (device='cuda')"
    if req.scheme != "multilinear" or req.n_in != 3:
        return "fused kernel supports trilinear 3D lookups only"
    if not req.uniform or len(set(req.sizes)) != 1:
        return "fused kernel needs a uniform cubic grid"
    return None


def run(vol, x, axes, scheme):
    import torch
    from robust_hald.fused_kernel import fused_segmented_apply
    i2 = x.float().contiguous().reshape(-1, 1, 3)           # kernel reads I[px*3]
    luts = vol.float().contiguous().unsqueeze(0)            # (1, D, D, D, 3)
    mask = torch.ones(1, i2.shape[0], 1, device=x.device)
    return fused_segmented_apply(i2, luts, mask).reshape(-1, 3)


register(Backend(name="fused", run=run, unsupported=_unsupported,
                 available=_available, priority=40, torch_native=True,
                 differentiable=False, auto=False))
