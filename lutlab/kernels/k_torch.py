"""k_torch -- differentiable, device-agnostic kernels written in plain torch.

Works on CPU, CUDA, ROCm and MPS, for any number of inputs and for uniform or
rectilinear grids. Gradients flow to both the LUT values and the query points
(the latter through the interpolation weights, as with `grid_sample`).

Implemented with gather-and-blend rather than `grid_sample` so one code path
covers every dimensionality and scheme. A hardware-trilinear fast path for
3D uniform grids can be added behind the same registry entry; the backend
agreement tests will validate it.
"""
from __future__ import annotations

import importlib.util

import numpy as np

from ._common import strides_of
from ._registry import Backend, Request, register


def _available():
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed (pip install lutlab[torch])"
    return None


def _locate(torch, x, sizes, axes):
    n = len(sizes)
    x = x.clamp(0.0, 1.0)
    if axes is None:
        s = torch.tensor(sizes, dtype=x.dtype, device=x.device)
        xf = x * (s - 1)
        i0 = torch.minimum(xf.detach().floor(), s - 2).long()
        return i0, xf - i0.to(x.dtype)
    idx, frac = [], []
    for k in range(n):
        pos = axes[k].to(device=x.device, dtype=x.dtype)
        xk = x[:, k].contiguous()
        i = (torch.searchsorted(pos, xk.detach(), right=True) - 1).clamp(0, sizes[k] - 2)
        frac.append((xk - pos[i]) / (pos[i + 1] - pos[i]))
        idx.append(i)
    return torch.stack(idx, 1), torch.stack(frac, 1)


def run(vol, x, axes, scheme):
    import torch
    dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    x = x.to(dtype)
    vol = vol.to(dtype)
    sizes = tuple(int(s) for s in vol.shape[:-1])
    n = len(sizes)
    strides = [int(s) for s in strides_of(sizes)]
    st = torch.tensor(strides, dtype=torch.long, device=x.device)
    vf = vol.reshape(-1, 3)
    i0, f = _locate(torch, x, sizes, axes)
    base = (i0 * st).sum(1)

    if scheme == "nearest":
        sel = i0 + (f >= 0.5).long()
        return vf[(sel * st).sum(1)]

    if scheme == "multilinear":
        out = x.new_zeros((x.shape[0], 3))
        for c in range(1 << n):
            off, w = 0, None
            for k in range(n):
                if (c >> k) & 1:
                    off += strides[k]
                    wk = f[:, k]
                else:
                    wk = 1.0 - f[:, k]
                w = wk if w is None else w * wk
            out = out + w.unsqueeze(1) * vf[base + off]
        return out

    # tetrahedral (n == 3, enforced upstream)
    order = torch.argsort(f, dim=1, descending=True)
    fs = torch.gather(f, 1, order)
    w = torch.stack([1.0 - fs[:, 0], fs[:, 0] - fs[:, 1], fs[:, 1] - fs[:, 2], fs[:, 2]], dim=1)
    o1 = st[order[:, 0]]
    o2 = o1 + st[order[:, 1]]
    o3 = sum(strides)
    idx = torch.stack([base, base + o1, base + o2, base + o3], dim=1)
    return (w.unsqueeze(2) * vf[idx]).sum(1)


def _auto_when(req: Request) -> bool:
    # On CPU with numpy input numba/numpy avoid a pointless host<->tensor copy.
    return req.kind != "cpu" or req.grad or req.from_torch


register(Backend(name="torch", run=run, available=_available, priority=30,
                 torch_native=True, differentiable=True, auto_when=_auto_when))
