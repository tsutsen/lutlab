"""k_fused -- single-pass HIP kernel for trilinear 3-D LUT apply (opt-in).

Native to lutlab: the HIP source below is compiled on first use via
``torch.utils.cpp_extension.load_inline``. Opt-in only (never chosen by
``backend="auto"``): it needs torch + a CUDA/ROCm device (and the matching
nvcc/hipcc toolchain), supports only cubic uniform 3-D trilinear lookups and
has no backward pass. The kernel is written for a *segmented* apply (S LUTs
masked per pixel); the lutlab backend drives it with ``S=1`` for a plain
cubic LUT.
"""
from __future__ import annotations

import importlib.util

from ._registry import Backend, Request, register

_CUDA_SRC = r"""
#include <hip/hip_runtime.h>

__global__ void fused_seg_lut_kernel(
    const float* __restrict__ I,
    const float* __restrict__ luts,
    const float* __restrict__ masks,
    float* __restrict__ Y,
    int H, int W, int S, int D)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= H * W) return;

    float x = I[px*3], y = I[px*3+1], z = I[px*3+2];
    float sc = (float)(D - 1);
    float fx = x * sc, fy = y * sc, fz = z * sc;
    int xi = (int)fx; if (xi > D-2) xi = D-2;
    int yi = (int)fy; if (yi > D-2) yi = D-2;
    int zi = (int)fz; if (zi > D-2) zi = D-2;
    float dx = fx - xi, dy = fy - yi, dz = fz - zi;
    float iDX = 1.f - dx, iDY = 1.f - dy, iDZ = 1.f - dz;

    float oR = 0.f, oG = 0.f, oB = 0.f;
    size_t lsz = (size_t)D * D * D * 3;

    for (int s = 0; s < S; s++) {
        float m = masks[(size_t)s * H * W + px];
        if (m < 1e-4f) continue;
        const float* L = luts + (size_t)s * lsz;
        int b000 = (xi*D+yi)*D+zi,     b100 = ((xi+1)*D+yi)*D+zi;
        int b010 = (xi*D+yi+1)*D+zi,   b110 = ((xi+1)*D+yi+1)*D+zi;
        int b001 = (xi*D+yi)*D+zi+1,   b101 = ((xi+1)*D+yi)*D+zi+1;
        int b011 = (xi*D+yi+1)*D+zi+1, b111 = ((xi+1)*D+yi+1)*D+zi+1;

        float v, v2, v3, v4;
        v  = L[b000*3]*iDX   + L[b100*3]*dx;
        v2 = L[b010*3]*iDX   + L[b110*3]*dx;
        v3 = L[b001*3]*iDX   + L[b101*3]*dx;
        v4 = L[b011*3]*iDX   + L[b111*3]*dx;
        oR += m * ((v*iDY + v2*dy)*iDZ + (v3*iDY + v4*dy)*dz);

        v  = L[b000*3+1]*iDX + L[b100*3+1]*dx;
        v2 = L[b010*3+1]*iDX + L[b110*3+1]*dx;
        v3 = L[b001*3+1]*iDX + L[b101*3+1]*dx;
        v4 = L[b011*3+1]*iDX + L[b111*3+1]*dx;
        oG += m * ((v*iDY + v2*dy)*iDZ + (v3*iDY + v4*dy)*dz);

        v  = L[b000*3+2]*iDX + L[b100*3+2]*dx;
        v2 = L[b010*3+2]*iDX + L[b110*3+2]*dx;
        v3 = L[b001*3+2]*iDX + L[b101*3+2]*dx;
        v4 = L[b011*3+2]*iDX + L[b111*3+2]*dx;
        oB += m * ((v*iDY + v2*dy)*iDZ + (v3*iDY + v4*dy)*dz);
    }
    Y[px*3]   = oR;
    Y[px*3+1] = oG;
    Y[px*3+2] = oB;
}

extern "C" void launch_fused_seg_lut(
    const float* I, const float* luts, const float* masks,
    float* Y, int H, int W, int S, int D)
{
    int total = H * W, threads = 256;
    fused_seg_lut_kernel<<<(total+threads-1)/threads, threads>>>(
        I, luts, masks, Y, H, W, S, D);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
extern "C" void launch_fused_seg_lut(
    const float*, const float*, const float*,
    float*, int, int, int, int);
void fused_seg_lut(torch::Tensor I, torch::Tensor luts,
                   torch::Tensor masks, torch::Tensor Y) {
    int H = I.size(0), W = I.size(1);
    int S = luts.size(0), D = luts.size(1);
    launch_fused_seg_lut(
        I.data_ptr<float>(), luts.data_ptr<float>(),
        masks.data_ptr<float>(), Y.data_ptr<float>(),
        H, W, S, D);
}
"""

_mod = None


def _get_mod():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(
            name="lutlab_fused_seg_lut",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["fused_seg_lut"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _mod


def _available():
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
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


def run(volume, x, axes, scheme):
    import torch
    i2 = x.float().contiguous().reshape(-1, 1, 3)           # kernel reads I[px*3]
    luts = volume.float().contiguous().unsqueeze(0)         # (1, D, D, D, 3)
    mask = torch.ones(1, i2.shape[0], 1, device=x.device)
    Y = torch.zeros_like(i2)
    _get_mod().fused_seg_lut(i2, luts, mask, Y)
    return Y.reshape(-1, 3)


register(Backend(name="fused", run=run, unsupported=_unsupported,
                 available=_available, priority=40, torch_native=True,
                 differentiable=False, auto=False))
