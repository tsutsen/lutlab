"""k_fused -- single-pass HIP kernels for cubic uniform 3-D LUTs (opt-in).

Native to lutlab: the HIP source below is compiled on first use via
``torch.utils.cpp_extension.load_inline``. Opt-in only (never chosen by
``backend="auto"``): it needs torch + a CUDA/ROCm device (and the matching
nvcc/hipcc toolchain), supports cubic uniform 3-D grids, and has no backward
pass.  The kernels are written for a *segmented* apply (S LUTs masked per
pixel); the lutlab backend drives them with ``S=1`` for a plain cubic LUT.

Schemes:
  * ``nearest``      -- one node, rounds to the nearest grid point;
  * ``multilinear``  -- eight nodes, same result as the multi-linear oracle;
  * ``tetrahedral``  -- four nodes, same barycentric subdivision as the
                        numba/torch backends (and libplacebo's custom LUT).

Each scheme is a separate kernel (rather than a mode branch inside one kernel)
so the register allocation and occupancy of the hot path are not inflated by
the other two schemes.
"""
from __future__ import annotations

import importlib.util

from ._registry import Backend, Request, register

_CUDA_SRC = r"""
#include <hip/hip_runtime.h>

__device__ __forceinline__ int clamp_int(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float masked_value(
    const float* __restrict__ masks, int S, int px, int H, int W, int s)
{
    return (S == 1) ? 1.f : masks[(size_t)s * H * W + px];
}

__global__ void fused_nearest_kernel(
    const float* __restrict__ I,
    const float* __restrict__ luts,
    const float* __restrict__ masks,
    float* __restrict__ Y,
    int H, int W, int S, int D)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= H * W) return;

    float x = fminf(fmaxf(I[px*3],     0.f), 1.f);
    float y = fminf(fmaxf(I[px*3 + 1], 0.f), 1.f);
    float z = fminf(fmaxf(I[px*3 + 2], 0.f), 1.f);
    float sc = (float)(D - 1);
    int xi = clamp_int((int)(x * sc + 0.5f), 0, D - 1);
    int yi = clamp_int((int)(y * sc + 0.5f), 0, D - 1);
    int zi = clamp_int((int)(z * sc + 0.5f), 0, D - 1);
    int idx = (xi * D + yi) * D + zi;
    size_t lsz = (size_t)D * D * D * 3;
    float oR = 0.f, oG = 0.f, oB = 0.f;
    for (int s = 0; s < S; s++) {
        float m = masked_value(masks, S, px, H, W, s);
        if (m < 1e-4f) continue;
        const float* L = luts + (size_t)s * lsz;
        oR += m * L[idx*3];
        oG += m * L[idx*3 + 1];
        oB += m * L[idx*3 + 2];
    }
    Y[px*3] = oR; Y[px*3 + 1] = oG; Y[px*3 + 2] = oB;
}

__global__ void fused_multilinear_kernel(
    const float* __restrict__ I,
    const float* __restrict__ luts,
    const float* __restrict__ masks,
    float* __restrict__ Y,
    int H, int W, int S, int D)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= H * W) return;

    float x = fminf(fmaxf(I[px*3],     0.f), 1.f);
    float y = fminf(fmaxf(I[px*3 + 1], 0.f), 1.f);
    float z = fminf(fmaxf(I[px*3 + 2], 0.f), 1.f);
    float sc = (float)(D - 1);
    float fx = x * sc, fy = y * sc, fz = z * sc;
    int xi = clamp_int((int)fx, 0, D - 2);
    int yi = clamp_int((int)fy, 0, D - 2);
    int zi = clamp_int((int)fz, 0, D - 2);
    float dx = fx - xi, dy = fy - yi, dz = fz - zi;
    float iDX = 1.f - dx, iDY = 1.f - dy, iDZ = 1.f - dz;
    int base = (xi * D + yi) * D + zi;
    size_t lsz = (size_t)D * D * D * 3;
    float oR = 0.f, oG = 0.f, oB = 0.f;

    for (int s = 0; s < S; s++) {
        float m = masked_value(masks, S, px, H, W, s);
        if (m < 1e-4f) continue;
        const float* L = luts + (size_t)s * lsz;
        int b000 = base,     b100 = b000 + D * D;
        int b010 = b000 + D, b110 = b100 + D;
        int b001 = b000 + 1, b101 = b100 + 1;
        int b011 = b010 + 1, b111 = b110 + 1;

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
    Y[px*3] = oR; Y[px*3 + 1] = oG; Y[px*3 + 2] = oB;
}

__global__ void fused_tetrahedral_kernel(
    const float* __restrict__ I,
    const float* __restrict__ luts,
    const float* __restrict__ masks,
    float* __restrict__ Y,
    int H, int W, int S, int D)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= H * W) return;

    float x = fminf(fmaxf(I[px*3],     0.f), 1.f);
    float y = fminf(fmaxf(I[px*3 + 1], 0.f), 1.f);
    float z = fminf(fmaxf(I[px*3 + 2], 0.f), 1.f);
    float sc = (float)(D - 1);
    float fx = x * sc, fy = y * sc, fz = z * sc;
    int xi = clamp_int((int)fx, 0, D - 2);
    int yi = clamp_int((int)fy, 0, D - 2);
    int zi = clamp_int((int)fz, 0, D - 2);
    float dx = fx - xi, dy = fy - yi, dz = fz - zi;
    int base = (xi * D + yi) * D + zi;

    // Sort fractional coordinates descending; (a,b,c) are input-axis indices.
    int a = 0, b = 1, c = 2;
    float fa = dx, fb = dy, fc = dz;
    if (fa < fb) { int t = a; a = b; b = t; float f = fa; fa = fb; fb = f; }
    if (fb < fc) { int t = b; b = c; c = t; float f = fb; fb = fc; fc = f; }
    if (fa < fb) { int t = a; a = b; b = t; float f = fa; fa = fb; fb = f; }
    int st[3] = {D * D, D, 1};
    int v1 = base + st[a];
    int v2 = v1 + st[b];
    int v3 = base + D * D + D + 1;
    float w0 = 1.f - fa, w1 = fa - fb, w2 = fb - fc, w3 = fc;
    size_t lsz = (size_t)D * D * D * 3;
    float oR = 0.f, oG = 0.f, oB = 0.f;

    for (int s = 0; s < S; s++) {
        float m = masked_value(masks, S, px, H, W, s);
        if (m < 1e-4f) continue;
        const float* L = luts + (size_t)s * lsz;
        oR += m * (w0*L[base*3]   + w1*L[v1*3]   + w2*L[v2*3]   + w3*L[v3*3]);
        oG += m * (w0*L[base*3+1] + w1*L[v1*3+1] + w2*L[v2*3+1] + w3*L[v3*3+1]);
        oB += m * (w0*L[base*3+2] + w1*L[v1*3+2] + w2*L[v2*3+2] + w3*L[v3*3+2]);
    }
    Y[px*3] = oR; Y[px*3 + 1] = oG; Y[px*3 + 2] = oB;
}

extern "C" void launch_fused_seg_lut(
    const float* I, const float* luts, const float* masks,
    float* Y, int H, int W, int S, int D, int mode)
{
    int total = H * W, threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (mode == 0) {
        fused_nearest_kernel<<<blocks, threads>>>(I, luts, masks, Y, H, W, S, D);
    } else if (mode == 2) {
        fused_tetrahedral_kernel<<<blocks, threads>>>(I, luts, masks, Y, H, W, S, D);
    } else {
        fused_multilinear_kernel<<<blocks, threads>>>(I, luts, masks, Y, H, W, S, D);
    }
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
extern "C" void launch_fused_seg_lut(
    const float*, const float*, const float*,
    float*, int, int, int, int, int);
void fused_seg_lut(torch::Tensor I, torch::Tensor luts,
                   torch::Tensor masks, torch::Tensor Y, int mode) {
    int H = I.size(0), W = I.size(1);
    int S = luts.size(0), D = luts.size(1);
    launch_fused_seg_lut(
        I.data_ptr<float>(), luts.data_ptr<float>(),
        masks.numel() ? masks.data_ptr<float>() : nullptr,
        Y.data_ptr<float>(), H, W, S, D, mode);
}
"""

_mod = None
_MODES = {"nearest": 0, "multilinear": 1, "tetrahedral": 2}


def _get_mod():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(
            name="lutlab_fused_seg_lut_v5",
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
    if req.scheme not in _MODES or req.n_in != 3:
        return "fused kernel supports nearest/multilinear/tetrahedral 3D lookups only"
    if not req.uniform or len(set(req.sizes)) != 1:
        return "fused kernel needs a uniform cubic grid"
    return None


def run(volume, x, axes, scheme):
    import torch
    if x.dtype == torch.float32 and x.is_contiguous():
        i2 = x.reshape(-1, 1, 3)                            # kernel reads I[px*3]
    else:
        i2 = x.float().contiguous().reshape(-1, 1, 3)
    if volume.dtype == torch.float32 and volume.is_contiguous():
        luts = volume.unsqueeze(0)                          # (1, D, D, D, 3)
    else:
        luts = volume.float().contiguous().unsqueeze(0)
    # S=1: the kernel supplies m=1 and never reads the mask.  A zero-length
    # tensor avoids allocating a 1920x1080 float mask per call.
    mask = torch.empty(0, dtype=torch.float32, device=x.device)
    Y = torch.empty_like(i2)
    _get_mod().fused_seg_lut(i2, luts, mask, Y, _MODES[scheme])
    return Y.reshape(-1, 3)


register(Backend(name="fused", run=run, unsupported=_unsupported,
                 available=_available, priority=40, torch_native=True,
                 differentiable=False, auto=False))
