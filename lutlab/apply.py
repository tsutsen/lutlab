"""One `apply()` over every backend + interpolation scheme.

The two knobs are independent:

  interp   -- WHICH corners to blend: nearest, bilinear, trilinear,
              quadrilinear, multilinear, tetrahedral (see lutlab.interp).
  backend  -- WHERE the math runs:
              * gpu     -- hardware trilinear via torch `grid_sample`
                           (trilinear, 3D only)
              * fused   -- fused HIP kernel (trilinear, 3D; fastest, needs
                           ROCm dev packages)
              * numba   -- fused CPU kernels (all schemes, 3D fastest)
              * numpy   -- reference implementation (all schemes, any dim)
              * auto    -- gpu if it applies, else numba, else numpy

A LUT and a bare (D,)*n+(3,) array both work; a bare array is treated as the
LUT volume.
"""
from __future__ import annotations

import numpy as np

from . import interp as _interp
from .interp import interp_name
from .lut import LUT

__all__ = ["apply", "segmented_apply", "backends"]


def _vol(lut: LUT | np.ndarray) -> np.ndarray:
    return getattr(lut, "vol", lut)


def _gpu_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:                              # pragma: no cover
        return False


def backends() -> dict[str, bool]:
    """Which backends are usable right now (gpu needs a live device)."""
    return {"numpy": True, "numba": _interp._HAS_NUMBA,
            "gpu": _gpu_ok(), "fused": _gpu_ok()}


def _is_cuda(x) -> bool:
    try:
        import torch
        return torch.is_tensor(x) and x.is_cuda
    except Exception:
        return False


def _to_cuda(x):
    """numpy array or torch tensor -> CUDA float32 tensor (no copy if present)."""
    import torch
    if torch.is_tensor(x):
        return x.float().contiguous().cuda() if not x.is_cuda else x.float().contiguous()
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x, np.float32))).cuda()


def _apply_gs(v: np.ndarray, img):
    """Trilinear 3D via grid_sample. CUDA-in -> CUDA tensor out (no round-trip)."""
    import torch
    from robust_hald.gpu_apply import lut_apply_gpu
    orig = tuple(img.shape)
    I = _to_cuda(img).reshape(1, -1, 3)
    out = lut_apply_gpu(I, torch.from_numpy(np.ascontiguousarray(v)).cuda())
    out = out.reshape(orig)
    return out if _is_cuda(img) else out.cpu().numpy()


def _apply_fused(v: np.ndarray, img):
    """Trilinear 3D via the fused HIP kernel (fastest; needs ROCm dev pkgs)."""
    import torch
    from robust_hald.fused_kernel import fused_segmented_apply
    orig = tuple(img.shape)
    I = _to_cuda(img)
    I2 = I.reshape(-1, 1, 3)                      # (N, 1, 3): kernel reads I[px*3]
    luts = torch.from_numpy(np.ascontiguousarray(v)).cuda().unsqueeze(0)
    mask = torch.ones(1, I2.shape[0], 1, device=I.device)
    out = fused_segmented_apply(I2, luts, mask).reshape(orig)
    return out if _is_cuda(img) else out.cpu().numpy()


def apply(lut: LUT | np.ndarray, img: np.ndarray | "torch.Tensor",
          interp: str = "multilinear", backend: str = "auto"):
    """Apply `lut` to `img`. `img` is (..., n) in [0,1], n = lut.ndim.

    `img` may be a numpy array or a CUDA tensor; a CUDA tensor is used in place
    (no CPU round-trip) and a CUDA tensor is returned. Returns a numpy array for
    numpy input.

    backend: 'auto' (grid_sample GPU when it applies, else fastest CPU), 'gpu'
    (grid_sample), 'fused' (fused HIP kernel, fastest, needs ROCm dev packages),
    'numba', 'numpy'.
    """
    v = np.asarray(_vol(lut), dtype=np.float32)
    n = v.ndim - 1
    import torch as _torch
    nc = img.shape[-1] if _torch.is_tensor(img) else np.asarray(img).shape[-1]
    if nc != n:
        raise ValueError(f"input has {nc} channels, LUT takes {n}")
    kernel = interp_name(interp, n)

    # Fast GPU paths: trilinear 3D only.
    if kernel == "multilinear" and n == 3 and _gpu_ok() and \
            backend in ("auto", "gpu", "fused"):
        if backend == "fused":
            return _apply_fused(v, img)
        try:
            return _apply_gs(v, img)
        except Exception:                          # pragma: no cover
            if backend == "gpu":
                raise
            # backend == 'auto' -> fall through to the software path

    if backend == "fused":
        raise ValueError("fused backend needs trilinear 3D + ROCm dev packages; "
                         "use backend='numba' or 'numpy'")
    xa = np.asarray(img, dtype=np.float32)
    engine = backend if backend in ("numpy", "numba") else "auto"
    return _interp.interp(v, xa, mode=interp, engine=engine)


def segmented_apply(lut_or_luts: np.ndarray | list,
                    img: np.ndarray, masks: np.ndarray,
                    backend: str = "auto") -> np.ndarray:
    """Y = sum_s masks[s] * LUT_s(img).

    `lut_or_luts` is (S,D,D,D,3) or a list of LUTs/vols; `masks` is (S,H,W)
    (or (S,N) broadcastable to the last axes of `img`) summing to 1.
    backend: auto/gpu -> grid_sample; 'fused' -> the HIP kernel (fastest,
    compiles on first use); 'numpy' -> per-segment trilinear loop.
    """
    luts = lut_or_luts
    if not isinstance(luts, np.ndarray):
        luts = np.stack([np.asarray(_vol(L), np.float32) for L in luts])
    luts = np.asarray(luts, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    img = np.asarray(img, dtype=np.float32)

    if backend in ("auto", "gpu") and _gpu_ok():
        try:
            import torch
            from robust_hald.gpu_apply import segmented_apply_gpu
            I_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()
            L_t = torch.from_numpy(np.ascontiguousarray(luts)).cuda()
            M_t = torch.from_numpy(np.ascontiguousarray(masks)).cuda()
            return segmented_apply_gpu(I_t, L_t, M_t).cpu().numpy()
        except Exception:                          # pragma: no cover
            if backend == "gpu":
                raise
    elif backend == "fused":
        import torch
        from robust_hald.fused_kernel import fused_segmented_apply
        I_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()
        L_t = torch.from_numpy(np.ascontiguousarray(luts)).cuda()
        M_t = torch.from_numpy(np.ascontiguousarray(masks)).cuda()
        return fused_segmented_apply(I_t, L_t, M_t).cpu().numpy()

    # numpy fallback
    out = np.zeros_like(img)
    for s in range(luts.shape[0]):
        out = out + masks[s][..., None] * _interp.interp(
            luts[s], img, mode="trilinear")
    return out
