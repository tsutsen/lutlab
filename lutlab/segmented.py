"""Segmented (promptable) LUTs -- Art 5 pipeline.

    Y(x) = sum_s  m_s(x) * P(x; theta_s)

The one thing that changes across the papers is HOW the regions `m_s` come
from. That is a pluggable *segmentation backend*:

    segment_image(img, n_segments, backend="kmeans")   # default: colour K-means in OKLab
    segment_image(img, n_segments, backend="sam")      # frozen SAM (stub until a model is wired)
    segment_image(img, n_segments, backend="mything")  # anything you register

Fitting consumes pre-computed soft masks, so ANY backend that returns masks
summing to 1 drops straight in -- the fit does not care who made the masks.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from robust_hald.segmented_lut import (
    kmeans_segment, fit_segmented, segmented_forward, apply_params,
    forward_oklab, oklab_to_rgb, IDX, IDENT, N_PARAMS,
)
from robust_hald.seg_format import SegmentedLUT, GLOBAL_DIALS
from robust_hald.fast_apply import bake_lut, bake_segmented_luts

from .apply import segmented_apply

__all__ = ["segment_image", "register_segmenter", "available_backends",
           "fit_with_masks", "fit_segmented", "fit_and_apply",
           "segmented_forward", "SegmentedLUT", "GLOBAL_DIALS",
           "bake_lut", "bake_segmented_luts", "segmented_apply"]


def _sam_segment(img: np.ndarray, n_segments: int = 4, **kw) -> tuple[np.ndarray, str]:
    """Frozen SAM segmentation (STUB).

    No frozen-SAM model lives in this environment, so this raises until you
    plug one in:

        import lutlab.segmented as seg
        seg.register_segmenter("sam", my_sam_fn)
        # my_sam_fn(img, n_segments, **kw) -> (masks (S,H,W) soft, meta)
        # masks must be float32 and sum to 1 per pixel (feather SAM's hard
        # masks if you want the soft blend the pipeline expects).
    """
    raise NotImplementedError(
        "SAM segmentation is a stub in lutlab (no frozen-SAM model here). "
        "Enable it with lutlab.segmented.register_segmenter('sam', fn); "
        "fn(img, n_segments) must return (soft_masks (S,H,W), meta). The "
        "prototype uses backend='kmeans' as the SAM stand-in.")


_SEGMENTERS = {"kmeans": kmeans_segment, "sam": _sam_segment}


def register_segmenter(name: str, fn: Callable | None = None) -> Callable | None:
    """Register (or decorate) a segmentation backend.

    Use it as a decorator, or call it with the function:

    .. code-block:: python

       @segmented.register_segmenter("myseg")
       def myseg(img, n_segments, **kw):
           return masks, meta
    """
    if fn is None:
        return lambda f: _SEGMENTERS.__setitem__(str(name), f)
    _SEGMENTERS[str(name)] = fn
    return fn


def available_backends() -> list[str]:
    """Names of the segmentation backends available right now."""
    return list(_SEGMENTERS)


def segment_image(img: np.ndarray, n_segments: int = 4, backend: str = "kmeans",
                  **kw) -> tuple[np.ndarray, str]:
    """Segment `img` into `n_segments` regions -> (masks, meta).

    Returns soft `masks` (S,H,W) float32 summing to 1 per pixel, plus a
    backend-specific `meta` (kmeans -> OKLab centers). `backend` is 'kmeans'
    (default) or any name registered via `register_segmenter`.
    """
    try:
        seg = _SEGMENTERS[backend]
    except KeyError:
        raise ValueError(f"unknown segmentation backend {backend!r}; "
                         f"available: {', '.join(_SEGMENTERS)}") from None
    return seg(img, n_segments, **kw)


def fit_with_masks(I: np.ndarray, J: np.ndarray, masks: np.ndarray,
                   params_to_fit: list[str] | None = None, sample_n: int = 10000,
                   max_iter: int = 100, seed: int = 42,
                   verbose: bool = False) -> dict:
    """Fit per-segment params given pre-computed soft `masks` (S,H,W).

    This is what makes the segmentation backend pluggable: whatever produced
    `masks` (kmeans, SAM, ...) can be fitted. Ported from
    ``robust_hald.segmented_lut.fit_segmented``, which just prepends a kmeans
    step before this same loop.
    """
    if params_to_fit is None:
        params_to_fit = ["luminance", "contrast", "contrast_pivot",
                         "shadow_falloff", "highlight_falloff",
                         "temperature", "tint", "hue", "saturation", "vibrance"]
    S = masks.shape[0]
    H, W, _ = I.shape
    rng = np.random.RandomState(seed)
    n_pix = H * W
    n_samp = min(sample_n, n_pix)
    idx = rng.choice(n_pix, n_samp, replace=False)
    I_s = I.reshape(-1, 3)[idx]
    J_s = J.reshape(-1, 3)[idx]
    m_s = masks.reshape(S, -1)[:, idx]
    P_IDX = [IDX[p] for p in params_to_fit]
    n_p = len(P_IDX)
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0.2, 0.8), (-0.5, 0.5), (-0.5, 0.5),
              (-0.3, 0.3), (-0.3, 0.3), (-30, 30), (0.3, 2.0), (-0.5, 0.5)][:n_p]
    thetas = [IDENT.copy() for _ in range(S)]

    def full_forward(theta_list):
        Y = np.zeros_like(I_s)
        for s in range(S):
            ok = forward_oklab(I_s, theta_list[s], tonal_ranges=3)
            rgb = np.clip(oklab_to_rgb(ok), 0, 1)
            Y += m_s[s, :, None] * rgb
        return Y

    def total_loss():
        return float(np.mean((full_forward(thetas) - J_s) ** 2))

    loss_history = [total_loss()]
    if verbose:
        print(f"  initial loss: {loss_history[0]:.6f}", flush=True)
    for cycle in range(3):
        for s in range(S):
            theta_base = thetas[s].copy()

            def seg_loss(x):
                th = theta_base.copy()
                for i, pidx in enumerate(P_IDX):
                    th[pidx] = x[i]
                thetas[s] = th
                Y = full_forward(thetas)
                return float(np.mean((Y - J_s) ** 2 * m_s[s, :, None]))

            x0 = np.array([thetas[s][pidx] for pidx in P_IDX])
            res = minimize(seg_loss, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-10})
            for i, pidx in enumerate(P_IDX):
                thetas[s][pidx] = res.x[i]
            loss_history.append(total_loss())
            if verbose:
                print(f"  cycle {cycle} seg {s}: loss={loss_history[-1]:.6f} "
                      f"({res.nit} iters)", flush=True)
        if len(loss_history) >= 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-8:
            if verbose:
                print("  converged")
            break
    Y_full = segmented_forward(I, masks, thetas)
    psnr_full = float(10 * np.log10(1.0 / max(np.mean((Y_full - J) ** 2), 1e-10)))
    return {"thetas": thetas, "masks": masks, "psnr": psnr_full,
            "loss_history": loss_history, "params_fitted": params_to_fit}


def fit_and_apply(img: np.ndarray, target: np.ndarray, n_segments: int = 4,
                  d: int = 17, backend: str = "kmeans",
                  apply_backend: str = "auto", masks: np.ndarray | None = None,
                  **fit_kw) -> dict:
    """Segment (via `backend`), fit per-segment params, bake + apply.

    `backend` picks the segmentation (kmeans / sam / registered); `apply_backend`
    picks how the finished segmented LUT is applied (gpu / fused / numpy / auto).
    Pass `masks=` to use your own segmentation and skip the backend step.
    Returns {out, masks, meta, thetas, luts, psnr, backend}.
    """
    if masks is None:
        masks, meta = segment_image(img, n_segments, backend=backend)
    else:
        meta = None
    fit = fit_with_masks(img, target, masks, **fit_kw)
    thetas = fit["thetas"]
    luts = bake_segmented_luts(thetas, D=d)
    out = segmented_apply(luts, img, masks, backend=apply_backend)
    return {"out": out, "masks": masks, "meta": meta, "thetas": thetas,
            "luts": luts, "psnr": fit["psnr"], "backend": backend}
