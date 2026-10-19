"""Segmented LUTs: ``Y(x) = sum_s m_s(x) * LUT_s(x)`` with per-segment parametric grades.

Segmentation is a pluggable backend (``kmeans`` default; ``sam`` is a stub until
you register a model). Fitting consumes soft masks, so any backend returning
masks that sum to 1 plugs in. Masks are channel-last ``(H, W, S)``.

Needs `robust_hald` (parametric forward model + baking) and scipy.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._deps import require
from ..transforms.lut import LUT
from ..transforms.mixture import LUTMixture
from . import register

__all__ = ["segment_image", "register_segmenter", "available_segmenters",
           "fit_with_masks", "fit_segmented", "SegmentedResult"]


def _kmeans(img: np.ndarray, n_segments: int = 4, **kw):
    masks, meta = require("robust_hald.segmented_lut").kmeans_segment(img, n_segments, **kw)
    return np.moveaxis(np.asarray(masks, np.float32), 0, -1), meta     # (S,H,W) -> (H,W,S)


def _sam(img: np.ndarray, n_segments: int = 4, **kw):
    """Frozen-SAM segmentation (STUB): register a real one with `register_segmenter`."""
    raise NotImplementedError(
        "SAM segmentation is a stub. Register one: "
        "register_segmenter('sam', fn) with fn(img, n_segments, **kw) -> "
        "(soft_masks (H, W, S) summing to 1, meta).")


_SEGMENTERS: dict[str, Callable] = {"kmeans": _kmeans, "sam": _sam}


def register_segmenter(name: str, fn: Callable | None = None):
    """Register (or decorate) a segmenter ``fn(img, n_segments, **kw) -> (masks (H,W,S), meta)``."""
    if fn is None:
        def deco(f):
            _SEGMENTERS[str(name)] = f
            return f
        return deco
    _SEGMENTERS[str(name)] = fn
    return fn


def available_segmenters() -> list[str]:
    return list(_SEGMENTERS)


def segment_image(img: np.ndarray, n_segments: int = 4, backend: str = "kmeans", **kw):
    """Segment `img` -> soft masks ``(H, W, S)`` summing to 1, plus backend meta."""
    try:
        seg = _SEGMENTERS[backend]
    except KeyError:
        raise ValueError(f"unknown segmentation backend {backend!r}; "
                         f"available: {', '.join(_SEGMENTERS)}") from None
    return seg(img, n_segments, **kw)


_DEFAULT_PARAMS = ["luminance", "contrast", "contrast_pivot", "shadow_falloff",
                   "highlight_falloff", "temperature", "tint", "hue", "saturation", "vibrance"]
_BOUNDS = [(-0.5, 0.5), (-0.5, 0.5), (0.2, 0.8), (-0.5, 0.5), (-0.5, 0.5),
           (-0.3, 0.3), (-0.3, 0.3), (-30, 30), (0.3, 2.0), (-0.5, 0.5)]


def fit_with_masks(I: np.ndarray, J: np.ndarray, masks: np.ndarray,
                   params_to_fit: list[str] | None = None, sample_n: int = 10000,
                   max_iter: int = 100, seed: int = 42, verbose: bool = False) -> dict:
    """Fit per-segment grade parameters given soft ``masks (H, W, S)``."""
    sl = require("robust_hald.segmented_lut")
    minimize = require("scipy.optimize").minimize
    params_to_fit = params_to_fit or _DEFAULT_PARAMS
    masks_s = np.moveaxis(np.asarray(masks, np.float32), -1, 0)          # (S, H, W)
    S = masks_s.shape[0]
    H, W, _ = I.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(H * W, min(sample_n, H * W), replace=False)
    I_s, J_s = I.reshape(-1, 3)[idx], J.reshape(-1, 3)[idx]
    m_s = masks_s.reshape(S, -1)[:, idx]
    P_IDX = [sl.IDX[p] for p in params_to_fit]
    n_p = len(P_IDX)
    bounds = _BOUNDS[:n_p]
    thetas = [sl.IDENT.copy() for _ in range(S)]

    def full_forward(theta_list):
        Y = np.zeros_like(I_s)
        for s in range(S):
            ok = sl.forward_oklab(I_s, theta_list[s], tonal_ranges=3)
            Y += m_s[s, :, None] * np.clip(sl.oklab_to_rgb(ok), 0, 1)
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
                return float(np.mean((full_forward(thetas) - J_s) ** 2 * m_s[s, :, None]))

            x0 = np.array([thetas[s][p] for p in P_IDX])
            res = minimize(seg_loss, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": max_iter, "ftol": 1e-10})
            for i, pidx in enumerate(P_IDX):
                thetas[s][pidx] = res.x[i]
            loss_history.append(total_loss())
            if verbose:
                print(f"  cycle {cycle} seg {s}: loss={loss_history[-1]:.6f} ({res.nit} iters)",
                      flush=True)
        if len(loss_history) >= 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-8:
            break
    Y_full = sl.segmented_forward(I, masks_s, thetas)
    psnr_full = float(10 * np.log10(1.0 / max(np.mean((Y_full - J) ** 2), 1e-10)))
    return {"thetas": thetas, "psnr": psnr_full, "loss_history": loss_history,
            "params_fitted": params_to_fit}


@dataclass
class SegmentedResult:
    transform: LUTMixture                 # bound to the fitted masks
    masks: np.ndarray                     # (H, W, S)
    thetas: list = field(default_factory=list)
    psnr: float = 0.0
    meta: Any = None
    backend: str = "kmeans"

    def apply(self, img: np.ndarray, **kw) -> np.ndarray:
        return self.transform(img, **kw)


def fit_segmented(img: np.ndarray, target: np.ndarray, n_segments: int = 4, d: int = 17,
                  backend: str = "kmeans", masks: np.ndarray | None = None,
                  **fit_kw) -> SegmentedResult:
    """Segment (via `backend`), fit per-segment parameters, bake one LUT per segment.

    Pass ``masks=`` ``(H, W, S)`` to skip segmentation. The returned
    ``result.transform`` is a `LUTMixture` bound to the masks of this image.
    """
    meta = None
    if masks is None:
        masks, meta = segment_image(img, n_segments, backend=backend)
    fit = fit_with_masks(img, target, masks, **fit_kw)
    baked = require("robust_hald.fast_apply").bake_segmented_luts(fit["thetas"], D=d)
    luts = [LUT(np.asarray(v), name=f"segment_{s}") for s, v in enumerate(np.asarray(baked))]
    return SegmentedResult(LUTMixture(luts, name="segmented", weights=masks), masks,
                           fit["thetas"], fit["psnr"], meta, backend)


@register("segmented", requires=("robust_hald", "scipy"))
def build_segmented(*, image, target, **cfg) -> LUTMixture:
    """Per-image segmented LUT fitted to (image -> target); masks bound to this image."""
    return fit_segmented(image, target, **cfg).transform
