"""Image-space and LUT-space accuracy metrics."""
from __future__ import annotations

import numpy as np

from .._deps import require
from ..transforms.lut import LUT

__all__ = ["e00", "de76", "psnr", "ssim", "lpips", "accuracy",
           "lut_space_accuracy", "tone_curve"]

# linear-sRGB (Rec.709 primaries, D65) -> XYZ
_LIN_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])


def _lab(rgb, space: str) -> np.ndarray:
    color = require("skimage.color")
    a = np.clip(np.asarray(rgb, np.float64), 0.0, 1.0)
    shape = a.shape
    a = a.reshape(-1, 1, 3)                       # skimage wants an image-like array
    if space == "srgb":
        lab = color.rgb2lab(a)
    elif space == "linear":
        lab = color.xyz2lab(a @ _LIN_TO_XYZ.T)
    else:
        raise ValueError(f"unknown colour space {space!r}; use 'srgb' or 'linear'")
    return lab.reshape(shape)


def e00(a, b, *, space: str = "srgb") -> np.ndarray:
    """Per-pixel CIEDE2000 between two ``(..., 3)`` arrays in [0, 1].

    ``space`` says how to read the RGB values: ``"srgb"`` (encoded) or
    ``"linear"`` (linear light, Rec.709 primaries). 0 = identical.
    """
    color = require("skimage.color")
    la, lb = _lab(a, space), _lab(b, space)
    return color.deltaE_ciede2000(la.reshape(-1, 1, 3), lb.reshape(-1, 1, 3)).reshape(la.shape[:-1])


def de76(a, b, *, space: str = "srgb") -> np.ndarray:
    """Per-pixel CIE76 colour difference: the L2 distance in CIELAB (the paper-style ΔE_ab)."""
    return np.linalg.norm(_lab(a, space) - _lab(b, space), axis=-1)


def psnr(a, b) -> float:
    """PSNR in dB for data range 1.0 (+inf when identical)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def ssim(a, b) -> float:
    """Mean SSIM over an ``(H, W, 3)`` pair, data range 1.0."""
    metrics = require("skimage.metrics")
    a, b = np.asarray(a), np.asarray(b)
    ws = min(7, a.shape[0], a.shape[1])
    ws -= (ws % 2 == 0)
    if ws < 3:
        raise ValueError(f"image too small for SSIM: {a.shape[:2]}")
    return float(metrics.structural_similarity(a, b, channel_axis=2, data_range=1.0, win_size=ws))


_LPIPS_NET = None


def lpips(a, b, *, net: str = "alex") -> float:
    """LPIPS distance for ``(H, W, 3)`` images in [0, 1]. Needs `pip install lpips` + torch."""
    global _LPIPS_NET
    torch = require("torch")
    if _LPIPS_NET is None:
        _LPIPS_NET = require("lpips").LPIPS(net=net, verbose=False).eval()

    def prep(x):
        t = torch.as_tensor(np.asarray(x, np.float32)).permute(2, 0, 1)[None]
        return t * 2.0 - 1.0

    with torch.no_grad():
        return float(_LPIPS_NET(prep(a), prep(b)).item())


def accuracy(out, ref, *, space: str = "srgb") -> dict[str, float]:
    """Image-space accuracy of ``out`` vs ``ref`` (both ``(H, W, 3)`` in [0, 1])."""
    d = e00(out, ref, space=space)
    return {"e00_mean": float(d.mean()), "e00_max": float(d.max()),
            "psnr_db": psnr(out, ref), "ssim": ssim(out, ref)}


def lut_space_accuracy(lut, ref_lut=None, *, space: str | None = None) -> dict[str, float]:
    """How far ``lut`` sits from ``ref_lut`` (default: identity), on the grid.

    ``max_channel`` is the largest per-channel change the LUT makes at any grid
    point (``delta_from_identity`` in 0.2).
    """
    ref = ref_lut if ref_lut is not None else LUT.identity(sizes=lut.sizes, axes=lut.axes)
    if lut.sizes != ref.sizes:
        raise ValueError(f"grids differ: {lut.sizes} vs {ref.sizes}; resample first")
    d = e00(lut.volume.reshape(-1, 3), ref.volume.reshape(-1, 3), space=space or lut.space)
    return {"e00_mean": float(d.mean()), "e00_max": float(d.max()),
            "max_channel": float(np.abs(lut.volume - ref.volume).max())}


def tone_curve(lut, n: int = 33) -> tuple[np.ndarray, np.ndarray]:
    """Output of ``lut`` along the grey diagonal ``(t, ..., t)``: returns ``(t, curve[n, 3])``."""
    from ..kernels.dispatch import interpolate
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = np.repeat(t[:, None], lut.n_in, axis=1)
    return t, interpolate(lut.volume, x, axes=lut.axes, scheme="multilinear")
