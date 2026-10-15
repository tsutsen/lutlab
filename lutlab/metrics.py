"""Metrics: accuracy (CIEDE2000 / PSNR / SSIM), speed (ms -> fps), and power.

Accuracy answers "how close is the result to the reference?". Speed answers
"how fast is the apply" (ms and, for video work, fps). Power is a coarse
ballpark (watts / mJ per call) -- see `power_sample` for the honest caveats.
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

import numpy as np
from skimage import color as skcolor
from skimage.metrics import (peak_signal_noise_ratio,
                             structural_similarity)

from .lut import LUT, identity

__all__ = ["e00", "psnr", "ssim", "accuracy", "lut_space_accuracy",
           "time_apply", "to_fps", "apply_speed", "gpu_power_watts",
           "power_sample", "cpu_energy_uj", "evaluate"]

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #
def e00(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel CIEDE2000 colour distance between two (...,3) arrays in [0,1].

    ELI5: the "how different are these two colours" number the eye uses. 0 is
    identical; single digits is a subtle shift; 10+ is clearly different.
    """
    la = skcolor.rgb2lab(np.clip(np.asarray(a), _EPS, 1 - _EPS))
    lb = skcolor.rgb2lab(np.clip(np.asarray(b), _EPS, 1 - _EPS))
    return skcolor.deltaE_ciede2000(la, lb)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (higher is better, +inf is identical)."""
    a, b = np.asarray(a), np.asarray(b)
    if np.array_equal(a, b):                       # identical -> inf, no log(1/0)
        return float("inf")
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity, 0..1 (1 is identical)."""
    a, b = np.asarray(a), np.asarray(b)
    ws = min(7, min(a.shape[:2]) // 2 | 1)
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0,
                                       win_size=ws))


def accuracy(out: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Image-space accuracy of `out` vs `ref` (both (H,W,3) in [0,1])."""
    d = e00(out, ref)
    return {"e00_mean": float(d.mean()), "e00_max": float(d.max()),
            "psnr_db": psnr(out, ref), "ssim": ssim(out, ref)}


def lut_space_accuracy(lut: LUT | np.ndarray,
                       ref_lut: LUT | np.ndarray | None = None) -> dict[str, float]:
    """LUT-space distance: how far `lut` sits from `ref_lut` (default identity).

    Cheap way to characterise a LUT without an image: "how much does this LUT
    move colours, on average and at worst?"
    """
    ref = ref_lut if ref_lut is not None else identity(lut.D, lut.ndim)
    a, b = lut.vol.reshape(-1, 3), ref.vol.reshape(-1, 3)
    d = e00(a, b)
    return {"e00_mean": float(d.mean()), "e00_max": float(d.max()),
            "max_channel": float(np.abs(lut.vol - ref.vol).max())}


# --------------------------------------------------------------------------- #
# speed
# --------------------------------------------------------------------------- #
def time_apply(fn: Callable, warmup: int = 3, repeat: int = 20) -> float:
    """Time a zero-arg callable: `warmup` throws, then `repeat` timed runs.

    Returns the MEDIAN in milliseconds (median, not mean, so one hiccup doesn't
    skew the number).
    """
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(ts))


def to_fps(ms: float) -> float:
    """Convert a per-frame apply time (ms) to frames per second.

    ELI5: if applying the LUT takes 0.26 ms, you can do 1000/0.26 ~= 3846
    frames per second.  For a 24/30/60 fps video the apply is a tiny fraction
    of your frame budget when fps is high.
    """
    return 1000.0 / ms if ms and ms > 0 else float("inf")


def apply_speed(lut: LUT | np.ndarray, img: np.ndarray,
                interp: str = "multilinear", backend: str = "auto",
                warmup: int = 3, repeat: int = 20) -> dict[str, float | str]:
    """Measure one apply: returns {ms, fps, backend, Mp_s}."""
    from .apply import apply as _apply
    img = np.asarray(img)
    npix = img.size // img.shape[-1]
    ms = time_apply(lambda: _apply(lut, img, interp=interp, backend=backend),
                    warmup, repeat)
    return {"ms": ms, "fps": to_fps(ms), "backend": backend,
            "Mp_s": round(npix / (ms * 1000.0), 1) if ms > 0 else float("inf")}


# --------------------------------------------------------------------------- #
# power (coarse)
# --------------------------------------------------------------------------- #
def gpu_power_watts(device: str = "0") -> float | None:
    """Current GPU graphics-package power in watts (via `rocm-smi`).

    ELI5: asks the driver "how many watts is the GPU using right now?". The
    number is the WHOLE package (GPU + memory) averaged over the driver's own
    window, so it is a ballpark, not a per-kernel joule count.
    """
    try:
        out = subprocess.run(["rocm-smi", "--showpower", "-d", str(device)],
                             capture_output=True, text=True, timeout=15)
        for line in out.stdout.splitlines():
            if "Power (W)" in line:
                return float(line.rsplit(":", 1)[1].strip())
    except Exception:
        return None
    return None


def power_sample(fn: Callable, warmup: int = 3, repeat: int = 20,
                 device: str = "0") -> dict[str, float | None]:
    """Sample GPU watts around `fn` and estimate energy per call.

    Returns {watts, ms, mJ_per_call, caveat}.

    .. warning::
       ``rocm-smi`` has ~ms latency and reports the whole package, so on a
       GPU that is already busy (e.g. an LLM server) this mostly measures the
       *other* workload, not a sub-millisecond LUT apply. It is only
       meaningful on an otherwise-idle GPU -- exactly the edge-device case it
       is meant for.
    """
    for _ in range(warmup):
        fn()
    p0 = gpu_power_watts(device)
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    p1 = gpu_power_watts(device)
    watts = None
    if p0 is not None and p1 is not None:
        watts = (p0 + p1) / 2.0
    elif p0 is not None or p1 is not None:
        watts = p0 if p0 is not None else p1
    ms = float(np.median(ts))
    mj = (watts * ms) if watts is not None else None   # W * ms = mJ
    return {"watts": watts, "ms": ms, "mJ_per_call": mj,
            "caveat": "whole-package power; a busy GPU inflates the reading"}


def cpu_energy_uj() -> int | None:
    """CPU energy in microjoules from RAPL, or None if not readable.

    Reading `/sys/class/powercap/.../energy_uj` normally needs root, so on an
    unprivileged box this returns None.
    """
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
            return int(f.read())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# one-call convenience
# --------------------------------------------------------------------------- #
def evaluate(lut: LUT | np.ndarray, img: np.ndarray, ref: np.ndarray,
             interp: str = "multilinear", backend: str = "auto",
             measure_power: bool = False) -> dict:
    """Apply + score in one shot: accuracy vs `ref` plus apply speed.

    If `measure_power` is set (and it is safe to do so) also sample GPU power.
    """
    from .apply import apply as _apply
    out = _apply(lut, img, interp=interp, backend=backend)
    r = {"accuracy": accuracy(out, ref), **apply_speed(lut, img, interp, backend)}
    if measure_power:
        r["power"] = power_sample(lambda: _apply(lut, img, interp=interp,
                                                 backend=backend))
    return r
