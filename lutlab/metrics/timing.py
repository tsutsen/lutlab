"""Wall-clock timing with correct GPU synchronisation."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .._arrays import is_torch

__all__ = ["time_fn", "to_fps", "apply_speed", "cuda_sync"]


def cuda_sync() -> None:
    """Block until queued GPU work finishes (no-op without CUDA/ROCm)."""
    import sys
    t = sys.modules.get("torch")
    if t is not None and t.cuda.is_available():
        t.cuda.synchronize()


def time_fn(fn: Callable, *, warmup: int = 3, repeat: int = 20,
            sync: Callable | None = cuda_sync) -> dict[str, float]:
    """Time a zero-arg callable. GPU kernels launch asynchronously, so `sync`
    (default: cuda_sync) runs before and after each timed call; pass None to skip."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(repeat):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1000.0)
    a = np.asarray(ts)
    return {"median_ms": float(np.median(a)), "min_ms": float(a.min()),
            "p90_ms": float(np.percentile(a, 90)), "repeat": repeat}


def to_fps(ms: float) -> float:
    return 1000.0 / ms if ms > 0 else float("inf")


def apply_speed(transform, x, *, warmup: int = 3, repeat: int = 20, **apply_kw) -> dict:
    """Time ``apply(transform, x, **apply_kw)``: ms, fps and megapixels/s."""
    from ..kernels.dispatch import apply
    npix = int(np.prod(x.shape[:-1]))
    r = time_fn(lambda: apply(transform, x, **apply_kw), warmup=warmup, repeat=repeat,
                sync=cuda_sync if (is_torch(x) or apply_kw.get("device")) else None)
    ms = r["median_ms"]
    return {**r, "fps": to_fps(ms), "mpx_s": npix / (ms * 1000.0) if ms > 0 else float("inf")}
