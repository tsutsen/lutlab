"""Coarse power / energy sampling. Hardware-specific; see caveats on each function."""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

import numpy as np

__all__ = ["gpu_power_watts", "power_sample", "cpu_energy_uj"]


def gpu_power_watts(device: str = "0") -> float | None:
    """Current GPU package power in watts via ``rocm-smi`` (None if unavailable).

    Whole-package, driver-averaged: a ballpark, not a per-kernel joule count.
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


def power_sample(fn: Callable, *, warmup: int = 3, repeat: int = 20,
                 device: str = "0") -> dict:
    """Sample GPU watts around `fn` and estimate energy per call.

    Only meaningful on an otherwise idle GPU: ``rocm-smi`` has ~ms latency and
    reports the whole package, so on a busy GPU it mostly measures the other work.
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
    ps = [p for p in (p0, p1) if p is not None]
    watts = float(np.mean(ps)) if ps else None
    ms = float(np.median(ts))
    return {"watts": watts, "ms": ms, "mJ_per_call": None if watts is None else watts * ms,
            "caveat": "whole-package power; a busy GPU inflates the reading"}


def cpu_energy_uj() -> int | None:
    """CPU package energy in microjoules from Intel RAPL (usually needs root)."""
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
            return int(f.read())
    except Exception:
        return None
