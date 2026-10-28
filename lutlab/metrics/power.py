"""Coarse power / energy sampling. Hardware-specific; see caveats on each function."""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

import numpy as np

from .timing import cuda_sync

__all__ = ["gpu_power_watts", "power_sample", "cpu_energy_uj"]


def _index(device: str) -> str:
    return str(device).split(":")[-1]


def _rocm_power(device: str) -> float | None:
    out = subprocess.run(["rocm-smi", "--showpower", "-d", _index(device)],
                         capture_output=True, text=True, timeout=15)
    for line in out.stdout.splitlines():
        if "Power (W)" in line:
            try:
                return float(line.rsplit(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _nvidia_power(device: str) -> float | None:
    out = subprocess.run(["nvidia-smi", f"--id={_index(device)}",
                          "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=15)
    line = out.stdout.strip().splitlines()
    if out.returncode != 0 or not line:
        return None
    try:
        return float(line[0])
    except ValueError:
        return None


def gpu_power_watts(device: str = "0") -> float | None:
    """Current GPU package power in watts via ``rocm-smi`` or ``nvidia-smi``.

    Whole-package, driver-averaged: a ballpark, not a per-kernel joule count.
    Returns ``None`` when neither tool is available.
    """
    for probe in (_rocm_power, _nvidia_power):
        try:
            watts = probe(device)
        except Exception:
            watts = None
        if watts is not None:
            return watts
    return None


def power_sample(fn: Callable, *, warmup: int = 3, repeat: int = 20,
                 device: str = "0") -> dict:
    """Sample GPU watts around `fn` and estimate energy per call.

    Only meaningful on an otherwise idle GPU: the power tools have ~ms latency and
    report the whole package, so on a busy GPU it mostly measures the other work.
    The GPU is synchronised around each call, so ``ms`` is wall time, not launch time.
    """
    for _ in range(warmup):
        fn()
    cuda_sync()
    p0 = gpu_power_watts(device)
    ts = []
    for _ in range(repeat):
        cuda_sync()
        t = time.perf_counter()
        fn()
        cuda_sync()
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
