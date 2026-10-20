"""Helpers for testing LUTs, backends and methods -- also used by lutlab's own suite.

* `random_lut`     smooth random LUT (any n_in, optional rectilinear axes)
* `check_backends` max difference of every backend/scheme vs the numpy oracle
* `conformance`    the checks every `transform` must pass
* `selftest`       quick sanity run behind ``python -m lutlab selftest``
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .kernels import BackendError, backends
from .kernels.dispatch import apply, interpolate
from .transforms.lut import LUT

__all__ = ["random_lut", "check_backends", "conformance", "selftest"]


def _positions(sizes, axes):
    if axes is None:
        return [np.linspace(0.0, 1.0, s) for s in sizes]
    return [np.asarray(a, np.float64) for a in axes]


def random_lut(size=17, n_in: int = 3, *, seed: int = 0, strength: float = 0.08,
               axes=None, name: str = "random") -> LUT:
    """Identity plus smooth random perturbation, clipped to [0, 1]."""
    sizes = (int(size),) * n_in if np.isscalar(size) else tuple(int(s) for s in size)
    n_in = len(sizes)
    rng = np.random.default_rng(seed)
    coarse = (rng.standard_normal((4,) * n_in + (3,)) * strength).astype(np.float32)
    pts = np.stack(np.meshgrid(*_positions(sizes, axes), indexing="ij"), -1)
    noise = interpolate(coarse, pts.reshape(-1, n_in).astype(np.float32),
                        scheme="multilinear").reshape(sizes + (3,))
    base = LUT.identity(sizes=sizes, axes=axes).volume
    return LUT(np.clip(base + noise, 0.0, 1.0), name=name, axes=axes)


def check_backends(lut: LUT, *, n: int = 500, seed: int = 0, atol: float = 1e-5) -> dict:
    """Compare every available backend/scheme with the numpy oracle.

    Returns ``{"backend/scheme": max_abs_diff | "skipped: reason"}``. A value
    above `atol` is a bug.
    """
    rng = np.random.default_rng(seed)
    x = rng.random((n, lut.n_in)).astype(np.float32)
    schemes = ["nearest", "multilinear"] + (["tetrahedral"] if lut.n_in == 3 else [])
    out: dict = {}
    for scheme in schemes:
        ref = apply(lut, x, interp=scheme, backend="numpy")
        for name in backends():
            try:
                y = apply(lut, x, interp=scheme, backend=name)
            except BackendError as e:
                out[f"{name}/{scheme}"] = f"skipped: {e}"
                continue
            out[f"{name}/{scheme}"] = float(np.abs(np.asarray(y) - ref).max())
    return out


def conformance(transform, *, n: int = 256, seed: int = 0, to_lut_grid_size: int | None = None,
                to_lut_tol: float = 1e-4, to_lut_kw: dict | None = None,
                skip_to_lut: bool = False) -> list[str]:
    """Checks every `transform` must pass. Returns failure messages ([] = pass).

    * output is ``(..., 3)``, finite, and leading shapes are preserved
    * calls are deterministic
    * ``to_lut(grid_size)`` (the frozen dense LUT) reproduces the native output
      within `to_lut_tol`. `to_lut_grid_size` defaults to the transform's own grid
      size (exact for a LUT on its own grid) and 17 otherwise. Freezing onto a
      grid whose cells do not align with the native one has an inherent gap (and
      so do neural methods): set `to_lut_tol` to the accepted gap. Transforms
      without ``to_lut`` (context-dependent ones) skip the check; so does
      ``skip_to_lut``.
    """
    fails: list[str] = []
    rng = np.random.default_rng(seed)
    n_in = transform.n_in
    x = rng.random((n, 3)).astype(np.float32)
    ctx = rng.random((n, n_in - 3)).astype(np.float32) if n_in > 3 else None
    try:
        y = np.asarray(transform(x, context=ctx))
    except Exception as e:
        return [f"call failed: {type(e).__name__}: {e}"]
    if y.shape != (n, 3):
        fails.append(f"output shape {y.shape}, expected {(n, 3)}")
        return fails
    if not np.isfinite(y).all():
        fails.append("output contains non-finite values")
    if not np.array_equal(y, np.asarray(transform(x, context=ctx))):
        fails.append("non-deterministic: two identical calls differ")
    try:
        x4 = x[: n // 4 * 4].reshape(4, -1, 3)
        c4 = None if ctx is None else ctx[: n // 4 * 4].reshape(4, -1, ctx.shape[-1])
        y4 = np.asarray(transform(x4, context=c4))
        if y4.shape != (4, n // 4, 3):
            fails.append(f"leading shape not preserved: got {y4.shape}")
    except Exception as e:
        fails.append(f"multi-dim input failed: {type(e).__name__}: {e}")
    if not skip_to_lut and ctx is None and hasattr(transform, "to_lut"):
        try:
            grid_size = to_lut_grid_size
            if grid_size is None:
                try:
                    grid_size = int(transform.grid_size)
                except Exception:
                    grid_size = 17
            baked = transform.to_lut(grid_size, **(to_lut_kw or {}))
            gap = float(np.abs(np.asarray(baked(x)) - y).max())
            if gap > to_lut_tol:
                fails.append(f"to_lut({grid_size}) differs from native by {gap:.2e} > {to_lut_tol:.0e}")
        except Exception as e:
            fails.append(f"to_lut failed: {type(e).__name__}: {e}")
    return fails


def selftest() -> int:
    """Quick sanity run: backend agreement + file round-trips. Returns an exit code."""
    from . import formats
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")

    cases = [("3D uniform", random_lut(17, 3)),
             ("3D rectilinear", random_lut((9, 11, 13), 3, axes=[np.linspace(0, 1, s) ** 1.5
                                                                  for s in (9, 11, 13)])),
             ("4D uniform", random_lut(7, 4))]
    for label, lut in cases:
        res = check_backends(lut)
        skipped = sorted({k.split("/")[0] for k, v in res.items() if isinstance(v, str)})
        for key, v in res.items():
            if not isinstance(v, str) and not key.startswith("numpy/"):
                check(f"{label} {key} vs numpy", v < 1e-5, f"(max diff {v:.1e})")
        if skipped:
            print(f"  [skip] {label}: {', '.join(skipped)} (unavailable or unsupported)")
    lut = random_lut(9, 3)
    with tempfile.TemporaryDirectory() as td:
        for ext, tol in ((".json", 0.0), (".npy", 0.0), (".cube", 2e-6)):
            p = Path(td) / f"t{ext}"
            lut.save(p)
            back = formats.load(p)
            err = float(np.abs(back.volume - lut.volume).max())
            check(f"{ext} round-trip", err <= tol, f"(max err {err:.1e})")
    print("selftest:", "OK" if ok else "FAILED")
    return 0 if ok else 1
