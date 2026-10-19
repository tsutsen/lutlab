"""Evaluate any transform on paired data and collect a method x task report."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from .. import __version__
from .._arrays import is_torch
from ..kernels.dispatch import apply
from . import quality
from .timing import apply_speed

__all__ = ["SCHEMA", "evaluate", "run_matrix"]

SCHEMA = "lutlab eval v2"
_METRICS = ("psnr", "ssim", "e00", "de76", "lpips")


def _np(x):
    return x.detach().cpu().numpy() if is_torch(x) else np.asarray(x)


def evaluate(transform, pairs: Iterable, *, metrics: tuple[str, ...] = ("psnr", "ssim", "e00"),
             space: str = "srgb", timing: dict | None = None, apply_kw: dict | None = None,
             max_pairs: int | None = None) -> dict:
    """Score `transform` on ``(input, target[, context])`` pairs (float, [0, 1], (H, W, 3)).

    Returns metric means over pairs, optional timing (measured on the first
    pair; ``timing={"warmup": 2, "repeat": 10}`` enables it) and the
    transform's static cost.
    """
    for m in metrics:
        if m not in _METRICS:
            raise ValueError(f"unknown metric {m!r}; options: {', '.join(_METRICS)}")
    kw = dict(apply_kw or {})
    acc: dict[str, list[float]] = {}
    first = None
    n = 0
    for item in pairs:
        if max_pairs is not None and n >= max_pairs:
            break
        inp, tgt, *rest = item
        ctx = rest[0] if rest else None
        if first is None:
            first = (inp, ctx)
        out = _np(apply(transform, inp, context=ctx, **kw))
        tgt = _np(tgt)
        if "psnr" in metrics:
            acc.setdefault("psnr_db", []).append(quality.psnr(out, tgt))
        if "ssim" in metrics:
            acc.setdefault("ssim", []).append(quality.ssim(out, tgt))
        if "e00" in metrics:
            d = quality.e00(out, tgt, space=space)
            acc.setdefault("e00_mean", []).append(float(d.mean()))
        if "de76" in metrics:
            acc.setdefault("de76_mean", []).append(float(quality.de76(out, tgt, space=space).mean()))
        if "lpips" in metrics:
            acc.setdefault("lpips", []).append(quality.lpips(out, tgt))
        n += 1
    if n == 0:
        raise ValueError("no pairs to evaluate")
    res: dict = {"n": n}
    for k, v in acc.items():
        finite = [x for x in v if np.isfinite(x)]
        res[k] = float(np.mean(finite)) if finite else float("inf")
    res["timing"] = None
    if timing is not None and first is not None:
        res["timing"] = apply_speed(transform, first[0], context=first[1], **timing, **kw)
    res["cost"] = transform.cost().as_dict() if hasattr(transform, "cost") else None
    return res


def _is_transform(obj) -> bool:
    return hasattr(obj, "resample") and hasattr(obj, "n_in")


def run_matrix(methods: Mapping[str, object], tasks: Mapping[str, Iterable], *,
               out: str | None = None, seed: int = 0, meta: dict | None = None,
               **eval_kw) -> dict:
    """Run every method on every task -> JSON-serialisable report.

    ``methods`` values are transforms or zero-arg factories returning one.
    ``tasks`` values must be re-iterable (list, `PairFolder`, ...). A failing
    cell is recorded as ``{"error": ...}`` and the sweep continues.
    """
    rows = []
    for mname, m in methods.items():
        for tname, pairs in tasks.items():
            row: dict = {"method": mname, "task": tname}
            try:
                t = m if _is_transform(m) else m()
                row.update(evaluate(t, pairs, **eval_kw))
            except Exception as e:                      # keep sweeping
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
    cfg = json.dumps({"meta": meta or {}, "eval": {k: repr(v) for k, v in eval_kw.items()},
                      "seed": seed}, sort_keys=True)
    report = {"schema": SCHEMA, "lutlab_version": __version__, "seed": seed,
              "meta": meta or {}, "config_hash": hashlib.sha1(cfg.encode()).hexdigest()[:12],
              "rows": rows}
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=float)
    return report
