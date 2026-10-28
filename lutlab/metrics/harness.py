"""Evaluate any transform on paired data and collect a method x task report."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from .. import __version__
from .._arrays import is_torch
from ..kernels.dispatch import apply
from ..transforms.base import ColorTransform
from . import quality
from .timing import apply_speed

__all__ = ["SCHEMA", "evaluate", "run_matrix"]

SCHEMA = "lutlab eval v2"
_METRICS = ("psnr", "ssim", "e00", "de76", "lpips")


def _np(x):
    return x.detach().cpu().numpy() if is_torch(x) else np.asarray(x)


def _json_safe(obj):
    """Replace non-finite floats with None so the report is valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def evaluate(transform, pairs: Iterable, *, metrics: tuple[str, ...] = ("psnr", "ssim", "e00"),
             space: str = "srgb", timing: dict | None = None, apply_kw: dict | None = None,
             max_pairs: int | None = None) -> dict:
    """Score `transform` on ``(input, target[, context])`` pairs (float, [0, 1], (H, W, 3)).

    Returns metric means over pairs, optional timing (measured on the first
    pair; ``timing={"warmup": 2, "repeat": 10}`` enables it) and the
    transform's static cost. A non-finite score is kept: one perfect PSNR makes
    the mean infinite (``run_matrix`` writes that as JSON ``null``).
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
        res[k] = float(np.mean(v))
    res["timing"] = None
    if timing is not None and first is not None:
        res["timing"] = apply_speed(transform, first[0], context=first[1], **timing, **kw)
    res["cost"] = transform.cost().as_dict() if isinstance(transform, ColorTransform) else None
    return res


def _is_transform(obj) -> bool:
    return isinstance(obj, ColorTransform)


def run_matrix(methods: Mapping[str, object], tasks: Mapping[str, Iterable], *,
               out: str | None = None, seed: int = 0, meta: dict | None = None,
               **eval_kw) -> dict:
    """Run every method on every task -> JSON-serialisable report.

    ``methods`` values are transforms or zero-arg factories returning one.
    ``tasks`` values must be re-iterable (list, `PairFolder`, ...). A failing
    cell is recorded as ``{"error": ...}`` and the sweep continues.

    ``seed`` seeds Python, numpy and (when torch is already imported) torch for
    construction and evaluation. The previous RNG states are restored afterwards.
    Methods that use their own generator are unaffected.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch = sys.modules.get("torch")
    torch_state = torch.get_rng_state() if torch is not None else None
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch is not None and torch.cuda.is_available() else None)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    try:
        rows = []
        for mname, m in methods.items():
            try:
                t = m if _is_transform(m) else m()
                build_error = None
            except Exception as e:                      # keep sweeping
                t, build_error = None, f"{type(e).__name__}: {e}"
            for tname, pairs in tasks.items():
                row: dict = {"method": mname, "task": tname}
                if build_error:
                    row["error"] = build_error
                else:
                    try:
                        row.update(evaluate(t, pairs, **eval_kw))
                    except Exception as e:              # keep sweeping
                        row["error"] = f"{type(e).__name__}: {e}"
                rows.append(row)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
    cfg = json.dumps({"meta": meta or {}, "eval": {k: repr(v) for k, v in eval_kw.items()},
                      "seed": seed}, sort_keys=True)
    report = {"schema": SCHEMA, "lutlab_version": __version__, "seed": seed,
              "meta": meta or {}, "config_hash": hashlib.sha1(cfg.encode()).hexdigest()[:12],
              "rows": rows}
    if out:
        with open(out, "w") as f:
            json.dump(_json_safe(report), f, indent=2, default=float)
    return report
