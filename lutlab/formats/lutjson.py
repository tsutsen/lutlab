"""Lossless JSON container. Any dimensionality, axes, domain, bit depth.

Schema v3 stores the LUT's compact grid (uint8 / uint16 / float32) plus an
explicit ``bit_depth`` -- exact round-trip at every quantisation. Schema v2
(base64 float32) is still readable.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from ..transforms.lut import LUT

__all__ = ["loads", "dumps", "read", "write"]


def dumps(lut: LUT) -> str:
    lo, hi = lut.domain
    raw = np.ascontiguousarray(lut.raw)
    obj = {
        "format": "lutlab", "version": 3, "name": lut.name, "space": lut.space,
        "n_in": lut.n_in, "sizes": list(lut.sizes),
        "axes": None if lut.axes is None else [a.astype(float).tolist() for a in lut.axes],
        "domain": {"min": list(lo), "max": list(hi)},
        "bit_depth": lut.bit_depth,
        "dtype": raw.dtype.str,
        "data": base64.b64encode(raw.tobytes()).decode(),
    }
    return json.dumps(obj)


def loads(text: str, *, name: str | None = None) -> LUT:
    obj = json.loads(text)
    fmt, ver = obj.get("format"), obj.get("version")
    if fmt not in ("lutlab v1", "lutlab") or (fmt == "lutlab" and ver not in (2, 3)):
        raise ValueError(f"unrecognised LUT json (format={fmt!r}, version={ver!r})")
    if fmt == "lutlab v1":                                        # lutlab 0.1
        raw = np.frombuffer(base64.b64decode(obj["data"]), dtype="<f4")
        volume = raw.reshape((obj["D"],) * obj["ndim"] + (3,))
        return LUT(volume, name=obj.get("name") or name or "lut")
    raw = np.frombuffer(base64.b64decode(obj["data"]), dtype=obj.get("dtype", "<f4"))
    raw = raw.reshape(tuple(obj["sizes"]) + (3,))
    n = obj.get("bit_depth") if ver >= 3 else None
    if n is not None and raw.dtype.kind in "iu":
        top = float((2 ** int(n)) - 1)
        volume = raw.astype(np.float32) / top
    else:
        volume, n = raw.astype(np.float32), None
    dom = obj.get("domain")
    return LUT(volume, name=obj.get("name") or name or "lut", axes=obj.get("axes"),
               domain=None if dom is None else (dom["min"], dom["max"]),
               space=obj.get("space", "srgb"), bit_depth=n)


def read(path, **kw) -> LUT:
    p = Path(path)
    return loads(p.read_text(encoding="utf-8"), name=p.stem, **kw)


def write(lut: LUT, path) -> None:
    Path(path).write_text(dumps(lut), encoding="utf-8")
