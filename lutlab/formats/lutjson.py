"""Lossless JSON container (base64 float32 grid). Any dimensionality, axes, domain.

Schema v2 stores sizes / axes / domain / space; lutlab 0.1 files (``"lutlab v1"``)
are still readable.
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
    obj = {
        "format": "lutlab", "version": 2, "name": lut.name, "space": lut.space,
        "n_in": lut.n_in, "sizes": list(lut.sizes),
        "axes": None if lut.axes is None else [a.astype(float).tolist() for a in lut.axes],
        "domain": {"min": list(lo), "max": list(hi)},
        "dtype": "<f4",
        "data": base64.b64encode(np.ascontiguousarray(lut.vol, dtype="<f4").tobytes()).decode(),
    }
    return json.dumps(obj)


def loads(text: str, *, name: str | None = None) -> LUT:
    obj = json.loads(text)
    fmt, ver = obj.get("format"), obj.get("version")
    if fmt not in ("lutlab v1", "lutlab") or (fmt == "lutlab" and ver != 2):
        raise ValueError(f"unrecognised LUT json (format={fmt!r}, version={ver!r})")
    raw = np.frombuffer(base64.b64decode(obj["data"]), dtype="<f4")
    if fmt == "lutlab v1":                                        # lutlab 0.1
        vol = raw.reshape((obj["D"],) * obj["ndim"] + (3,))
        return LUT(vol, name=obj.get("name") or name or "lut")
    vol = raw.reshape(tuple(obj["sizes"]) + (3,))
    dom = obj.get("domain")
    return LUT(vol, name=obj.get("name") or name or "lut", axes=obj.get("axes"),
               domain=None if dom is None else (dom["min"], dom["max"]),
               space=obj.get("space", "srgb"))


def read(path, **kw) -> LUT:
    p = Path(path)
    return loads(p.read_text(encoding="utf-8"), name=p.stem, **kw)


def write(lut: LUT, path) -> None:
    Path(path).write_text(dumps(lut), encoding="utf-8")
