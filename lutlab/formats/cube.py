""".cube (Adobe/Resolve) reader and writer. 3D LUTs only.

Rows are R-fastest (``for b: for g: for r``). Both the standard header
(``TITLE``, ``LUT_3D_SIZE``, ``DOMAIN_MIN/MAX``) and the header written by
lutlab 0.1 (``Title:``, ``DataSize:``, ``DomainMin:``) are read. Unknown
header lines produce a warning instead of being skipped silently.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np

from ..transforms.lut import LUT

__all__ = ["loads", "dumps", "read", "write"]

_IGNORED = {"lut3d", "description", "method"}


def loads(text: str, *, name: str | None = None, space: str = "srgb") -> LUT:
    title, d = None, None
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    rows: list[tuple[float, float, float]] = []
    warned: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line[0] in "+-.0123456789":
            p = line.split()
            if len(p) < 3:
                raise ValueError(f"bad .cube data row: {line!r}")
            rows.append((float(p[0]), float(p[1]), float(p[2])))
            continue
        kw, _, rest = line.partition(" ")
        rest = rest.strip()
        key = kw.lower().rstrip(":").replace("_", "")
        if key == "title":
            title = rest.lstrip(":").strip().strip('"').strip()
        elif key in ("lut3dsize", "datasize"):
            d = int(rest.split()[0])
        elif key == "lut1dsize":
            raise ValueError("1D LUTs / shapers in .cube files are not supported")
        elif key == "domainmin":
            lo = tuple(float(v) for v in rest.split()[:3])
        elif key == "domainmax":
            hi = tuple(float(v) for v in rest.split()[:3])
        elif key == "lut3dinputrange":
            a, b = (float(v) for v in rest.split()[:2])
            lo, hi = (a,) * 3, (b,) * 3
        elif key in _IGNORED:
            continue
        elif key not in warned:
            warned.add(key)
            warnings.warn(f"ignoring unknown .cube header keyword {kw!r}", stacklevel=2)
    if d is None:
        c = round(len(rows) ** (1 / 3)) if rows else 0
        d = c if c ** 3 == len(rows) else None
    if d is None or d ** 3 != len(rows):
        raise ValueError(f"not a 3D .cube LUT: {len(rows)} data rows do not form a "
                         f"{d}^3 grid" if d else
                         f"not a 3D .cube LUT: {len(rows)} data rows is not a perfect cube")
    arr = np.asarray(rows, np.float32).reshape(d, d, d, 3)      # indexed [b, g, r]
    volume = arr.transpose(2, 1, 0, 3)                           # -> [r, g, b]
    default = lo == (0.0,) * 3 and hi == (1.0,) * 3
    return LUT(volume, name=title or name or "cube", domain=None if default else (lo, hi),
               space=space)


def dumps(lut: LUT, *, precision: int = 6) -> str:
    """Return the .cube text. 6 decimals => round-trip error ~1e-6; use JSON for lossless."""
    if lut.n_in != 3:
        raise ValueError(".cube holds 3D LUTs only; save 1D/2D/4D LUTs as .json")
    if len(set(lut.sizes)) != 1:
        raise ValueError(f".cube needs equal axis sizes, got {lut.sizes}; resample first")
    if not lut.is_uniform:
        raise ValueError(".cube needs a uniform grid; lut.resample(grid_size) resamples to one")
    lo, hi = lut.domain
    title = lut.name.replace('"', "'")
    head = [f'TITLE "{title}"', f"LUT_3D_SIZE {lut.grid_size}",
            "DOMAIN_MIN " + " ".join(f"{v:g}" for v in lo),
            "DOMAIN_MAX " + " ".join(f"{v:g}" for v in hi)]
    buf = io.StringIO()
    np.savetxt(buf, lut.volume.transpose(2, 1, 0, 3).reshape(-1, 3), fmt=f"%.{precision}f")
    return "\n".join(head) + "\n" + buf.getvalue()


def read(path, **kw) -> LUT:
    p = Path(path)
    return loads(p.read_text(encoding="utf-8-sig"), name=p.stem, **kw)


def write(lut: LUT, path, **kw) -> None:
    Path(path).write_text(dumps(lut, **kw), encoding="utf-8")
