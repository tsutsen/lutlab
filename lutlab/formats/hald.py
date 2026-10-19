"""Hald CLUT image codec -- the standard Hald-cube file (a.k.a. a "16-cube" PNG).

A Hald CLUT is a square image of ``grid_w x grid_w`` cells, ``grid_w = ceil(sqrt(L**3))``,
where cell ``(i, j)`` with ``n = i*grid_w + j < L**3`` holds the transform's output at the
input colour ``(n//L**2, (n//L) % L, n % L) / (L-1)``. Writing renders the LUT onto this
standard pattern; reading takes each cell's mean back. Same convention as ImageMagick's
``hald:N``.

The file's ``bit_depth`` is a LUT attribute (``lut.bit_depth``): ``write`` picks the PNG
container (8-bit for ``bit_depth<=8``, 16-bit above) and ``read`` recovers it from the file
(or you override it with ``bit_depth=``). Needs OpenCV (handles 8- and 16-bit RGB PNG; PIL
cannot).
"""
from __future__ import annotations

import numpy as np

from .._deps import require
from ..transforms.lut import LUT
from ..kernels import apply

__all__ = ["make_hald", "read", "write"]


def make_hald(levels: int = 17, cell: int = 1):
    """Standard Hald pattern -> ``(img (H, W, 3) float32 [0, 1], cell_bin (H, W) int32)``."""
    n = levels ** 3
    grid_w = int(np.ceil(np.sqrt(n)))
    idx = np.arange(n, dtype=np.float32)
    colors = np.stack([idx // levels ** 2, (idx // levels) % levels, idx % levels], 1) / (levels - 1)
    grid = np.full(grid_w * grid_w, -1, dtype=np.int32)
    grid[:n] = np.arange(n, dtype=np.int32)
    grid = grid.reshape(grid_w, grid_w)
    cell_bin = np.repeat(np.repeat(grid, cell, axis=0), cell, axis=1)
    img = colors[cell_bin.clip(0, n - 1)].astype(np.float32)
    return img, cell_bin


def _open_rgb(path) -> np.ndarray:
    """Read a PNG as uint8/uint16 (H, W, 3) RGB (cv2 stores BGR)."""
    cv2 = require("cv2")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image {path!r}")
    return img[..., ::-1]


def _write_png(path, arr: np.ndarray, container: int) -> None:
    top = (2 ** container) - 1
    q = np.clip(np.rint(arr * top), 0, top).astype("uint8" if container == 8 else "uint16")
    require("cv2").imwrite(str(path), q[..., ::-1])   # RGB -> BGR


def _infer_levels(img: np.ndarray, cell: int | None) -> int:
    grid_w = img.shape[0] // cell if cell else img.shape[0]
    return max(2, int(round(grid_w ** (2.0 / 3.0))))


def read(path, *, levels: int | None = None, cell: int | None = None,
         bit_depth: int | None = None, **kw) -> LUT:
    """Read a LUT from a Hald CLUT PNG.

    ``bit_depth`` is taken from the file (8- or 16-bit) unless overridden with
    ``bit_depth=`` (converted via `LUT.to_bit_depth`, which warns on up-conversion).
    Each cell's mean is taken back.
    """
    img = _open_rgb(path)
    container = 8 if img.dtype == np.uint8 else 16
    J = img.astype(np.float32) / ((2 ** container) - 1)
    if levels is None:
        levels = _infer_levels(J, cell)
    _, cell_bin = make_hald(levels, cell or 1)
    if cell_bin.shape != J.shape[:2]:
        raise ValueError(
            f"Hald cell grid {cell_bin.shape[::-1]} != image {J.shape[::-1]}; pass cell="
            f"(written with a different cell) or method='grid_pcell'")
    flat, pix = cell_bin.ravel(), J.reshape(-1, 3)
    valid = flat >= 0
    volume = np.zeros((levels ** 3, 3), np.float32)
    for c in range(3):
        np.add.at(volume, (flat[valid], c), pix[valid, c])
    volume /= np.maximum(np.bincount(flat[valid], minlength=levels ** 3), 1)[:, None]
    lut = LUT(volume.reshape(levels, levels, levels, 3), name=f"hald_L{levels}",
              bit_depth=container)
    if bit_depth is not None and bit_depth != container:
        lut = lut.to_bit_depth(bit_depth)
    return lut


def write(lut: LUT, path, *, cell: int = 1, bit_depth: int | None = None, **kw) -> None:
    """Render ``lut`` onto the standard Hald pattern and save as a PNG.

    ``bit_depth`` defaults to the LUT's own (else 8). The PNG container is 8-bit for
    ``bit_depth<=8`` and 16-bit above. Writing above the LUT's depth adds no precision
    (warns).
    """
    if lut.n_in != 3 or len(set(lut.sizes)) != 1:
        raise ValueError(f"Hald CLUT needs a cubic 3-D LUT, got n_in={lut.n_in} sizes={lut.sizes}")
    bd = bit_depth if bit_depth is not None else (lut.bit_depth if lut.bit_depth is not None else 8)
    lut_bd = lut if lut.bit_depth == bd else lut.to_bit_depth(bd)
    container = 8 if bd <= 8 else 16
    pattern, _ = make_hald(lut.grid_size, cell)
    _write_png(path, apply(lut_bd, pattern, interp="multilinear", backend="numpy"), container)
