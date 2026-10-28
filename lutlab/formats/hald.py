"""Hald CLUT image codec -- two pixel layouts, one file format.

``layout="lutlab"`` (default)
    A lutlab VARIANT of the Hald CLUT: the ``L**3`` nodes are laid out on a
    ``grid_w x grid_w`` grid, ``grid_w = ceil(sqrt(L**3))``, where cell ``(i, j)``
    with ``n = i*grid_w + j < L**3`` holds the transform's output at the input colour
    ``(n//L**2, (n//L) % L, n % L) / (L-1)``; every node occupies a ``cell x cell``
    square of pixels, so the image is ``grid_w*cell`` pixels per side.  Reading a file
    written with ``cell>1`` requires passing ``cell=`` (and ``levels=``) explicitly.

``layout="canonical"``
    The standard Hald layout (ImageMagick ``-hald-clut``, the classic Hald
    specification): level ``L`` gives an image of ``L**3`` pixels per side
    (``hald:4`` -> 64x64, ``hald:8`` -> 512x512, ``hald:16`` -> 4096x4096) holding
    ``L**2`` values per axis -- one pixel per node -- laid out as a flattened 3-D
    buffer (width = red, height = green, depth = blue; blue outermost, red
    innermost).  The pixel at ``(x, y)`` holds the node
    ``r = x % C``, ``g = (y % L) * L + x // C``, ``b = y // L`` with ``C = L**2``.
    Files written this way are interchangeable with ImageMagick.

The file's ``bit_depth`` is a LUT attribute (``lut.bit_depth``): ``write`` picks the PNG
container (8-bit for ``bit_depth<=8``, 16-bit above). ``read`` can only recover the two
container depths, so a 10/12/17..32-bit LUT reads back as 16-bit and a 1..7-bit LUT as
8-bit (the values stay on the same levels); override with ``bit_depth=`` if needed.
The pattern stores output values over the unit cube only: a LUT with a custom ``domain``
or non-uniform ``axes`` is rejected by ``write`` -- use .json for those.
Needs OpenCV (handles 8- and 16-bit RGB PNG; PIL cannot).
"""
from __future__ import annotations

import numpy as np

from .._deps import require
from ..transforms.lut import LUT
from ..kernels import apply

__all__ = ["make_hald", "canonical_level", "canonical_nodes", "canonical_pattern",
           "read", "write"]


def make_hald(levels: int = 17, cell: int = 1):
    """lutlab Hald pattern -> ``(img (H, W, 3) float32 [0, 1], cell_bin (H, W) int32)``."""
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
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return img[..., :3][..., ::-1]


def _write_png(path, arr: np.ndarray, container: int) -> None:
    top = (2 ** container) - 1
    q = np.clip(np.rint(arr * top), 0, top).astype("uint8" if container == 8 else "uint16")
    if not require("cv2").imwrite(str(path), q[..., ::-1]):   # RGB -> BGR
        raise OSError(f"cannot write image {path!r}")


def _infer_levels(img: np.ndarray, cell: int | None) -> int:
    grid_w = img.shape[0] // cell if cell else img.shape[0]
    return max(2, int(round(grid_w ** (2.0 / 3.0))))


# --------------------------------------------------------------------------- canonical
# The canonical Hald CLUT (the layout ImageMagick's ``-hald-clut`` and the classic
# Hald specification use):
#
#   * level ``L``  -> image side ``L**3`` px  (hald:4 -> 64x64, hald:8 -> 512x512,
#     hald:16 -> 4096x4096);
#   * the table has ``L**2`` values per axis (hald:8 -> a 64x64x64 table), i.e.
#     ``L**6`` nodes, **one pixel per node**;
#   * the pixels are a flattened 3-D buffer -- width = red, height = green, depth =
#     blue -- with blue outermost and red innermost.  The pixel at ``(x, y)`` holds
#     the node
#         r = x % C,   g = (y % L) * L + x // C,   b = y // L,     C = L**2.
#
# For an identity CLUT the pixel colour is the node's own input coordinate, so the
# image is the familiar Hald ramp.  See https://www.quelsolaar.com/technology/clut.html
# (the layout this file implements) and ImageMagick's ``-hald-clut``.


def canonical_level(image_side: int) -> int:
    """Level of a canonical Hald image: the cube root of its side."""
    level = int(round(image_side ** (1.0 / 3.0)))
    if level < 2 or level ** 3 != image_side:
        raise ValueError(f"side {image_side} is not a cube -- not a canonical Hald image")
    return level


def canonical_nodes(level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The node coordinates ``(r, g, b)`` of every pixel of a canonical Hald image.

    Returns three ``(L**3, L**3)`` arrays indexed by ``[y, x]``.
    """
    if level < 2:
        raise ValueError(f"level must be >= 2, got {level}")
    c = level ** 2
    side = level ** 3
    x = np.arange(side)
    y = np.arange(side)
    r = np.broadcast_to(x % c, (side, side))
    g = (y[:, None] % level) * level + x[None, :] // c
    b = np.broadcast_to((y // level)[:, None], (side, side))
    return r, g, b


def canonical_pattern(level: int) -> np.ndarray:
    """The identity canonical Hald image of the given level, ``(L**3, L**3, 3)`` float32."""
    r, g, b = canonical_nodes(level)
    c = level ** 2
    out = np.empty(r.shape + (3,), np.float32)
    out[..., 0] = r / (c - 1)
    out[..., 1] = g / (c - 1)
    out[..., 2] = b / (c - 1)
    return out


def read(path, *, levels: int | None = None, cell: int | None = None,
         bit_depth: int | None = None, layout: str = "lutlab", **kw) -> LUT:
    """Read a LUT from a Hald CLUT PNG.

    ``layout`` selects the pixel layout:

    * ``"lutlab"`` (default) -- the lutlab variant: ``ceil(sqrt(L**3))`` cells per
      side, one node per ``cell x cell`` pixel block (needs ``cell=``);
    * ``"canonical"`` -- the standard Hald layout (ImageMagick ``-hald-clut``):
      side ``L**3`` px, one pixel per node, ``L**2`` values per axis.  The level is
      derived from the image side unless ``levels=`` is given.

    ``bit_depth`` is taken from the file (8- or 16-bit) unless overridden with
    ``bit_depth=`` (converted via `LUT.to_bit_depth`, which warns on up-conversion).
    Each cell's mean is taken back (the canonical layout has one pixel per node,
    so the mean is the node itself).
    """
    img = _open_rgb(path)
    container = 8 if img.dtype == np.uint8 else 16
    J = img.astype(np.float32) / ((2 ** container) - 1)
    if layout == "canonical":
        side = J.shape[0]
        if J.shape[0] != J.shape[1]:
            raise ValueError(f"canonical Hald images are square, got {J.shape[1]}x{J.shape[0]}")
        level = levels if levels is not None else canonical_level(side)
        if level ** 3 != side:
            raise ValueError(f"side {side} does not match level {level} (expected {level ** 3})")
        r, g, b = canonical_nodes(level)
        c = level ** 2
        volume = np.empty((c, c, c, 3), np.float32)
        volume[r, g, b] = J                   # pixel (y, x) -> node (r, g, b)
        lut = LUT(volume, name=f"hald_canonical_L{level}", bit_depth=container)
        if bit_depth is not None and bit_depth != container:
            lut = lut.to_bit_depth(bit_depth)
        return lut
    if layout != "lutlab":
        raise ValueError(f"unknown Hald layout {layout!r}: use 'lutlab' or 'canonical'")
    side = int(J.shape[0])
    if levels is None and cell is None and int(round(side ** (1.0 / 3.0))) ** 3 == side:
        # a canonical Hald image has the same geometry as a variant image with
        # levels = side**(2/3) and cell = 1 -- the two must not be confused
        raise ValueError(
            f"image side {side} is a perfect cube: this may be a canonical Hald image "
            f"(ImageMagick layout) -- pass layout='canonical'; for the lutlab variant "
            f"pass levels= and/or cell=")
    if levels is None:
        levels = _infer_levels(J, cell)
    _, cell_bin = make_hald(levels, cell or 1)
    if cell_bin.shape != J.shape[:2]:
        raise ValueError(
            f"Hald cell grid {cell_bin.shape[::-1]} != image {J.shape[::-1]}; pass cell="
            f"(written with a different cell)")
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


def write(lut: LUT, path, *, cell: int = 1, bit_depth: int | None = None,
          layout: str = "lutlab", level: int | None = None, **kw) -> None:
    """Render ``lut`` onto a Hald pattern and save as a PNG.

    ``layout`` selects the pixel layout:

    * ``"lutlab"`` (default) -- the lutlab variant (see `make_hald`); the image side
      is ``ceil(sqrt(L**3)) * cell`` px;
    * ``"canonical"`` -- the standard Hald layout read by ImageMagick's
      ``-hald-clut``: side ``L**3`` px, one pixel per node, ``L**2`` values per axis.
      ``level`` selects the size (``L=4`` -> 64x64, ``L=8`` -> 512x512, ``L=16`` ->
      4096x4096); the table is resampled to ``L**2`` values per axis if needed.

    ``bit_depth`` defaults to the LUT's own (else 8). The PNG container is 8-bit for
    ``bit_depth<=8`` and 16-bit above, so a 17..32-bit LUT is written to a 16-bit PNG
    (the tag is not stored). Writing above the LUT's depth adds no precision (warns).
    The pattern stores values over the unit cube, so a custom ``domain`` or a
    non-uniform grid cannot be represented; those raise -- use .json for them.
    """
    if lut.n_in != 3 or len(set(lut.sizes)) != 1:
        raise ValueError(f"Hald CLUT needs a cubic 3-D LUT, got n_in={lut.n_in} sizes={lut.sizes}")
    if not lut.is_uniform:
        raise ValueError("Hald CLUT stores a uniform grid only; "
                         "resample first or use .json")
    if not lut.has_default_domain:
        raise ValueError("Hald CLUT stores the unit domain only; use .json for a custom domain")
    bd = bit_depth if bit_depth is not None else (lut.bit_depth if lut.bit_depth is not None else 8)
    lut_bd = lut if lut.bit_depth == bd else lut.to_bit_depth(bd)
    container = 8 if bd <= 8 else 16
    if layout == "canonical":
        if level is None:
            # the smallest canonical level that keeps at least the table's own
            # resolution (level L carries L**2 values per axis)
            level = max(2, int(np.ceil(np.sqrt(lut.grid_size))))
            while level ** 2 < lut.grid_size and level < 64:
                level += 1
        c = level ** 2
        lut_bd = lut_bd.resample(c)
        r, g, b = canonical_nodes(level)
        img = lut_bd.volume[r, g, b]           # node (r, g, b) -> pixel (y, x)
        _write_png(path, img, container)
        return
    if layout != "lutlab":
        raise ValueError(f"unknown Hald layout {layout!r}: use 'lutlab' or 'canonical'")
    pattern, _ = make_hald(lut.grid_size, cell)
    _write_png(path, apply(lut_bd, pattern, interp="multilinear", backend="numpy"), container)
