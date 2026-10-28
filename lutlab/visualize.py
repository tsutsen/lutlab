"""Static 3D pictures of a LUT as a coloured volume.

A LUT maps the RGB cube into RGB. Plotting its nodes in *output* space and
colouring each node by its own output makes the transformation visible: an
identity LUT is a regular cube, a real grade is a deformed one. Points are
subsampled above ``max_points`` so 65^3 and 129^3 grids stay fast.

matplotlib is imported lazily; it is not a hard lutlab dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .transforms import LUT

__all__ = ["volume", "comparison", "grid"]


def _as_lut(lut) -> LUT:
    return lut if isinstance(lut, LUT) else LUT.load(lut)


def _points(lut: LUT, max_points: int) -> np.ndarray:
    """LUT outputs as an ``(N, 3)`` array, deterministically subsampled."""
    if lut.n_in != 3:
        raise ValueError(f"volume() needs a 3-input LUT, got n_in={lut.n_in}")
    vol = lut.volume
    sizes = vol.shape[:3]
    if int(np.prod(sizes)) > max_points:
        step = int(np.ceil((int(np.prod(sizes)) / max_points) ** (1 / 3)))
        idx = [np.unique(np.r_[np.arange(0, d, step), d - 1]) for d in sizes]
        vol = vol[np.ix_(*idx)]
    return vol.reshape(-1, 3)


def _point_size(n_points: int, base: float | None) -> float:
    """Keep the same visual density when a small grid has few nodes."""
    if base is not None:
        return float(base)
    return float(min(100.0, 8.0 * (35_937 / max(n_points, 1)) ** (2 / 3)))


def _limits(points: list[np.ndarray]) -> tuple[float, float]:
    lo = min(0.0, *(float(p.min()) for p in points if p.size))
    hi = max(1.0, *(float(p.max()) for p in points if p.size))
    pad = 0.03 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _draw(ax, pts: np.ndarray, *, title: str, limits, reference: bool,
          point_size: float, alpha: float, show_axes: bool) -> None:
    colour = np.clip(pts, 0.0, 1.0)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colour, s=point_size,
               alpha=alpha, linewidths=0, depthshade=True)
    if reference:
        for a in (0.0, 1.0):
            for b in (0.0, 1.0):
                ax.plot([a, a], [b, b], [0.0, 1.0], color="#9aa1ab", lw=0.35, alpha=0.8)
                ax.plot([a, a], [0.0, 1.0], [b, b], color="#9aa1ab", lw=0.35, alpha=0.8)
                ax.plot([0.0, 1.0], [a, a], [b, b], color="#9aa1ab", lw=0.35, alpha=0.8)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.grid(False)
    if show_axes:
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_zticklabels(["0", "0.5", "1"])
        ax.tick_params(pad=-2, length=1.2, width=0.35, labelsize=4.5)
        ax.set_xlabel("R", fontsize=5.5, labelpad=-4)
        ax.set_ylabel("G", fontsize=5.5, labelpad=-4)
        ax.set_zlabel("B", fontsize=5.5, labelpad=-4)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
            pane.line.set_color("#d5d9de")
    else:
        ax.set_axis_off()


def _save(fig, out, dpi: int):
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def volume(lut, out=None, *, title: str | None = None, max_points: int = 60_000,
           dpi: int = 300, figsize=(4.4, 4.4), point_size: float | None = None,
           alpha: float = 0.55, reference: bool = True, show_axes: bool = True):
    """Render one LUT as a coloured 3D point volume; optionally save it to ``out``."""
    import matplotlib.pyplot as plt

    lut = _as_lut(lut)
    pts = _points(lut, max_points)
    lim = _limits([pts])
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _draw(ax, pts, title=title or lut.name, limits=lim, reference=reference,
          point_size=_point_size(len(pts), point_size), alpha=alpha,
          show_axes=show_axes)
    return _save(fig, out, dpi)


def comparison(luts, out=None, *, titles=None, max_points: int = 60_000,
               dpi: int = 300, figsize=(8.2, 4.1), point_size: float | None = None,
               alpha: float = 0.55, reference: bool = True, show_axes: bool = True):
    """Render one or more LUTs side by side on shared limits; optionally save."""
    import matplotlib.pyplot as plt

    luts = [_as_lut(l) for l in luts]
    pts = [_points(l, max_points) for l in luts]
    lim = _limits(pts)
    if titles is None:
        titles = [l.name for l in luts]
    if len(titles) != len(luts):
        raise ValueError(f"{len(titles)} titles for {len(luts)} LUTs")
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i, (p, t) in enumerate(zip(pts, titles), 1):
        ax = fig.add_subplot(1, len(luts), i, projection="3d")
        _draw(ax, p, title=t, limits=lim, reference=reference,
              point_size=_point_size(len(p), point_size), alpha=alpha,
              show_axes=show_axes)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.93, wspace=0.02)
    return _save(fig, out, dpi)


def grid(rows, out=None, *, row_labels=None, col_labels=None,
         max_points: int = 60_000, dpi: int = 300, figsize=None,
         point_size: float | None = None, alpha: float = 0.55,
         reference: bool = True, show_axes: bool = False, note: str | None = None):
    """Render a rectangular grid of LUTs, e.g. one row per LUT and one column per size.

    ``rows`` is a sequence of equal-length sequences of LUTs (or paths).  Row and
    column labels are drawn outside the panels; ``point_size=None`` scales the
    marker per panel to keep the visual density roughly constant.
    """
    import matplotlib.pyplot as plt

    rows = [[_as_lut(l) for l in row] for row in rows]
    if not rows or not rows[0]:
        raise ValueError("rows must be a non-empty rectangular sequence")
    ncols = len(rows[0])
    if any(len(row) != ncols for row in rows):
        raise ValueError("all rows must have the same number of LUTs")
    if row_labels is not None and len(row_labels) != len(rows):
        raise ValueError(f"{len(row_labels)} row labels for {len(rows)} rows")
    if col_labels is not None and len(col_labels) != ncols:
        raise ValueError(f"{len(col_labels)} column labels for {ncols} columns")

    pts = [[_points(l, max_points) for l in row] for row in rows]
    lim = _limits([p for row in pts for p in row])
    nrows = len(rows)
    if figsize is None:
        figsize = (3.1 * ncols, 2.9 * nrows)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for r, row in enumerate(pts):
        for c, p in enumerate(row):
            ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1, projection="3d")
            _draw(ax, p, title="", limits=lim, reference=reference,
                  point_size=_point_size(len(p), point_size), alpha=alpha,
                  show_axes=show_axes)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93,
                        bottom=0.04 if note else 0.01,
                        wspace=0.01, hspace=0.01)
    if col_labels is not None:
        for c, lab in enumerate(col_labels):
            box = fig.axes[c].get_position()
            fig.text((box.x0 + box.x1) / 2, 0.965, lab, ha="center", va="bottom",
                     fontsize=9)
    if row_labels is not None:
        for r, lab in enumerate(row_labels):
            box = fig.axes[r * ncols].get_position()
            fig.text(0.028, (box.y0 + box.y1) / 2, lab, ha="center", va="center",
                     rotation=90, fontsize=8)
    if note:
        fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=7, color="#555")
    return _save(fig, out, dpi)
