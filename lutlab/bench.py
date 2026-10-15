"""Method-matrix benchmark runner -> `results.json`.

Builds a Hald, grades it, warps it by `s` pixels of smooth random displacement,
extracts the LUT with each method, and scores LUT-space accuracy against the
true grade (the ground-truth LUT). The schema is fixed so numbers from different
papers/boxes are directly comparable.

Run it:  `python -m lutlab.bench`   (writes `results.json` next to the script)
"""
from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

from .lut import LUT, identity
from .metrics import lut_space_accuracy
from .hald import (make_hald, make_displacement, remap,
                   grade_warm_cinematic)

__all__ = ["run_benchmark", "main"]

_SCHEMA = "lutlab bench v1"


def _extract(method: str, H: np.ndarray, warped: np.ndarray, levels: int,
             cell: int, radius: int, d: int) -> LUT:
    from . import extract as ex
    from .lut import LUT
    if method == "naive":
        return LUT(ex.naive_extract(H, warped, d), name="naive")
    if method == "farneback":
        return LUT(ex.flow_aligned_extract(H, warped, d), name="farneback")
    if method == "grid_remap":
        return LUT(ex.grid_aligned_extract(H, warped, d, levels=levels,
                                           cell=cell, radius=radius),
                   name="grid_remap")
    if method == "grid_pcell":
        return ex.hald_extract(warped, levels=levels, cell=cell, radius=radius,
                               method="grid_pcell")
    raise ValueError(f"unknown method {method!r}")


def run_benchmark(s_list: tuple[int, ...] = (0, 2, 4, 8, 16), levels: int = 17,
                  cell: int = 24, radius: int = 5, d: int = 17,
                  grade: Callable = grade_warm_cinematic, seed: int = 0,
                  methods: tuple[str, ...] = ("naive", "farneback", "grid_remap",
                                              "grid_pcell"),
                  out: str | None = None) -> dict:
    """Run the misalignment sweep; returns the report dict, writes JSON if `out`."""
    H, _cell_bin, _colors = make_hald(levels, cell)
    render = grade(H)
    gt = grade(np.ascontiguousarray(identity(d, 3).vol))   # (d,d,d,3)
    rows = {}
    for s in s_list:
        disp = make_displacement(H.shape[:2], s, seed=seed)
        warped = remap(render, disp)
        row = {"s_px": s}
        for m in methods:
            try:
                row[m] = lut_space_accuracy(
                    _extract(m, H, warped, levels, cell, radius, d),
                    ref_lut=LUT(gt))["e00_mean"]
            except Exception as e:                       # pragma: no cover
                row[m] = {"error": f"{e.__class__.__name__}: {e}"}
        rows[str(s)] = row
    report = {"schema": _SCHEMA, "grade": getattr(grade, "__name__", "grade"),
              "levels": levels, "cell": cell, "d": d, "seed": seed,
              "methods": list(methods),
              "columns": "LUT-space CIEDE2000 vs the true grade (lower = better)",
              "rows": rows}
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
    return report


def main() -> None:
    rep = run_benchmark(out="results.json")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
