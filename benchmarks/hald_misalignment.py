"""Hald-extraction misalignment sweep -> results.json  (needs `robust_hald`).

Builds a Hald, grades it, warps it by `s` pixels of smooth random displacement,
extracts the LUT with each method, and scores LUT-space accuracy against the
true grade. This is experiment scaffolding, not library API: it lives outside
the package on purpose.

    python benchmarks/hald_misalignment.py            # writes results.json
"""
from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

import lutlab
from lutlab import LUT, identity
from lutlab.methods.hald_extract import extract_from_hald, extract_from_pair, make_hald
from lutlab.metrics import lut_space_accuracy

SCHEMA = "lutlab bench v2"


def _extract(method: str, H, warped, levels: int, cell: int, radius: int, d: int) -> LUT:
    if method in ("naive", "farneback"):
        return extract_from_pair(H, warped, d, method=method)
    if method == "grid_remap":
        return extract_from_pair(H, warped, d, method="grid_remap",
                                 levels=levels, cell=cell, radius=radius)
    if method == "grid_pcell":
        return extract_from_hald(warped, levels=levels, cell=cell, radius=radius,
                                 method="grid_pcell")
    raise ValueError(f"unknown method {method!r}")


def run_benchmark(s_list=(0, 2, 4, 8, 16), levels: int = 17, cell: int = 24, radius: int = 5,
                  d: int = 17, grade: Callable | None = None, seed: int = 0,
                  methods=("naive", "farneback", "grid_remap", "grid_pcell"),
                  out: str | None = None) -> dict:
    from robust_hald.hald import grade_warm_cinematic
    from robust_hald.warp import make_displacement, remap
    grade = grade or grade_warm_cinematic
    H, _cell_bin, _colors = make_hald(levels, cell)
    render = grade(H)
    gt = LUT(grade(np.ascontiguousarray(identity(d, 3).vol)), name="truth")
    rows = {}
    for s in s_list:
        warped = remap(render, make_displacement(H.shape[:2], s, seed=seed))
        row = {"s_px": s}
        for m in methods:
            try:
                row[m] = lut_space_accuracy(_extract(m, H, warped, levels, cell, radius, d),
                                            gt)["e00_mean"]
            except Exception as e:
                row[m] = {"error": f"{type(e).__name__}: {e}"}
        rows[str(s)] = row
    report = {"schema": SCHEMA, "lutlab_version": lutlab.__version__,
              "grade": getattr(grade, "__name__", "grade"), "levels": levels, "cell": cell,
              "d": d, "seed": seed, "methods": list(methods),
              "columns": "LUT-space CIEDE2000 vs the true grade (lower = better)", "rows": rows}
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    print(json.dumps(run_benchmark(out="results.json"), indent=2))
