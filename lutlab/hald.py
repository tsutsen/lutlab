"""Hald/CLUT generation + misalignment (re-exported from `robust_hald`).

This is the "where the LUT comes from" side, shared by the papers. Two things
live here: building a Hald/CLUT grid, and modelling misalignment (a smooth
random warp of amplitude `s` px) so you can test how robust an extraction is.

    hald  = make_hald(17, 24)                    # 1704 x 1704 CLUT grid
    disp  = make_displacement(hald.shape[:2], s=8, seed=3)
    warped = remap(rendered_grade, disp)         # the "misaligned after"
    lut   = lutlab.extract.hald_extract(warped)  # -> 17^3 LUT
"""
from robust_hald.hald import make_hald, grade_warm_cinematic, grade_cool_fade
from robust_hald.warp import make_displacement, remap

__all__ = ["make_hald", "grade_warm_cinematic", "grade_cool_fade",
           "make_displacement", "remap"]
