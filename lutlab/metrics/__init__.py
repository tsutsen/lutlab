"""Metrics: quality (CIEDE2000, PSNR, SSIM, LPIPS), timing, power, and the
evaluation harness (run a transform over a paired-image dataset).

`quality` needs scikit-image (``pip install lutlab[metrics]``), imported lazily.
"""
from . import power, quality, timing
from .quality import accuracy, de76, e00, lpips, lut_space_accuracy, psnr, ssim, tone_curve
from .timing import apply_speed, time_fn, to_fps
from .harness import SCHEMA, evaluate, run_matrix
from .datasets import PairFolder

__all__ = ["quality", "timing", "power", "e00", "de76", "psnr", "ssim", "lpips", "accuracy",
           "lut_space_accuracy", "tone_curve", "time_fn", "to_fps", "apply_speed",
           "evaluate", "run_matrix", "PairFolder", "SCHEMA"]
