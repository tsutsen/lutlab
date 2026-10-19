"""Metrics: quality (CIEDE2000, PSNR, SSIM, LPIPS), timing, power.

`quality` needs scikit-image (``pip install lutlab[metrics]``), imported lazily.
"""
from . import power, quality, timing
from .quality import accuracy, de76, e00, lpips, lut_space_accuracy, psnr, ssim
from .timing import apply_speed, time_fn, to_fps

__all__ = ["quality", "timing", "power", "e00", "de76", "psnr", "ssim", "lpips", "accuracy",
           "lut_space_accuracy", "time_fn", "to_fps", "apply_speed"]
