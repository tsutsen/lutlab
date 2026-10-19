from __future__ import annotations

import numpy as np


def strides_of(sizes) -> np.ndarray:
    """C-order element strides for a grid of the given per-axis sizes."""
    s = np.ones(len(sizes), np.int64)
    for k in range(len(sizes) - 2, -1, -1):
        s[k] = s[k + 1] * sizes[k + 1]
    return s
