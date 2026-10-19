"""Methods with no external dependencies."""
from __future__ import annotations

from ..transforms.lut import LUT
from . import register


@register("identity")
def build_identity(*, grid_size: int = 17, n_in: int = 3, **_ignored):
    """The do-nothing LUT; the baseline every other method is compared against."""
    return LUT.identity(grid_size, n_in)
