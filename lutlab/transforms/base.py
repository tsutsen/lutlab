"""The unit of comparison: a `ColorTransform` maps colours to colours.

A dense LUT is one implementation. NILUT-style MLPs, parametric grades,
basis-LUT mixtures and image-adaptive methods are others. Anything that
satisfies this protocol can be benchmarked, baked to a grid and
conformance-tested the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Cost", "ColorTransform"]


@dataclass(frozen=True)
class Cost:
    """Static cost of a transform (what you pay before any pixel is touched)."""
    params: int                      # learnable / stored scalars
    nbytes: int                      # storage of those scalars
    reads_per_px: int | None = None  # memory reads per pixel, if meaningful
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"params": self.params, "nbytes": self.nbytes,
                "reads_per_px": self.reads_per_px, **self.meta}


@runtime_checkable
class ColorTransform(Protocol):
    """Protocol every colour transform follows.

    ``__call__`` takes ``x`` of shape ``(..., C)`` with values in the
    transform's input domain and returns ``(..., 3)``. ``context`` carries
    extra per-pixel or global inputs (4D-LUT context, mixture weights, style
    codes). Implementations should accept and ignore unknown keyword
    arguments such as ``interp=`` and ``backend=``.
    """
    name: str
    n_in: int

    def __call__(self, x, *, context=None, **kw): ...

    def bake(self, D: int = 17, **kw) -> Any:
        """Return an equivalent dense `LUT` on a D-point-per-axis grid."""

    def cost(self) -> Cost: ...
