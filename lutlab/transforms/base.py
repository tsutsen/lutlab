"""`Cost` and the `ColorTransform` contract.

A dense `LUT` is the central transform. NILUT-style MLPs, parametric grades and
image-adaptive methods are others. `ColorTransform` is a `Protocol` (not a base
class): any object that is *callable* and has `resample`, `cost`, `name` and
`n_in` satisfies it -- that contract is what :func:`lutlab.testing.conformance`
checks.
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
    """The contract every colour transform follows.

    ``__call__`` takes ``x`` of shape ``(..., C)`` in the transform's input
    domain and returns ``(..., 3)``. ``context`` carries extra per-pixel or
    global inputs (mixture weights, 4D-LUT context, style codes).
    ``cost()`` is a static property (no arguments). Implementations should
    accept and ignore unknown keyword arguments such as ``interp=`` and
    ``backend=``.

    Optional capability: ``to_lut(...)`` freezes the transform into one dense
    `LUT` (exact or approximate) -- a mixture blends its LUTs, a learned method
    bakes its parameters. Not part of the contract: context-dependent
    transforms that need an image to define themselves have none. The
    behavioural checks live in :func:`lutlab.testing.conformance`.

    A structural `Protocol`, not a base class: `LUT` and `LUTMixture` satisfy it
    without inheriting from it.
    """
    name: str
    n_in: int

    def __call__(self, x, *, context=None, **kw): ...

    def cost(self) -> Cost: ...
