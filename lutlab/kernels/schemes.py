"""Interpolation scheme names: user-facing sugar -> canonical kernel."""
from __future__ import annotations

__all__ = ["SCHEMES", "canonical_scheme"]

# sugar -> (kernel, required n_in or None)
_SUGAR = {
    "nearest": ("nearest", None),
    "multilinear": ("multilinear", None),
    "linear": ("multilinear", None),
    "bilinear": ("multilinear", 2),
    "trilinear": ("multilinear", 3),
    "quadrilinear": ("multilinear", 4),
    "tetrahedral": ("tetrahedral", 3),
}
SCHEMES = tuple(sorted(_SUGAR))


def canonical_scheme(mode: str, n_in: int) -> str:
    """Map a user scheme name to a kernel name, checking dimensionality.

    >>> canonical_scheme("trilinear", 3)
    'multilinear'
    """
    try:
        kernel, want = _SUGAR[str(mode).lower()]
    except KeyError:
        raise ValueError(f"unknown interp mode {mode!r}; options: {', '.join(SCHEMES)}") from None
    if want is not None and want != n_in:
        raise ValueError(f"{mode!r} needs a {want}-input LUT, got {n_in}")
    return kernel
