"""lutlab -- a small library and testbed for colour LUTs and LUT-based methods.

The public surface is small: the `LUT` and `LUTMixture` transforms, the
`ColorTransform` protocol they follow and the `Cost` type (plus `BackendError`),
and the submodules `transforms`, `kernels`, `formats`, `methods`, `metrics`,
`testing`. A *transform* is any object that is callable and has `resample`,
`cost`, `name` and `n_in` -- `LUT` is the dense implementation, learned methods
build the others.
"""
__version__ = "0.3.0"

from .transforms import LUT, LUTMixture, ColorTransform, Cost
from .kernels import BackendError
from . import formats, kernels, methods, metrics, transforms, testing, visualize

__all__ = ["LUT", "LUTMixture", "ColorTransform", "Cost", "BackendError",
           "transforms", "kernels", "formats", "methods", "metrics", "testing",
           "visualize", "__version__"]
