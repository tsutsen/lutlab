"""lutlab -- a small library and testbed for colour LUTs and LUT-based methods."""
__version__ = "0.3.0"

from .transforms import LUT, ColorTransform, Cost, LUTMixture, identity
from .kernels import BackendError, apply, backends
from .formats import load, save
from ._arrays import to_float
from . import analysis, evaluation, formats, kernels, methods, metrics, testing, transforms

__all__ = ["LUT", "LUTMixture", "ColorTransform", "Cost", "identity", "apply", "backends",
           "BackendError", "load", "save", "to_float", "analysis", "evaluation", "formats",
           "kernels", "methods", "metrics", "testing", "transforms", "__version__"]
