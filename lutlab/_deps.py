"""Optional-dependency helpers: fail late, with an actionable message."""
from __future__ import annotations

import importlib
import importlib.util

__all__ = ["have", "require"]

_HINTS = {
    "skimage": "pip install lutlab[metrics]",
    "scipy": "pip install lutlab[fit]",
    "imageio": "pip install lutlab[eval]",
    "torch": "pip install lutlab[torch] (or a ROCm/CUDA build of torch)",
    "numba": "pip install lutlab[fast]",
    "lpips": "pip install lpips",
}


def have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def require(module: str):
    """Import and return `module`, or raise ImportError with an install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        hint = _HINTS.get(module.split(".")[0], f"pip install {module}")
        raise ImportError(f"{module} is required for this feature ({hint})") from e
