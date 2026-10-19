"""PG-IA-NILUT: prompt-guided image-adaptive neural implicit lookup tables.

S. Kosugi, "Prompt-Guided Image-Adaptive Neural Implicit Lookup Tables for Interpretable Image
Enhancement", ACM MM 2024 (arXiv 2408.11055).

    from lutlab.methods import pg_ia_nilut as pg
    model = pg.PGIANILUT(pg.IANILUT(), pg.CnnPredictor())
    t = model.transform()                          # image-adaptive lutlab transform
    w = model.filters.weights(exposure=0.4)        # interpretable controls
    lut = model.filters.lut(w, N=33)               # LUT bypassing: a plain lutlab LUT -> .cube

Importing this package does not import torch; the heavy modules load on first attribute access.
"""
from __future__ import annotations

import importlib

from .. import register
from .prompts import FILTER_NAMES, GUIDING_PROMPTS, guiding_prompts, name_key

_LAZY = {
    "IANILUT": "model", "CnnPredictor": "model", "ResNetPredictor": "model",
    "PGIANILUT": "model", "IANILUTTransform": "model", "make_predictor": "model",
    "ClipScorer": "guidance", "score_jacobian": "guidance",
    "prompt_guidance_loss": "guidance", "guidance_step_loss": "guidance",
    "TrainConfig": "train", "ImagePairDataset": "train", "TrainResult": "train",
    "train_stage1": "train", "train_stage2": "train", "train_stage3": "train",
    "train_pg_ia_nilut": "train", "build_method": "train",
}
__all__ = ["FILTER_NAMES", "GUIDING_PROMPTS", "guiding_prompts", "name_key", *_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        try:
            mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        except ImportError as e:
            raise ImportError(f"lutlab.methods.pg_ia_nilut.{name} needs torch "
                              f"(pip install lutlab[torch]): {e}") from e
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@register("pg_ia_nilut", requires=("torch",))
def build(**kw):
    """PG-IA-NILUT (Kosugi, MM'24). ``checkpoint=`` loads, ``data=`` trains, neither -> identity."""
    return __getattr__("build_method")(**kw)
