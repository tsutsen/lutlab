"""Paired-image datasets for evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .._arrays import to_float
from .._deps import require

__all__ = ["PairFolder"]

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


class PairFolder:
    """``root/input/<name>.ext`` + ``root/target/<name>.ext`` -> ``(input, target)`` float32 pairs.

    Pairs are matched by file stem. Re-iterable; images load lazily. Works for
    FiveK / PPR10K-style exports (one folder of inputs, one of retouched targets).
    """

    def __init__(self, root, input_dir: str = "input", target_dir: str = "target",
                 exts: tuple[str, ...] = _EXTS) -> None:
        root = Path(root)
        ins = {p.stem: p for p in sorted((root / input_dir).iterdir())
               if p.suffix.lower() in exts}
        tgs = {p.stem: p for p in sorted((root / target_dir).iterdir())
               if p.suffix.lower() in exts}
        self.names = sorted(set(ins) & set(tgs))
        if not self.names:
            raise ValueError(f"no matching pairs under {root}/{input_dir} and /{target_dir}")
        self._ins, self._tgs = ins, tgs

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        img = to_float(require("imageio.v3").imread(path))
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        return img[..., :3]

    def __getitem__(self, i: int):
        n = self.names[i]
        return self._read(self._ins[n]), self._read(self._tgs[n])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
