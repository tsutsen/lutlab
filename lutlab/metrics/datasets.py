"""Paired-image datasets for evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .._arrays import to_float, quantize
from .._deps import require

__all__ = ["PairFolder"]

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


class PairFolder:
    """``root/input/<name>.ext`` + ``root/target/<name>.ext`` -> ``(input, target)`` float32 pairs.

    Pairs are matched by file stem. Re-iterable; images load lazily. Works for
    FiveK / PPR10K-style exports (one folder of inputs, one of retouched targets).

    ``source_depth`` is the bit depth actually used inside the stored integer
    image (a 12-bit TIFF stored as uint16 needs ``source_depth=12``); ``bit_depth``
    then quantises the loaded float image to that many levels. ``cache`` keeps up
    to that many decoded pairs in memory (0 disables it; useful for a method x
    task sweep that reads the same folder many times).
    """

    def __init__(self, root, input_dir: str = "input", target_dir: str = "target",
                 exts: tuple[str, ...] = _EXTS, bit_depth: int | None = None,
                 source_depth: int | None = None, cache: int = 0) -> None:
        root = Path(root)
        ins = {p.stem: p for p in sorted((root / input_dir).iterdir())
               if p.suffix.lower() in exts}
        tgs = {p.stem: p for p in sorted((root / target_dir).iterdir())
               if p.suffix.lower() in exts}
        self.names = sorted(set(ins) & set(tgs))
        if not self.names:
            raise ValueError(f"no matching pairs under {root}/{input_dir} and /{target_dir}")
        self._ins, self._tgs = ins, tgs
        self._bit_depth, self._source_depth = bit_depth, source_depth
        self._cache_size = max(0, int(cache))
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = (
            {} if self._cache_size else None)

    def __len__(self) -> int:
        return len(self.names)

    def _read(self, path: Path) -> np.ndarray:
        img = to_float(require("imageio.v3").imread(path), source_depth=self._source_depth)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = img[..., :3]
        if self._bit_depth is not None:
            img = quantize(img, self._bit_depth)
        return img

    def __getitem__(self, i: int):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        n = self.names[i]
        item = self._read(self._ins[n]), self._read(self._tgs[n])
        if self._cache is not None:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = item
        return item

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
