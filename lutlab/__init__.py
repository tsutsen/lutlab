"""lutlab -- a small, single-source-of-truth LUT library shared across the papers.

One import point for everything a colour LUT needs:

    lutlab.LUT          the typed LUT: (D,)*n + (3,) volume + one apply path
    lutlab.apply        apply / segmented_apply over every backend
    lutlab.interp       interpolation kernels: nearest, (multi)linear, tetrahedral
    lutlab.formats      .cube / JSON round-trips, resample, backend validation
    lutlab.metrics      accuracy (CIEDE2000/PSNR/SSIM), speed (ms->fps), power
    lutlab.hald         Hald/CLUT generation + misalignment models
    lutlab.extract      LUT extraction (Hald + before/after), one import point
    lutlab.segmented    Art 5 segmented (promptable) LUTs, pluggable seg backend
    lutlab.bench        method-matrix benchmark runner -> results.json

Design rules:
  * canonical volume layout `(D,)*n + (3,)`, axis 0 = input dim 0 (R for 3D);
    every format round-trips through it, so axis-order bugs die here.
  * `extract` / `hald` / `segmented` are THIN re-exports of the working
    `robust_hald` code (we import it, we don't modify it).
  * `import lutlab` stays light (numpy + skimage); the GPU / extraction
    submodules pull in their own deps lazily.
"""
import os
import sys

# Make the sibling packages importable without touching them. `robust_hald`
# (the regular package) lives one level down, `xlut` in Experiments/.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # lutlab repo root
# robust_hald is a thin dependency (extract/hald/segmented re-export it). Locate it
# via env var, a sibling copy, or the known in-repo path -- so `import lutlab` works
# whether lutlab lives next to Experiments/ or standalone.
_robust = [
    os.environ.get("ROBUST_HALD_DIR"),
    os.path.join(_here, "robust_hald"),
    "/home/leon/Projects/PhD/Experiments/robust_hald",
]
for _p in _robust:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
if _here not in sys.path:
    sys.path.insert(0, _here)

from .lut import LUT, identity
from .apply import apply, segmented_apply, backends
from . import interp
from . import formats
from . import metrics

__version__ = "0.1.0"
__all__ = ["LUT", "identity", "apply", "segmented_apply", "backends",
           "interp", "formats", "metrics", "hald", "extract", "segmented",
           "bench", "__version__"]
