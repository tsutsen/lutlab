# lutlab

A small library and testbed for colour LUTs and LUT-based methods: load and save
LUTs, apply them on CPU or GPU with several interpolation schemes, measure
them, and compare methods (such as a learned NILUT) under one evaluation
harness.

```python
import numpy as np
import lutlab

lut = lutlab.testing.random_lut(17, seed=0)        # or: lutlab.LUT.load("grade.cube")
lut.save("grade.cube")                             # .cube / .json / .npy
lut = lutlab.LUT.load("grade.cube")

img = np.random.rand(1080, 1920, 3).astype(np.float32)   # float in [0, 1]
out = lut(img, interp="tetrahedral")               # backend="auto": numba if installed
print(lutlab.metrics.e00(out, img).mean())         # CIEDE2000; needs the `metrics` extra
```

## Install

```
pip install lutlab                 # numpy only: reference backend, formats, LUT type
pip install "lutlab[fast]"         # + numba CPU kernels
pip install "lutlab[torch]"        # + differentiable / GPU kernels (or install your own ROCm/CUDA torch)
pip install "lutlab[metrics]"      # + CIEDE2000 / SSIM        ([eval] adds imageio for PairFolder)
```

The opt-in `fused` backend is a native HIP kernel compiled from source in
`lutlab/kernels/` (no separate package). `python -m lutlab info` shows what is
available on your machine.

## Layout

| package | what lives there |
|---|---|
| `lutlab.transforms` | `LUT`, `LUTMixture`, `Cost` |
| `lutlab.kernels` | `apply()`, `interpolate()`, `backends()`, backend registry; `k_numpy`, `k_numba`, `k_torch`, `k_fused` |
| `lutlab.formats` | `load` / `save` for `.cube`, `.json`, `.npy`, `.png` (Hald) |
| `lutlab.metrics` | quality (e00, psnr, ssim, lpips), timing, power, and the evaluation harness (`evaluate`, `run_matrix`, `PairFolder`) |
| `lutlab.methods` | method registry: things that *produce* a transform (`identity`, `pg_ia_nilut`) |
| `lutlab.testing` | `random_lut`, `check_backends`, `conformance` |

## Conventions

* `LUT.volume` is `(D0, ..., Dn-1, 3)`: input axes first, output RGB last. Axes may differ in size and may
  be non-uniform (`axes=`); a 4D LUT is `n_in=4` with the 4th input passed as `context=`.
* Inputs are float in the LUT's `domain` (default `[0, 1]`); out-of-range values clamp on every backend;
  NaN/inf and integer images raise instead of silently misbehaving.
* The return type follows the input: numpy in, numpy out; torch in, torch out (differentiable on `torch`).
* `device=None` means "where the data is". Pass `device="cuda"` to send numpy input through a GPU backend.

| backend | runs on | grids | n_in | differentiable |
|---|---|---|---|---|
| `numpy` (reference) | CPU | any | any | no |
| `numba` | CPU, multi-threaded | uniform | any | no |
| `torch` | CPU / CUDA / ROCm / MPS | any | any | yes |
| `fused` (opt-in) | CUDA / ROCm | cubic uniform, nearest/trilinear/tetrahedral | 3 | no |

## Included methods

| method | needs | what it is |
|---|---|---|
| `identity` (`LUT.identity`) | - | the do-nothing LUT, the baseline |
| `pg_ia_nilut` | torch (+ `open_clip` to train) | Kosugi, MM'24 ([arXiv 2408.11055](https://arxiv.org/abs/2408.11055)) |

**PG-IA-NILUT**: an MLP filter `E(x, w) = x + e([x, sort(x), w]) - e([x, sort(x), 0])` whose five
parameters `w` get interpretable names (Exposure, Contrast, Saturation, Color temperature, Tint) from a
CLIP-based prompt guidance loss. At test time the MLP is baked into a 33^3 LUT ("LUT bypassing"), which
here is simply a lutlab `LUT`:

```python
from lutlab.methods import pg_ia_nilut as pg

model = pg.PGIANILUT(pg.IANILUT(), pg.CnnPredictor())      # filters E + parameter predictor F
w = model.filters.weights(exposure=0.4, color_temperature=-0.2)
model.filters.lut(w, N=33).save("look.cube")                # LUT bypassing -> .cube
out = model.transform(w)(img)                               # numpy image through lutlab's CPU kernels
out = model.transform()(img)                                # w = F(img): image-adaptive enhancement
```

Training runs the paper's three stages (`pg.train_pg_ia_nilut`, see `examples/train_pg_ia_nilut.py`).
The paper does not state optimiser settings, loss weights or the CLIP backbone; `TrainConfig` marks
which values are from the paper and which are placeholders.

## Adding things

```python
# a backend: a function (volume, x[N, n], axes, scheme) -> [N, 3] plus capability checks
from lutlab.kernels import Backend, register
register(Backend(name="mine", run=run, unsupported=lambda req: None, priority=25))

# a method: anything that builds a transform
from lutlab import methods
@methods.register("nilut", requires=("torch",))
def build(*, image=None, target=None, **cfg): ...
```

Every transform should pass `lutlab.testing.conformance(t)`; the test suite parametrizes over
backends, so a new backend is checked against the numpy oracle and scipy for free.

## Development

```
pip install -e ".[dev]" && pytest
```

See `MIGRATION.md` for what changed from 0.1.
