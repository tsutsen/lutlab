# Migrating from lutlab 0.1 to 0.2

> **Note (0.3):** this page is the historical 0.1 -> 0.2 rename guide. Some
> helpers mentioned below were removed or renamed again in 0.3; the table and
> notes have been updated to the current names where they still exist.

## Renames and moves

| 0.1 | 0.2 |
|---|---|
| `lutlab.lut.LUT`, `.ndim` | `lutlab.LUT`, `.n_in` |
| `lutlab.lut.identity(D, ndim)` | `lutlab.LUT.identity(D, n_in)` (3-channel output; 2D has no identity) |
| `lutlab.interp` + `lutlab.apply` (two dispatchers) | `lutlab.kernels` (one registry; `lutlab.kernels.apply` is the function) |
| `lutlab.apply.apply(lut, img, interp, backend)` | `lut(img, interp=..., backend=...)` or `lutlab.kernels.apply(lut, img, ...)` |
| backend `"gpu"` (grid_sample) | `backend="torch"` (+ `device="cuda"` for numpy input) |
| backend `"fused"` | unchanged key, opt-in only |
| `lutlab.formats.from_cube / to_cube / from_json ...`, `LUT.to_cube()` | `lutlab.load(path)`, `lut.save(path)`, `lutlab.formats.cube.dumps/loads` |
| `lutlab.formats.resample`, `.validate` | `lutlab.analysis.resample`; backend agreement: `lutlab.testing.check_backends` |
| `lutlab.metrics.e00 / psnr / ssim / accuracy` | same names; also `lutlab.metrics.quality`, `.timing`, `.power` |
| `lutlab.extract`, `lutlab.hald` (re-exports of robust_hald) | removed in 0.3; Hald read/write lives in `lutlab.formats.hald` |
| `lutlab.segmented.segmented_apply` | `lutlab.LUTMixture(luts, weights=masks)` |
| `lutlab.segmented.fit_and_apply` | removed in 0.3; use `lutlab.LUTMixture` |
| `python -m lutlab bench` | removed; `python -m lutlab selftest` runs the sanity checks |
| `lutlab.selftest` | `python -m lutlab selftest` and `pytest` |

## Behaviour changes (all deliberate)

* `interp="nearest"` rounds to the nearest grid point; 0.1 floored (a bias toward black).
* `apply` no longer imports torch for CPU backends.
* `backend="auto"` follows the data: numpy -> CPU (numba if installed), torch tensor -> its device.
  0.1 sent numpy input to the GPU whenever one existed.
* NaN/inf raise `ValueError`; integer images raise `TypeError` (use `lutlab.to_float`).
  Unknown backends and impossible backend/scheme combinations raise `BackendError` with the reason.
* `.cube` `DOMAIN_MIN/MAX` is honoured. `.cube` files are written in the standard header
  (`TITLE`, `LUT_3D_SIZE`, ...); the 0.1 header is still read.
* Segment masks are channel-last `(H, W, S)` (0.1: `(S, H, W)`). Custom segmenters must follow.
* `LUT` is immutable and copies its input.
* `ModuleNotFoundError: robust_hald` can no longer happen at `import lutlab`; the `sys.path`
  hack and hard-coded paths are gone. `robust_hald` is not a lutlab dependency.
* `numba`, `scikit-image`, `scipy` are optional extras; core needs only numpy.
* The numpy tetrahedral kernel is vectorised (0.1: a Python loop per pixel).
