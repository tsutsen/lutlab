# Migrating from lutlab 0.1 to 0.2

## Renames and moves

| 0.1 | 0.2 |
|---|---|
| `lutlab.lut.LUT`, `.ndim` | `lutlab.LUT`, `.n_in` |
| `lutlab.lut.identity(D, ndim)` | `lutlab.identity(D, n_in)` (3-channel output; 2D has no identity) |
| `lutlab.interp` + `lutlab.apply` (two dispatchers) | `lutlab.kernels` (one registry; `lutlab.apply` is the function) |
| `lutlab.apply.apply(lut, img, interp, backend)` | `lut(img, interp=..., backend=...)` or `lutlab.apply(lut, img, ...)` |
| backend `"gpu"` (grid_sample) | `backend="torch"` (+ `device="cuda"` for numpy input) |
| backend `"fused"` | unchanged key, opt-in only, needs `robust_hald` |
| `lutlab.formats.from_cube / to_cube / from_json ...`, `LUT.to_cube()` | `lutlab.load(path)`, `lut.save(path)`, `lutlab.formats.cube.dumps/loads` |
| `lutlab.formats.resample`, `.validate` | `lutlab.analysis.resample`; backend agreement: `lutlab.testing.check_backends` |
| `lutlab.metrics.e00 / psnr / ssim / accuracy` | same names; also `lutlab.metrics.quality`, `.timing`, `.power` |
| `lutlab.extract`, `lutlab.hald` (re-exports of robust_hald) | `lutlab.methods.hald_extract.extract_from_hald / extract_from_pair / make_hald`; import other robust_hald helpers from `robust_hald` directly |
| `lutlab.segmented.segmented_apply` | `lutlab.LUTMixture(luts, weights=masks)` |
| `lutlab.segmented.fit_and_apply` | `lutlab.methods.segmented.fit_segmented` -> `SegmentedResult` |
| `python -m lutlab bench` | `benchmarks/hald_misalignment.py` |
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
  hack and hard-coded paths are gone. Install `robust_hald` normally.
* `numba`, `scikit-image`, `scipy` are optional extras; core needs only numpy.
* The numpy tetrahedral kernel is vectorised (0.1: a Python loop per pixel).
