# lutlab

A small Python library for 3D colour lookup tables (LUTs): **load** them in the
major formats, **apply** them to images on CPU or GPU, **interpolate** between
table nodes, **measure** quality (ΔE₀₀, PSNR, SSIM, LPIPS) and speed, and
**extract** a LUT from a Hald CLUT image.

The motivation and the benchmark results are in the companion paper,
[`lutlab-paper`](https://github.com/tsutsen/lutlab-paper).

## Install

```bash
# CPU (numpy / numba backends) — nothing else needed:
uv venv && uv pip install -e .        # or: pip install -e .

# GPU (grid_sample / fused backends) — additionally:
#   1. a ROCm or CUDA build of torch (2.x), and
#   2. the `robust_hald` package importable (it provides grid_sample + the fused kernel).
#      e.g. put it on PYTHONPATH, or `uv pip install -e /path/to/robust_hald`.
```

## Quickstart

```python
import numpy as np, lutlab

lut  = lutlab.load_lut("sunset.cube")          # or .npy / .json / a LUT object
img  = np.random.rand(1080, 1920, 3)           # float, in [0, 1]
out  = lutlab.apply(lut, img, interp="trilinear", backend="auto")

# quality of `out` against a reference:
lutlab.metrics.e00(out, ref).mean()

# build a Hald CLUT image and read a LUT back from it:
hald = lutlab.hald.make_hald(levels=17, cell=12)
lut2 = lutlab.extract(hald)
```

`backend=` picks the accelerator: `auto` (GPU if present, else numba), `numpy`,
`numba`, `gpu` (grid_sample), `fused` (single-pass HIP/CUDA kernel — fastest on
the GPU, no PCIe round-trip).

## Modules

| module | what it does |
|---|---|
| `lutlab.lut` | the `LUT` container (volume, name, metadata) |
| `lutlab.apply` | apply a LUT to an image (`backend=`, `interp=`) |
| `lutlab.interp` | `nearest`, `trilinear` (default `multilinear`), `tetrahedral` |
| `lutlab.formats` | load/save `.cube`, `.npy`, `.json` (resample on load) |
| `lutlab.metrics` | ΔE₀₀, PSNR, SSIM, LPIPS, apply-speed, accuracy |
| `lutlab.hald` | generate a Hald CLUT image and extract a LUT from one |
| `lutlab.extract` | convenience extraction entry points |
| `lutlab.segmented` | segmented LUTs (K-means segments + per-segment parametric grade) |

Run the self-test: `python -m lutlab.selftest`.

## Interpolation kernels

- **nearest** — 1 read, no smoothness (banding).
- **trilinear** (= `multilinear`, the default) — 8 reads, smooth, on every backend.
- **tetrahedral** — 4 reads, smoothest; on CPU only in the current build
  (a GPU tetrahedral path is not exposed yet — see the paper's *Limitations*).

## GPU / fused kernel

The `fused` backend compiles a small HIP/CUDA kernel at first use (~1 s, cached).
On ROCm it needs `hipcc` and the ROCm dev libraries (`hipsparse`, `hipblas`,
`hipsolver`, `hipfft`, `hiprand`, `thrust`). On Arch/CachyOS: `sudo pacman -S rocm`
pulls them. The kernel is benchmarked in the paper (fastest measured apply path).

## License

MIT — see [LICENSE](LICENSE).
