"""The single entry point for applying LUTs: `apply` (LUT-aware) and
`interpolate` (raw arrays).

Contract
--------
* Input ``x`` is ``(..., C)`` float in the LUT's domain (default [0, 1]).
  Out-of-range values are clamped to the grid edge on every backend.
  Integer images are rejected (convert them to float in [0, 1] first).
* Non-finite inputs raise for numpy arrays (``check_finite=None`` default);
  torch inputs are not checked by default to avoid a device sync.
* The return type follows the input type: numpy in -> numpy out,
  torch in -> torch out (on the same device, differentiable when the
  selected backend is).
* ``device=None`` means "where the data is": numpy -> ``"cpu"``,
  torch tensor -> its device. Pass ``device="cuda"`` to push numpy input
  through a GPU backend.
"""
from __future__ import annotations

import weakref

import numpy as np

from .._arrays import is_torch
from .._deps import have
from ..transforms.lut import LUT, _normalise_axes
from ._registry import BackendError, Request, select
from .schemes import canonical_scheme

__all__ = ["apply", "interpolate"]

# LUT is immutable, so one float tensor per (LUT, device, dtype) is safe to reuse.
# Weak keys keep the cache from extending a LUT's lifetime (and its device memory).
_LUT_TENSORS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _lut_tensor(lut: LUT, torch, device, dtype):
    cache = _LUT_TENSORS.setdefault(lut, {})
    key = (str(device), str(dtype))
    tensor = cache.get(key)
    if tensor is None:
        array = np.array(lut.volume, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array).to(device=device, dtype=dtype)
        cache[key] = tensor
    return tensor


def _as_array(x):
    return x if (isinstance(x, np.ndarray) or is_torch(x)) else np.asarray(x)


def _require_float(x):
    ok = x.dtype.is_floating_point if is_torch(x) else np.issubdtype(x.dtype, np.floating)
    if not ok:
        raise TypeError(f"expected a floating-point array in [0, 1], got dtype {x.dtype}; "
                        "convert integer images to float in [0, 1] first")


def _to_tensor(a):
    """numpy -> float32 torch tensor; copies when the array is read-only (e.g. `LUT.volume`)."""
    import torch
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.from_numpy(a if a.flags.writeable else a.copy())


def _cat_context(x, ctx, n_in):
    if ctx is None:
        return x
    k = n_in - x.shape[-1]
    if is_torch(x):
        import torch
        ctx = torch.as_tensor(ctx, dtype=x.dtype, device=x.device)
    else:
        ctx = np.asarray(ctx, dtype=np.float32)
    if k <= 0 or ctx.shape[-1] != k:
        raise ValueError(f"LUT takes {n_in} input channels: x has {x.shape[-1]}, so "
                         f"context must supply {max(k, 0)}, got {ctx.shape[-1]}")
    shape = tuple(x.shape[:-1]) + (k,)
    if is_torch(x):
        import torch
        return torch.cat([x, ctx.expand(shape)], dim=-1)
    return np.concatenate([x, np.broadcast_to(ctx, shape)], axis=-1)


def interpolate(volume, x, *, axes=None, scheme: str = "multilinear", backend: str = "auto",
                device=None, check_finite=None):
    """Interpolate grid ``volume (D0..Dn-1, 3)`` at points ``x (..., n)`` in [0, 1].

    ``volume`` / ``x`` may be numpy arrays or torch tensors (``volume`` can be a
    trainable parameter). ``axes`` are optional rectilinear sample positions,
    validated with the same rules as a `LUT` (right length, 0..1, increasing).
    """
    volume, x = _as_array(volume), _as_array(x)
    xt, vt = is_torch(x), is_torch(volume)
    if volume.ndim < 2 or volume.shape[-1] != 3:
        raise ValueError(f"volume must have shape (D0, ..., Dn-1, 3); got {tuple(volume.shape)}")
    n_in = volume.ndim - 1
    sizes = tuple(int(s) for s in volume.shape[:-1])
    if axes is not None:
        # Same rules as LUT: right length, 0..1, strictly increasing. Torch axes
        # are converted for the check only; the backend still gets the originals.
        _normalise_axes([a.detach().cpu().numpy() if is_torch(a) else a for a in axes], sizes)
    if x.ndim < 1 or x.shape[-1] != n_in:
        got = x.shape[-1] if x.ndim else 0
        hint = " (pass context= for the extra channels)" if got < n_in else ""
        raise ValueError(f"input has {got} channels but the LUT takes {n_in}{hint}")
    _require_float(x)
    kernel = canonical_scheme(scheme, n_in)

    if check_finite is None:
        check_finite = not xt
    if check_finite:
        finite = bool(x.isfinite().all()) if xt else bool(np.isfinite(x).all())
        if not finite:
            raise ValueError("input contains NaN or inf")

    if device is not None:
        dev = str(device)
    elif xt:
        dev = str(x.device)
    elif vt:
        dev = str(volume.device)
    else:
        dev = "cpu"

    grad = False
    if xt or vt:
        import torch
        grad = torch.is_grad_enabled() and bool(
            (xt and x.requires_grad) or (vt and volume.requires_grad))
    req = Request(scheme=kernel, n_in=n_in, sizes=sizes, uniform=axes is None,
                  device=dev, grad=grad, from_torch=xt)
    b = select(req, backend)

    lead = tuple(x.shape[:-1])
    if b.torch_native:
        import torch
        td = torch.device(dev)
        xin = x if xt else _to_tensor(x)
        vin = volume if vt else _to_tensor(volume)
        xin, vin = xin.to(td), vin.to(td)
        ax = None if axes is None else tuple(
            (a if is_torch(a) else _to_tensor(a)).to(td) for a in axes)
        out = b.run(vin, xin.reshape(-1, n_in), ax, kernel).reshape(lead + (3,))
        return out if xt else out.detach().cpu().numpy()

    xa = x.detach().cpu().numpy() if xt else x
    va = volume.detach().cpu().numpy() if vt else volume
    xa = np.ascontiguousarray(xa, dtype=np.float32).reshape(-1, n_in)
    va = np.ascontiguousarray(va, dtype=np.float32)
    ax = None if axes is None else tuple(
        np.ascontiguousarray(a.detach().cpu().numpy() if is_torch(a) else a, np.float32)
        for a in axes)
    out = b.run(va, xa, ax, kernel).reshape(lead + (3,))
    if xt:
        import torch
        return torch.from_numpy(out).to(x.device)
    return out


def apply_mixture(luts, x, weights, *, interp: str = "multilinear", backend: str = "auto",
                  device=None, check_finite=None, domain=None):
    """Apply S LUTs on one grid and blend them per pixel in one lookup.

    ``luts`` must share sizes, axes and domain; ``weights`` is ``(..., S)``. This
    is the one-pass path behind :class:`LUTMixture`; use ``backend="torch"`` for
    the differentiable/GPU version and ``backend="numpy"`` for the CPU version.
    """
    from . import k_numpy, k_torch

    x = _as_array(x)
    _require_float(x)
    S = len(luts)
    n_in = luts[0].n_in
    if x.ndim < 1 or x.shape[-1] != n_in:
        raise ValueError(f"input has {x.shape[-1] if x.ndim else 0} channels but the "
                         f"LUTs take {n_in}")
    wshape = getattr(weights, "shape", None)
    if wshape is None:
        wshape = np.asarray(weights).shape
    if len(wshape) < 1 or wshape[-1] != S:
        raise ValueError(f"weights have {wshape[-1] if wshape else 0} channels for {S} LUTs")
    kernel = canonical_scheme(interp, n_in)
    lead = tuple(x.shape[:-1])

    if check_finite is None:
        check_finite = not is_torch(x)
    if check_finite:
        finite = bool(x.isfinite().all()) if is_torch(x) else bool(np.isfinite(x).all())
        if not finite:
            raise ValueError("input contains NaN or inf")

    if domain is not None:
        lo, hi = domain
        if is_torch(x):
            import torch
            lo_a = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
            hi_a = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
        else:
            lo_a, hi_a = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
        x = (x - lo_a) / (hi_a - lo_a)

    torch_path = (is_torch(x) or backend == "torch"
                  or (device is not None and str(device).split(":")[0] != "cpu"))
    if torch_path:
        if not (is_torch(x) or have("torch")):
            raise BackendError("torch is not installed (pip install lutlab[torch])")
        import torch
        td = torch.device(device) if device is not None else (
            x.device if is_torch(x) else "cpu")
        dtype = (x.dtype if is_torch(x) and x.dtype in (torch.float32, torch.float64)
                 else torch.float32)
        xt = (x if is_torch(x) else torch.as_tensor(x, dtype=dtype)).to(td, dtype)
        wt = (weights if is_torch(weights) else torch.as_tensor(weights, dtype=dtype)).to(td)
        volumes = torch.stack([_lut_tensor(l, torch, td, dtype) for l in luts])
        axes = None if luts[0].axes is None else tuple(
            torch.as_tensor(a, dtype=dtype, device=td) for a in luts[0].axes)
        x2, w2 = xt.reshape(-1, n_in), wt.reshape(-1, S)
        if x2.shape[0] != w2.shape[0]:
            raise ValueError("weights leading shape does not match the input")
        out = k_torch.mix(volumes, x2, w2, axes, kernel).reshape(lead + (3,))
        return out if is_torch(x) else out.detach().cpu().numpy()

    if is_torch(x):
        raise BackendError("a torch input needs backend='torch' or a non-CPU device")
    xa = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, n_in)
    wa = (weights.detach().cpu().numpy() if is_torch(weights)
          else np.asarray(weights, np.float32))
    wa = np.ascontiguousarray(wa, dtype=np.float32).reshape(-1, S)
    if xa.shape[0] != wa.shape[0]:
        raise ValueError("weights leading shape does not match the input")
    volumes = np.stack([l.volume for l in luts])
    out = k_numpy.mix(volumes, xa, wa, luts[0].axes, kernel).reshape(lead + (3,))
    return out


def apply(lut, x, *, context=None, interp: str = "multilinear", backend: str = "auto",
          device=None, check_finite=None):
    """Apply a colour transform to ``x``.

    ``lut`` may be a `LUT`, a bare ``(D..., 3)`` array (uniform grid, unit
    domain) or any other transform (called as ``lut(x, context=...)``;
    ``interp``/``backend``/``device`` are forwarded only if non-default).
    ``context`` supplies extra input channels (e.g. the 4th axis of a 4D LUT).
    """
    x = _as_array(x)
    if isinstance(lut, LUT):
        volume, axes, (lo, hi) = lut.volume, lut.axes, lut.domain
        default_domain = lut.has_default_domain
    elif isinstance(lut, np.ndarray) or is_torch(lut):
        volume, axes, default_domain = lut, None, True
    else:
        kw = {k: v for k, v in (("interp", interp), ("backend", backend), ("device", device))
              if v not in ("multilinear", "auto", None)}
        return lut(x, context=context, **kw)

    _require_float(x)
    x = _cat_context(x, context, volume.ndim - 1)

    use_tensor = isinstance(lut, LUT) and backend not in ("numpy", "numba") and (
        is_torch(x) or backend == "torch"
        or (device is not None and str(device).split(":")[0] != "cpu"))
    if use_tensor and not (is_torch(x) or have("torch")):
        use_tensor = False
    if use_tensor:
        import torch
        dev = device if device is not None else (x.device if is_torch(x) else "cpu")
        dtype = (x.dtype if is_torch(x) and x.dtype in (torch.float32, torch.float64)
                 else torch.float32)
        if backend == "fused":
            dtype = torch.float32
        volume = _lut_tensor(lut, torch, torch.device(dev), dtype)

    if not default_domain:
        if is_torch(x):
            import torch
            lo_a = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
            hi_a = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
        else:
            lo_a, hi_a = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
        x = (x - lo_a) / (hi_a - lo_a)
    return interpolate(volume, x, axes=axes, scheme=interp, backend=backend,
                       device=device, check_finite=check_finite)
