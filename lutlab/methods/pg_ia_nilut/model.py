"""IA-NILUT: an MLP colour filter conditioned on interpretable parameters, plus LUT bypassing.

Paper: S. Kosugi, "Prompt-Guided Image-Adaptive Neural Implicit Lookup Tables for
Interpretable Image Enhancement", ACM MM 2024 (arXiv 2408.11055).

Eq. (3)   y = x + e([x, sort(x), w]) - e([x, sort(x), 0])
Eq. (5)   LUT bypassing: O = E_hat(I, w) on an N^3 grid I, then y = Lookup(x, {I, O}).

Colour values are channel-LAST ``(..., 3)`` in [0, 1] throughout, like the rest of lutlab.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ...kernels.dispatch import interpolate
from ...transforms.base import Cost
from ...transforms.lut import LUT
from .prompts import FILTER_NAMES, name_key

__all__ = ["IANILUT", "CnnPredictor", "ResNetPredictor", "PGIANILUT", "IANILUTTransform"]

_CHUNK = 1 << 18          # pixels per MLP call when bypassing is off


def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class IANILUT(nn.Module):
    """The filter bank E: ``(rgb, sort(rgb), w) -> rgb`` with identity at ``w = 0``.

    Paper settings: 5 fully connected layers, 256 hidden units, tanh, J = 5 parameters.

    Note: Eq. (3) subtracts ``e(., 0)``, so the last layer's bias cancels exactly and never receives a
    gradient (3 of the 201,219 parameters are inert by construction; kept to match the paper's count).
    """

    def __init__(self, n_params: int = 5, hidden: int = 256, n_layers: int = 5,
                 names: tuple[str, ...] | None = None) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        names = tuple(names) if names is not None else (
            FILTER_NAMES if n_params == len(FILTER_NAMES)
            else tuple(f"filter_{j + 1}" for j in range(n_params)))
        if len(names) != n_params:
            raise ValueError(f"{len(names)} names for {n_params} parameters")
        self.n_params, self.hidden, self.n_layers, self.names = n_params, hidden, n_layers, names
        dims = [3 + 3 + n_params] + [hidden] * (n_layers - 1) + [3]
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < n_layers - 1:
                layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    # -- Eq. (3) -------------------------------------------------------------------- #
    @staticmethod
    def _features(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, torch.sort(x, dim=-1).values, w], dim=-1)

    @staticmethod
    def _expand(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Broadcast w ``(J,)`` or ``(B, J)`` over the pixels of x ``(..., 3)`` / ``(B, ..., 3)``."""
        w = w.to(dtype=x.dtype, device=x.device)
        if w.ndim == 1:
            return w.expand(*x.shape[:-1], w.shape[0])
        if w.ndim != 2 or x.ndim < 2 or w.shape[0] != x.shape[0]:
            raise ValueError(f"w {tuple(w.shape)} does not match x {tuple(x.shape)}: "
                             "use w of shape (J,) or (B, J) with x of shape (B, ..., 3)")
        w = w.reshape(w.shape[0], *([1] * (x.ndim - 2)), w.shape[1])
        return w.expand(*x.shape[:-1], w.shape[-1])

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply the MLP directly to every pixel (no LUT bypassing)."""
        if x.shape[-1] != 3:
            raise ValueError(f"expected channel-last RGB, got shape {tuple(x.shape)}")
        w = self._expand(x, w)
        out = self.mlp(torch.stack([self._features(x, w), self._features(x, torch.zeros_like(w))]))
        return x + (out[0] - out[1])

    # -- Eq. (5): LUT bypassing ------------------------------------------------------- #
    def grid(self, N: int, device=None, dtype=torch.float32) -> torch.Tensor:
        """The N^3 sampling points I as an ``(N, N, N, 3)`` grid, indexed ``[r, g, b]``."""
        t = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=-1)

    def lut_volume(self, w: torch.Tensor, N: int = 33) -> torch.Tensor:
        """``O = E_hat(I, w)`` as an ``(N, N, N, 3)`` tensor (differentiable)."""
        p = next(self.parameters())
        I = self.grid(N, p.device, p.dtype)
        return self(I.reshape(-1, 3), w.to(p.device)).reshape(N, N, N, 3)

    @torch.no_grad()
    def lut(self, w, N: int = 33, name: str = "ia-nilut") -> LUT:
        """Bake the filter for parameters ``w (J,)`` into a lutlab `LUT` (N = 33 in the paper)."""
        w = torch.as_tensor(np.asarray(w, np.float32) if not torch.is_tensor(w) else w)
        if w.ndim != 1:
            raise ValueError("lut() takes one parameter vector of shape (J,)")
        return LUT(self.lut_volume(w, N).cpu().numpy(), name=name)

    def enhance(self, x: torch.Tensor, w: torch.Tensor, *, bypass: bool = True,
                N: int = 33) -> torch.Tensor:
        """``E(X, w)``. With ``bypass`` the MLP only sees the N^3 grid; the image goes through
        a differentiable trilinear lookup (lutlab's torch backend). ``w`` is ``(J,)`` or ``(B, J)``."""
        if not bypass:
            return self(x, w)
        if w.ndim == 1:
            return interpolate(self.lut_volume(w, N), x, scheme="multilinear", backend="torch")
        if w.shape[0] != x.shape[0]:
            raise ValueError(f"w has batch {w.shape[0]} but x has {x.shape[0]}")
        return torch.stack([interpolate(self.lut_volume(w[b], N), x[b], scheme="multilinear",
                                        backend="torch") for b in range(x.shape[0])])

    def weights(self, **named: float) -> torch.Tensor:
        """Parameter vector from filter names: ``weights(exposure=0.4, color_temperature=-0.2)``."""
        keys = {name_key(n): i for i, n in enumerate(self.names)}
        w = torch.zeros(self.n_params)
        for k, v in named.items():
            if k not in keys:
                raise ValueError(f"unknown filter {k!r}; options: {', '.join(keys)}")
            w[keys[k]] = float(v)
        return w


# ---- parameter predictors F ------------------------------------------------------- #
def _to_nchw(x: torch.Tensor, size: int) -> torch.Tensor:
    x = x.permute(0, 3, 1, 2)
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


class CnnPredictor(nn.Module):
    """Five-layer CNN predictor F (FiveK setting), after the image-adaptive 3D-LUT classifier of
    Zeng et al.: 256x256 input, five stride-2 convs, global pooling, 1x1 conv to J outputs."""

    def __init__(self, n_params: int = 5, size: int = 256) -> None:
        super().__init__()
        self.size = size

        def block(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
            if norm:
                layers.append(nn.InstanceNorm2d(o, affine=True))
            return layers

        self.net = nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64), *block(64, 128),
                                 *block(128, 128, norm=False), nn.Dropout(p=0.5),
                                 nn.AdaptiveAvgPool2d(1), nn.Conv2d(128, n_params, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x ``(B, H, W, 3)`` in [0, 1] -> w ``(B, J)``."""
        return self.net(_to_nchw(x, self.size)).flatten(1)


class ResNetPredictor(nn.Module):
    """ResNet-18 predictor F (PPR10K setting). Needs torchvision."""

    def __init__(self, n_params: int = 5, size: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import resnet18
        self.size = size
        self.net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_params)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((_to_nchw(x, self.size) - self.mean) / self.std)


def make_predictor(kind: str, n_params: int) -> nn.Module:
    if kind == "cnn":
        return CnnPredictor(n_params)
    if kind == "resnet18":
        return ResNetPredictor(n_params)
    raise ValueError(f"unknown predictor {kind!r}; options: cnn, resnet18")


# ---- the full model + lutlab integration ------------------------------------------ #
class PGIANILUT(nn.Module):
    """Filters E and (optionally) the parameter predictor F: ``Y = E(X, F(X))``."""

    def __init__(self, filters: IANILUT | None = None, predictor: nn.Module | None = None) -> None:
        super().__init__()
        self.filters = filters or IANILUT()
        self.predictor = predictor

    def predict_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.predictor is None:
            raise ValueError("this model has no parameter predictor")
        return self.predictor(x)

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None, *, bypass: bool = True,
                N: int = 33) -> torch.Tensor:
        return self.filters.enhance(x, self.predict_weights(x) if w is None else w,
                                    bypass=bypass, N=N)

    def transform(self, w=None, *, N: int = 33, bypass: bool = True) -> "IANILUTTransform":
        """A lutlab `transform`. Fixed parameters ``w``, or (``w=None``) predicted per image."""
        return IANILUTTransform(self, w, N=N, bypass=bypass)

    def num_parameters(self) -> dict[str, int]:
        f, p = _n_params(self.filters), (_n_params(self.predictor) if self.predictor else 0)
        return {"filters": f, "predictor": p, "total": f + p}

    # -- checkpoints ------------------------------------------------------------------ #
    def save(self, path) -> None:
        f = self.filters
        cfg = {"n_params": f.n_params, "hidden": f.hidden, "n_layers": f.n_layers,
               "names": list(f.names),
               "predictor": None if self.predictor is None else type(self.predictor).__name__}
        torch.save({"config": cfg, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "PGIANILUT":
        ck = torch.load(path, map_location=map_location, weights_only=True)
        c = ck["config"]
        pred = None
        if c["predictor"] == "CnnPredictor":
            pred = CnnPredictor(c["n_params"])
        elif c["predictor"] == "ResNetPredictor":
            pred = ResNetPredictor(c["n_params"])
        m = cls(IANILUT(c["n_params"], c["hidden"], c["n_layers"], tuple(c["names"])), pred)
        m.load_state_dict(ck["state"])
        return m.eval()


class IANILUTTransform:
    """lutlab `transform` view of a `PGIANILUT`.

    ``x`` is ``(..., 3)`` float in [0, 1] (numpy in -> numpy out, torch in -> torch out).
    Parameters come from ``context=`` (``(J,)``), else the bound ``w``, else -- for images of
    shape ``(H, W, 3)`` or ``(B, H, W, 3)`` -- the predictor F. With ``bypass`` (default) the MLP is
    baked into an N^3 lutlab `LUT` and numpy images use lutlab's CPU kernels; extra keyword
    arguments (``interp=``, ``backend=``) are forwarded to that lookup.
    """
    n_in = 3

    def __init__(self, model: PGIANILUT, w=None, *, N: int = 33, bypass: bool = True) -> None:
        self.model, self.N, self.bypass = model, N, bypass
        self.w = None if w is None else torch.as_tensor(np.asarray(
            w.detach().cpu() if torch.is_tensor(w) else w, np.float32))
        self.name = f"pg_ia_nilut(N={N}{'' if bypass else ', direct'})"

    def _dev(self):
        return next(self.model.parameters()).device

    def _w_for(self, x: torch.Tensor, context):
        if context is not None:
            return torch.as_tensor(np.asarray(context, np.float32)).to(self._dev())
        if self.w is not None:
            return self.w.to(self._dev())
        if self.model.predictor is None:
            raise ValueError("no parameters: pass context=w, bind w=, or give the model a predictor")
        if x.ndim < 3:
            raise ValueError("predicted parameters need an image (H, W, 3) or batch (B, H, W, 3)")
        return self.model.predict_weights(x if x.ndim == 4 else x[None])

    @torch.no_grad()
    def __call__(self, x, *, context=None, **kw):
        is_t = torch.is_tensor(x)
        xt = x if is_t else torch.as_tensor(np.asarray(x, np.float32))
        xt = xt.to(self._dev(), torch.float32)
        was_training = self.model.training
        self.model.eval()
        try:
            w = self._w_for(xt, context)
            if w.ndim == 2 and w.shape[0] == 1 and xt.ndim == 3:
                w = w[0]
            if self.bypass and not is_t and w.ndim == 1:                # numpy -> lutlab kernels
                return self.model.filters.lut(w, self.N)(np.asarray(x, np.float32), **kw)
            if self.bypass:
                out = self.model.filters.enhance(xt, w, bypass=True, N=self.N)
            elif w.ndim == 1:
                out = torch.cat([self.model.filters(c, w) for c in xt.reshape(-1, 3).split(_CHUNK)])
                out = out.reshape(xt.shape)
            else:
                out = self.model.filters(xt, w)
        finally:
            self.model.train(was_training)
        return out if is_t else out.cpu().numpy()

    def to_lut(self, grid_size: int | None = None, *, weights=None, **_kw) -> LUT:
        """The filter as one `LUT` for global parameters (per-image predicted ones need an image)."""
        w = weights if weights is not None else self.w
        if w is None:
            raise ValueError("to_lut() needs global parameters: pass weights= or bind w=")
        return self.model.filters.lut(w, grid_size or self.N, name=self.name)

    def cost(self) -> Cost:
        n = self.model.num_parameters()
        return Cost(params=n["total"], nbytes=4 * n["total"],
                    reads_per_px=8 if self.bypass else None,
                    meta={"filters": n["filters"], "predictor": n["predictor"], "lut_N": self.N})
