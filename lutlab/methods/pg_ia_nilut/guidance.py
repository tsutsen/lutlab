"""Prompt guidance loss (paper Sec. 4.2, Eq. 6-7).

For random weights w_bar the enhanced image is scored with J CLIP prompt pairs, ``c = CLIPscore(E(X, w_bar))``.
The loss pushes the score Jacobian ``dc_j'/dw_j`` toward the identity::

    L_PG = sum_j  lambda_j * |dc_j/dw_j - 1|  +  lambda * sum_{j' != j} |dc_j'/dw_j|

so that changing w_j moves only the j-th impression score ("Exposure", "Contrast", ...).

`ClipScorer` is one scorer; any callable ``images (B, H, W, 3) -> (B, J)`` works, which is
how the loss is tested without CLIP weights.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..._deps import require
from .model import IANILUT
from .prompts import guiding_prompts

__all__ = ["ClipScorer", "score_jacobian", "prompt_guidance_loss", "guidance_step_loss"]


class ClipScorer(nn.Module):
    """CLIP impression scores ``exp(s+) / (exp(s+) + exp(s-))`` for J (positive, negative) prompt pairs.

    ``s`` is CLIP's scaled cosine similarity (``logit_scale * cos``), as in CLIP-IQA. Fully
    differentiable w.r.t. the image; CLIP's weights stay frozen. Needs ``pip install open_clip_torch``
    and downloads weights on first use. The paper does not state the backbone; RN50 (the
    CLIP-IQA choice) is an assumption.
    """

    def __init__(self, prompts=None, *, dataset: str = "fivek", model_name: str = "RN50",
                 pretrained: str = "openai", device: str | torch.device = "cpu") -> None:
        super().__init__()
        oc = require("open_clip")
        prompts = prompts if prompts is not None else guiding_prompts(dataset)
        model, _, _ = oc.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval().requires_grad_(False)
        tok = oc.get_tokenizer(model_name)
        with torch.no_grad():
            flat = [p for pair in prompts for p in pair]
            t = F.normalize(model.encode_text(tok(flat).to(device)).float(), dim=-1)
        self.model = model
        self.register_buffer("text", t.view(len(prompts), 2, -1))
        self.register_buffer("mean", torch.tensor(oc.constants.OPENAI_DATASET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(oc.constants.OPENAI_DATASET_STD).view(1, 3, 1, 1))
        self.scale = float(model.logit_scale.exp())
        size = model.visual.image_size
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images ``(B, H, W, 3)`` in [0, 1] -> scores ``(B, J)`` in (0, 1)."""
        x = F.interpolate(images.permute(0, 3, 1, 2), size=self.size, mode="bicubic",
                          align_corners=False, antialias=True)
        f = F.normalize(self.model.encode_image((x - self.mean) / self.std).float(), dim=-1)
        sims = torch.einsum("bd,jkd->bjk", f, self.text)
        return torch.sigmoid(self.scale * (sims[..., 0] - sims[..., 1]))


def score_jacobian(c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Per-sample Jacobian ``J[b, k, j] = dc[b, k] / dw[b, j]`` for ``c (B, K)`` computed from
    ``w (B, J)``. Differentiable (``create_graph``), so it can be used inside a loss. Samples must
    be independent, which makes summing over the batch before differentiating exact."""
    rows = []
    for k in range(c.shape[1]):
        (g,) = torch.autograd.grad(c[:, k].sum(), w, create_graph=True, retain_graph=True,
                                   allow_unused=True)
        rows.append(torch.zeros_like(w) if g is None else g)
    return torch.stack(rows, dim=1)


def prompt_guidance_loss(jac: torch.Tensor, lambdas=1.0, lam: float = 1.0) -> torch.Tensor:
    """Eq. (7) from a Jacobian ``(B, J, J)`` (rows: scores, columns: weights); mean over the batch.

    ``lambdas`` is the per-filter weight lambda_j (scalar or length J); ``lam`` weights the
    off-diagonal (cross-talk) terms.
    """
    B, J, J2 = jac.shape
    if J != J2:
        raise ValueError(f"need as many prompt pairs as parameters, got Jacobian {tuple(jac.shape)}")
    lam_j = torch.as_tensor(lambdas, dtype=jac.dtype, device=jac.device).expand(J)
    diag = jac.diagonal(dim1=1, dim2=2)
    off = jac - torch.diag_embed(diag)
    per_sample = (lam_j * (diag - 1.0).abs()).sum(1) + lam * off.abs().sum((1, 2))
    return per_sample.mean()


def guidance_step_loss(filters: IANILUT, scorer, images: torch.Tensor, *, lambdas=1.0,
                       lam: float = 1.0, w_range: float = 1.0, bypass: bool = True, N: int = 33,
                       generator: torch.Generator | None = None):
    """One L_PG evaluation on a batch ``images (B, H, W, 3)``: sample w_bar ~ U(-w_range, w_range)^J,
    enhance, score, differentiate. Returns ``(loss, jacobian)``."""
    B, dev = images.shape[0], images.device
    w = ((torch.rand(B, filters.n_params, device=dev, generator=generator) * 2 - 1) * w_range
         ).requires_grad_(True)
    c = scorer(filters.enhance(images, w, bypass=bypass, N=N))
    if c.shape[1] != filters.n_params:
        raise ValueError(f"scorer returned {c.shape[1]} scores for {filters.n_params} parameters")
    jac = score_jacobian(c, w)
    return prompt_guidance_loss(jac, lambdas, lam), jac
