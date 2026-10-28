"""Three-stage training of PG-IA-NILUT (paper Sec. 4.3, Eq. 8-10)::

    stage 1   E                 <- L_PG                                              (Eq. 8)
    stage 2   E, w_1..w_I       <- L_PG + sum_i MSE(T_i, E(X_i, w_i))
                                        + sum_i MSE(X_i, E(T_i, -w_i))               (Eq. 9)
    stage 3   F (E frozen)      <- sum_i MSE(T_i, E(X_i, F(X_i)))                    (Eq. 10)

Settings the paper states: J = 5 parameters, 5-layer / 256-unit / tanh MLP, N = 33 for LUT bypassing,
FiveK vs PPR10K contrast prompts, the loss structure above. Settings it does NOT state (they are
placeholders here, see `TrainConfig`): optimiser and learning rates, number of steps/epochs, batch size,
lambda_j and lambda, the distribution of the random weights w_bar, the CLIP backbone, the weighting of
L_PG against the MSE terms, and the resolution the guidance loss sees.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .guidance import ClipScorer, guidance_step_loss
from .model import IANILUT, PGIANILUT, make_predictor

__all__ = ["TrainConfig", "ImagePairDataset", "TrainResult", "train_stage1", "train_stage2",
           "train_stage3", "train_pg_ia_nilut", "build_method"]


@dataclass
class TrainConfig:
    # -- stated in the paper -------------------------------------------------------- #
    n_params: int = 5
    hidden: int = 256
    n_layers: int = 5
    lut_size: int = 33
    dataset: str = "fivek"                  # 'fivek' | 'ppr10k' (only the Contrast prompts differ)
    predictor: str = "cnn"                  # 'cnn' (FiveK) | 'resnet18' (PPR10K)
    # -- NOT stated: placeholders, tune them ------------------------------------------ #
    size: tuple[int, int] = (256, 256)      # resolution of (X, T) while training
    pg_size: int = 224                      # resolution the guidance loss sees
    batch_size: int = 8
    lambda_j: float | tuple[float, ...] = 1.0
    lam: float = 1.0
    w_range: float = 1.0                    # w_bar ~ U(-w_range, w_range)^J
    pg_weight: float = 1.0                  # L_PG weight in stage 2 relative to the MSE terms
    stage1_steps: int = 500
    stage2_epochs: int = 20
    stage3_epochs: int = 20
    lr_filters: float = 1e-4
    lr_w: float = 1e-2
    lr_predictor: float = 1e-4
    bypass_train: bool = True               # train through the N^3 LUT (cheap) instead of per-pixel MLP
    seed: int = 0


def _resize(x: torch.Tensor, size) -> torch.Tensor:
    """Channel-last batch ``(B, H, W, 3)`` -> ``(B, h, w, 3)`` (bilinear, antialiased)."""
    size = (size, size) if isinstance(size, int) else tuple(size)
    if tuple(x.shape[1:3]) == size:
        return x
    y = F.interpolate(x.permute(0, 3, 1, 2), size=size, mode="bilinear", align_corners=False,
                      antialias=True)
    return y.permute(0, 2, 3, 1)


class ImagePairDataset(Dataset):
    """``(index, X, T)`` with X, T float32 ``(H, W, 3)`` tensors in [0, 1] at a fixed size.

    ``pairs`` is anything indexable yielding ``(input, target, ...)`` numpy images, e.g.
    ``lutlab.metrics.PairFolder`` or a list.
    """

    def __init__(self, pairs: Sequence, size: tuple[int, int] = (256, 256)) -> None:
        self.pairs, self.size = pairs, tuple(size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        x, t = self.pairs[i][:2]
        out = []
        for im in (x, t):
            tens = torch.as_tensor(im).float()
            out.append(_resize(tens[None], self.size)[0])
        return i, out[0], out[1]


@dataclass
class TrainResult:
    model: PGIANILUT
    w_table: torch.Tensor | None = None        # per-image parameters w_i from stage 2, (I, J)
    history: dict[str, list[float]] = field(default_factory=dict)


def _cycle(loader):
    while True:
        yield from loader


def _log(history, key, value):
    history.setdefault(key, []).append(float(value))


def train_stage1(model: PGIANILUT, scorer, loader, cfg: TrainConfig, *, device, history=None,
                 log: Callable | None = None) -> dict:
    """Stage 1 (Eq. 8): fit the filters E with the prompt guidance loss only."""
    history = {} if history is None else history
    if len(loader) == 0:
        raise ValueError("empty dataset: stage 1 needs at least one batch")
    model.filters.train()
    opt = torch.optim.Adam(model.filters.parameters(), lr=cfg.lr_filters)
    it = _cycle(loader)
    for step in range(cfg.stage1_steps):
        _, x, _ = next(it)
        x = _resize(x.to(device), cfg.pg_size)
        loss, _ = guidance_step_loss(model.filters, scorer, x, lambdas=cfg.lambda_j, lam=cfg.lam,
                                     w_range=cfg.w_range, bypass=cfg.bypass_train, N=cfg.lut_size)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _log(history, "stage1_pg", loss.item())
        if log and (step % 50 == 0 or step == cfg.stage1_steps - 1):
            log(f"[stage 1] step {step + 1}/{cfg.stage1_steps}  L_PG={loss.item():.4f}")
    return history


def train_stage2(model: PGIANILUT, scorer, dataset: Dataset, cfg: TrainConfig, *, device,
                 history=None, log: Callable | None = None) -> torch.Tensor:
    """Stage 2 (Eq. 9): jointly fit E and one parameter vector w_i per training image."""
    history = {} if history is None else history
    n = len(dataset)
    w_table = torch.nn.Parameter(torch.zeros(n, cfg.n_params, device=device))
    model.filters.train()
    opt = torch.optim.Adam([{"params": model.filters.parameters(), "lr": cfg.lr_filters},
                            {"params": [w_table], "lr": cfg.lr_w}])
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    E = lambda x, w: model.filters.enhance(x, w, bypass=cfg.bypass_train, N=cfg.lut_size)
    for epoch in range(cfg.stage2_epochs):
        tot = 0.0
        for idx, x, t in loader:
            x, t, w = x.to(device), t.to(device), w_table[idx.to(device)]
            pg, _ = guidance_step_loss(model.filters, scorer, _resize(x, cfg.pg_size),
                                       lambdas=cfg.lambda_j, lam=cfg.lam, w_range=cfg.w_range,
                                       bypass=cfg.bypass_train, N=cfg.lut_size)
            fwd = F.mse_loss(E(x, w), t)
            rev = F.mse_loss(E(t, -w), x)
            loss = cfg.pg_weight * pg + fwd + rev
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]
            _log(history, "stage2_pg", pg.item())
            _log(history, "stage2_fwd_mse", fwd.item())
            _log(history, "stage2_rev_mse", rev.item())
        _log(history, "stage2_epoch_loss", tot / n)
        if log:
            log(f"[stage 2] epoch {epoch + 1}/{cfg.stage2_epochs}  loss={tot / n:.5f}")
    return w_table.detach().cpu()


def train_stage3(model: PGIANILUT, dataset: Dataset, cfg: TrainConfig, *, device, history=None,
                 log: Callable | None = None) -> dict:
    """Stage 3 (Eq. 10): fit the predictor F with E frozen."""
    history = {} if history is None else history
    if model.predictor is None:
        raise ValueError("model has no predictor to train")
    was_trainable = [p.requires_grad for p in model.filters.parameters()]
    model.filters.requires_grad_(False)
    model.filters.eval()
    model.predictor.train()
    try:
        opt = torch.optim.Adam(model.predictor.parameters(), lr=cfg.lr_predictor)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(cfg.seed + 1))
        for epoch in range(cfg.stage3_epochs):
            tot = 0.0
            for _, x, t in loader:
                x, t = x.to(device), t.to(device)
                y = model.filters.enhance(x, model.predict_weights(x), bypass=cfg.bypass_train,
                                          N=cfg.lut_size)
                loss = F.mse_loss(y, t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * x.shape[0]
            _log(history, "stage3_mse", tot / len(dataset))
            if log:
                log(f"[stage 3] epoch {epoch + 1}/{cfg.stage3_epochs}  mse={tot / len(dataset):.5f}")
        model.predictor.eval()
        return history
    finally:
        for p, trainable in zip(model.filters.parameters(), was_trainable):
            p.requires_grad_(trainable)


def train_pg_ia_nilut(data, scorer=None, cfg: TrainConfig | None = None, *, device=None,
                      log: Callable | None = print) -> TrainResult:
    """Run all three stages. ``data``: an `ImagePairDataset`, or pairs for one (e.g. a `PairFolder`).
    ``scorer``: callable ``(B, H, W, 3) -> (B, J)``; default builds a `ClipScorer` for ``cfg.dataset``."""
    cfg = cfg or TrainConfig()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    ds = data if isinstance(data, Dataset) else ImagePairDataset(data, cfg.size)
    if len(ds) == 0:
        raise ValueError("empty dataset: no (input, target) pairs to train on")
    scorer = scorer if scorer is not None else ClipScorer(dataset=cfg.dataset, device=device)
    model = PGIANILUT(IANILUT(cfg.n_params, cfg.hidden, cfg.n_layers),
                      make_predictor(cfg.predictor, cfg.n_params)).to(device)
    history: dict[str, list[float]] = {}
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    train_stage1(model, scorer, loader, cfg, device=device, history=history, log=log)
    w_table = train_stage2(model, scorer, ds, cfg, device=device, history=history, log=log)
    train_stage3(model, ds, cfg, device=device, history=history, log=log)
    return TrainResult(model.eval().cpu(), w_table, history)


def build_method(*, checkpoint=None, data=None, scorer=None, device=None,
                 config: TrainConfig | None = None, **cfg_overrides):
    """Registry entry point (``methods.build('pg_ia_nilut', ...)``) -> a lutlab `transform`.

    * ``checkpoint=path``  load a trained model (image-adaptive if it has a predictor)
    * ``data=pairs``       train all three stages, then return the adaptive transform
    * neither              an untrained filter with ``w = 0`` (exactly the identity)
    """
    if checkpoint is not None:
        return PGIANILUT.load(checkpoint).transform()
    if data is not None:
        cfg = config or TrainConfig(**cfg_overrides)
        return train_pg_ia_nilut(data, scorer, cfg, device=device).model.transform()
    cfg = config or TrainConfig(**cfg_overrides)
    model = PGIANILUT(IANILUT(cfg.n_params, cfg.hidden, cfg.n_layers)).eval()
    return model.transform(w=torch.zeros(cfg.n_params), N=cfg.lut_size)
