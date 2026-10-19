"""Backend registry. One place decides which backend runs a request."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["BackendError", "Request", "Backend", "BackendInfo",
           "register", "infos", "select", "names"]


class BackendError(ValueError):
    """No (suitable) backend for the request."""


@dataclass(frozen=True)
class Request:
    """What the caller wants, as seen by backends."""
    scheme: str                 # canonical: nearest | multilinear | tetrahedral
    n_in: int
    sizes: tuple[int, ...]
    uniform: bool
    device: str                 # "cpu", "cuda", "cuda:1", "mps", ...
    grad: bool = False          # autograd needed through the call
    from_torch: bool = False    # caller passed a torch tensor

    @property
    def kind(self) -> str:
        return self.device.split(":")[0]


def _ok(_req: Request) -> str | None:
    return None


def _avail() -> str | None:
    return None


def _auto_always(_req: Request) -> bool:
    return True


@dataclass(frozen=True)
class Backend:
    """A registered kernel implementation.

    ``run(volume, x, axes, scheme)`` takes ``volume (D0..Dn-1, 3)`` and ``x (N, n)``
    (numpy float32, or torch tensors when ``torch_native``) and returns
    ``(N, 3)``. ``unsupported`` / ``available`` return a reason string, or None.
    """
    name: str
    run: Callable
    unsupported: Callable[[Request], str | None] = _ok
    available: Callable[[], str | None] = _avail
    priority: int = 0
    torch_native: bool = False
    differentiable: bool = False
    reference: bool = False          # the correctness oracle
    auto: bool = True                # may `backend="auto"` pick it?
    auto_when: Callable[[Request], bool] = _auto_always


@dataclass(frozen=True)
class BackendInfo:
    name: str
    available: bool
    reason: str | None
    differentiable: bool
    reference: bool
    auto: bool
    priority: int


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    _REGISTRY[backend.name] = backend
    return backend


def names() -> list[str]:
    return sorted(_REGISTRY, key=lambda n: -_REGISTRY[n].priority)


def infos() -> dict[str, BackendInfo]:
    """Which backends exist and whether each is usable right now."""
    out = {}
    for n in names():
        b = _REGISTRY[n]
        reason = b.available()
        out[n] = BackendInfo(n, reason is None, reason, b.differentiable,
                             b.reference, b.auto, b.priority)
    return out


def _why_not(b: Backend, req: Request) -> str | None:
    why = b.available()
    if why:
        return why
    if req.grad and not b.differentiable:
        return "not differentiable (gradients requested)"
    return b.unsupported(req)


def select(req: Request, name: str = "auto") -> Backend:
    if name != "auto":
        if name not in _REGISTRY:
            raise BackendError(f"unknown backend {name!r}; "
                               f"options: auto, {', '.join(names())}")
        b = _REGISTRY[name]
        why = _why_not(b, req)
        if why:
            raise BackendError(f"backend {name!r} cannot run this request: {why}")
        return b
    reasons = []
    for n in names():
        b = _REGISTRY[n]
        if not b.auto:
            continue
        why = _why_not(b, req)
        if why:
            reasons.append(f"{n}: {why}")
        elif not b.auto_when(req):
            reasons.append(f"{n}: not preferred for this input")
        else:
            return b
    raise BackendError("no backend can run this request -- " + "; ".join(reasons))
