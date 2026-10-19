"""Methods: things that *produce* a `transform` from data.

A method is a function ``build(**cfg) -> transform`` registered under a
name. Per-image and trained global methods (NILUT, 4D LUT, AdaInt, ...) all fit
this shape, so the
evaluation harness and the conformance tests treat them uniformly::

    @methods.register("my_method", requires=("torch",))
    def build(*, image=None, target=None, **cfg) -> transform: ...

    t = methods.build("my_method", ...)
    methods.available()        # {name: None if usable else reason}
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .._deps import have

__all__ = ["MethodSpec", "register", "get", "build", "available", "names"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    build: Callable
    requires: tuple[str, ...] = ()
    doc: str = ""

    def unavailable_reason(self) -> str | None:
        missing = [m for m in self.requires if not have(m)]
        return f"missing: {', '.join(missing)}" if missing else None


_METHODS: dict[str, MethodSpec] = {}


def register(name: str, *, requires: tuple[str, ...] = ()):
    """Decorator: register ``build(**cfg) -> transform`` under `name`."""
    def deco(fn: Callable) -> Callable:
        _METHODS[name] = MethodSpec(name, fn, tuple(requires), (fn.__doc__ or "").strip())
        return fn
    return deco


def get(name: str) -> MethodSpec:
    try:
        return _METHODS[name]
    except KeyError:
        raise ValueError(f"unknown method {name!r}; registered: "
                         f"{', '.join(sorted(_METHODS)) or '(none)'}") from None


def names() -> list[str]:
    return sorted(_METHODS)


def available() -> dict[str, str | None]:
    return {n: _METHODS[n].unavailable_reason() for n in names()}


def build(name: str, **cfg):
    spec = get(name)
    why = spec.unavailable_reason()
    if why:
        raise ImportError(f"method {name!r} is unavailable ({why})")
    return spec.build(**cfg)


from . import builtin, pg_ia_nilut   # noqa: E402,F401  (register methods)
