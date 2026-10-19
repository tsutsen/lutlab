"""``python -m lutlab <command>``

    info                    versions, backend availability, registered methods
    selftest                quick sanity checks (backend agreement, round-trips)
    convert SRC DST         convert between .cube / .json / .npy
"""
from __future__ import annotations

import sys


def _info() -> int:
    import numpy as np
    import lutlab
    print(f"lutlab {lutlab.__version__}  (numpy {np.__version__})")
    print("\nbackends:")
    for n, i in lutlab.kernels.backends().items():
        flags = ", ".join(f for f, on in (("reference", i.reference), ("differentiable",
                          i.differentiable), ("opt-in", not i.auto)) if on)
        state = "available" if i.available else f"unavailable: {i.reason}"
        print(f"  {n:<6} {state}" + (f"  [{flags}]" if flags else ""))
    print("\nmethods:")
    for n, why in lutlab.methods.available().items():
        print(f"  {n:<14} {'available' if why is None else 'unavailable (' + why + ')'}")
    return 0


def _convert(args: list[str]) -> int:
    if len(args) != 2:
        print("usage: python -m lutlab convert SRC DST")
        return 2
    import lutlab
    lutlab.LUT.load(args[0]).save(args[1])
    print(f"wrote {args[1]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    if cmd == "info":
        return _info()
    if cmd == "selftest":
        from .testing import selftest
        return selftest()
    if cmd == "convert":
        return _convert(argv[1:])
    print(__doc__.strip())
    return 0 if cmd in ("help", "-h", "--help") else 2


if __name__ == "__main__":
    raise SystemExit(main())
