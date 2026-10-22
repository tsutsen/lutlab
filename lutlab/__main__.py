"""``python -m lutlab <command>``

    info                    versions, backend availability, registered methods
    selftest                quick sanity checks (backend agreement, round-trips)
    convert SRC DST         convert between .cube / .json / .npy / Hald PNG
                            --hald-layout {lutlab,canonical}  (Hald PNG only)
                            --hald-level N                    (canonical layout)
                            --hald-cell C                     (lutlab variant)
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
    positional = [a for a in args if not a.startswith("--")]
    opts = [a for a in args if a.startswith("--")]
    if len(positional) != 2:
        print("usage: python -m lutlab convert SRC DST [--hald-layout ...]")
        return 2
    layout = "lutlab"
    level = cell = None
    for o in opts:
        key, _, val = o.partition("=")
        if key == "--hald-layout":
            layout = val
        elif key == "--hald-level":
            level = int(val)
        elif key == "--hald-cell":
            cell = int(val)
    import lutlab
    from pathlib import Path
    read_kw, write_kw = {}, {}
    if Path(positional[0]).suffix.lower() == ".png":
        read_kw["layout"] = layout
    if Path(positional[1]).suffix.lower() == ".png":
        if layout == "canonical":
            write_kw = dict(layout="canonical", level=level)
        else:
            write_kw = dict(cell=cell if cell is not None else 1)
    try:
        lut = lutlab.LUT.load(positional[0], **read_kw)
    except (ValueError, TypeError) as e:
        print(f"cannot read {positional[0]}: {e}")
        return 1
    lut.save(positional[1], **write_kw)
    print(f"wrote {positional[1]}")
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
