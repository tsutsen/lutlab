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
    _gpu_info()
    return 0


def _gpu_info() -> None:
    """Print what torch can see, when torch is installed."""
    try:
        import torch
    except Exception as exc:                      # noqa: BLE001
        print(f"\ngpu: torch not importable ({type(exc).__name__}: {exc})")
        return
    build = ("HIP/ROCm " + torch.version.hip) if torch.version.hip else (
            ("CUDA " + torch.version.cuda) if torch.version.cuda else "CPU-only")
    print(f"\ngpu: torch {torch.__version__} ({build})")
    if not torch.cuda.is_available():
        print("     no accelerator visible")
        return
    total, free = torch.cuda.mem_get_info(0)
    print(f"     {torch.cuda.get_device_name(0)}  "
          f"({free / 2**30:.2f} / {total / 2**30:.2f} GiB free)")


def _convert(args: list[str]) -> int:
    positional: list[str] = []
    opts: list[tuple[str, str]] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            key, sep, val = a.partition("=")
            if not sep:
                if i + 1 >= len(args):
                    print(f"missing value for {key}")
                    return 2
                val = args[i + 1]
                i += 1
            opts.append((key, val))
        else:
            positional.append(a)
        i += 1
    if len(positional) != 2:
        print("usage: python -m lutlab convert SRC DST [--hald-layout ...]")
        return 2
    layout = "lutlab"
    level = cell = None
    for key, val in opts:
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
        if level is not None:
            read_kw["levels"] = level
        if cell is not None:
            read_kw["cell"] = cell
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
