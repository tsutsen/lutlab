"""`python -m lutlab <command>`

    docs      build the HTML docs (Sphinx + Read the Docs theme) -> ./lutlab_docs/
    selftest  run the quick correctness checks (interp, round-trips, backends)
    bench     run the method-matrix benchmark -> results.json
"""
from __future__ import annotations

import os
import subprocess
import sys


def _docs(args: list[str]) -> int:
    out = args[0] if args else "lutlab_docs"
    try:
        import sphinx  # noqa: F401
    except ImportError:
        print("sphinx is not installed -- "
              "`pip install sphinx sphinx-rtd-theme` and retry.")
        return 1
    # Sphinx source lives in lutlab/docs/ (Read the Docs theme, Godot-like).
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    r = subprocess.run([sys.executable, "-m", "sphinx", "-b", "html",
                        src, out])
    print(f"wrote docs -> {out}/")
    return r.returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    if cmd == "docs":
        return _docs(argv[1:])
    if cmd == "selftest":
        from .selftest import run
        return run(argv[1:])
    if cmd == "bench":
        from .bench import main as bmain
        bmain()
        return 0
    print(__doc__.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
