import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_import_is_light():
    code = ("import sys, lutlab;"
            "bad = {'numba','torch','skimage','scipy','imageio'} & set(sys.modules);"
            "print(sorted(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT, check=True).stdout.strip()
    assert out == "[]", f"`import lutlab` pulled in heavy modules: {out}"


def test_public_api_resolves():
    import lutlab
    for name in lutlab.__all__:
        assert hasattr(lutlab, name), name


def test_selftest_and_info_cli(capsys):
    from lutlab.__main__ import main
    assert main(["selftest"]) == 0
    assert main(["info"]) == 0
    assert "numpy" in capsys.readouterr().out
    assert main(["bogus"]) == 2


def test_readme_quickstart_runs(tmp_path, monkeypatch):
    pytest.importorskip("skimage")
    block = re.search(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S).group(1)
    monkeypatch.chdir(tmp_path)
    exec(compile(block, "README.md", "exec"), {})
