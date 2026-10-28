import numpy as np
import pytest

from lutlab import LUT, visualize


def test_points_are_subsampled_and_bounded():
    lut = LUT.identity(65)
    pts = visualize._points(lut, max_points=60_000)
    assert pts.shape == (35_937, 3)
    assert np.all((pts >= 0) & (pts <= 1))
    with pytest.raises(ValueError):
        visualize._points(LUT.identity(17, n_in=4), max_points=1_000)


def test_volume_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path / "volume.png"
    visualize.volume(LUT.identity(17), out=out, dpi=80)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_grid_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path / "grid.png"
    rows = [[LUT.identity(9), LUT.identity(17)],
            [LUT.identity(9), LUT.identity(17)]]
    visualize.grid(rows, out=out, row_labels=["a", "b"], col_labels=["9", "17"],
                   dpi=60, figsize=(3.0, 2.6))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        visualize.grid(rows, row_labels=["a"])
