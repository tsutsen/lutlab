import numpy as np
import pytest

import lutlab
from lutlab import LUT, identity


def test_identity_applies_to_itself_all_schemes(rng):
    lut = identity(9)
    x = rng.random((200, 3)).astype(np.float32)
    for scheme in ("nearest", "multilinear", "tetrahedral"):
        y = lut(x, interp=scheme, backend="numpy")
        tol = 0.5 / 8 + 1e-6 if scheme == "nearest" else 1e-6
        assert np.abs(y - x).max() <= tol


@pytest.mark.parametrize("n_in", [1, 3, 4])
def test_identity_shapes(n_in):
    lut = identity(5, n_in)
    assert lut.shape == (5,) * n_in + (3,) and lut.n_in == n_in
    assert lutlab.analysis.max_delta(lut) == 0.0


def test_identity_2d_has_no_canonical_form():
    with pytest.raises(ValueError):
        identity(5, 2)


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.zeros((4, 4, 4, 2)), np.zeros(3)])
def test_bad_layout_rejected(bad):
    with pytest.raises(ValueError):
        LUT(bad)


def test_min_size_and_nonfinite_rejected():
    with pytest.raises(ValueError, match=">= 2"):
        LUT(np.zeros((1, 4, 4, 3)))
    v = np.zeros((3, 3, 3, 3))
    v[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        LUT(v)


@pytest.mark.parametrize("axes", [
    [[0, 0.5, 0.4, 1]] * 3,           # not increasing
    [[0.1, 0.5, 0.9, 1]] * 3,         # does not start at 0
    [[0, 0.5, 1]] * 3,                # wrong length
])
def test_bad_axes_rejected(axes):
    with pytest.raises(ValueError):
        LUT(np.zeros((4, 4, 4, 3)), axes=axes)


def test_uniform_axes_collapse_to_none():
    lut = LUT(np.zeros((5, 5, 5, 3)), axes=[np.linspace(0, 1, 5)] * 3)
    assert lut.is_uniform and lut.axes is None


def test_immutable_and_copying():
    src = np.zeros((3, 3, 3, 3), np.float32)
    lut = LUT(src)
    src[0, 0, 0, 0] = 9.0
    assert lut.vol[0, 0, 0, 0] == 0.0
    with pytest.raises(ValueError):
        lut.vol[0, 0, 0, 0] = 1.0


def test_D_requires_equal_sizes():
    assert identity(5).D == 5
    with pytest.raises(ValueError):
        LUT(np.zeros((3, 4, 5, 3))).D


def test_domain_rescales_input(rng):
    base = lutlab.testing.random_lut(9, 3, seed=1)
    wide = base.replace(domain=(0.0, 2.0))
    x = rng.random((50, 3)).astype(np.float32) * 2
    assert np.allclose(wide(x, backend="numpy"), base(x / 2, backend="numpy"), atol=1e-6)


def test_4d_lut_needs_context(rng):
    lut = lutlab.testing.random_lut(5, 4, seed=2)
    x = rng.random((20, 3)).astype(np.float32)
    ctx = rng.random((20, 1)).astype(np.float32)
    full = np.concatenate([x, ctx], axis=-1)
    assert np.allclose(lut(x, context=ctx), lut(full), atol=1e-6)
    assert np.allclose(lut(x, context=np.float32(0.3) * np.ones(1, np.float32)),
                       lut(np.concatenate([x, np.full((20, 1), 0.3, np.float32)], -1)),
                       atol=1e-6)
    with pytest.raises(ValueError, match="context"):
        lut(x)


def test_bake_and_replace():
    lut = lutlab.testing.random_lut(9, 3, seed=4)
    assert lut.bake(9) is lut and lut.bake() is lut
    b = lut.bake(17)
    assert b.sizes == (17,) * 3
    assert lut.replace(name="x").name == "x" and lut.replace(name="x").allclose(lut)


def test_cost():
    lut = identity(17)
    c = lut.cost()
    assert c.params == 17 ** 3 * 3 and c.nbytes == c.params * 4 and c.reads_per_px == 8
    assert lut.cost("tetrahedral").reads_per_px == 4 and lut.cost("nearest").reads_per_px == 1


def test_is_a_colortransform():
    assert isinstance(identity(5), lutlab.ColorTransform)
    assert isinstance(lutlab.LUTMixture([identity(5)]), lutlab.ColorTransform)
