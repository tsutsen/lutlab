import numpy as np
import pytest

import lutlab
from lutlab import LUT


def test_identity_applies_to_itself_all_schemes(rng):
    lut = LUT.identity(9)
    x = rng.random((200, 3)).astype(np.float32)
    for scheme in ("nearest", "multilinear", "tetrahedral"):
        y = lut(x, interp=scheme, backend="numpy")
        tol = 0.5 / 8 + 1e-6 if scheme == "nearest" else 1e-6
        assert np.abs(y - x).max() <= tol


@pytest.mark.parametrize("n_in", [1, 3, 4])
def test_identity_shapes(n_in):
    lut = LUT.identity(5, n_in)
    assert lut.shape == (5,) * n_in + (3,) and lut.n_in == n_in
    assert np.abs(lut.volume - LUT.identity(5, n_in).volume).max() == 0.0


def test_identity_2d_has_no_canonical_form():
    with pytest.raises(ValueError):
        LUT.identity(5, 2)


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
    assert lut.volume[0, 0, 0, 0] == 0.0
    with pytest.raises(ValueError):
        lut.volume[0, 0, 0, 0] = 1.0


def test_D_requires_equal_sizes():
    assert LUT.identity(5).grid_size == 5
    with pytest.raises(ValueError):
        LUT(np.zeros((3, 4, 5, 3))).grid_size


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


def test_resample_and_replace():
    lut = lutlab.testing.random_lut(9, 3, seed=4)
    assert lut.resample(9) is lut and lut.resample() is lut
    b = lut.resample(17)
    assert b.sizes == (17,) * 3
    assert lut.replace(name="x").name == "x" and lut.replace(name="x").allclose(lut)


def test_cost():
    lut = LUT.identity(17)
    c = lut.cost()
    assert c.params == 17 ** 3 * 3 and c.nbytes == c.params * 4 and c.reads_per_px == 8


def test_transform_contract():
    # ColorTransform is a runtime-checkable Protocol: both transforms satisfy it.
    assert isinstance(LUT.identity(5), lutlab.ColorTransform)
    assert isinstance(lutlab.LUTMixture([LUT.identity(5)]), lutlab.ColorTransform)
    # The contract: callable with cost/name/n_in; to_lut freezes it to a dense LUT.
    for t in (LUT.identity(5), lutlab.LUTMixture([LUT.identity(5)])):
        for attr in ("__call__", "cost", "name", "n_in", "to_lut"):
            assert hasattr(t, attr)
    i5 = LUT.identity(5)
    assert i5.to_lut(5) is i5


def test_bit_depth_quantizes_and_converts():
    import warnings
    base = LUT.identity(9).volume.copy()
    base[4, 2, 1, 0] = 0.417          # off-grid value to prove quantisation happens

    lut = LUT(base, bit_depth=8)
    assert lut.bit_depth == 8 and LUT(base).bit_depth is None
    assert "8-bit" in repr(lut) and "float" in repr(LUT(base))
    scaled = np.round(lut.volume.ravel() * 255)          # lies exactly on the 8-bit grid
    assert np.allclose(lut.volume.ravel(), scaled / 255)

    # down-conversion 16 -> 8: lossy round, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        l8 = LUT(base, bit_depth=16).to_bit_depth(8)
    assert l8.bit_depth == 8

    # up-conversion 8 -> 16: adds no precision -> warns; round-trips back
    with pytest.warns(UserWarning, match="no precision"):
        up = LUT(base, bit_depth=8).to_bit_depth(16)
    assert up.bit_depth == 16
    assert np.allclose(up.to_bit_depth(8).volume, LUT(base, bit_depth=8).volume)

    # float -> 8: a real quantisation, must NOT warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LUT(base).to_bit_depth(8)

    # replace preserves bit_depth
    assert LUT(base, bit_depth=8).replace().bit_depth == 8


def test_compact_storage_memory():
    """A quantised LUT stores int (compact) and reports the smaller size."""
    volume = np.random.default_rng(0).random((16, 16, 16, 3), dtype=np.float32)
    lf, l8, l16 = LUT(volume), LUT(volume, bit_depth=8), LUT(volume, bit_depth=16)
    assert l8._volume.dtype == np.uint8 and l16._volume.dtype == np.uint16 and lf._volume.dtype == np.float32
    # the memory win: 4x / 2x smaller, surfaced via .nbytes and cost()
    assert l8.nbytes == lf.nbytes // 4 and l16.nbytes == lf.nbytes // 2
    assert l8.cost().nbytes == l8.nbytes and l16.cost().nbytes == l16.nbytes
    # public .volume is still float32 in [0,1]
    for l in (l8, l16):
        assert l.volume.dtype == np.float32
        assert float(l.volume.min()) >= 0.0 and float(l.volume.max()) <= 1.0


def test_bit_depth_32_is_the_cap_and_uses_float32_storage():
    volume = np.random.default_rng(0).random((5, 5, 5, 3), dtype=np.float32)
    l32 = LUT(volume, bit_depth=32)
    assert l32.bit_depth == 32 and l32.raw.dtype == np.float32 and l32.raw_top is None
    assert l32.volume.dtype == np.float32
    assert np.abs(l32.volume - volume).max() < 1e-6
    with pytest.raises(ValueError, match="between 1 and 32"):
        LUT(volume, bit_depth=33)
    with pytest.raises(ValueError, match="between 1 and 32"):
        l32.to_bit_depth(33)


def test_1bit_volume_is_still_float32():
    l1 = LUT(np.random.default_rng(1).random((5, 5, 5, 3), dtype=np.float32), bit_depth=1)
    assert l1.raw.dtype == np.uint8 and l1.volume.dtype == np.float32


def test_random_lut_supports_two_inputs():
    lut = lutlab.testing.random_lut(5, 2, seed=0)
    assert lut.shape == (5, 5, 3) and lut.n_in == 2


def test_allclose_checks_axes_and_domain():
    volume = np.random.default_rng(2).random((5, 5, 5, 3), dtype=np.float32)
    assert LUT(volume).allclose(LUT(volume))
    rect = LUT(volume, axes=[np.linspace(0, 1, 5) ** 2] * 3)
    assert not LUT(volume).allclose(rect)
    assert not LUT(volume).allclose(LUT(volume, domain=(0, 2)))


def test_to_float_source_depth():
    from lutlab._arrays import to_float
    assert to_float(np.array([4095], np.uint16), source_depth=12)[0] == 1.0
    assert to_float(np.array([255], np.uint8), source_depth=8)[0] == 1.0
    with pytest.raises(ValueError, match="source_depth"):
        to_float(np.array([1], np.uint16), source_depth=17)
    with pytest.raises(ValueError, match="integer"):
        to_float(np.ones(1, np.float32), source_depth=12)
