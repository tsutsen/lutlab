import numpy as np
import pytest

import lutlab
from lutlab import LUT, testing
from lutlab._arrays import to_float
from lutlab.kernels import BackendError, interpolate
from lutlab.kernels import _registry as reg

scipy_interp = pytest.importorskip("scipy.interpolate")


def _axes(sizes, axes):
    return axes if axes is not None else [np.linspace(0, 1, s) for s in sizes]


# ---- independent oracle: scipy ------------------------------------------------ #
@pytest.mark.parametrize("n_in,size", [(1, 9), (3, 9), (4, 5)])
@pytest.mark.parametrize("rectilinear", [False, True])
def test_multilinear_matches_scipy(n_in, size, rectilinear, rng):
    sizes = (size,) * n_in
    axes = [np.linspace(0, 1, size) ** 1.4 for _ in range(n_in)] if rectilinear else None
    if n_in == 1:                       # random_lut has no 1D identity base path issue
        volume = rng.random((size, 3)).astype(np.float32)
    else:
        volume = testing.random_lut(sizes, n_in, seed=5, axes=axes).volume
    x = rng.random((300, n_in)).astype(np.float32)
    ref = scipy_interp.RegularGridInterpolator(_axes(sizes, axes), volume, method="linear")(x)
    got = interpolate(volume, x, axes=None if axes is None else [a.astype(np.float32) for a in axes],
                      scheme="multilinear", backend="numpy")
    assert np.abs(got - ref).max() < 1e-5


def test_nearest_matches_scipy_away_from_ties(rng):
    sizes = (7, 7, 7)
    volume = rng.random(sizes + (3,)).astype(np.float32)
    x = rng.random((400, 3)).astype(np.float32)
    ref = scipy_interp.RegularGridInterpolator(_axes(sizes, None), volume, method="nearest")(x)
    got = interpolate(volume, x, scheme="nearest", backend="numpy")
    assert np.abs(got - ref).max() == 0.0


def test_nearest_rounds_not_floors():
    lut = LUT.identity(3)                                   # grid values 0, .5, 1
    x = np.array([[0.24] * 3, [0.26] * 3, [0.74] * 3, [0.76] * 3, [0.99] * 3], np.float32)
    got = lut(x, interp="nearest", backend="numpy")[:, 0]
    assert np.allclose(got, [0.0, 0.5, 0.5, 1.0, 1.0])


# ---- tetrahedral invariants --------------------------------------------------- #
@pytest.mark.parametrize("rectilinear", [False, True])
def test_tetrahedral_exact_for_linear_functions(rectilinear, rng):
    D = 12
    pos = np.linspace(0, 1, D) ** (1.5 if rectilinear else 1.0)
    A, b = rng.standard_normal((3, 3)), rng.standard_normal(3)
    grid = np.stack(np.meshgrid(pos, pos, pos, indexing="ij"), -1)
    volume = (grid @ A + b).astype(np.float32)
    x = rng.random((2000, 3)).astype(np.float32)
    axes = [pos.astype(np.float32)] * 3 if rectilinear else None
    got = interpolate(volume, x, axes=axes, scheme="tetrahedral", backend="numpy")
    assert np.abs(got - (x @ A + b)).max() < 2e-5


def test_tetrahedral_hits_vertices_and_is_a_lerp_on_the_diagonal(rng):
    lut = testing.random_lut(6, 3, seed=7)
    g = np.stack(np.meshgrid(*[np.linspace(0, 1, 6)] * 3, indexing="ij"), -1).reshape(-1, 3)
    assert np.abs(lut(g.astype(np.float32), interp="tetrahedral", backend="numpy")
                  - lut.volume.reshape(-1, 3)).max() < 1e-6
    # on (t, t, t) the interpolant is the straight line between the cell's diagonal corners
    t = (rng.random(100) * 0.999).astype(np.float32)
    xf = t * 5
    i = np.floor(xf).astype(int)
    f = (xf - i)[:, None]
    expect = (1 - f) * lut.volume[i, i, i] + f * lut.volume[i + 1, i + 1, i + 1]
    got = lut(np.repeat(t[:, None], 3, 1), interp="tetrahedral", backend="numpy")
    assert np.abs(got - expect).max() < 1e-6


# ---- edges, validation -------------------------------------------------------- #
def test_edges_and_clamping():
    lut = testing.random_lut(5, 3, seed=8)
    corner = lambda a: lut(np.array([a], np.float32), backend="numpy")[0]
    assert np.allclose(corner([0, 0, 0]), lut.volume[0, 0, 0])
    assert np.allclose(corner([1, 1, 1]), lut.volume[-1, -1, -1])
    assert np.allclose(corner([2.5, 9, 1e9]), corner([1, 1, 1]))
    assert np.allclose(corner([-3, -0.1, -1e9]), corner([0, 0, 0]))


def test_nonfinite_rejected_and_ints_rejected():
    lut = LUT.identity(5)
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="NaN or inf"):
            lut(np.array([[0.1, bad, 0.2]], np.float32))
    with pytest.raises(TypeError, match="floating-point array"):
        lut(np.zeros((4, 3), np.uint8))
    assert to_float(np.full((2, 3), 255, np.uint8)).max() == 1.0


def test_shapes_dtypes_and_empty(rng):
    lut = testing.random_lut(7, 3, seed=9)
    assert lut(np.zeros((2, 3, 4, 3), np.float32)).shape == (2, 3, 4, 3)
    assert lut(np.zeros(3, np.float32)).shape == (3,)
    assert lut(np.zeros((0, 3), np.float32)).shape == (0, 3)
    y = lut(rng.random((5, 3)))                          # float64 in
    assert y.dtype == np.float32
    with pytest.raises(ValueError, match="takes 3"):
        lut(np.zeros((4, 5), np.float32))


def test_user_errors_are_actionable():
    lut = LUT.identity(5)
    x = np.zeros((4, 3), np.float32)
    with pytest.raises(BackendError, match="unknown backend 'typo'.*numpy"):
        lut(x, backend="typo")
    with pytest.raises(ValueError, match="unknown interp"):
        lut(x, interp="cubic")
    with pytest.raises(ValueError, match="3-input"):
        testing.random_lut(5, 4)(np.zeros((2, 4), np.float32), interp="tetrahedral")
    with pytest.raises(BackendError, match="CPU only"):
        lut(x, backend="numpy", device="cuda")
    rect = testing.random_lut((5, 5, 5), 3, axes=[np.linspace(0, 1, 5) ** 2] * 3)
    want = "uniform" if lutlab.kernels.backends()["numba"].available else "not installed"
    with pytest.raises(BackendError, match=want):
        rect(x, backend="numba")


# ---- numba vs oracle ---------------------------------------------------------- #
@pytest.mark.parametrize("n_in,size", [(3, 17), (4, 6), (1, 12)])
@pytest.mark.parametrize("scheme", ["nearest", "multilinear", "tetrahedral"])
def test_numba_matches_numpy(n_in, size, scheme, rng):
    pytest.importorskip("numba")
    if scheme == "tetrahedral" and n_in != 3:
        pytest.skip("tetrahedral is 3D only")
    volume = rng.random((size,) * n_in + (3,)).astype(np.float32)
    x = rng.random((5000, n_in)).astype(np.float32)
    a = interpolate(volume, x, scheme=scheme, backend="numpy")
    b = interpolate(volume, x, scheme=scheme, backend="numba")
    assert np.abs(a - b).max() < 1e-5


def test_check_backends_all_agree(lut3):
    for key, v in testing.check_backends(lut3).items():
        assert isinstance(v, str) or v < 1e-5, key


# ---- registry ----------------------------------------------------------------- #
def test_backends_listing_and_auto_choice():
    info = lutlab.kernels.backends()
    assert info["numpy"].available and info["numpy"].reference
    assert not info["fused"].auto
    req = reg.Request("multilinear", 3, (9, 9, 9), True, "cpu")
    chosen = reg.select(req).name
    assert chosen == ("numba" if info["numba"].available else "numpy")


def test_custom_backend_can_be_registered(monkeypatch):
    calls = []

    def run(volume, x, axes, scheme):
        calls.append(scheme)
        return np.zeros((x.shape[0], 3), np.float32)

    monkeypatch.setitem(reg._REGISTRY, "zeros", reg.Backend("zeros", run, priority=999))
    out = LUT.identity(5)(np.full((4, 3), 0.5, np.float32))
    assert calls == ["multilinear"] and out.max() == 0.0
    assert LUT.identity(5)(np.full((4, 3), 0.5, np.float32), backend="numpy").max() > 0.4


def test_no_backend_error_lists_reasons():
    req = reg.Request("multilinear", 3, (5, 5, 5), True, "cuda")
    if not lutlab.kernels.backends()["torch"].available:
        with pytest.raises(BackendError, match="numpy: .*CPU only"):
            reg.select(req)


# ---- torch (skipped here if torch is not installed) --------------------------- #
@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("n_in,size", [(3, 9), (4, 5)])
@pytest.mark.parametrize("rectilinear", [False, True])
@pytest.mark.parametrize("scheme", ["nearest", "multilinear", "tetrahedral"])
def test_torch_matches_numpy(torch, n_in, size, rectilinear, scheme, rng):
    if scheme == "tetrahedral" and n_in != 3:
        pytest.skip("3D only")
    axes = [np.linspace(0, 1, size) ** 1.5] * n_in if rectilinear else None
    lut = testing.random_lut(size, n_in, seed=11, axes=axes)
    x = rng.random((3000, n_in)).astype(np.float32)
    a = lut(x, interp=scheme, backend="numpy")
    b = lut(x, interp=scheme, backend="torch")
    assert isinstance(b, np.ndarray) and np.abs(a - b).max() < 1e-5


def test_torch_in_torch_out(torch, rng):
    lut = testing.random_lut(9, 3, seed=1)
    x = torch.as_tensor(rng.random((4, 5, 3)).astype(np.float32))
    y = lut(x)
    assert torch.is_tensor(y) and y.shape == (4, 5, 3)
    assert np.abs(y.numpy() - lut(x.numpy(), backend="numpy")).max() < 1e-5


def test_torch_gradients(torch, rng):
    D = 4
    volume = torch.tensor(rng.random((D, D, D, 3)), dtype=torch.float64, requires_grad=True)
    x = torch.tensor(rng.random((6, 3)) * 0.9 + 0.05, dtype=torch.float64, requires_grad=True)
    f = lambda v, p: interpolate(v, p, scheme="multilinear", backend="torch")
    assert torch.autograd.gradcheck(f, (volume, x), eps=1e-6, atol=1e-5)
    g = lambda v: interpolate(v, x.detach(), scheme="tetrahedral", backend="torch")
    assert torch.autograd.gradcheck(g, (volume,), eps=1e-6, atol=1e-5)
    with pytest.raises(BackendError, match="differentiable"):
        interpolate(volume, x, backend="numba")
    out = interpolate(volume, x)                            # auto -> torch when grads needed
    out.sum().backward()
    assert volume.grad is not None and x.grad is not None
