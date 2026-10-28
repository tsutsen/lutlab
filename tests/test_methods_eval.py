import json

import numpy as np
import pytest

from lutlab import LUT, LUTMixture, methods, metrics, testing

pytest.importorskip("skimage")


# ---- methods registry ---------------------------------------------------------- #
def test_registry_roundtrip_and_errors():
    assert "identity" in methods.names()
    assert methods.build("identity", grid_size=9).sizes == (9, 9, 9)
    with pytest.raises(ValueError, match="unknown method.*identity"):
        methods.get("nope")

    @methods.register("needs_ghost", requires=("definitely_not_installed_pkg",))
    def build(**_):
        return LUT.identity(3)

    try:
        assert "missing" in methods.available()["needs_ghost"]
        with pytest.raises(ImportError, match="unavailable"):
            methods.build("needs_ghost")
    finally:
        methods._METHODS.pop("needs_ghost")


# ---- LUTMixture ---------------------------------------------------------------- #
def test_mixture_blends_and_bakes(rng):
    a, b = testing.random_lut(9, 3, seed=1), testing.random_lut(9, 3, seed=2)
    mix = LUTMixture([a, b])
    x = rng.random((50, 3)).astype(np.float32)
    w = np.array([0.3, 0.7], np.float32)
    expect = 0.3 * a(x) + 0.7 * b(x)
    assert np.abs(mix(x, context=w) - expect).max() < 1e-6
    per_px = np.stack([np.full(50, 0.3), np.full(50, 0.7)], -1).astype(np.float32)
    assert np.abs(mix(x, context=per_px) - expect).max() < 1e-6
    assert np.abs(mix.to_lut(9, weights=w)(x) - expect).max() < 1e-5         # trilinear is linear in the grid
    assert mix.cost().meta["n_luts"] == 2
    with pytest.raises(ValueError, match="needs weights"):
        mix(x)
    with pytest.raises(ValueError, match="3 channels for 2 LUTs"):
        mix(x, context=np.ones((50, 3), np.float32))
    with pytest.raises(ValueError, match="same grid"):
        LUTMixture([a, testing.random_lut(7, 3)]).to_lut(9, weights=w)


# ---- conformance (and that conformance is not vacuous) ------------------------- #
_RECT = [np.linspace(0, 1, s) ** 1.5 for s in (5, 7, 9)]
_CASES = {
    "identity": (lambda: LUT.identity(9), {}),
    "random": (lambda: testing.random_lut(11, 3, seed=1), {"to_lut_tol": 1e-5}),
    # rectilinear -> uniform resampling has an inherent gap; 33 keeps it small
    "rectilinear": (lambda: testing.random_lut((5, 7, 9), 3, seed=1, axes=_RECT),
                    {"to_lut_grid_size": 33, "to_lut_tol": 5e-2}),
    "4d": (lambda: testing.random_lut(5, 4, seed=1), {}),
    "mixture": (lambda: LUTMixture([testing.random_lut(9, 3, seed=1), testing.random_lut(9, 3, seed=2)],
                                   weights=np.array([0.5, 0.5], np.float32)),
                {"to_lut_grid_size": 9, "to_lut_tol": 1e-5}),
    "method-identity": (lambda: methods.build("identity"), {}),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_conformance_passes(case):
    make, kw = _CASES[case]
    assert testing.conformance(make(), **kw) == []


def test_to_lut_gap_is_real_when_grids_do_not_align():
    lut = testing.random_lut(11, 3, seed=1)
    assert testing.conformance(lut, to_lut_grid_size=17, to_lut_tol=1e-5) != []      # 11 -> 17: cells misalign
    assert testing.conformance(lut, to_lut_grid_size=21, to_lut_tol=1e-5) == []      # 21 = 2*10+1: aligned


class _Broken:
    n_in = 3
    name = "broken"

    def __init__(self, mode):
        self.mode, self.rng = mode, np.random.default_rng()

    def __call__(self, x, *, context=None, **kw):
        x = np.asarray(x, np.float32)
        if self.mode == "shape":
            return x[..., :2]
        if self.mode == "nan":
            return np.full(x.shape, np.nan, np.float32)
        if self.mode == "random":
            return x + self.rng.random(x.shape).astype(np.float32) * 0.01
        return x

    def to_lut(self, grid_size=17, **kw):
        return LUT.identity(grid_size) if self.mode != "badresample" else LUT.identity(grid_size).replace(volume=LUT.identity(grid_size).volume * 0.5)

    def cost(self):
        return None


@pytest.mark.parametrize("mode,needle", [("shape", "output shape"), ("nan", "non-finite"),
                                          ("random", "non-deterministic"), ("badresample", "to_lut(")])
def test_conformance_catches_broken_transforms(mode, needle):
    fails = testing.conformance(_Broken(mode))
    assert any(needle in f for f in fails), fails


# ---- evaluation ---------------------------------------------------------------- #
def _pairs(rng, n=3):
    return [(im, im) for im in (rng.random((24, 24, 3)).astype(np.float32) for _ in range(n))]


def test_evaluate_identity_vs_perturbed(rng):
    pairs = _pairs(rng)
    r = metrics.evaluate(LUT.identity(17), pairs, timing={"warmup": 1, "repeat": 2})
    assert r["n"] == 3 and r["psnr_db"] > 60 and r["e00_mean"] < 0.01 and r["ssim"] > 0.999
    assert r["timing"]["fps"] > 0 and r["cost"]["params"] == 17 ** 3 * 3
    worse = metrics.evaluate(testing.random_lut(17, 3, seed=1, strength=0.3), pairs)
    assert worse["psnr_db"] < r["psnr_db"] and worse["e00_mean"] > r["e00_mean"]
    assert metrics.evaluate(LUT.identity(5), pairs, max_pairs=1)["n"] == 1
    with pytest.raises(ValueError, match="unknown metric"):
        metrics.evaluate(LUT.identity(5), pairs, metrics=("fid",))
    with pytest.raises(ValueError, match="no pairs"):
        metrics.evaluate(LUT.identity(5), [])


def test_evaluate_with_context(rng):
    lut4 = testing.random_lut(5, 4, seed=1)
    pairs = [(rng.random((8, 8, 3)).astype(np.float32), rng.random((8, 8, 3)).astype(np.float32),
              rng.random((8, 8, 1)).astype(np.float32))]
    assert metrics.evaluate(lut4, pairs)["n"] == 1


def test_run_matrix_records_errors_and_writes_json(rng, tmp_path):
    def boom():
        raise RuntimeError("no gpu")

    out = tmp_path / "r.json"
    rep = metrics.run_matrix(
        {"identity": LUT.identity(9), "factory": lambda: testing.random_lut(9, 3, seed=1), "boom": boom},
        {"synthetic": _pairs(rng)}, out=str(out), meta={"note": "t"}, seed=7)
    assert rep["schema"] == metrics.SCHEMA and len(rep["rows"]) == 3
    rows = {r["method"]: r for r in rep["rows"]}
    assert "psnr_db" in rows["identity"] and "psnr_db" in rows["factory"]
    assert rows["boom"]["error"] == "RuntimeError: no gpu"
    saved = json.loads(out.read_text())
    assert saved["seed"] == 7 and saved["config_hash"] == rep["config_hash"]


def test_pairfolder(tmp_path, rng):
    iio = pytest.importorskip("imageio.v3")
    for d in ("input", "target"):
        (tmp_path / d).mkdir()
    for n in ("a", "b"):
        for d in ("input", "target"):
            iio.imwrite(tmp_path / d / f"{n}.png", (rng.random((8, 8, 3)) * 255).astype(np.uint8))
    iio.imwrite(tmp_path / "input" / "orphan.png", np.zeros((8, 8, 3), np.uint8))
    pf = metrics.PairFolder(tmp_path)
    pairs = list(pf)
    assert len(pf) == 2 and pf.names == ["a", "b"] and len(list(pf)) == 2        # re-iterable
    assert pairs[0][0].dtype == np.float32 and 0.0 <= pairs[0][0].max() <= 1.0
    assert metrics.evaluate(LUT.identity(5), pf)["n"] == 2
    cached = metrics.PairFolder(tmp_path, cache=1)
    first = cached[0]
    assert cached[0] is first                                  # decoded pair reused
    (tmp_path / "e1").mkdir(), (tmp_path / "e1" / "input").mkdir(), (tmp_path / "e1" / "target").mkdir()
    with pytest.raises(ValueError, match="no matching pairs"):
        metrics.PairFolder(tmp_path / "e1")


def test_evaluate_keeps_infinite_psnr_and_json_is_strict(tmp_path):
    rng = np.random.default_rng(0)
    im = rng.random((8, 8, 3)).astype(np.float32)
    r = metrics.evaluate(LUT.identity(5), [(im, im), (im, np.zeros_like(im))],
                         metrics=("psnr",))
    assert r["psnr_db"] == float("inf")
    out = tmp_path / "strict.json"
    rep = metrics.run_matrix({"identity": LUT.identity(5)}, {"same": [(im, im)]}, out=str(out))
    assert rep["rows"][0]["psnr_db"] == float("inf")
    text = out.read_text()
    assert "Infinity" not in text
    assert json.loads(text)["rows"][0]["psnr_db"] is None


def test_run_matrix_builds_a_factory_once():
    calls = []

    def make():
        calls.append(1)
        return LUT.identity(5)

    pairs = _pairs(np.random.default_rng(0))
    rep = metrics.run_matrix({"m": make}, {"a": pairs, "b": pairs})
    assert len(calls) == 1 and len(rep["rows"]) == 2


def test_mixture_to_lut_requires_the_same_domain():
    a = LUT.identity(5).replace(domain=(0, 2))
    b = LUT.identity(5)
    with pytest.raises(ValueError, match="same input domain"):
        LUTMixture([a, b]).to_lut(weights=np.array([0.5, 0.5], np.float32))


def test_lpips_caches_per_network(monkeypatch):
    torch = pytest.importorskip("torch")
    from lutlab.metrics import quality

    seen = []

    class Net:
        def __init__(self, net):
            self.net = net

        def eval(self):
            return self

        def __call__(self, a, b):
            return torch.tensor([[0.0]])

    class FakeLPIPS:
        @staticmethod
        def LPIPS(net, verbose):
            seen.append(net)
            return Net(net)

    monkeypatch.setattr(quality, "require", lambda name: torch if name == "torch" else FakeLPIPS)
    quality._LPIPS_NETS.clear()
    try:
        a = np.zeros((2, 2, 3), np.float32)
        quality.lpips(a, a, net="alex")
        quality.lpips(a, a, net="vgg")
        assert seen == ["alex", "vgg"]
    finally:
        quality._LPIPS_NETS.clear()


def test_mixture_one_pass_matches_the_loop(rng):
    a, b = testing.random_lut(9, 3, seed=1), testing.random_lut(9, 3, seed=2)
    mix = LUTMixture([a, b])
    x = rng.random((50, 3)).astype(np.float32)
    global_w = np.array([0.3, 0.7], np.float32)
    per_px = rng.random((50, 2)).astype(np.float32)

    def loop(w, backend):
        y0, y1 = a(x, backend=backend), b(x, backend=backend)
        return y0 * w[..., 0:1] + y1 * w[..., 1:2]

    from lutlab._deps import have
    backends = ("numpy", "torch") if have("torch") else ("numpy",)
    for backend in backends:
        assert np.abs(mix(x, context=global_w, backend=backend)
                      - loop(global_w, backend)).max() < 1e-6
        assert np.abs(mix(x, context=per_px, backend=backend)
                      - loop(per_px, backend)).max() < 1e-6


def test_mixture_one_pass_torch_gradients():
    torch = pytest.importorskip("torch")
    a, b = testing.random_lut(9, 3, seed=1), testing.random_lut(9, 3, seed=2)
    mix = LUTMixture([a, b])
    x = torch.rand(20, 3, requires_grad=True)
    w = torch.rand(20, 2, requires_grad=True)
    out = mix(x, context=w, backend="torch")
    out.sum().backward()
    assert x.grad is not None and w.grad is not None


def test_mixture_resample_uses_bound_global_weights(rng):
    a, b = testing.random_lut(9, 3, seed=1), testing.random_lut(9, 3, seed=2)
    weights = np.array([0.3, 0.7], np.float32)
    mix = LUTMixture([a, b], weights=weights)
    x = rng.random((20, 3)).astype(np.float32)
    baked = mix.resample(9)
    assert baked.sizes == (9, 9, 9)
    assert np.abs(baked(x) - mix(x)).max() < 1e-5
    with pytest.raises(ValueError, match="global weights"):
        LUTMixture([a, b]).resample(9)


def test_run_matrix_seed_is_real_and_restored():
    def make():
        return testing.random_lut(5, 3, seed=int(np.random.randint(1_000_000)))

    pairs = [(im, im) for im in (np.random.default_rng(0).random((8, 8, 3),
                                                               dtype=np.float32),)]
    np.random.seed(123)
    expected = np.random.rand()
    r1 = metrics.run_matrix({"m": make}, {"t": pairs}, seed=7, metrics=("psnr",))
    r2 = metrics.run_matrix({"m": make}, {"t": pairs}, seed=7, metrics=("psnr",))
    r3 = metrics.run_matrix({"m": make}, {"t": pairs}, seed=8, metrics=("psnr",))
    np.random.seed(123)
    assert np.random.rand() == expected                       # RNG state restored
    assert r1["rows"][0]["psnr_db"] == r2["rows"][0]["psnr_db"]
    assert r1["rows"][0]["psnr_db"] != r3["rows"][0]["psnr_db"]


def test_pairfolder_source_depth(tmp_path):
    iio = pytest.importorskip("imageio.v3")
    for d in ("input", "target"):
        (tmp_path / d).mkdir()
    image = np.full((8, 8), 4095, np.uint16)
    for d in ("input", "target"):
        iio.imwrite(tmp_path / d / "a.png", image)
    x, _ = next(iter(metrics.PairFolder(tmp_path, source_depth=12)))
    assert x.max() == pytest.approx(1.0)
    x_default, _ = next(iter(metrics.PairFolder(tmp_path)))
    assert x_default.max() < 0.1
