import json

import numpy as np
import pytest

import lutlab
from lutlab import LUT, LUTMixture, evaluation, identity, methods, testing

pytest.importorskip("skimage")


# ---- methods registry ---------------------------------------------------------- #
def test_registry_roundtrip_and_errors():
    assert "identity" in methods.names()
    assert methods.build("identity", D=9).sizes == (9, 9, 9)
    with pytest.raises(ValueError, match="unknown method.*identity"):
        methods.get("nope")

    @methods.register("needs_ghost", requires=("definitely_not_installed_pkg",))
    def build(**_):
        return identity(3)

    try:
        assert "missing" in methods.available()["needs_ghost"]
        with pytest.raises(ImportError, match="unavailable"):
            methods.build("needs_ghost")
    finally:
        methods._METHODS.pop("needs_ghost")


def test_robust_hald_methods_fail_with_a_hint_not_a_traceback():
    if lutlab._deps.have("robust_hald"):
        pytest.skip("robust_hald installed")
    assert "robust_hald" in methods.available()["hald_extract"]
    with pytest.raises(ImportError, match="robust_hald"):
        methods.hald_extract.make_hald()
    assert methods.segmented.available_segmenters() == ["kmeans", "sam"]
    with pytest.raises(NotImplementedError, match="register"):
        methods.segmented.segment_image(np.zeros((4, 4, 3)), backend="sam")
    with pytest.raises(ValueError, match="unknown segmentation"):
        methods.segmented.segment_image(np.zeros((4, 4, 3)), backend="zzz")


def test_custom_segmenter_registration():
    @methods.segmented.register_segmenter("halves")
    def halves(img, n_segments=2, **kw):
        m = np.zeros(img.shape[:2] + (2,), np.float32)
        m[:, : img.shape[1] // 2, 0] = 1
        m[:, img.shape[1] // 2:, 1] = 1
        return m, None

    try:
        masks, _ = methods.segmented.segment_image(np.zeros((4, 6, 3)), backend="halves")
        assert masks.shape == (4, 6, 2) and np.allclose(masks.sum(-1), 1)
    finally:
        methods.segmented._SEGMENTERS.pop("halves")


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
    assert np.abs(mix.bake(9, weights=w)(x) - expect).max() < 1e-5      # trilinear is linear in the grid
    assert mix.cost().meta["S"] == 2
    with pytest.raises(ValueError, match="needs weights"):
        mix(x)
    with pytest.raises(ValueError, match="3 channels for 2 LUTs"):
        mix(x, context=np.ones((50, 3), np.float32))
    with pytest.raises(ValueError, match="same grid"):
        LUTMixture([a, testing.random_lut(7, 3)]).bake(9, weights=w)


# ---- conformance (and that conformance is not vacuous) ------------------------- #
_RECT = [np.linspace(0, 1, s) ** 1.5 for s in (5, 7, 9)]
_CASES = {
    "identity": (lambda: identity(9), {}),
    "random": (lambda: testing.random_lut(11, 3, seed=1), {"bake_tol": 1e-5}),
    # rectilinear -> uniform resampling has an inherent gap; 33 keeps it small
    "rectilinear": (lambda: testing.random_lut((5, 7, 9), 3, seed=1, axes=_RECT),
                    {"bake_D": 33, "bake_tol": 5e-2}),
    "4d": (lambda: testing.random_lut(5, 4, seed=1), {}),
    "mixture": (lambda: LUTMixture([testing.random_lut(9, 3, seed=1), testing.random_lut(9, 3, seed=2)],
                                   weights=np.array([0.5, 0.5], np.float32)),
                {"bake_D": 9, "bake_tol": 1e-5}),
    "method-identity": (lambda: methods.build("identity"), {}),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_conformance_passes(case):
    make, kw = _CASES[case]
    assert testing.conformance(make(), **kw) == []


def test_bake_gap_is_real_when_grids_do_not_align():
    lut = testing.random_lut(11, 3, seed=1)
    assert testing.conformance(lut, bake_D=17, bake_tol=1e-5) != []      # 11 -> 17: cells misalign
    assert testing.conformance(lut, bake_D=21, bake_tol=1e-5) == []      # 21 = 2*10+1: aligned


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

    def bake(self, D=17, **kw):
        return identity(D) if self.mode != "badbake" else identity(D).replace(vol=identity(D).vol * 0.5)

    def cost(self):
        return None


@pytest.mark.parametrize("mode,needle", [("shape", "output shape"), ("nan", "non-finite"),
                                          ("random", "non-deterministic"), ("badbake", "bake(")])
def test_conformance_catches_broken_transforms(mode, needle):
    fails = testing.conformance(_Broken(mode))
    assert any(needle in f for f in fails), fails


# ---- evaluation ---------------------------------------------------------------- #
def _pairs(rng, n=3):
    return [(im, im) for im in (rng.random((24, 24, 3)).astype(np.float32) for _ in range(n))]


def test_evaluate_identity_vs_perturbed(rng):
    pairs = _pairs(rng)
    r = evaluation.evaluate(identity(17), pairs, timing={"warmup": 1, "repeat": 2})
    assert r["n"] == 3 and r["psnr_db"] > 60 and r["e00_mean"] < 0.01 and r["ssim"] > 0.999
    assert r["timing"]["fps"] > 0 and r["cost"]["params"] == 17 ** 3 * 3
    worse = evaluation.evaluate(testing.random_lut(17, 3, seed=1, strength=0.3), pairs)
    assert worse["psnr_db"] < r["psnr_db"] and worse["e00_mean"] > r["e00_mean"]
    assert evaluation.evaluate(identity(5), pairs, max_pairs=1)["n"] == 1
    with pytest.raises(ValueError, match="unknown metric"):
        evaluation.evaluate(identity(5), pairs, metrics=("fid",))
    with pytest.raises(ValueError, match="no pairs"):
        evaluation.evaluate(identity(5), [])


def test_evaluate_with_context(rng):
    lut4 = testing.random_lut(5, 4, seed=1)
    pairs = [(rng.random((8, 8, 3)).astype(np.float32), rng.random((8, 8, 3)).astype(np.float32),
              rng.random((8, 8, 1)).astype(np.float32))]
    assert evaluation.evaluate(lut4, pairs)["n"] == 1


def test_run_matrix_records_errors_and_writes_json(rng, tmp_path):
    def boom():
        raise RuntimeError("no gpu")

    out = tmp_path / "r.json"
    rep = evaluation.run_matrix(
        {"identity": identity(9), "factory": lambda: testing.random_lut(9, 3, seed=1), "boom": boom},
        {"synthetic": _pairs(rng)}, out=str(out), meta={"note": "t"}, seed=7)
    assert rep["schema"] == evaluation.SCHEMA and len(rep["rows"]) == 3
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
    pf = evaluation.PairFolder(tmp_path)
    pairs = list(pf)
    assert len(pf) == 2 and pf.names == ["a", "b"] and len(list(pf)) == 2        # re-iterable
    assert pairs[0][0].dtype == np.float32 and 0.0 <= pairs[0][0].max() <= 1.0
    assert evaluation.evaluate(identity(5), pf)["n"] == 2
    (tmp_path / "e1").mkdir(), (tmp_path / "e1" / "input").mkdir(), (tmp_path / "e1" / "target").mkdir()
    with pytest.raises(ValueError, match="no matching pairs"):
        evaluation.PairFolder(tmp_path / "e1")
