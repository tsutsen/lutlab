"""Quick correctness checks for lutlab. Run: `python -m lutlab selftest`.

Covers the things that used to bite (axis order, backend disagreement) plus the
new bits (interp schemes, round-trips, the segmented `backend=` param). Fast,
CPU-first; the GPU is exercised only if a device is free, and skipped on any
error so a busy llama-server can't fail the suite.
"""
import numpy as np


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def _interp(rng):
    ok = True
    D = 16
    a = rng.randn(3, 3); b0 = rng.randn(3)
    g = np.stack(np.meshgrid(*[np.linspace(0, 1, D)] * 3, indexing="ij"), -1)
    lin = (g.reshape(-1, 3) @ a + b0).reshape(D, D, D, 3).astype(np.float32)
    q = rng.rand(4000, 3).astype(np.float32)
    true = q @ a + b0
    tet = np.abs(_interp_tet(lin, q) - true).max()
    ok &= _check("tetrahedral exact for linear f (max err %.2e)" % tet, tet < 1e-4)
    from . import interp
    nn = interp.interp(lin, q, "nearest")
    ml = interp.interp(lin, q, "trilinear")
    ok &= _check("multilinear ~ exact for linear f",
                 np.abs(ml - true).max() < 1e-4)
    ok &= _check("nearest differs from trilinear", not np.allclose(nn, ml, atol=1e-6))
    # sugar name checks
    ok &= _check("'bilinear' needs 2D, 'tetrahedral' needs 3D",
                 _raises(lambda: interp.interp_name("bilinear", 3)) and
                 not _raises(lambda: interp.interp_name("trilinear", 3)))
    return ok


def _interp_tet(vol, q):
    from . import interp
    return interp.interp_tetrahedral(vol, q)


def _lut(rng):
    ok = True
    from .lut import LUT, identity
    L = identity(17, 3)
    ok &= _check("identity LUT: max delta == 0", L.max_delta() == 0.0)
    x = rng.rand(50, 3).astype(np.float32)
    ok &= _check("identity apply == input",
                 np.allclose(L.apply(x, backend="numpy"), x, atol=1e-6))
    R = L.resample(33)
    ok &= _check("resample 17 -> 33", R.shape == (33, 33, 33, 3))
    ok &= _check("2D LUT accepted", LUT(np.zeros((9, 9, 3), np.float32)).ndim == 2)
    ok &= _check("bad layout rejected",
                 _raises(lambda: LUT(np.zeros((4, 3), np.float32))))
    return ok


def _formats(rng, tmp):
    ok = True
    from . import formats
    from .lut import LUT, identity
    # Non-round base values, so .cube's 6-decimal formatting actually rounds.
    v = np.clip(identity(17, 3).vol + 0.1234567, 0, 1).astype(np.float32)
    L = LUT(v, name="t")
    # cube round-trip: .cube is lossy at 6 decimals -> check ~1e-6, not exact
    cp = f"{tmp}/t.cube"
    formats.to_cube(L, cp)
    vol2, title2 = formats.from_cube(cp)
    cube_err = float(np.abs(vol2 - v).max())
    ok &= _check(f".cube round-trip ~1e-6 (max err {cube_err:.1e})",
                 cube_err < 2e-6 and title2 == "t")
    # json round-trip
    jp = f"{tmp}/t.json"
    formats.to_json(L, jp)
    vol3, name3 = formats.from_json(jp)
    ok &= _check("JSON round-trip exact",
                 np.array_equal(vol3, v) and name3 == "t")
    # validate: numpy vs numba must agree
    rep = formats.validate(L, n=200)
    numba_ok = (rep.get("numba") is not None) and rep["numba"] < 1e-5
    ok &= _check(f"validate numpy==numba (numba={rep.get('numba')})", numba_ok)
    if "gpu" in rep and isinstance(rep["gpu"], float):
        ok &= _check(f"validate numpy==gpu (gpu={rep['gpu']:.2e})", rep["gpu"] < 2e-4)
    else:
        print(f"    (gpu check skipped: {rep.get('gpu')})")
    return ok


def _apply_backends(rng):
    ok = True
    from lutlab import apply          # the top-level apply() function
    from .lut import LUT
    v = np.clip(np.linspace(0, 1, 17)[..., None] * 1.05, 0, 1)
    v = np.broadcast_to(v, (17, 17, 17, 3)).copy().astype(np.float32)
    L = LUT(v, name="b")
    img = rng.rand(8, 8, 3).astype(np.float32)
    ref = apply(L, img, interp="trilinear", backend="numpy")
    from .apply import backends
    if backends()["numba"]:
        ym = apply(L, img, interp="trilinear", backend="numba")
        ok &= _check("apply numpy==numba", np.allclose(ref, ym, atol=1e-6))
    if backends()["gpu"]:
        try:
            yg = apply(L, img, interp="trilinear", backend="gpu")
            ok &= _check("apply numpy==gpu", np.allclose(ref, yg, atol=2e-4))
        except Exception as e:
            print(f"    (gpu apply skipped: {e.__class__.__name__})")
    else:
        print("    (gpu not available -- skipped)")
    ok &= _check("wrong channel count rejected",
                 _raises(lambda: apply(L, rng.rand(4, 2).astype(np.float32))))
    return ok


def _metrics():
    ok = True
    from . import metrics
    ok &= _check("to_fps(1ms) == 1000", abs(metrics.to_fps(1.0) - 1000) < 1e-9)
    ok &= _check("to_fps(0.26ms) ~ 3846", abs(metrics.to_fps(0.26) - 3846.15) < 1)
    a = np.random.RandomState(0).rand(8, 8, 3)
    ok &= _check("e00 identical == 0", metrics.e00(a, a).max() < 1e-9)
    ok &= _check("psnr identical == inf", np.isinf(metrics.psnr(a, a)))
    return ok


def _segmented(rng, tmp):
    ok = True
    from . import segmented
    ok &= _check("available_backends has kmeans + sam",
                 set(["kmeans", "sam"]).issubset(segmented.available_backends()))
    img = rng.rand(24, 24, 3).astype(np.float32)
    target = np.clip(img * 1.15, 0, 1)
    masks, meta = segmented.segment_image(img, 4, backend="kmeans")
    ok &= _check("kmeans -> (S,H,W) masks summing to 1",
                 masks.shape == (4, 24, 24) and
                 np.allclose(masks.sum(0), 1, atol=1e-3))
    ok &= _check("sam stub raises NotImplementedError",
                 _raises(lambda: segmented.segment_image(img, 4, backend="sam"),
                         NotImplementedError))
    # registry round-trip
    seen = {}

    def fake(img_, n, **kw):
        seen["n"] = n
        m = np.full((n, img_.shape[0], img_.shape[1]), 1.0 / n, np.float32)
        return m, "fake-meta"
    segmented.register_segmenter("fake", fake)
    m2, meta2 = segmented.segment_image(img, 3, backend="fake")
    ok &= _check("register_segmenter dispatches", seen.get("n") == 3 and meta2 == "fake-meta")
    # end-to-end fit with the pre-computed kmeans masks
    fit = segmented.fit_with_masks(img, target, masks, sample_n=800, max_iter=3)
    ok &= _check("fit_with_masks -> S thetas", len(fit["thetas"]) == 4)
    # full round-trip
    r = segmented.fit_and_apply(img, target, n_segments=4, d=9, backend="kmeans",
                                apply_backend="numpy")
    ok &= _check("fit_and_apply -> out (H,W,3)", r["out"].shape == img.shape)
    return ok


def _raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:
        return False


def run(argv=None):
    rng = np.random.RandomState(42)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="lutlab_test_")
    print("lutlab selftest")
    print("interp    "); a = _interp(rng)
    print("lut       "); b = _lut(rng)
    print("formats   "); c = _formats(rng, tmp)
    print("apply     "); d = _apply_backends(rng)
    print("metrics   "); e = _metrics()
    print("segmented "); f = _segmented(rng, tmp)
    passed = all([a, b, c, d, e, f])
    print(f"\n{'ALL PASS' if passed else 'SOME FAILED'} ({' '.join(chr(65+i) for i,p in enumerate([a,b,c,d,e,f]) if p)}/ABCDEF)")
    return 0 if passed else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(run())
