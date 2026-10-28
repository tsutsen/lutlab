import numpy as np
import pytest

import lutlab
from lutlab import LUT, metrics, testing

pytest.importorskip("skimage")


def test_resample_keeps_identity_and_metadata():
    lut = LUT.identity(17)
    r = lut.resample(33)
    assert r.sizes == (33,) * 3 and metrics.lut_space_accuracy(r)["max_channel"] < 1e-6
    meta = testing.random_lut(5, 3, seed=1).replace(domain=(0, 2), space="linear")
    r2 = meta.resample((7, 8, 9))
    assert r2.sizes == (7, 8, 9) and r2.domain == meta.domain and r2.space == "linear"
    with pytest.raises(ValueError):
        meta.resample((7, 8))


def test_resample_from_rectilinear_matches_direct_lookup(rng):
    axes = [np.linspace(0, 1, 9) ** 1.5] * 3
    lut = testing.random_lut(9, 3, seed=2, axes=axes)
    r = lut.resample(33)
    x = rng.random((200, 3)).astype(np.float32)
    assert np.abs(lut(x, backend="numpy") - r(x, backend="numpy")).max() < 0.02


def test_delta_and_tone_curve():
    for lut in (LUT.identity(5), LUT.identity(5, 4), LUT.identity(sizes=(4, 5, 6), axes=[np.linspace(0, 1, s) ** 2 for s in (4, 5, 6)])):
        assert metrics.lut_space_accuracy(lut)["max_channel"] == 0.0
    t, curve = metrics.tone_curve(LUT.identity(9), n=17)
    assert np.allclose(curve, t[:, None], atol=1e-6)
    assert metrics.lut_space_accuracy(testing.random_lut(5, 3, seed=1))["max_channel"] > 0


def test_e00_psnr_ssim_basics(rng):
    a = rng.random((16, 16, 3))
    assert metrics.e00(a, a).max() == 0.0 and metrics.e00(a, a).shape == (16, 16)
    assert metrics.e00(np.zeros(3), np.ones(3)).shape == ()
    assert 95 < float(metrics.e00(np.zeros(3), np.ones(3))) < 105
    assert metrics.psnr(a, a) == float("inf")
    assert metrics.psnr(np.zeros((4, 4, 3)), np.full((4, 4, 3), 0.1)) == pytest.approx(20.0)
    assert metrics.ssim(a, a) == pytest.approx(1.0)
    assert metrics.ssim(a, 1 - a) < 0.5
    with pytest.raises(ValueError, match="too small"):
        metrics.ssim(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))


def test_e00_colour_space_matters():
    mid, grey = np.full(3, 0.5), np.full(3, 0.6)
    assert metrics.e00(mid, grey, space="srgb") != pytest.approx(metrics.e00(mid, grey, space="linear"))
    with pytest.raises(ValueError, match="colour space"):
        metrics.e00(mid, grey, space="aces")


def test_accuracy_and_lut_space(rng):
    a = rng.random((16, 16, 3))
    acc = metrics.accuracy(a, a)
    assert acc["e00_mean"] == 0 and acc["ssim"] == pytest.approx(1.0)
    z = metrics.lut_space_accuracy(LUT.identity(5))
    assert z["e00_max"] == 0.0 and z["max_channel"] == 0.0
    assert metrics.lut_space_accuracy(testing.random_lut(5, 3, seed=1))["e00_mean"] > 0
    with pytest.raises(ValueError, match="grids differ"):
        metrics.lut_space_accuracy(LUT.identity(5), LUT.identity(7))


def test_timing_helpers():
    r = metrics.time_fn(lambda: sum(range(1000)), warmup=1, repeat=5)
    assert set(r) == {"median_ms", "min_ms", "p90_ms", "repeat"} and r["min_ms"] <= r["median_ms"]
    assert metrics.to_fps(10.0) == 100.0 and metrics.to_fps(0.0) == float("inf")
    s = metrics.apply_speed(LUT.identity(9), np.zeros((64, 64, 3), np.float32), warmup=1, repeat=3)
    assert s["fps"] > 0 and s["mpx_s"] > 0


def test_lut_space_accuracy_2d_needs_a_reference():
    lut = LUT(np.zeros((3, 3, 3), np.float32))
    with pytest.raises(ValueError, match="ref_lut"):
        metrics.lut_space_accuracy(lut)
    assert metrics.lut_space_accuracy(lut, lut)["max_channel"] == 0.0


def test_psnr_and_ssim_match_scikit_image(rng):
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    a = rng.random((16, 16, 3))
    b = rng.random((16, 16, 3))
    assert metrics.psnr(a, b) == pytest.approx(sk_psnr(a, b, data_range=1.0))
    assert metrics.ssim(a, b) == pytest.approx(
        sk_ssim(a, b, channel_axis=2, data_range=1.0))


def test_gpu_power_parses_nvidia_smi(monkeypatch):
    from lutlab.metrics import power

    class Result:
        returncode = 0
        stdout = "123.4\n"

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "rocm-smi":
            raise FileNotFoundError
        return Result()

    monkeypatch.setattr(power.subprocess, "run", fake_run)
    assert power.gpu_power_watts("cuda:1") == 123.4
    assert calls[-1][:2] == ["nvidia-smi", "--id=1"]


def test_power_sample_synchronises(monkeypatch):
    from lutlab.metrics import power
    calls = []
    monkeypatch.setattr(power, "cuda_sync", lambda: calls.append(1))
    monkeypatch.setattr(power, "gpu_power_watts", lambda device: 10.0)
    r = power.power_sample(lambda: None, warmup=1, repeat=2)
    assert r["watts"] == 10.0 and len(calls) >= 4
