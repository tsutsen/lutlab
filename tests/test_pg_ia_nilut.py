"""PG-IA-NILUT (Kosugi, MM'24, arXiv 2408.11055): architecture, LUT bypassing, guidance loss, training."""
import numpy as np
import pytest

import lutlab
from lutlab import evaluation, methods, testing
from lutlab.methods import pg_ia_nilut as pg

# ---- torch-free parts ----------------------------------------------------------- #

def test_prompts_match_table_1():
    assert pg.FILTER_NAMES == ("Exposure", "Contrast", "Saturation", "Color temperature",
                               "Tint correction")
    fk, pp = pg.guiding_prompts("fivek"), pg.guiding_prompts("PPR10K")
    assert fk[0] == ("Overexposed photo.", "Underexposed photo.")
    assert fk[1] == ("Clear photo.", "Unclear photo.") and pp[1] == ("High contrast photo.", "Low contrast photo.")
    assert fk[2:] == pp[2:] and fk[3] == ("Yellow tinted photo.", "Blue tinted photo.")
    assert fk[4] == ("Magenta tinted photo.", "Green tinted photo.")
    with pytest.raises(ValueError, match="unknown dataset"):
        pg.guiding_prompts("imagenet")
    assert pg.name_key("Color temperature") == "color_temperature"


def test_registered_and_import_is_lazy():
    assert "pg_ia_nilut" in methods.names()
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", "import sys, lutlab.methods.pg_ia_nilut as m;"
                          "print('torch' in sys.modules)"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    with pytest.raises(AttributeError):
        pg.does_not_exist


def test_de76_metric():
    pytest.importorskip("skimage")
    a = np.random.default_rng(0).random((8, 8, 3))
    assert lutlab.metrics.de76(a, a).max() == 0.0
    assert float(lutlab.metrics.de76(np.zeros(3), np.ones(3))) == pytest.approx(100.0, abs=0.5)
    r = evaluation.evaluate(lutlab.identity(9), [(a.astype(np.float32),) * 2], metrics=("de76",))
    assert r["de76_mean"] == pytest.approx(0.0, abs=1e-4)


# ---- everything below needs torch ------------------------------------------------ #
torch = pytest.importorskip("torch")


@pytest.fixture
def E():
    torch.manual_seed(0)
    return pg.IANILUT().eval()


def test_parameter_counts_match_the_paper():
    n = pg.PGIANILUT(pg.IANILUT(), pg.CnnPredictor()).num_parameters()
    assert n["filters"] == 201_219            # 11->256, 3x 256->256, 256->3
    assert abs(n["total"] - 449_300) / 449_300 < 0.01      # paper, Table A: 449.3 K


def test_eq3_is_identity_at_zero_weights(E):
    x = torch.rand(5000, 3)
    assert torch.equal(E(x, torch.zeros(5)), x)
    xb = torch.rand(3, 8, 8, 3)
    assert torch.equal(E(xb, torch.zeros(3, 5)), xb)
    assert (E(x, torch.full((5,), 0.7)) - x).abs().max() > 1e-4          # and w does something


def test_features_are_rgb_sorted_rgb_and_w():
    f = pg.IANILUT._features(torch.tensor([[0.3, 0.1, 0.2]]), torch.arange(5.0)[None])
    assert f.shape == (1, 11)
    assert torch.allclose(f[0, :6], torch.tensor([0.3, 0.1, 0.2, 0.1, 0.2, 0.3]))
    assert torch.equal(f[0, 6:], torch.arange(5.0))


def test_weight_broadcasting(E):
    x = torch.rand(2, 4, 4, 3)
    w = torch.rand(5)
    assert torch.allclose(E(x, w), E(x, w.expand(2, 5)), atol=1e-6)
    per = torch.stack([w, -w])
    out = E(x, per)
    assert torch.allclose(out[0], E(x[0], w), atol=1e-6) and torch.allclose(out[1], E(x[1], -w), atol=1e-6)
    with pytest.raises(ValueError, match="does not match"):
        E(x, torch.rand(3, 5))
    with pytest.raises(ValueError, match="channel-last"):
        E(torch.rand(3, 4, 4), w)


def test_lut_bypassing_builds_a_real_lutlab_lut(E):
    w = E.weights(exposure=0.5, color_temperature=-0.3)
    lut = E.lut(w, N=9)
    assert isinstance(lut, lutlab.LUT) and lut.sizes == (9, 9, 9)
    g = E.grid(9)
    with torch.no_grad():
        direct = E(g.reshape(-1, 3), w).reshape(9, 9, 9, 3)
    assert np.abs(lut.vol - direct.numpy()).max() < 1e-6                   # O = E_hat(I, w), indexed [r, g, b]
    assert np.abs(E.lut(torch.zeros(5), 9).vol - lutlab.identity(9).vol).max() == 0.0
    with pytest.raises(ValueError, match="one parameter vector"):
        E.lut(torch.zeros(2, 5))


def test_bypass_error_depends_on_how_smooth_the_mlp_is(E):
    x, w = torch.rand(20000, 3), torch.rand(5) * 2 - 1
    with torch.no_grad():
        smooth = (E.enhance(x, w, bypass=True, N=33) - E(x, w)).abs().max().item()
        rough_net = pg.IANILUT().eval()
        for p in rough_net.parameters():
            p.mul_(3.0)
        rough = (rough_net.enhance(x, w, bypass=True, N=33) - rough_net(x, w)).abs().max().item()
    assert smooth < 1e-4            # measured ~2e-5 for default init
    assert rough > 1e-3 > smooth    # a rougher MLP loses more when baked: the bake gap is real


def test_numpy_path_through_lutlab_kernels_equals_torch_path(E):
    t = pg.PGIANILUT(E).transform(w=E.weights(exposure=0.4, saturation=-0.5, tint_correction=0.2))
    img = np.random.default_rng(0).random((32, 40, 3)).astype(np.float32)
    a = t(img)                                                       # numpy -> LUT -> lutlab backend
    b = t(torch.as_tensor(img))                                      # torch -> differentiable lookup
    assert isinstance(a, np.ndarray) and torch.is_tensor(b)
    assert np.abs(a - b.numpy()).max() < 1e-5
    assert np.abs(a - t(img, backend="numpy", interp="multilinear")).max() < 1e-5   # kwargs reach lutlab


def test_transform_conformance(E):
    w = E.weights(exposure=0.4, contrast=0.2)
    m = pg.PGIANILUT(E)
    assert testing.conformance(m.transform(w), bake_D=33, bake_tol=1e-5) == []      # exact: same N=33 LUT
    assert testing.conformance(m.transform(w, bypass=False), bake_D=33, bake_tol=1e-3) == []


def test_weight_sources_and_errors(E):
    m = pg.PGIANILUT(E)
    img = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    bound = m.transform(w=E.weights(exposure=0.5))
    assert np.abs(bound(img, context=np.zeros(5, np.float32)) - img).max() < 1e-6    # context beats bound w
    with pytest.raises(ValueError, match="no parameters"):
        m.transform()(img)
    m2 = pg.PGIANILUT(E, pg.CnnPredictor())
    t = m2.transform()
    out = t(img)                                                      # w = F(img)
    assert out.shape == img.shape and np.isfinite(out).all()
    assert np.allclose(t(np.stack([img, img])[0]), out, atol=1e-6)
    with pytest.raises(ValueError, match="need an image"):
        t(img.reshape(-1, 3))
    with pytest.raises(ValueError, match="global parameters"):
        t.bake()
    assert t.cost().meta["predictor"] == m2.num_parameters()["predictor"] and t.cost().reads_per_px == 8
    assert m2.transform(bypass=False).cost().reads_per_px is None


def test_named_weights(E):
    w = E.weights(exposure=0.4, tint_correction=-1.0)
    assert w.tolist() == pytest.approx([0.4, 0, 0, 0, -1.0])
    with pytest.raises(ValueError, match="unknown filter 'sharpness'"):
        E.weights(sharpness=1)


def test_score_jacobian_matches_analytic_and_autograd():
    torch.manual_seed(0)
    A, b = torch.randn(4, 4, dtype=torch.float64), torch.randn(4, dtype=torch.float64)
    w = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    jac = pg.score_jacobian(w @ A.T + b, w)
    assert jac.shape == (3, 4, 4) and torch.allclose(jac, A.expand(3, 4, 4))
    w2 = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    f = lambda v: torch.sigmoid(v @ A.T)
    jac2 = pg.score_jacobian(f(w2), w2)
    for i in range(2):
        ref = torch.autograd.functional.jacobian(lambda v: f(v[None])[0], w2[i].detach())
        assert torch.allclose(jac2[i], ref, atol=1e-10)


def test_prompt_guidance_loss_eq7_by_hand():
    jac = torch.tensor([[[0.5, 0.2], [0.1, 1.0]]])
    # sum_j lambda_j |d_jj - 1| = 1*0.5 + 2*0 ;  lambda * sum_{j'!=j} |d_j'j| = 3*(0.2+0.1)
    assert pg.prompt_guidance_loss(jac, lambdas=[1.0, 2.0], lam=3.0).item() == pytest.approx(1.4)
    assert pg.prompt_guidance_loss(torch.eye(5).expand(4, 5, 5)).item() == 0.0
    off = torch.eye(3).expand(2, 3, 3) + 0.5 * (1 - torch.eye(3))
    assert pg.prompt_guidance_loss(off, lam=2.0).item() == pytest.approx(2.0 * 6 * 0.5)
    with pytest.raises(ValueError, match="as many prompt pairs"):
        pg.prompt_guidance_loss(torch.zeros(1, 3, 5))


def _mean_rgb(images):                       # synthetic scorer: c_j = mean of channel j
    return images.mean((1, 2))


@pytest.mark.parametrize("bypass", [False, True])
def test_guidance_loss_is_differentiable_through_the_whole_filter(bypass):
    torch.manual_seed(0)
    E3 = pg.IANILUT(n_params=3, hidden=16, n_layers=3)
    x = torch.rand(2, 8, 8, 3)
    loss, jac = pg.guidance_step_loss(E3, _mean_rgb, x, bypass=bypass, N=9)
    assert jac.shape == (2, 3, 3) and torch.isfinite(loss)
    loss.backward()                           # needs double backward through sort / tanh / lookup
    dead = E3.mlp[-1].bias                    # Eq. 3 subtracts e(.,0): the output bias cancels exactly
    for p in E3.parameters():
        if p is dead:
            assert p.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0


def test_stage1_can_reach_the_guidance_target_with_a_reachable_scorer():
    # c_j = mean channel j, J = 3: E(x, w) = x + w satisfies dc_j/dw_k = delta_jk exactly, so L_PG -> 0.
    torch.manual_seed(0)
    cfg = pg.TrainConfig(n_params=3, hidden=32, n_layers=3, stage1_steps=250, lr_filters=3e-3,
                         batch_size=4, pg_size=8, bypass_train=False)
    model = pg.PGIANILUT(pg.IANILUT(3, 32, 3))
    data = [(np.random.default_rng(i).random((8, 8, 3)).astype(np.float32),) * 2 for i in range(8)]
    ds = pg.ImagePairDataset(data, (8, 8))
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    hist = pg.train_stage1(model, _mean_rgb, loader, cfg, device=torch.device("cpu"))
    first, last = np.mean(hist["stage1_pg"][:5]), np.mean(hist["stage1_pg"][-5:])
    assert first > 0.5 and last < 0.25 * first, (first, last)


def _toy_pairs(n=6, size=16):
    rng = np.random.default_rng(0)
    xs = [rng.random((size, size, 3)).astype(np.float32) * 0.7 for _ in range(n)]
    return [(x, np.clip(x * 1.25 + 0.05, 0, 1).astype(np.float32)) for x in xs]


def test_three_stage_pipeline_end_to_end(tmp_path):
    torch.manual_seed(0)
    cfg = pg.TrainConfig(n_params=3, hidden=16, n_layers=3, size=(16, 16), pg_size=16, batch_size=3,
                         stage1_steps=5, stage2_epochs=25, stage3_epochs=4, lr_filters=1e-2,
                         lr_w=5e-2, lr_predictor=1e-3, lut_size=9)
    res = pg.train_pg_ia_nilut(_toy_pairs(), _mean_rgb, cfg, device="cpu", log=None)
    h = res.history
    assert len(h["stage1_pg"]) == 5 and len(h["stage2_epoch_loss"]) == 25 and len(h["stage3_mse"]) == 4
    assert res.w_table.shape == (6, 3) and res.w_table.abs().max() > 0
    assert h["stage2_epoch_loss"][-1] < 0.7 * h["stage2_epoch_loss"][0]       # Eq. 9 is being minimised
    # reconstruction term of Eq. 9: E(T, -w) should recover X better than doing nothing
    X, T = _toy_pairs()[0]
    t = res.model.transform(w=res.w_table[0].numpy(), N=9)
    fwd = np.mean((t(X) - T) ** 2)
    assert fwd < np.mean((X - T) ** 2)                                         # forward pass moved X toward T
    # adaptive transform + checkpoint round trip
    adaptive = res.model.transform(N=9)
    out = adaptive(X)
    p = tmp_path / "m.pt"
    res.model.save(p)
    again = pg.PGIANILUT.load(p).transform(N=9)(X)
    assert np.abs(out - again).max() < 1e-6


def test_stage3_freezes_the_filters_and_trains_only_the_predictor():
    torch.manual_seed(0)
    cfg = pg.TrainConfig(n_params=3, hidden=8, n_layers=3, size=(16, 16), batch_size=3,
                         stage3_epochs=2, lut_size=5)
    model = pg.PGIANILUT(pg.IANILUT(3, 8, 3), pg.CnnPredictor(3))
    f0 = {k: v.clone() for k, v in model.filters.state_dict().items()}
    p0 = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    pg.train_stage3(model, pg.ImagePairDataset(_toy_pairs(), (16, 16)), cfg, device=torch.device("cpu"))
    assert all(torch.equal(v, f0[k]) for k, v in model.filters.state_dict().items())
    assert any(not torch.equal(v, p0[k]) for k, v in model.predictor.state_dict().items())
    with pytest.raises(ValueError, match="no predictor"):
        pg.train_stage3(pg.PGIANILUT(pg.IANILUT(3, 8, 3)), pg.ImagePairDataset(_toy_pairs(), (16, 16)),
                        cfg, device=torch.device("cpu"))


def test_registry_entry_points(tmp_path):
    t = methods.build("pg_ia_nilut")                          # untrained: w = 0 -> identity
    img = np.random.default_rng(0).random((20, 20, 3)).astype(np.float32)
    assert np.abs(t(img) - img).max() < 1e-5
    assert testing.conformance(t, bake_D=33, bake_tol=1e-5) == []
    m = pg.PGIANILUT(pg.IANILUT(), pg.CnnPredictor())
    m.save(tmp_path / "c.pt")
    adaptive = methods.build("pg_ia_nilut", checkpoint=tmp_path / "c.pt")
    assert adaptive(img).shape == img.shape


def test_plugs_into_the_evaluation_harness_and_exports_cube(tmp_path):
    pytest.importorskip("skimage")
    m = pg.PGIANILUT(pg.IANILUT(), pg.CnnPredictor())
    pairs = [(x, y) for x, y in _toy_pairs(2, 32)]
    r = evaluation.evaluate(m.transform(), pairs, metrics=("psnr", "ssim", "e00", "de76"),
                            timing={"warmup": 1, "repeat": 2})
    assert r["n"] == 2 and {"psnr_db", "ssim", "e00_mean", "de76_mean"} <= set(r)
    assert r["cost"]["params"] == m.num_parameters()["total"] and r["timing"]["fps"] > 0
    rep = evaluation.run_matrix({"pg_ia_nilut": m.transform()}, {"toy": pairs})
    assert "error" not in rep["rows"][0]
    # the interpretable filter, frozen at chosen controls, is just a .cube file
    lut = m.filters.lut(m.filters.weights(exposure=0.6), N=17)
    lut.save(tmp_path / "exposure_plus.cube")
    assert lutlab.load(tmp_path / "exposure_plus.cube").shape == (17, 17, 17, 3)


def test_dataset_wrapper_resizes_pairs():
    ds = pg.ImagePairDataset([(np.zeros((40, 30, 3), np.float32), np.ones((50, 20, 3), np.float32))], (24, 24))
    i, x, t = ds[0]
    assert i == 0 and x.shape == t.shape == (24, 24, 3) and float(t.min()) > 0.99


def test_resnet_predictor_is_optional():
    pytest.importorskip("torchvision")
    p = pg.ResNetPredictor(5)
    assert p(torch.rand(2, 64, 64, 3)).shape == (2, 5)


@pytest.mark.skipif(not __import__("os").environ.get("LUTLAB_TEST_CLIP"),
                    reason="needs open_clip + downloaded CLIP weights; set LUTLAB_TEST_CLIP=1")
def test_clip_scorer_smoke():
    s = pg.ClipScorer(dataset="fivek")
    c = s(torch.rand(2, 64, 64, 3))
    assert c.shape == (2, 5) and ((c > 0) & (c < 1)).all()
