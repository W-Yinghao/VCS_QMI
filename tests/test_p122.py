"""P122 (v6 V6-EVIDENCE) unit tests: pools, augmentation determinism, nested step 0, symmetrisation, losses, bins, increments, diagnostics."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import vcs_measure.evidence as ev  # noqa: E402

VIEWS = {"random_resized_crop": {"size": 32, "scale": [0.2, 1.0], "ratio": [0.75, 1.3333333333], "interpolation": "bilinear", "antialias": True},
         "horizontal_flip_p": 0.5, "color_jitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1, "p": 0.8},
         "grayscale_p": 0.2, "gaussian_blur_p": 0.0, "solarize_p": 0.0}


def toy_pairs(n=256, d=8, seed=0, h=True):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, d, generator=g); nz = lambda: torch.randn(n, d, generator=g) * 0.6
    n1 = lambda x: torch.nn.functional.normalize(x, dim=1)
    z1, z2, zb = n1(base + nz()), n1(base + nz()), n1(torch.randn(n, d, generator=g))
    H = (lambda: torch.randn(n, 2 * d, generator=g)) if h else (lambda: None)
    return ev.Pairs(z1, z2, zb, H(), H(), H())


def test_pools_disjoint_and_shared():
    uids = np.arange(1000)
    p1 = ev.make_pools(uids, {"fit": 200, "tune": 100, "eval": 200}, 5); p2 = ev.make_pools(uids, {"fit": 200, "tune": 100, "eval": 200}, 5)
    allu = np.concatenate([np.concatenate([v["anchors"], v["partners"]]) for v in p1.values()])
    assert len(np.unique(allu)) == len(allu) == 500
    assert all((p1[k][r] == p2[k][r]).all() for k in p1 for r in ("anchors", "partners"))       # shared explicit indices (deterministic)
    assert ev.pools_hash(p1) == ev.pools_hash(p2)
    assert len(p1["fit"]["anchors"]) == len(p1["fit"]["partners"]) == 100
    with pytest.raises(AssertionError):
        ev.make_pools(uids, {"fit": 800, "tune": 300}, 5)


def test_augment_deterministic_order_independent():
    rng = np.random.default_rng(0); imgs = rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
    a = ev.augment(imgs, np.array([3, 7, 11]), VIEWS, 9, 1); b = ev.augment(imgs, np.array([11, 3, 7]), VIEWS, 9, 1)
    assert torch.equal(a[0], b[1]) and torch.equal(a[2], b[0])                                  # per-image seed, not position
    c = ev.augment(imgs, np.array([3, 7, 11]), VIEWS, 9, 2)
    assert not torch.equal(a, c)                                                                # different view tag -> different view
    s0 = torch.random.get_rng_state(); ev.augment(imgs, np.array([3]), VIEWS, 9, 1); assert torch.equal(s0, torch.random.get_rng_state())


def test_squared_risk_identity_and_js():
    fp, fq = torch.randn(100), torch.randn(100)
    r = ev.readouts(fp, fq); tp, tq = torch.tanh(fp.double()), torch.tanh(fq.double())
    sq = 0.5 * ((1 - tp) ** 2).mean() + 0.5 * ((-1 - tq) ** 2).mean()
    assert abs(r["sq_risk"] - float(sq)) < 1e-12 and abs(r["J"] - (1 - float(sq))) < 1e-12
    z = torch.zeros(10); assert abs(float(ev.own_risk(z, z, "js")) - 2 * math.log(2)) < 1e-6 and abs(ev.readouts(z, z)["js_native"]) < 1e-12


def test_bins_closed_form():
    P = toy_pairs(); b = ev.fit_bins(P, 8)["fn"]; sp, sq = P.s()
    t = torch.tanh(b.on_s(torch.cat([sp, sq])))
    assert (t.abs() <= 1).all()
    edges = torch.quantile(torch.cat([sp, sq]).double(), torch.linspace(0, 1, 9, dtype=torch.float64))[1:-1].float()
    bp, bq = torch.bucketize(sp, edges), torch.bucketize(sq, edges)
    k = int(bp[0]); npk, nqk = int((bp == k).sum()), int((bq == k).sum())
    assert abs(float(torch.tanh(b.on_s(sp[:1]))) - (npk - nqk) / (npk + nqk)) < 1e-5


def test_symmetric_residual_and_nested_step0():
    torch.manual_seed(0); q = ev.SymResidual(8)
    u, v = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.allclose(q(u, v), torch.zeros(5))                                              # zero-initialised output layer
    for p in q.net[-1].parameters():
        torch.nn.init.normal_(p)
    assert torch.allclose(q(u, v), q(v, u), atol=1e-6)                                          # symmetrised
    P = toy_pairs(); parent = ev.fit_bins(P, 8)["fn"]
    nf = ev.NestedFn(parent, ev.SymResidual(8), "z")
    fp0, fq0 = parent(P); fp1, fq1 = nf(P)
    assert torch.allclose(fp0, fp1) and torch.allclose(fq0, fq1)                                # step 0 equals the parent


def test_fit_nested_parent_is_candidate_and_selection_on_tune():
    ev.STEPS, ev.LRS, ev.INITS = 30, (5e-3,), (0,)
    Pf, Pt = toy_pairs(seed=1), toy_pairs(seed=2)
    parent = ev.fit_bins(Pf, 8)["fn"]
    cands = ev.fit_nested(parent, Pf, Pt, "vcs", "z", torch.device("cpu"))
    assert cands[0]["params"]["residual"] is False and len(cands) == 2
    best, rows = ev.select(cands, Pt, "vcs")
    assert best["tune_risk"] <= rows[0]["tune_risk"] + 1e-12                                    # never worse than the parent on TUNE
    mu, sd = torch.zeros(1, 16), torch.ones(1, 16)
    hc = ev.fit_nested(best["fn"], Pf, Pt, "js", "h", torch.device("cpu"), mu=mu, sd=sd)
    hb, _ = ev.select(hc, Pt, "js"); assert hb["tune_risk"] <= hc[0]["tune_risk"] + 1e-12


def test_increment_and_density_diag():
    jb, ja = np.array([0.3, 0.5, 0.1]), np.array([0.1, 0.2, 0.2])
    inc = ev.boot_increment(jb, ja, reps=200)
    assert abs(inc["mean"] - float((jb - ja).mean())) < 1e-12 and inc["ci95"][0] <= inc["mean"] <= inc["ci95"][1]
    P = toy_pairs(n=64)
    zero = ev.ScalarFn(lambda s: torch.zeros_like(s), "zero", {})
    d = ev.density_diag(zero, P, 16, 16)
    assert abs(d["global_log_mean_exp_2f_Q"]) < 1e-12 and abs(d["per_anchor_mean"]) < 1e-12 and abs(d["per_anchor_ess_mean"] - 16) < 1e-9


def test_affine_and_mlp_fit_run():
    ev.STEPS, ev.LRS, ev.INITS = 30, (5e-3,), (0,)
    Pf, Pt = toy_pairs(seed=3), toy_pairs(seed=4)
    for loss in ("vcs", "js"):
        aff = ev.fit_affine(Pf, Pt, loss); mlp = ev.fit_mlp1d(Pf, Pt, loss, torch.device("cpu"))
        best, rows = ev.select(aff + mlp, Pt, loss)
        assert len(rows) == 4 and np.isfinite(best["tune_risk"])
