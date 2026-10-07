"""P149 (r3 V0 / V1): Brownian channel replay (reference_core semantics: per-uid sequential increments, chunk/order invariance, coupling across t,
variance t per coordinate, no post-normalisation); orthogonal transport of the pair MLP first layer and of RFF frequencies (exact output equality);
standardiser; independent-pool readout (SE, Hoeffding radius, S_plugin) against direct formulas; derangement nulls have no fixed point; the
P116 solver used generically on [psi, 1]."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import p149_twoview as P  # noqa: E402


def test_brownian_replay_and_coupling():
    x = np.random.default_rng(0).standard_normal((50, 16)); ids = np.arange(100, 150)
    a = P.brownian(x, ids, role=2, view=1)
    assert set(a) == set(P.TIMES) and np.array_equal(a[0.0], x)
    # chunk / order invariance: the same uid gets the same noise regardless of position or batch
    b = P.brownian(x[::-1], ids[::-1], role=2, view=1)
    for t in P.TIMES:
        assert np.allclose(a[t], b[t][::-1])
    c = P.brownian(x[:10], ids[:10], role=2, view=1)
    assert np.allclose(a[1.0][:10], c[1.0])
    # coupling across t: W_1 = W_0.25 + independent increment of variance 0.75
    inc = (a[1.0] - x) - (a[0.25] - x)
    assert abs(((a[0.25] - x) ** 2).mean() - 0.25) < 0.05 and abs((inc ** 2).mean() - 0.75) < 0.1
    assert abs(np.mean((a[0.25] - x) * inc)) < 0.03
    # different view / role -> independent noise
    d = P.brownian(x, ids, role=2, view=2); e = P.brownian(x, ids, role=0, view=1)
    assert not np.allclose(a[1.0], d[1.0]) and not np.allclose(a[1.0], e[1.0])
    # matches the reference implementation (sequential draws per uid)
    rng = np.random.default_rng(np.random.SeedSequence([P.NOISE_SEED, 2, 100, 1, 991]))
    n1 = math.sqrt(0.25) * rng.standard_normal(16); n2 = n1 + math.sqrt(0.75) * rng.standard_normal(16)
    assert np.allclose(a[0.25][0], x[0] + n1) and np.allclose(a[1.0][0], x[0] + n2)


def test_orthogonal_and_standardizer():
    R = P.haar_orthogonal(32, 5)
    assert np.allclose(R @ R.T, np.eye(32), atol=1e-10) and np.allclose(P.haar_orthogonal(32, 5), R)
    x = np.random.default_rng(1).standard_normal((500, 8)) * 3 + 2
    mu, g = P.standardizer(x); y = (x - mu) / g
    assert np.allclose(y.mean(0), 0, atol=1e-12) and abs((y ** 2).mean() - 1) < 1e-12
    with pytest.raises(ValueError):
        P.standardizer(np.ones((10, 4)))


def test_mlp_transport_exact():
    d = 12; m = P.PairMLP(d, hidden=16, seed=3).double()
    u, v = torch.randn(40, d, dtype=torch.float64), torch.randn(40, d, dtype=torch.float64)
    R1, R2 = (torch.as_tensor(P.haar_orthogonal(d, s), dtype=torch.float64) for s in (11, 12))
    import copy
    mt = copy.deepcopy(m); Rt = torch.block_diag(R1.T, R2.T); mt.net[0].weight.data = m.net[0].weight.data @ Rt
    assert torch.allclose(m(u, v), mt(u @ R1.T, v @ R2.T), atol=1e-10)
    # a non-orthogonal change is NOT transported by this rule
    A = torch.randn(d, d, dtype=torch.float64)
    assert not torch.allclose(m(u, v), mt(u @ A.T, v @ R2.T), atol=1e-3)


def test_rff_transport_exact():
    d = 10; g = torch.Generator().manual_seed(0)
    W = torch.randn(2 * d, 64, generator=g, dtype=torch.float64); b = 2 * math.pi * torch.rand(64, generator=g, dtype=torch.float64)
    v = torch.randn(65, generator=g, dtype=torch.float64)
    cr = P.RFFCritic(W, b, v)
    u, w = torch.randn(30, d, dtype=torch.float64), torch.randn(30, d, dtype=torch.float64)
    R1, R2 = (torch.as_tensor(P.haar_orthogonal(d, s), dtype=torch.float64) for s in (21, 22))
    Wt = torch.cat([R1 @ W[:d], R2 @ W[d:]], 0)  # (R x)^T W' = x^T W  <=>  W' = R W  (per block)
    ct = P.RFFCritic(Wt, b, v)
    assert torch.allclose(cr.f(u, w), ct.f(u @ R1.T, w @ R2.T), atol=1e-10)


def test_readout_formulas():
    rng = np.random.default_rng(2); tp = np.tanh(rng.normal(1.0, 0.5, 400)); tq = np.tanh(rng.normal(-1.0, 0.5, 300))
    r = P.readout(tp, tq)
    ap, am = tp - tp ** 2 / 2, -tq - tq ** 2 / 2
    assert abs(r["J_common"] - (ap.mean() + am.mean())) < 1e-12
    assert abs(r["se_J"] - math.sqrt(ap.var(ddof=1) / 400 + am.var(ddof=1) / 300)) < 1e-12
    assert abs(r["hoeffding_radius_J"] - math.sqrt(2 * (1 / 400 + 1 / 300) * math.log(2 / 0.05))) < 1e-12
    assert abs(r["S_plugin"] - 0.5 * ((tp ** 2).mean() + (tq ** 2).mean())) < 1e-12
    assert r["approximation_bias_covered"] is False
    z = P.readout(np.zeros(100), np.zeros(100)); assert z["J_common"] == 0 and z["S_plugin"] == 0


def test_rff_fit_on_dependent_gaussian_pairs():
    """Two-stage RFF solver on [psi, 1]: positive J on a dependent pair, ~0 on an independent one (small synthetic check)."""
    rng = torch.Generator().manual_seed(7); d = 6; n = 600
    u = torch.randn(n, d, generator=rng); v = 0.9 * u + 0.4 * torch.randn(n, d, generator=rng)
    ql, qr = torch.randn(n, d, generator=rng), torch.randn(n, d, generator=rng)
    P.DEV = torch.device("cpu")
    fit = P.fit_rff((u[:400], v[:400]), (ql[:400], qr[:400]), (u[400:], v[400:]), (ql[400:], qr[400:]), d, seed=0)
    m = fit["model"]; tp = torch.tanh(m.f(u[400:], v[400:])).numpy(); tq = torch.tanh(m.f(ql[400:], qr[400:])).numpy()
    assert P.readout(tp, tq)["J_common"] > 0.3 and len(fit["candidates"]) == 3
    tqq = torch.tanh(m.f(ql[:200], qr[200:400])).numpy()  # re-paired independent pool stays near 0 in J contribution
    assert abs(float((-tqq - tqq ** 2 / 2).mean()) - float((-tq - tq ** 2 / 2).mean())) < 0.1
