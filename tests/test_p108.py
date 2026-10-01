"""P108 (package v4 module E) — exact identities and plumbing."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import p108  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_population_identities_exact(seed):
    p, q = p108.discrete_toy(12, seed); r = np.random.default_rng(seed + 100)
    for _ in range(5):
        T = np.tanh(r.normal(size=12) * 2)
        o = p108.population_readouts(T, p, q)
        assert abs(o["J_bias"] - o["pred_J_bias"]) < 1e-12
        assert abs(o["S_plug_bias"] - o["pred_S_plug_bias"]) < 1e-12


def test_eta_is_optimum_and_both_readouts_exact():
    p, q = p108.discrete_toy(12, 3); eta = (p - q) / (p + q); o = p108.population_readouts(eta, p, q)
    assert abs(o["J"] - o["S"]) < 1e-14 and abs(o["S_plug"] - o["S"]) < 1e-14


@pytest.mark.parametrize("t", [0.01, 0.1, 0.5, 1.0])
def test_mechanism_orders_exact(t):
    p, q = p108.discrete_toy(12, 4); eta = (p - q) / (p + q); m = 0.5 * (p + q); S = float((m * eta ** 2).sum())
    o = p108.population_readouts((1 - t) * eta, p, q)                     # U = 0
    assert abs(o["J_bias"] - (-t * t * S)) < 1e-14
    assert abs(o["S_plug_bias"] - ((-2 * t + t * t) * S)) < 1e-14
    U = np.tanh(np.linspace(-1, 1, 12)); e = t * (U - eta)                # nonzero U
    o = p108.population_readouts(eta + e, p, q)
    assert abs(o["J_bias"] + t * t * float((m * (U - eta) ** 2).sum())) < 1e-13
    assert abs(o["S_plug_bias"] - (2 * t * float((m * eta * (U - eta)).sum()) + t * t * float((m * (U - eta) ** 2).sum()))) < 1e-13


def test_readouts_sample_formula():
    g = torch.Generator().manual_seed(0); tp, tq = torch.rand(1000, generator=g, dtype=torch.float64) * 2 - 1, torch.rand(800, generator=g, dtype=torch.float64) * 2 - 1
    r = p108.readouts(tp, tq)
    assert math.isclose(r["J"], float((tp - tp ** 2 / 2).mean() + (-tq - tq ** 2 / 2).mean()), rel_tol=1e-12)
    assert math.isclose(r["S_plug"], float(0.5 * (tp ** 2).mean() + 0.5 * (tq ** 2).mean()), rel_tol=1e-12)


def test_u_fixed_bounded_and_data_independent():
    g = torch.Generator().manual_seed(1); x, y = torch.randn(500, 7, generator=g), torch.randn(500, 7, generator=g)
    u1, u2 = p108.u_fixed(x, y), p108.u_fixed(x, y)
    assert torch.equal(u1, u2) and float(u1.abs().max()) < 0.8


def test_rotation_orthogonal_and_distance_preserving():
    R = p108.orthogonal(20, "x")
    assert float((R @ R.T - torch.eye(20, dtype=R.dtype)).abs().max()) < 1e-12
    assert not torch.allclose(p108.orthogonal(20, "x"), p108.orthogonal(20, "y"))
    g = torch.Generator().manual_seed(2); a = torch.randn(10, 20, generator=g, dtype=torch.float64)
    assert torch.allclose(torch.cdist(a, a), torch.cdist(a @ R.T, a @ R.T), atol=1e-10)


def test_rotated_roles_keep_truth_and_eta():
    from vcs_estim.benchmark import build_roles, eta_of
    from vcs_estim.data import setting_from_mi
    st = setting_from_mi("gaussian", 1.5, 2); roles, _ = build_roles(st, 2, 10, 256, 0, smoke=True)
    rot, info = p108.rotate_roles(roles, 10); E0 = info["orig_eval"]
    assert rot["TRUTH"] is roles["TRUTH"] and info["orthogonality_error"] < 1e-12
    assert torch.equal(eta_of(st, E0.xp, E0.yp, 2), eta_of(st, roles["EVAL"].xp, roles["EVAL"].yp, 2))
    assert not torch.allclose(rot["EVAL"].xp, roles["EVAL"].xp)


def test_mechanism_mc_smoke_orders():
    r = p108.mechanism_mc("C1_gauss_mid", smoke=True)
    z, u = r["U"]["zero"], r["U"]["fixed"]
    assert abs(z["loglog_slope_t_le_0.1"]["J"] - 2.0) < 1e-6 and abs(z["loglog_slope_t_le_0.1"]["S_plug"] - 1.0) < 0.05   # U = 0: -t^2 S vs (-2t + t^2) S
    assert 1.9 < u["loglog_slope_t_le_0.1"]["J"] < 2.1                                                               # J second order for any U
    assert z["max_abs_S_plug_identity_residual"] < 1e-12 and u["max_abs_S_plug_identity_residual"] < 1e-12            # plug-in identity is algebraic
    assert z["max_abs_D_J_z"] < 5 and u["max_abs_D_J_z"] < 5                                                          # J identity: sampling only
