"""Exact empirical statistics from cell counts (U / V), the saturated FIT table, and the backend U moments in the MF objective."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import t03_models as Mo
import t03_objectives as O
from v10 import mf_vcs as MF


def _data(n=200, m=5, seed=0):
    g = np.random.default_rng(seed)
    return g.integers(0, m, n), g.integers(0, m, n), m


def test_counts_weights():
    x, y, m = _data()
    c = O.Counts.from_xy(x, y, m)
    assert c.nxy.sum() == len(x) and np.array_equal(c.nx, np.bincount(x, minlength=m))
    assert np.isclose(c.p_hat().sum(), 1.0) and np.isclose(c.q_hat("u").sum(), 1.0) and np.isclose(c.q_hat("v").sum(), 1.0)
    assert np.all(c.q_hat("u") >= 0)                                   # N_x N_y >= N_xy
    with pytest.raises(ValueError):
        c.q_hat("w")


def test_table_j_equals_dense_pairs_u_and_v():
    x, y, m = _data()
    T = np.random.default_rng(1).uniform(-1, 1, (m, m))
    c = O.Counts.from_xy(x, y, m)
    Tp = torch.as_tensor(T[x][:, y])                                  # Tp[i, j] = T(x_i, y_j)
    assert abs(O.empirical_j(T, c, "u") - float(O.dense_u_terms_rows(Tp).j)) < 1e-13
    assert abs(O.empirical_j(T, c, "v") - float(O.dense_v_terms_rows(Tp).j)) < 1e-13


def test_empirical_j_keeps_negative_values():
    x, y, m = _data(seed=3)
    c = O.Counts.from_xy(x, y, m)
    T = np.full((m, m), 0.5)                                           # constant critic: U-J = -(0.5^2) < 0
    assert O.empirical_j(T, c, "u") == pytest.approx(-0.25, abs=1e-12)


def test_saturated_table_is_the_empirical_maximiser():
    x, y, m = _data(n=150, m=6, seed=4)
    c = O.Counts.from_xy(x, y, m)
    for mode in ("u", "v"):
        Ts = O.saturated_table(c, mode)
        assert np.all(np.abs(Ts) <= 1)
        best = O.empirical_j(Ts, c, mode)
        g = np.random.default_rng(5)
        for _ in range(50):
            assert O.empirical_j(np.clip(Ts + g.normal(0, 0.05, Ts.shape), -1, 1), c, mode) <= best + 1e-15
        den = c.p_hat() + c.q_hat(mode)
        assert np.all(Ts[den == 0] == 0)


def test_mf_u_moments_equal_table_and_dense():
    x, y, m = _data(n=120, m=4, seed=6)
    model, _ = Mo.build_critic("MF_VCS_M0", "two_block", m, 0.25, 3, 1, dtype=torch.float64)
    with torch.no_grad():
        model.W.copy_(torch.as_tensor(np.random.default_rng(2).uniform(-1, 1, (3, 3))))
    X, Y = Mo.onehot(x, m, dtype=torch.float64), Mo.onehot(y, m, dtype=torch.float64)
    A, B = model.features(X, Y)
    t = O.mf_u_moments(A, B, model.W)
    T = Mo.critic_table(model, m).numpy()
    j = float(t.j.detach())
    assert abs(j - O.empirical_j(T, O.Counts.from_xy(x, y, m), "u")) < 1e-13
    assert abs(j - float(O.dense_u_terms_rows(A @ model.W @ B.T).j.detach())) < 1e-13
    s = O.mf_u_scene(A, B)
    assert s.q_mode == "u"
    h = MF.QuadraticHead([s], weights=[1.0])                          # the M1 snapshot of the same U objective
    assert abs(float(h.value_and_gradient(model.W.detach())[0]) - j) < 1e-12


def test_derangement_terms_validate_shapes():
    x, y, m = _data(n=10, m=3)
    model, _ = Mo.build_critic("MLP_VCS_perm1", "two_block", m, 0.25, None, 1, dtype=torch.float64)
    X, Y = Mo.onehot(x, m, dtype=torch.float64), Mo.onehot(y, m, dtype=torch.float64)
    with pytest.raises(ValueError):
        O.mlp_derangement_terms(model, X, Y, torch.zeros(9, dtype=torch.long))
    with pytest.raises(ValueError):
        O.mlp_derangement_terms(model, X, Y, torch.zeros(1, 9, dtype=torch.long))
