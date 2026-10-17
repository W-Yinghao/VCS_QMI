"""The M1 relative zero-head rule (decision 2026-10-17 2.1 / 2.3) as T03 calls it - src/v10/adapter.mf_objective('MF_M1') on U-mode
scenes - against the five targeted acceptance items of protocol P2 m1_rule.targeted_acceptance_new, replicated in U mode on one-hot
features (fp32, the training dtype).  These do not replace the BF T01 rows; they pin the T03 call path."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import t03_models as Mo
import t03_objectives as O
from v10 import adapter as A
from v10 import mf_vcs as MF


def _features(seed=0, n=256, m=8, K=6):
    g = np.random.default_rng(seed)
    x = g.integers(0, m, n)
    y = np.where(g.random(n) < 0.6, x, g.integers(0, m, n))                  # dependent pairs: y = x for about 60 %
    model, _ = Mo.build_critic("MF_VCS_M1", "diagonal_mixture", m, 0.75, K, 1017)
    model.W.requires_grad_(False)
    A_, B_ = model.features(Mo.onehot(x, m), Mo.onehot(y, m))
    return model, A_, B_


def _m1(scene, W, cap):
    return A.mf_objective("MF_M1", [scene], W, [1.0], m1=A.M1Config(max_iterations=cap))


def test_zero_head_start_iterates_and_trains_the_maps():
    model, A_, B_ = _features()
    terms, info = _m1(O.mf_u_scene(A_, B_), model.W, 30)
    assert info["m1_rule"] == "relative_zero_head" and info["m1_W_start_is_zero"] and info["m1_gap_ref_zero"] > 0
    assert info["m1_gap_start"] == info["m1_gap_ref_zero"]                     # from W = 0 the start gap is G(0) itself
    assert info["m1_iterations"] > 0 and info["m1_W_update_absmax"] > 0 and float(terms.j.detach()) > 0
    (-terms.f).backward()
    assert A.grad_l2(list(model.left.parameters()) + list(model.right.parameters())) > 0


def test_warm_start_gap_differs_from_the_zero_head_reference():
    model, A_, B_ = _features()
    _, info1 = _m1(O.mf_u_scene(A_.detach(), B_.detach()), model.W, 30)
    _, A2, B2 = _features(seed=1)                                             # next batch's features
    W1 = info1["m1_W"]
    _, info2 = _m1(O.mf_u_scene(A2.detach(), B2.detach()), W1, 30)
    head = MF.QuadraticHead([O.mf_u_scene(A2.detach(), B2.detach())], weights=[1.0])
    assert not info2["m1_W_start_is_zero"]
    assert info2["m1_gap_ref_zero"] == float(head.delta.abs().sum())           # G_b(0) of the CURRENT batch
    assert info2["m1_gap_start"] == pytest.approx(float(head.fw_gap(W1)), rel=1e-6) and info2["m1_gap_start"] != info2["m1_gap_ref_zero"]
    assert info2["m1_tolerance_effective"] == 0.01 * info2["m1_gap_ref_zero"]


def test_exactly_zero_reference_returns_w0_without_solver():
    n, K = 16, 4
    a = torch.zeros(n, K)
    a[:, 0] = 1.0                                                              # identical one-hot rows: Delta = 0 exactly in U mode
    scene = O.mf_u_scene(a, a.clone())
    head = MF.QuadraticHead([scene], weights=[1.0])
    assert not bool(head.delta.any())
    called = {"n": 0}
    orig = MF.solve_box_head

    def spy(*args, **kw):
        called["n"] += 1
        return orig(*args, **kw)
    MF.solve_box_head = spy
    try:
        terms, info = _m1(scene, torch.zeros(K, K), 100)
    finally:
        MF.solve_box_head = orig
    assert called["n"] == 0 and info["m1_stop_reason"] == "zero_reference_gap" and info["m1_converged"] is True
    assert info["m1_tolerance_effective"] is None and info["m1_gap_relative_to_zero"] is None and not bool(info["m1_W"].any())
    with pytest.raises(ValueError):
        MF.solve_box_head(head, tolerance=0.0)                                # the solver itself refuses a zero tolerance


def test_cap_reached_returns_the_approximate_head_with_its_true_gap():
    model, A_, B_ = _features()
    scene = O.mf_u_scene(A_.detach(), B_.detach())
    _, info = _m1(scene, model.W, 1)
    assert info["m1_iterations"] == 1 and info["m1_stop_reason"] == "iteration_budget_exhausted" and info["m1_converged"] is False
    head = MF.QuadraticHead([scene], weights=[1.0])
    assert info["m1_gap_final"] == pytest.approx(float(head.fw_gap(info["m1_W"])), rel=1e-6)
    assert info["m1_gap_final"] > info["m1_tolerance_effective"]


def test_hvp_and_moment_counts_are_actual_counts():
    model, A_, B_ = _features()
    scene = O.mf_u_scene(A_.detach(), B_.detach())
    calls = {"hvp": 0}
    orig = MF.QuadraticHead._hvp

    def counted(self, d):
        calls["hvp"] += 1
        return orig(self, d)
    MF.QuadraticHead._hvp = counted
    try:
        _, info = _m1(scene, model.W, 30)
    finally:
        MF.QuadraticHead._hvp = orig
    diag = info["m1_fp64_diagnostic"]
    assert calls["hvp"] == info["m1_hvp_calls"] + (diag["hvp_calls"] if diag.get("triggered") else 0)
    assert info["m1_moments_passes"] == 2
    assert info["m1_hvp_calls"] == 2 + 3 * info["m1_iterations"] + info["m1_momentum_restarts"]
