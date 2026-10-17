"""Uniform derangements: exhaustive small-n identities (mean over all derangements = complete U, values and gradients), the sampler
law, perm8 = mean of its perm1 parts, and a negative control showing that the law check detects a fixed-point-repair sampler."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats

import t03_gates as G
import t03_models as Mo
import t03_objectives as O


def test_counts_of_derangements():
    for n in range(2, 9):
        assert len(O.all_derangements(n)) == O.derangement_count(n)
    assert O.derangement_count(4) == 9 and O.derangement_count(5) == 44 and O.derangement_count(6) == 265
    assert abs(O.expected_attempts(20) - np.e) < 1e-12


def test_sampler_never_returns_fixed_points_and_is_reproducible():
    a = O.DerangementStream(np.random.default_rng(5), 257).draw(200)
    b = O.DerangementStream(np.random.default_rng(5), 257).draw(200)
    assert np.array_equal(a, b) and O.is_derangement_matrix(a, 257)
    s = O.DerangementStream(np.random.default_rng(6), 3)
    s.draw(3000)
    assert s.draws == 3000 and s.attempts >= 3000                    # n = 3: D_3 = 2 of 6, about 3 attempts per draw
    assert abs(s.attempts / s.draws - 3.0) < 0.2
    with pytest.raises(ValueError):
        O.sample_derangement(np.random.default_rng(0), 1)


def test_enumeration_mean_equals_complete_U_values_and_gradients():
    res = G.gate_enumeration(ns=(4, 5))
    assert res["status"] == "PASS", res
    for r in res["rows"]:
        assert r["derangement_mean_value_err"] < 1e-12 and r["derangement_mean_grad_err"] < 1e-12
        assert r["permutation_mean_minus_V_value_err"] < 1e-12       # control: randperm averages to V ...
        assert abs(r["permutation_mean_minus_U_value"]) > 1e-6       # ... which is not U


def test_sampler_law_gate_and_perm8_and_table_gates():
    assert G.gate_sampler_law(cases=((4, 9000),))["status"] == "PASS"
    assert G.gate_perm8()["status"] == "PASS"
    assert G.gate_table_u()["status"] == "PASS"
    r = G.gate_rejection_rate(n=4096, draws=300)
    assert r["status"] == "PASS"


def _repair_sampler(rng, n):
    """A biased 'fixed-point repair' sampler (NOT used anywhere): swap each fixed point with its right neighbour."""
    s = rng.permutation(n)
    for i in range(n):
        if s[i] == i:
            j = (i + 1) % n
            s[i], s[j] = s[j], s[i]
    return s if not np.any(s == np.arange(n)) else _repair_sampler(rng, n)


def test_law_check_detects_a_repair_sampler():
    n, draws = 4, 9000
    rng = np.random.default_rng(11)
    ders = O.all_derangements(n)
    idx = {tuple(d): i for i, d in enumerate(ders)}
    freq = np.bincount([idx[tuple(_repair_sampler(rng, n))] for _ in range(draws)], minlength=len(ders))
    assert stats.chisquare(freq).pvalue < 1e-3                        # the gate's pre-set threshold rejects it


def test_mf_and_mlp_derangement_terms_average_over_rows():
    torch.manual_seed(0)
    n, m, K = 40, 4, 3
    model, x, y = G._fixture("MF", n, m, K, 3)
    X, Y = (torch.nn.functional.one_hot(torch.as_tensor(v), m).double() for v in (x, y))
    sig = torch.as_tensor(O.DerangementStream(np.random.default_rng(1), n).draw(5))
    A, B = model.features(X, Y)
    j5 = O.mf_derangement_terms(A, B, model.W, sig).j
    js = torch.stack([O.mf_derangement_terms(A, B, model.W, sig[r:r + 1]).j for r in range(5)])
    assert torch.allclose(j5, js.mean(), atol=1e-14, rtol=0)
    tab = Mo.critic_table(model, m).numpy()
    est = O.perm_estimate_from_table(tab, x, y, sig.numpy())
    assert np.allclose(est, [float(O.mf_derangement_terms(A, B, model.W, sig[r:r + 1]).j.detach()) for r in range(5)], atol=1e-13, rtol=0)
