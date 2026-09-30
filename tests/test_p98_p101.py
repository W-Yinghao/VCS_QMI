"""P98 / P101 checks: noise scale (dot-product variance with the fourth-order term), calibration stops at tolerance, fixed PCA basis, role
disjointness, common-random-number noise, and the P98 FIT-internal hold-out never touching the selection split."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vcs_estim.frozen import Pairs  # noqa: E402
from vcs_estim.observation_frozen import (TARGETS, add_noise, calibrate, noise_fields, noisy_pairs, o_line_roles,  # noqa: E402
                                          pca_basis)


def test_dot_product_noise_variance_includes_fourth_order_term():
    d, tau, n = 128, 1.0, 200_000
    g = torch.Generator().manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(1, d, generator=g, dtype=torch.float64), dim=1)
    w = torch.nn.functional.normalize(torch.randn(1, d, generator=g, dtype=torch.float64), dim=1)
    u = add_noise(z.expand(n, d).contiguous(), tau, "iso", torch.Generator().manual_seed(1))
    v = add_noise(w.expand(n, d).contiguous(), tau, "iso", torch.Generator().manual_seed(2))
    var = float(((u * v).sum(-1) - (z * w).sum()).var())
    pred = (2 * tau ** 2 + tau ** 4) / d
    se = pred * math.sqrt(2 / n) * 3          # generous MC tolerance (variance estimator + kurtosis)
    assert abs(var - pred) < 3 * se + 0.01 * pred, (var, pred)
    f = noise_fields(tau, "iso", d, None); assert f["noise_total_rms"] == tau and abs(f["noise_coordinate_sd"] - tau / math.sqrt(d)) < 1e-15
    assert f["calibration_target"] == "J_proxy" and f["renormalised_after_noise"] is False


def test_subspace_noise_rms_and_fixed_basis():
    g = torch.Generator().manual_seed(3); d = 32
    z = torch.nn.functional.normalize(torch.randn(4000, d, generator=g) @ torch.diag(torch.linspace(3, 0.1, d)), dim=1)
    B1, i1 = pca_basis(z); B2, i2 = pca_basis(z.clone())
    assert i1["basis_sha256"] == i2["basis_sha256"] and i1["r"] <= 64 and i1["explained"] >= 0.9 - 1e-9
    assert torch.allclose(B1.T @ B1, torch.eye(i1["r"]), atol=1e-5)
    r = i1["r"]; tau = 0.6
    u = add_noise(z, tau, "pca", torch.Generator().manual_seed(4), B1); e = u - z
    assert abs(float(e.pow(2).sum(1).mean()) - tau ** 2) < 0.02 * tau ** 2     # total RMS tau inside the subspace
    resid = e - (e @ B1) @ B1.T; assert float(resid.abs().max()) < 1e-5            # noise lies in the fixed basis
    # the same basis is reused for every tau (the function takes it as an argument; nothing is recomputed per tau)
    u2 = add_noise(z, 1.5, "pca", torch.Generator().manual_seed(4), B1); assert float(((u2 - z) - ((u2 - z) @ B1) @ B1.T).abs().max()) < 1e-5


def test_common_random_numbers_across_checkpoints():
    P = Pairs(*(torch.zeros(8, 4) for _ in range(4)))
    a, b = noisy_pairs(P, 0.3, "iso", "EVAL"), noisy_pairs(P, 0.3, "iso", "EVAL")
    assert torch.equal(a.xp, b.xp) and torch.equal(a.yq, b.yq) and not torch.equal(a.xp, a.yp)
    c = noisy_pairs(P, 0.3, "iso", "FIT-CAL"); assert not torch.equal(a.xp, c.xp)


def test_calibration_stops_at_tolerance_and_records_unreachable():
    calls = []

    def j(t):
        calls.append(t); return 0.99 * math.exp(-0.8 * t)          # monotone decreasing proxy
    res = calibrate(j, targets=(0.95, 0.70, 0.50, 0.01), tol=0.03, max_refine=10)
    for tgt in (0.95, 0.70, 0.50):
        r = res["targets"][str(tgt)]; assert r["reachable"] and abs(r["J"] - tgt) <= 0.03 and r["refinements"] <= 10
    u = res["targets"]["0.01"]; assert u["reachable"] is False and u["tau"] is None      # outside the grid's J range: not extended
    assert max(calls) <= 2.0 + 1e-12                                                       # never evaluated beyond the grid
    # a non-monotone curve is kept raw (no smoothing): the grid curve is returned as measured
    res2 = calibrate(lambda t: 0.9 - 0.3 * abs(math.sin(3 * t)), targets=(0.7,), tol=0.03, max_refine=10)
    assert [c[1] for c in res2["grid_curve"]] == [0.9 - 0.3 * abs(math.sin(3 * t)) for t, _ in res2["grid_curve"]]


def test_roles_disjoint_and_fit_split_halves():
    uids = list(range(45000))
    r = o_line_roles(uids)
    sets = {k: set(v) for k, v in r.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (sets[names[i]] & sets[names[j]]), (names[i], names[j])
    assert len(r["FIT-CRITIC"]) == len(r["FIT-CAL"]) == 13500 and len(r["EVAL"]) == 6000
    assert o_line_roles(uids) == r                                  # deterministic


def test_p98_holdout_is_fit_internal():
    from probe_robustness import HOLDOUT_FRAC, holdout_split
    tr, ho = holdout_split(45000)
    assert len(set(tr) & set(ho)) == 0 and len(tr) + len(ho) == 45000 and abs(len(ho) - HOLDOUT_FRAC * 45000) <= 1
    assert tr.max() < 45000 and ho.max() < 45000                    # indices into FIT feature rows only (selection rows are a separate tensor)
    tr2, ho2 = holdout_split(45000); assert np.array_equal(ho, ho2)
