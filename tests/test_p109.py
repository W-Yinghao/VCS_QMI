"""P109 unit tests: X1 measurement family, I1 audit statistics / max-statistic, I2 nesting and exact enumeration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))

from vcs_measure import nested as NS  # noqa: E402
from vcs_measure.common import split_base_ids  # noqa: E402
from vcs_measure.xmeasure import PairSet, j_value, measure  # noqa: E402


def _pairs(n, d, dep, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g); e = torch.randn(n, d, generator=g)
    return torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(dep * x + e, dim=1)


def test_split_disjoint_and_deterministic():
    u = np.arange(1000)
    a = split_base_ids(u, {"fit": 500, "tune": 200, "eval": 300}, 7); b = split_base_ids(u, {"fit": 500, "tune": 200, "eval": 300}, 7)
    assert all((a[k] == b[k]).all() for k in a)
    s = [set(v.tolist()) for v in a.values()]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])


def test_pairset_q_never_same_row_or_cross_block():
    u1, u2 = _pairs(40, 8, 1.0, 0)
    ps = PairSet(torch.cat([u1, u1]), torch.cat([u2, u2]), 4, 0, block=40)
    assert (ps.qa != ps.qb).all()
    assert ((ps.qa // 40) == (ps.qb // 40)).all()           # Q partners stay inside one augmentation block
    assert ((ps.qa % 40) != (ps.qb % 40)).all()             # never the same base image


def test_measure_zero_candidate_and_dependence_detected():
    f, t, e = (PairSet(*_pairs(600, 8, 3.0, s), 4, s) for s in (1, 2, 3))
    m = measure(f, t, e, mlp_steps=100)
    assert set(m["candidates"]) == {"cosine", "mlp", "prod_ridge", "rff_ridge", "zero"}
    assert m["picked"] != "zero" and m["picked_eval_J"] > 0.05
    f0, t0, e0 = (PairSet(*_pairs(600, 8, 0.0, s), 4, s) for s in (4, 5, 6))
    m0 = measure(f0, t0, e0, mlp_steps=100)
    assert m0["candidates"]["zero"]["eval_J"] == 0.0
    assert m0["picked_eval_J"] < 0.05                       # no spurious dependence; may be < 0 and is not clipped


def test_j_value_bounds():
    tp = torch.ones(10); tq = -torch.ones(10)
    assert abs(j_value(tp, tq) - 1.0) < 1e-6 and j_value(torch.zeros(5), torch.zeros(5)) == 0.0


def test_weighted_readout_equals_unweighted_for_m1():
    rng = np.random.default_rng(0); tp, tq = rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50)
    w = np.ones((50, 1))
    assert abs(NS.j_w(tp, tq[:, None], w) - (np.mean(tp - tp ** 2 / 2) + np.mean(-tq - tq ** 2 / 2))) < 1e-12


def test_exact_enumeration_equals_expectation():
    # Q term with p1 enumerated equals the Monte-Carlo average over many sampled n'
    rng = np.random.default_rng(1); n = 200; p1 = rng.uniform(0.2, 0.8, n); t0, t1 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    ex = NS.j_w(np.zeros(n), np.stack([t0, t1], 1), np.stack([1 - p1, p1], 1))
    draws = rng.random((20000, n)) < p1
    mc = np.mean([np.mean(np.where(d, -t1 - t1 ** 2 / 2, -t0 - t0 ** 2 / 2)) for d in draws[:2000]])
    assert abs(ex - mc) < 5e-3


def test_nested_critic_contains_coarse_at_step0():
    from vcs_estim.increments import YPairLinear
    torch.manual_seed(0)
    coarse = YPairLinear(3, 4); W = torch.randn(6, 3)
    nested = NS.NestedCritic(coarse, lambda z: z @ W, YPairLinear(6, 4))
    z = torch.randn(20, 6); n = torch.randint(0, 2, (20,)); y = torch.randint(0, 4, (20,))
    assert torch.allclose(nested(z, n, y), coarse(z @ W, n, y))
    assert not any(p.requires_grad for p in nested.coarse.parameters())


def test_nested_fit_never_below_coarse_on_val():
    from vcs_estim.increments import YPairLinear
    torch.manual_seed(0); n = 400
    z = torch.randn(n, 6); y = torch.randint(0, 4, (n,)); nn_ = (torch.rand(n) < 0.5).long(); neg = {"n_neg": (torch.rand(n) < 0.5).long()}
    W = torch.randn(6, 3)
    coarse = NS.fit(z @ W, nn_, y, neg, objective="vcs", kind="linear", n_classes=4, steps=30, seed=1)
    fine = NS.fit(z, nn_, y, neg, objective="vcs", kind="linear", n_classes=4, steps=30, seed=1, coarse=coarse, cmap=lambda x: x @ W)
    # same VAL split (same seed): the nested fit can always return step 0 = the coarse model
    assert fine.val_J >= coarse.val_J - 1e-6


def test_audit_maxT_adjusted_p_not_below_raw():
    from vcs_measure import audit as A

    class D:
        pass
    rng = np.random.default_rng(0); n_all = 900; d = D()
    d.y = rng.integers(0, 10, n_all); d.part = np.repeat([0, 1, 2], 300)
    d.idx = {k: np.where(d.part == i)[0] for i, k in enumerate(("fit", "val", "eval"))}
    feats = {l: torch.randn(n_all, 6) for l in A.AUDIT_LAYERS}
    planted = {l: f + 0.8 * torch.ones(6) for l, f in feats.items()}
    d.F = {"clean": feats, "colour_s0.2": planted}
    d.feats = lambda v, l, rows: d.F[v][l][torch.as_tensor(rows)].float()
    out = A.run_repeat(d, 200, rng, mode="planted", version="colour_s0.2", perms=50, seed=3, device="cpu")
    for l in A.AUDIT_LAYERS:
        for s in ("vcs_closed", "js_exact"):
            r = out["layers"][l][s]; assert r["p_maxT"] >= r["p"] - 1e-12
    assert out["vcs_closed_any_layer_reject_maxT"]          # a strong planted shift is detected
