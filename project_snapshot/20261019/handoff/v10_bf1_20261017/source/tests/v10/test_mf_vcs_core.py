"""Port of the package's core/verify_numpy.py checks to pytest, run against the unchanged torch backend src/v10/mf_vcs.py.

The numpy dense pair-feature oracle, finite differences and SciPy box-QP of verify_numpy.py are reproduced here (fp64, CPU,
small sizes, seconds).  These tests do not replace T01 (real host batch, real warp chain, CUDA): they pin the backend copy
and its mathematics on the login node.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from v10 import mf_vcs as M

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "v10" / "package_v10_server_v1_20261016"
RNG = np.random.default_rng(20261016)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def make_scene(n, r, s, mode):
    za, zb = RNG.normal(size=(n, r)), RNG.normal(size=(n, s))
    a, b = softmax(za), softmax(zb)
    if mode == "independent":
        qa, qb = softmax(RNG.normal(size=(n + 1, r))), softmax(RNG.normal(size=(n + 3, s)))
    else:
        qa, qb = a, b
    return dict(a=a, b=b, qa=qa, qb=qb, mode=mode, za=za, zb=zb)


def dense_features(s):
    p = np.einsum("ir,is->irs", s["a"], s["b"]).reshape(len(s["a"]), -1)
    q = np.einsum("ir,js->ijrs", s["qa"], s["qb"])
    if s["mode"] == "u":
        q = q[~np.eye(len(s["a"]), dtype=bool)]
    return p, q.reshape(-1, p.shape[1])


def oracle_terms(s, w):
    p, q = dense_features(s)
    tp, tq = p @ w.ravel(), q @ w.ravel()
    return np.array([tp.mean(), (tp * tp).mean(), tq.mean(), (tq * tq).mean()])


def oracle_objective(scenes, weights, w, ridge=0.0):
    p1, p2, q1, q2 = sum(wt * oracle_terms(s, w) for s, wt in zip(scenes, weights))
    return p1 - q1 - (p2 + q2) / 2 - ridge * np.sum(w * w) / 2


def oracle_quadratic(scenes, weights, ridge):
    k = scenes[0]["a"].shape[1] * scenes[0]["b"].shape[1]
    delta, h = np.zeros(k), ridge * np.eye(k)
    for s, wt in zip(scenes, weights):
        p, q = dense_features(s)
        delta += wt * (p.mean(0) - q.mean(0))
        h += wt * (p.T @ p / len(p) + q.T @ q / len(q))
    return delta, h


def to_batch(s, dtype=torch.float64):
    t = lambda x: torch.tensor(x, dtype=dtype)  # noqa: E731
    if s["mode"] == "independent":
        return M.SceneBatch(t(s["a"]), t(s["b"]), "independent", t(s["qa"]), t(s["qb"]))
    return M.SceneBatch(t(s["a"]), t(s["b"]), s["mode"])


def test_backend_copy_is_the_package_file():
    manifest = json.loads((PACKAGE / "MANIFEST_SHA256.json").read_text())
    ours = hashlib.sha256((REPO / "src" / "v10" / "mf_vcs.py").read_bytes()).hexdigest()
    assert ours == manifest["core/mf_vcs.py"] == hashlib.sha256((PACKAGE / "core" / "mf_vcs.py").read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["v", "u", "independent"])
@pytest.mark.parametrize("dtype,limit", [(torch.float64, 2e-13), (torch.float32, 3e-6)])
def test_moments_and_dense_equal_pair_feature_oracle(mode, dtype, limit):
    for n, r, s in ((2, 2, 3), (5, 4, 2), (17, 7, 6)):
        sc = make_scene(n, r, s, mode)
        w = RNG.uniform(-1, 1, (r, s))
        ref = oracle_terms(sc, w)
        wt = torch.tensor(w, dtype=dtype)
        for fn in (M.objective_terms, M.dense_objective_terms):
            terms = fn([to_batch(sc, dtype)], wt, weights=[1.0])
            got = np.array([float(terms.p1), float(terms.p2), float(terms.q1), float(terms.q2)])
            assert np.max(np.abs(got - ref)) <= limit


@pytest.mark.parametrize("mode", ["v", "u", "independent"])
def test_hvp_and_w_gradient_match_dense_quadratic(mode):
    sc = make_scene(5, 3, 2, mode)
    w = RNG.uniform(-0.7, 0.7, (3, 2))
    delta, h = oracle_quadratic([sc], [1.0], 0.07)
    head = M.QuadraticHead([to_batch(sc)], weights=[1.0], ridge=0.07)
    d = RNG.normal(size=w.shape)
    assert np.max(np.abs(head.hvp(torch.tensor(d)).numpy().ravel() - h @ d.ravel())) <= 2e-13
    assert max(0.0, -np.linalg.eigvalsh(h).min()) <= 2e-13
    wt = torch.tensor(w, requires_grad=True)
    f = M.objective_terms([to_batch(sc)], wt, weights=[1.0], ridge=0.07).f
    g = torch.autograd.grad(f, wt)[0].numpy().ravel()
    assert np.max(np.abs(g - (delta - h @ w.ravel()))) <= 2e-12


@pytest.mark.parametrize("mode", ["v", "u", "independent"])
def test_logit_gradients_match_finite_differences(mode):
    sc = make_scene(5, 3, 2, mode)
    w = RNG.uniform(-0.7, 0.7, (3, 2))
    for side in ("a", "b"):
        z = torch.tensor(sc["z" + side], requires_grad=True)
        feats = {k: torch.tensor(sc[k]) for k in ("a", "b", "qa", "qb")}
        feats[side] = z.softmax(-1)
        if mode != "independent":
            feats["q" + side] = feats[side]
        batch = (M.SceneBatch(feats["a"], feats["b"], "independent", feats["qa"], feats["qb"]) if mode == "independent"
                 else M.SceneBatch(feats["a"], feats["b"], mode))
        f = M.objective_terms([batch], torch.tensor(w), weights=[1.0], ridge=0.07).f
        g = torch.autograd.grad(f, z)[0].numpy()

        def value(zz):
            ed = dict(sc)
            ed[side] = softmax(zz)
            if mode != "independent":
                ed["q" + side] = ed[side]
            return oracle_objective([ed], [1.0], w, 0.07)
        fd = np.empty_like(sc["z" + side])
        for idx in np.ndindex(fd.shape):
            lo, hi = sc["z" + side].copy(), sc["z" + side].copy()
            lo[idx] -= 1e-6
            hi[idx] += 1e-6
            fd[idx] = (value(hi) - value(lo)) / 2e-6
        assert np.max(np.abs(g - fd)) <= 2e-9


def test_fixed_scene_weights_and_pooling_counterexample():
    scenes = [make_scene(4, 3, 2, "v"), make_scene(7, 3, 2, "u"), make_scene(3, 3, 2, "independent")]
    weights, w = [0.2, 0.3, 0.5], RNG.uniform(-0.8, 0.8, (3, 2))
    got = float(M.objective_terms([to_batch(s) for s in scenes], torch.tensor(w), weights=weights, ridge=0.11).f)
    assert abs(got - oracle_objective(scenes, weights, w, 0.11)) <= 2e-13
    hard1, hard2 = np.tile([1.0, 0.0], (2, 1)), np.tile([0.0, 1.0], (3, 1))
    sw = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    local = M.objective_terms([M.SceneBatch(torch.tensor(x), torch.tensor(x)) for x in (hard1, hard2)], sw, weights=[0.4, 0.6])
    pooled_x = torch.tensor(np.concatenate((hard1, hard2)))
    pooled = M.objective_terms([M.SceneBatch(pooled_x, pooled_x)], sw, weights=[1.0])
    assert abs(float(local.j) - float(pooled.j)) > 0.5


def test_ordinary_permutations_average_to_v_and_derangements_to_u():
    sc = make_scene(5, 3, 2, "v")
    w = RNG.uniform(-1, 1, (3, 2))
    t = sc["a"] @ w @ sc["b"].T
    pos = np.diag(t)
    vals, offv = [], []
    for perm in itertools.permutations(range(5)):
        neg = t[np.arange(5), perm]
        f = pos.mean() - 0.5 * np.mean(pos ** 2) - neg.mean() - 0.5 * np.mean(neg ** 2)
        vals.append(f)
        if all(i != j for i, j in enumerate(perm)):
            offv.append(f)
    wt = torch.tensor(w)
    assert abs(np.mean(vals) - float(M.objective_terms([to_batch(sc)], wt, weights=[1.0]).j)) <= 2e-13
    assert abs(np.mean(offv) - float(M.objective_terms([to_batch(dict(sc, mode="u"))], wt, weights=[1.0]).j)) <= 2e-13
    perms = torch.tensor(list(itertools.permutations(range(5))))
    sampled = M.sampled_permutation_terms([to_batch(sc)], wt, weights=[1.0], permutations=[perms])
    assert abs(float(sampled.j) - np.mean(vals)) <= 2e-13
    assert np.var(vals) > 1e-8


def scipy_solve(delta, h):
    res = minimize(lambda x: (0.5 * x @ h @ x - delta @ x, h @ x - delta), np.zeros_like(delta), jac=True, method="L-BFGS-B",
                   bounds=[(-1.0, 1.0)] * len(delta), options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 50000, "maxls": 100})
    return res.x


@pytest.mark.parametrize("ridge", [0.0, 0.05])
def test_box_solver_matches_scipy_and_gap_bounds_improvement(ridge):
    cases = {"weighted_mixed": ([make_scene(4, 3, 2, "v"), make_scene(7, 3, 2, "u"), make_scene(3, 3, 2, "independent")], [0.2, 0.3, 0.5]),
             "active_box": ([dict(a=np.eye(2), b=np.eye(2), qa=np.eye(2), qb=np.eye(2), mode="u")], [1.0])}
    for scenes, weights in cases.values():
        delta, h = oracle_quadratic(scenes, weights, ridge)
        head = M.QuadraticHead([to_batch(s) for s in scenes], weights=weights, ridge=ridge)
        sol = M.solve_box_head(head, tolerance=1e-9, max_iterations=30000)
        assert sol.converged and sol.fw_gap <= 1e-9 and float(sol.w.abs().max()) <= 1.0
        x = scipy_solve(delta, h)
        f_oracle = delta @ x - 0.5 * x @ h @ x
        assert abs(sol.f - f_oracle) <= 3e-8
        w0 = RNG.uniform(-1, 1, len(delta))
        f0 = delta @ w0 - 0.5 * w0 @ h @ w0
        g0 = delta - h @ w0
        gap0 = np.where(g0 >= 0, g0 * (1 - w0), -g0 * (1 + w0)).sum()
        assert f_oracle - f0 <= gap0 + 2e-13


def test_m1_ridge_profile_envelope_and_strict_rejection():
    base = make_scene(7, 3, 2, "v")
    dz = torch.tensor(RNG.normal(size=base["za"].shape))
    za = torch.tensor(base["za"])
    b = torch.tensor(base["b"])

    def profile(theta):
        sc = M.SceneBatch((za + theta * dz).softmax(-1), b, "v")
        return M.profiled_objective([sc], weights=[1.0], ridge=0.2, tolerance=1e-11, max_iterations=30000)
    theta = torch.zeros((), dtype=torch.float64, requires_grad=True)
    env = torch.autograd.grad(profile(theta).terms.f, theta)[0]
    h = 1e-4
    fd = (profile(torch.tensor(h, dtype=torch.float64)).terms.f - profile(torch.tensor(-h, dtype=torch.float64)).terms.f) / (2 * h)
    assert abs(float(env - fd)) <= 2e-7
    hard = torch.eye(2, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        M.profiled_objective([M.SceneBatch(hard, hard, "u")], weights=[1.0], max_iterations=0)
    loose = M.profiled_objective([M.SceneBatch(hard, hard, "u")], weights=[1.0], max_iterations=0, require_converged=False)
    assert not loose.solution.converged and loose.solution.status == "iteration_budget_exhausted"


def test_contract_rejections():
    a = torch.tensor(softmax(RNG.normal(size=(9, 3))))
    b = torch.tensor(softmax(RNG.normal(size=(9, 2))))
    w = torch.zeros(3, 2, dtype=torch.float64)
    grouped = M.partition_scenes(a, b, torch.tensor([4, 4, 4, 8, 8, 8, 8, 8, 8]), q_mode="v")
    for fn in (lambda: M.objective_terms(grouped, w, weights=[-0.2, 1.2]), lambda: M.objective_terms(grouped, w, weights=[1.0, 1.0]),
               lambda: M.objective_terms(grouped, w, weights=torch.tensor([0.3, 0.7])), lambda: M.SceneBatch(a, b, forbidden_edges=torch.ones(9, 9)),
               lambda: M.SceneBatch(a[:1], b[:1], "u"), lambda: M.SceneBatch(a, b, "independent"),
               lambda: M.objective_terms(grouped, torch.full_like(w, 1.01), weights=[0.3, 0.7]), lambda: M.SceneBatch(-a, b),
               lambda: M.SceneBatch(2 * a, b), lambda: M.SceneBatch(a.half(), b.half())):
        with pytest.raises((ValueError, NotImplementedError)):
            fn()
    assert [s.scene_id for s in grouped] == ["4", "8"] and [len(s.a) for s in grouped] == [3, 6]
