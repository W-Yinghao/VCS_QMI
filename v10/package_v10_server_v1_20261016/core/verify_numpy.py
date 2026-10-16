"""Independent CPU mathematics audit; DOES NOT execute mf_vcs.py/PyTorch.

Dense pair features form the oracle. Moment formulas, analytic feature
derivatives, finite differences, and SciPy box-QP solves are cross-checked.
No neural training, GPU execution, timing claim, or data download is involved.
"""
from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261016)
CHECKS = []


def check(name, error, limit, note=""):
    row = {"name": name, "error": float(error), "limit": float(limit),
           "passed": bool(np.isfinite(error) and error <= limit), "note": note}
    CHECKS.append(row)
    if not row["passed"]:
        raise AssertionError(row)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def scene(n, r, s, mode, dtype=np.float64):
    za, zb = RNG.normal(size=(n, r)), RNG.normal(size=(n, s))
    a, b = softmax(za).astype(dtype), softmax(zb).astype(dtype)
    if mode == "independent":
        qa = softmax(RNG.normal(size=(n + 1, r))).astype(dtype)
        qb = softmax(RNG.normal(size=(n + 3, s))).astype(dtype)
    else:
        qa, qb = a, b
    return dict(a=a, b=b, qa=qa, qb=qb, mode=mode, za=za, zb=zb)


def dense_features(s):
    p = np.einsum("ir,is->irs", s["a"], s["b"]).reshape(len(s["a"]), -1)
    q = np.einsum("ir,js->ijrs", s["qa"], s["qb"])
    if s["mode"] == "u":
        q = q[~np.eye(len(s["a"]), dtype=bool)]
    return p, q.reshape(-1, p.shape[1])


def dense_terms(s, w):
    p, q = dense_features(s)
    tp, tq = p @ w.ravel(), q @ w.ravel()
    return np.array([tp.mean(), (tp * tp).mean(), tq.mean(), (tq * tq).mean()])


def moment_terms(s, w):
    a, b, qa, qb = (s[k] for k in ("a", "b", "qa", "qb"))
    t = np.einsum("ir,rs,is->i", a, w, b)
    q1 = qa.sum(0) @ w @ qb.sum(0)
    q2 = np.sum(w * ((qa.T @ qa) @ w @ (qb.T @ qb)))
    d = len(qa) * len(qb)
    if s["mode"] == "u":
        q1 -= t.sum()
        q2 -= np.sum(t * t)
        d -= len(a)
    return np.array([t.mean(), np.mean(t * t), q1 / d, q2 / d])


def objective(scenes, weights, w, ridge=0., moments=False):
    terms = sum(weight * (moment_terms(s, w) if moments else dense_terms(s, w)) for s, weight in zip(scenes, weights))
    p1, p2, q1, q2 = terms
    return p1 - q1 - (p2 + q2) / 2 - ridge * np.sum(w * w) / 2


def dense_quadratic(scenes, weights, ridge):
    k = scenes[0]["a"].shape[1] * scenes[0]["b"].shape[1]
    delta, h = np.zeros(k), ridge * np.eye(k)
    for s, weight in zip(scenes, weights):
        p, q = dense_features(s)
        delta += weight * (p.mean(0) - q.mean(0))
        h += weight * (p.T @ p / len(p) + q.T @ q / len(q))
    return delta, h


def moment_hvp(scenes, weights, v, ridge):
    out = ridge * v
    for s, weight in zip(scenes, weights):
        a, b, qa, qb = (s[k] for k in ("a", "b", "qa", "qb"))
        t = np.einsum("ir,rs,is->i", a, v, b)
        p = a.T @ (b * t[:, None])
        q = (qa.T @ qa) @ v @ (qb.T @ qb)
        d = len(qa) * len(qb)
        if s["mode"] == "u":
            q -= p
            d -= len(a)
        out = out + weight * (p / len(a) + q / d)
    return out


def fd(fun, x, h=1e-6):
    out = np.empty_like(x)
    for index in np.ndindex(x.shape):
        lo, hi = x.copy(), x.copy()
        lo[index] -= h
        hi[index] += h
        out[index] = (fun(hi) - fun(lo)) / (2 * h)
    return out


def pairwise_feature_gradient(s, w):
    a, b = s["a"], s["b"]
    p = np.einsum("ir,rs,is->i", a, w, b)
    ga = ((1 - p)[:, None] * (b @ w.T)) / len(a)
    gb = ((1 - p)[:, None] * (a @ w)) / len(a)
    if s["mode"] != "independent":
        t = a @ w @ b.T
        coeff = 1 + t
        if s["mode"] == "u":
            np.fill_diagonal(coeff, 0)
            d = len(a) * (len(a) - 1)
        else:
            d = len(a) ** 2
        ga -= coeff @ (b @ w.T) / d
        gb -= coeff.T @ (a @ w) / d
    return ga, gb


def gap_of(delta, h, w):
    g = delta - h @ w
    return np.where(g >= 0, g * (1 - w), -g * (1 + w)).sum()


def scipy_solve(delta, h):
    result = minimize(lambda w: (.5 * w @ h @ w - delta @ w, h @ w - delta),
                      np.zeros_like(delta), jac=True, method="L-BFGS-B",
                      bounds=[(-1., 1.)] * len(delta),
                      options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 50000, "maxls": 100})
    return result.x, result


def projected_solver(delta, h, ridge, tolerance=1e-9, max_iterations=30000):
    # Independent NumPy evaluation of the production solver's algorithm;
    # H is explicit here, whereas mf_vcs.py uses only moment HVPs.
    lipschitz = (np.trace(h) - ridge * len(delta) + ridge) * (1 + 64 * np.finfo(float).eps)
    w = np.zeros_like(delta)
    y, momentum = w.copy(), 1.
    f = delta @ w - .5 * w @ h @ w
    for iteration in range(max_iterations + 1):
        g = delta - h @ w
        gap = gap_of(delta, h, w)
        if gap <= tolerance or iteration == max_iterations:
            return w, f, gap, iteration
        c = np.clip(y + (delta - h @ y) / lipschitz, -1, 1)
        fc = delta @ c - .5 * c @ h @ c
        if fc < f - 64 * np.finfo(float).eps * max(1., abs(f)):
            momentum = 1.
            c = np.clip(w + g / lipschitz, -1, 1)
            fc = delta @ c - .5 * c @ h @ c
        nxt = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        y = c + ((momentum - 1) / nxt) * (c - w)
        w, f, momentum = c, fc, nxt


def run():
    for filename in ("mf_vcs.py", "verify_torch.py"):
        ast.parse((ROOT / filename).read_text(encoding="utf-8"), filename=filename)
        check(f"static_ast_{filename}", 0, 0, "Parsed only; no torch import or execution")
    for dtype, limit in ((np.float64, 2e-13), (np.float32, 3e-6)):
        for mode in ("v", "u", "independent"):
            value_error = 0.
            for n, r, ss in ((2, 2, 3), (5, 4, 2), (17, 7, 6)):
                s = scene(n, r, ss, mode, dtype)
                w = RNG.uniform(-1, 1, (r, ss)).astype(dtype)
                value_error = max(value_error, np.max(np.abs(moment_terms(s, w) - dense_terms(s, w))))
            check(f"dense_moments_{mode}_{np.dtype(dtype).name}", value_error, limit)
    for mode in ("v", "u", "independent"):
        s = scene(5, 3, 2, mode)
        w = RNG.uniform(-.7, .7, (3, 2))
        delta, h = dense_quadratic([s], [1.], .07)
        direction = RNG.normal(size=w.shape)
        check(f"dense_hessian_vs_moment_hvp_{mode}", np.max(np.abs(moment_hvp([s], [1.], direction, .07).ravel() - h @ direction.ravel())), 2e-13)
        check(f"PSD_dense_hessian_{mode}", max(0., -np.linalg.eigvalsh(h).min()), 2e-13)
        fd_w = fd(lambda z: objective([s], [1.], z, .07, True), w)
        check(f"W_gradient_finite_difference_{mode}", np.max(np.abs(fd_w.ravel() - (delta - h @ w.ravel()))), 2e-9)
        ga, gb = pairwise_feature_gradient(s, w)
        for side, raw_grad in (("a", ga), ("b", gb)):
            def changed(z):
                edited = dict(s)
                edited[side] = softmax(z)
                if mode != "independent":
                    edited["q" + side] = edited[side]
                return objective([edited], [1.], w, .07, True)
            expected = s[side] * (raw_grad - np.sum(raw_grad * s[side], axis=1, keepdims=True))
            observed = fd(changed, s["z" + side])
            check(f"logit_gradient_finite_difference_{mode}_{side}", np.max(np.abs(expected - observed)), 2e-9)
        if mode == "independent":
            qa, qb = s["qa"], s["qb"]
            t = qa @ w @ qb.T
            for side, expected in (("qa", -(1 + t) @ (qb @ w.T) / t.size), ("qb", -(1 + t).T @ (qa @ w) / t.size)):
                def changed_pool(z):
                    edited = dict(s)
                    edited[side] = z
                    return objective([edited], [1.], w, .07, True)
                # Features are perturbed in ambient coordinates for this
                # algebra derivative check, not as a simplex model proposal.
                observed = fd(changed_pool, s[side])
                check(f"independent_pool_gradient_{side}", np.max(np.abs(expected - observed)), 2e-9)
    scenes = [scene(4, 3, 2, "v"), scene(7, 3, 2, "u"), scene(3, 3, 2, "independent")]
    weights, w = [.2, .3, .5], RNG.uniform(-.8, .8, (3, 2))
    check("fixed_weighted_scene_objective", abs(objective(scenes, weights, w, .11, True) - objective(scenes, weights, w, .11)), 2e-13)
    delta, h = dense_quadratic(scenes, weights, .11)
    check("fixed_weighted_scene_hvp", np.max(np.abs(moment_hvp(scenes, weights, w, .11).ravel() - h @ w.ravel())), 2e-13)
    # Local marginals and pooled marginals are deliberately different here.
    hard1 = np.tile([1., 0.], (2, 1)); hard2 = np.tile([0., 1.], (3, 1))
    locals_ = [dict(a=a, b=a, qa=a, qb=a, mode="v") for a in (hard1, hard2)]
    pooled = np.concatenate((hard1, hard2))
    pooled_scene = dict(a=pooled, b=pooled, qa=pooled, qb=pooled, mode="v")
    special_w = np.array([[1., -1.], [-1., 1.]])
    difference = abs(objective(locals_, [.4, .6], special_w) - objective([pooled_scene], [1.], special_w))
    check("cross_scene_pooling_changes_Q_counterexample", 0 if difference > .5 else 1, 0, f"objective difference={difference}")
    s = scene(5, 3, 2, "v"); w = RNG.uniform(-1, 1, (3, 2))
    t = s["a"] @ w @ s["b"].T
    perms = list(itertools.permutations(range(5)))
    positive = np.diag(t)
    values = []
    off_values = []
    for perm in perms:
        neg = t[np.arange(5), perm]
        f = positive.mean() - .5 * np.mean(positive**2) - neg.mean() - .5 * np.mean(neg**2)
        values.append(f)
        if all(i != j for i, j in enumerate(perm)):
            off_values.append(f)
    check("all_120_ordinary_permutations_mean_equals_V", abs(np.mean(values) - objective([s], [1.], w, moments=True)), 2e-13)
    u = dict(s, mode="u")
    check("all_44_derangements_mean_equals_U", abs(np.mean(off_values) - objective([u], [1.], w, moments=True)), 2e-13, "Theory audit only; sampled U API is not implemented")
    check("ordinary_sampling_has_conditional_variance", 0 if np.var(values) > 1e-8 else 1, 0, f"variance={np.var(values):.9g}")
    for ridge in (0., .05):
        for label, cases, ww in (("weighted_mixed", scenes, weights), ("active_box", [dict(a=np.eye(2), b=np.eye(2), qa=np.eye(2), qb=np.eye(2), mode="u")], [1.])):
            delta, h = dense_quadratic(cases, ww, ridge)
            ws, fs, gap, iterations = projected_solver(delta, h, ridge)
            oracle, result = scipy_solve(delta, h)
            f_oracle = delta @ oracle - .5 * oracle @ h @ oracle
            check(f"projected_solver_gap_{label}_ridge{ridge}", gap, 1e-9, f"iterations={iterations}; independent NumPy algorithm")
            check(f"projected_solver_objective_vs_SciPy_{label}_ridge{ridge}", abs(fs - f_oracle), 3e-8, f"SciPy success={result.success}, scipy_gap={gap_of(delta,h,oracle):.6g}")
            w0 = RNG.uniform(-1, 1, len(delta))
            f0 = delta @ w0 - .5 * w0 @ h @ w0
            check(f"FW_gap_bounds_feasible_improvement_{label}_ridge{ridge}", max(0., f_oracle - f0 - gap_of(delta, h, w0)), 2e-13)
    # Smooth, strongly concave ridge-profile case: F envelope matches fixed-W
    # derivative; profiling raw J at the ridge optimum is a different function.
    base = scene(7, 3, 2, "v")
    dz = RNG.normal(size=base["za"].shape)
    ridge = .2
    def at(theta):
        a = softmax(base["za"] + theta * dz)
        return dict(base, a=a, qa=a)
    def profile(theta):
        ss = at(theta)
        d, hh = dense_quadratic([ss], [1.], ridge)
        opt, _ = scipy_solve(d, hh)
        return objective([ss], [1.], opt.reshape(3,2), ridge), objective([ss], [1.], opt.reshape(3,2), 0), opt.reshape(3,2)
    hstep = 1e-4
    center, raw, optimum = profile(0.)
    plus, minus = profile(hstep), profile(-hstep)
    profiled_fd = (plus[0] - minus[0]) / (2 * hstep)
    partial_fd = (objective([at(hstep)], [1.], optimum, ridge) - objective([at(-hstep)], [1.], optimum, ridge)) / (2 * hstep)
    raw_profiled_fd = (plus[1] - minus[1]) / (2 * hstep)
    check("M1_ridge_F_envelope_finite_difference", abs(profiled_fd - partial_fd), 2e-7, "One smooth finite example; no general approximate-gradient guarantee")
    check("M1_ridge_raw_J_profile_is_different", 0 if abs(raw_profiled_fd - partial_fd) > 1e-6 else 1, 0, f"F envelope partial={partial_fd:.9g}; raw J profile derivative={raw_profiled_fd:.9g}")
    p = np.array([[.30, .10], [.05, .55]])
    q = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    eta = (p-q)/(p+q)
    s_true = .5*np.sum((p-q)**2/(p+q))
    t = RNG.uniform(-1,1,p.shape)
    j = np.sum((p-q)*t) - .5*np.sum((p+q)*t*t)
    residual = .5*np.sum((p+q)*(t-eta)**2)
    check("original_J_gap_identity", abs(s_true-j-residual), 2e-14)
    single = dict(a=np.ones((5,1)),b=np.ones((5,1)),qa=np.ones((5,1)),qb=np.ones((5,1)),mode="v")
    d, hh = dense_quadratic([single],[1.],0.)
    opt, result = scipy_solve(d,hh)
    check("rank_one_exact_head_uninformative", abs(objective([single],[1.],opt.reshape(1,1))), 2e-14, "Perfect empirical head solution does not establish an expressive critic")
    report = {"status":"passed", "scope":"Independent NumPy/dense/finite-difference/SciPy mathematics checks; PyTorch backend NOT executed", "python":sys.executable,
              "seed":20261016, "n_checks":len(CHECKS), "torch_execution":"NOT_RUN: torch unavailable in primary runtime", "gpu_execution":"NOT_RUN", "checks":CHECKS}
    (ROOT/"MATH_NUMPY_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","n_checks","scope","torch_execution")},ensure_ascii=False))


if __name__ == "__main__":
    run()
