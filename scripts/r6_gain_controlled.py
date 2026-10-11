"""R6-02G — GAIN controlled validation (r6 package docs/05, docs/03 §3–5; reference math r6/package_r6_20261011/reference/r6_reference.py).

Balanced binary task C ∈ {−1, +1}; cheap observation U, extra observation V.  For each construction:
  exact      S_U, S_UV, Δ = S_UV − S_U = R*_U − R*_UV and the local gain d(u) = E[(η_UV − η_U)² | U = u] (exact enumeration, reference make_task)
  finite     predictors a(U), b(U, V) fitted on FIT_TASK draws in two families (square = least squares on C; logistic = balanced logistic,
             E[C | ·] = tanh f) and two classes (linear: no U×V interaction; interaction: with products — well specified here); on FIT_POLICY
             z = (b − a)², h = (C − a)² − (C − b)², res = h − z (= 2(b − a)(C − b)); policies decide from U only:
             never / always / random (same rate) / cheap uncertainty 1 − a(U)² / GAIN-square (predict z from U) / GAIN-utility (predict h from U) /
             oracle d(u) (exact; upper bound, not deployable) / finite-model oracle E[h | U] on DEV (upper bound for this predictor pair)
  evaluation DEV task risk E(C − pred)² at acquisition budgets {0, .25, .5, .75, 1} (top-scored fraction; ties broken by a fixed random key);
             calibration of predicted vs realised h; repeated over seeds (fresh FIT_TASK / FIT_POLICY / DEV draws).
Constructions: redundant, irrelevant, xor, context (package) + context_noise (added, labelled P: U = (K, X); K = 0: C pure noise; K = 1: C = X·V —
cheap uncertainty cannot tell the two strata apart, the gain can).  Label noise ε flips C with probability ε (balanced).  Negative checks:
a full predictor worse than the cheap one (z > 0, realised gain < 0); xor with constant d(u) (no policy should beat random at a fixed budget).
    python scripts/r6_gain_controlled.py --out /home/infres/yinwang/CS_QMI/outputs/R6_controlled [--seeds 20] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r6" / "package_r6_20261011" / "reference"))
import r6_reference as REF  # noqa: E402

BUDGETS = (0.0, 0.25, 0.5, 0.75, 1.0)
POLICIES = ("never", "always", "random", "cheap_uncertainty", "gain_square", "gain_utility", "oracle_d", "oracle_finite_h")
POLICY_INPUTS = {"random": [], "cheap_uncertainty": ["cheap_confidence"], "gain_square": ["cheap_input"], "gain_utility": ["cheap_input"]}


# --------------------------------------------------------------------------------------------- generators
def sample(name: str, n: int, eps: float, rng: np.random.Generator):
    """n draws of (K, X, V, C) for a construction; U = (K, X) (K ≡ 0 for the 1-d constructions); C flipped with probability eps."""
    s = lambda: rng.choice([-1.0, 1.0], size=n)
    k = np.zeros(n); x = s(); v = s()
    if name == "redundant":
        v = x.copy(); c = x.copy()
    elif name == "irrelevant":
        c = x.copy()
    elif name == "xor":
        c = x * v
    elif name == "context":
        k = rng.integers(0, 2, n).astype(float); c = np.where(k == 0, x, x * v)
    elif name == "context_noise":
        k = rng.integers(0, 2, n).astype(float); c = np.where(k == 0, s(), x * v)
    else:
        raise ValueError(name)
    flip = rng.random(n) < eps
    return k, x, v, np.where(flip, -c, c)


def exact(name: str, eps: float):
    """Exact Δ and d(u) by enumeration (reference make_task for the package constructions, with label noise folded in)."""
    if name == "context_noise":
        states = [REF.State(0.125 * 0.5, (0, x), v, cc) for x in (-1, 1) for v in (-1, 1) for cc in (-1, 1)]        # K = 0: C independent fair coin
        states += [REF.State(0.125, (1, x), v, x * v) for x in (-1, 1) for v in (-1, 1)]
    else:
        states = REF.make_task(name)
    if eps > 0:
        states = [REF.State(st.probability * (1 - eps), st.u, st.v, st.c) for st in states] + [REF.State(st.probability * eps, st.u, st.v, -st.c) for st in states]
    e = REF.evidence_summary(states)
    return {"S_cheap": e["S_cheap"], "S_full": e["S_full"], "delta": e["delta"], "risk_cheap": e["risk_cheap"], "risk_full": e["risk_full"],
            "d_u": {str(k): v for k, v in e["conditional_gain"].items()}}


# --------------------------------------------------------------------------------------------- predictors
def feats(k, x, v=None, cls="linear"):
    """Design matrices.  Cheap: [1, K, X, K·X] (exact for every U cell, so a(U) is well specified); full adds V (linear) or V and V·X, V·K, V·K·X
    (interaction — well specified for every construction here)."""
    cheap = np.stack([np.ones_like(k), k, x, k * x], 1)
    if v is None:
        return cheap
    extra = [v] if cls == "linear" else [v, v * x, v * k, v * k * x]
    return np.concatenate([cheap, np.stack(extra, 1)], 1)


def fit(Xd, c, family, l2=1e-3, iters=400):
    """square: ridge least squares of C (prediction clipped to [−1, 1]); logistic: balanced logistic σ(2f) (prediction tanh f), Newton steps."""
    if family == "square":
        w = np.linalg.solve(Xd.T @ Xd + l2 * len(c) * np.eye(Xd.shape[1]), Xd.T @ c)
        return lambda Z: np.clip(Z @ w, -1.0, 1.0)
    y = (c > 0).astype(float); w = np.zeros(Xd.shape[1])
    for _ in range(iters):
        f = Xd @ w; p = 1 / (1 + np.exp(-2 * f)); g = 2 * Xd.T @ (p - y) / len(c) + l2 * w
        H = 4 * (Xd * (p * (1 - p))[:, None]).T @ Xd / len(c) + l2 * np.eye(Xd.shape[1])
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-10:
            break
    return lambda Z: np.tanh(Z @ w)


def cell_mean(keys_fit, target, keys_eval, prior=0.0, alpha=1.0):
    """Policy regressor from U only: smoothed mean of the target per U cell (U is discrete here)."""
    out = np.full(len(keys_eval), prior, dtype=float); uniq = {}
    for kk, t in zip(keys_fit, target):
        s = uniq.setdefault(kk, [0.0, 0]); s[0] += t; s[1] += 1
    for i, kk in enumerate(keys_eval):
        if kk in uniq:
            s, n = uniq[kk]; out[i] = (s + alpha * prior) / (n + alpha)
    return out


def budget_risk(c, a, b, score, budget, tiebreak):
    n = len(c); m = int(round(budget * n))
    order = np.lexsort((tiebreak, -score)); take = np.zeros(n, bool); take[order[:m]] = True
    pred = np.where(take, b, a); return float(np.mean((c - pred) ** 2))


def run_cell(name, eps, family, cls, seed, n_fit, n_pol, n_dev, bad_full=False):
    rng = np.random.default_rng([20261011, int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big'), int(round(eps * 100)), seed, int(bad_full)])
    roles = {r: sample(name, n, eps, rng) for r, n in (("FIT_TASK", n_fit), ("FIT_POLICY", n_pol), ("DEV", n_dev))}
    k, x, v, c = roles["FIT_TASK"]
    a_fn = fit(feats(k, x), c, family); b_fn = fit(feats(k, x, v, cls), c, family)
    if bad_full:                                                                  # negative check: a full predictor worse than the cheap one
        b_fn = lambda Z: np.zeros(len(Z))
    out = {}
    for r in ("FIT_POLICY", "DEV"):
        k, x, v, c = roles[r]; a = a_fn(feats(k, x)); b = b_fn(feats(k, x, v, cls))
        z, res, h = REF.risk_components(c, a, b); out[r] = dict(k=k, x=x, c=c, a=a, b=b, z=z, h=h, res=res)
    P, D = out["FIT_POLICY"], out["DEV"]
    keys = lambda d: list(zip(d["k"].astype(int), d["x"].astype(int)))
    ex = exact(name, eps); d_true = np.array([ex["d_u"].get(str((int(kk), int(xx))) if name in ("context", "context_noise") else str(int(xx)), 0.0) for kk, xx in zip(D["k"], D["x"])])
    for pol, inp in POLICY_INPUTS.items():
        REF.validate_policy_inputs(inp)                                           # the policy never reads V, full logits, labels or oracle gain
    scores = {"never": np.zeros(n_dev), "always": np.ones(n_dev), "random": np.random.default_rng([seed, 7]).random(n_dev),
              "cheap_uncertainty": 1 - D["a"] ** 2, "gain_square": cell_mean(keys(P), P["z"], keys(D)), "gain_utility": cell_mean(keys(P), P["h"], keys(D)),
              "oracle_d": d_true, "oracle_finite_h": cell_mean(keys(D), D["h"], keys(D))}
    tie = np.random.default_rng([seed, 11]).random(n_dev)
    curves = {p: [budget_risk(D["c"], D["a"], D["b"], scores[p], bu, tie) for bu in BUDGETS] for p in POLICIES if p not in ("never", "always")}
    curves["never"] = [float(np.mean((D["c"] - D["a"]) ** 2))] * len(BUDGETS); curves["always"] = [float(np.mean((D["c"] - D["b"]) ** 2))] * len(BUDGETS)
    pred_h = scores["gain_utility"]; bins = np.quantile(pred_h, [0, 0.25, 0.5, 0.75, 1.0]); calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pred_h >= lo) & (pred_h <= hi)
        if m.any():
            calib.append({"pred": float(pred_h[m].mean()), "realised_h": float(D["h"][m].mean()), "n": int(m.sum())})
    return {"construction": name, "eps": eps, "family": family, "class": cls, "seed": seed, "bad_full": bad_full,
            "dev": {"mean_z": float(D["z"].mean()), "mean_h": float(D["h"].mean()), "mean_res": float(D["res"].mean()),
                    "risk_cheap": curves["never"][0], "risk_full": curves["always"][0]}, "curves": curves, "calibration_gain_utility": calib}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seeds = range(2 if a.smoke else a.seeds); n_fit, n_pol, n_dev = (200, 200, 400) if a.smoke else (2000, 2000, 4000)
    res = {"exact": {}, "cells": [], "counterexamples": {}}
    for name in ("redundant", "irrelevant", "xor", "context", "context_noise"):
        for eps in (0.0, 0.1):
            res["exact"][f"{name}|eps={eps}"] = exact(name, eps)
    for name in ("redundant", "irrelevant", "xor", "context", "context_noise"):
        for family in ("square", "logistic"):
            for cls in ("linear", "interaction"):
                for s in seeds:
                    res["cells"].append(run_cell(name, 0.1, family, cls, s, n_fit, n_pol, n_dev))
        for s in seeds:                                                           # negative check: a worse full predictor
            res["cells"].append(run_cell(name, 0.1, "square", "interaction", s, n_fit, n_pol, n_dev, bad_full=True))
    t = REF.make_task("redundant")
    res["counterexamples"]["reference_redundant_b_zero"] = REF.finite_model_utility(t, {-1: -1.0, 1: 1.0}, {(-1, -1): 0.0, (1, 1): 0.0})
    (out / "gain_controlled.json").write_text(json.dumps(res, indent=1)); print(f"[r6_gain] {len(res['cells'])} cells -> {out / 'gain_controlled.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
