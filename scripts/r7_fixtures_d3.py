"""r7 R7-20 D3 — finite conditional-distribution check (r7 package directions/D3_conditional_generation/SERVER_EXPERIMENTS_CN.md §B).

Finite laws with an exact truth.  Conditions c ∈ {0..15} (uniform π).  Observable states z = (block b, sub-mode k, quality q): 16 blocks × 3
sub-modes × {hi, lo} = 96 states; P_ref(z | c) lives on block c with sub-mode weights (0.5, 0.3, 0.2) and P(lo) = 0.1.
P(c, z) = π(c) P_ref(z | c); Q_m(c, z) = π(c) Q_m(z | c) for 13 prescribed Q_m:
  clean          Q = P_ref (S = 0)
  cond_mismatch  λ ∈ {0.2, 0.5, 0.8}: Q(z|c) = (1 − λ) P_ref(z|c) + λ P_ref(z|σ(c)), σ(c) = c + 8 mod 16 — the z-marginal is unchanged
  mode_missing   λ ∈ {1/3, 2/3, 1}: sub-modes 1, 2 of every block lose a fraction λ of their mass to sub-mode 0 (the z-marginal changes)
  quality        λ ∈ {0.2, 0.5, 0.8}: P(lo) = 0.1 + 0.8 λ
  contamination  λ ∈ {0.1, 0.3, 0.6}: Q = (1 − λ) P_ref + λ Uniform(Z)
Each fit sees n ∈ {8, 32, 128} FIT samples per condition per side, a fresh SELECT set of the same size and a fresh EVAL set (512 per condition
per side); 3 independent draws.  Observations reach the critics only through fixed random embeddings: g(c) ∈ R^8 and
e(z) = [0.7 g(b) + 0.7 u(b, k), onehot(q)] ∈ R^10 (an "encoder" in which an image carries a noisy trace of its own class).
Estimators on the same S (the S-error table): VCS (J on T = tanh f, 2-layer MLP), matched logistic (BCE on 2f, read through the same T and J),
the same-target closed-form kernel critic (T = wᵀψ, RFF ψ, (bandwidth, ridge) on SELECT), the null T ≡ 0 and the oracle T = η (sanity).
T is a function on the finite grid, so its population J, the posterior MSE S − J(T) = E_M (T − η)² and the per-condition versions are exact.
Discrimination only (never subtracted from S): grouped conditional RBF MMD² of e(z) given c, and the unconditional MMD² of e(z) (a CMMD-like
negative control that is blind to marginal-preserving mismatch by construction).
    python scripts/r7_fixtures_d3.py --out /home/infres/yinwang/CS_QMI/outputs/R7_D3 [--workers 8] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

C, K, NQ, D_G, SEED = 16, 3, 2, 8, 20261012
Z = C * K * NQ
W_REF, P_LO = (0.5, 0.3, 0.2), 0.1
N_EVAL, NS, DRAWS = 512, (8, 32, 128), 3
LEVELS = {"clean": [0.0], "cond_mismatch": [0.2, 0.5, 0.8], "mode_missing": [1 / 3, 2 / 3, 1.0], "quality": [0.2, 0.5, 0.8],
          "contamination": [0.1, 0.3, 0.6]}
ESTIMATORS = ("vcs", "logistic", "kernel", "null", "oracle")


def a_plus(t):
    return t - 0.5 * t * t


def a_minus(t):
    return -t - 0.5 * t * t


def zi(b, k, q):
    return (b * K + k) * NQ + q


def cond_law(w=W_REF, p_lo=P_LO):
    """[C, Z] conditionals: block c, sub-mode weights w, P(lo) = p_lo independent of the sub-mode."""
    out = np.zeros((C, Z))
    for c in range(C):
        for k in range(K):
            out[c, zi(c, k, 0)] = w[k] * (1 - p_lo); out[c, zi(c, k, 1)] = w[k] * p_lo
    return out


def p_ref_cond():
    return cond_law()


def q_cond(kind, lam):
    ref = cond_law()
    if kind == "clean":
        return ref.copy()
    if kind == "cond_mismatch":
        return (1 - lam) * ref + lam * ref[(np.arange(C) + 8) % C]
    if kind == "mode_missing":
        return cond_law(w=(W_REF[0] + (W_REF[1] + W_REF[2]) * lam, W_REF[1] * (1 - lam), W_REF[2] * (1 - lam)))
    if kind == "quality":
        return cond_law(p_lo=0.1 + 0.8 * lam)
    if kind == "contamination":
        return (1 - lam) * ref + lam / Z
    raise ValueError(kind)


def joint(cond):
    return cond / C                                                                     # π uniform


def truth(p_joint, q_joint):
    den = p_joint + q_joint; eta = np.divide(p_joint - q_joint, den, out=np.zeros_like(den), where=den > 0)
    return eta, float(np.sum(0.5 * den * eta * eta))


def pop_j(p_joint, q_joint, t):
    return float(np.sum(p_joint * a_plus(t) + q_joint * a_minus(t)))


def per_condition(p_cond, q_cond_, t):
    """(S_c, J_c(T)) for every condition (rows of the conditional laws)."""
    den = p_cond + q_cond_; eta = np.divide(p_cond - q_cond_, den, out=np.zeros_like(den), where=den > 0)
    return (0.5 * den * eta * eta).sum(1), (p_cond * a_plus(t) + q_cond_ * a_minus(t)).sum(1)


def embeddings():
    g = np.random.default_rng([SEED, 1]); gc = g.normal(size=(C, D_G)); u = g.normal(size=(C, K, D_G))
    ez = np.zeros((Z, D_G + NQ))
    for b in range(C):
        for k in range(K):
            for q in range(NQ):
                ez[zi(b, k, q), :D_G] = 0.7 * gc[b] + 0.7 * u[b, k]; ez[zi(b, k, q), D_G + q] = 1.0
    return gc, ez


def features(gc, ez):
    """[C, Z, D] observation of every grid cell (c, z)."""
    return np.concatenate([np.repeat(gc[:, None, :], Z, 1), np.repeat(ez[None, :, :], C, 0)], -1)


def sample(cond, n, rng):
    """n states per condition -> flat grid indices c·Z + z."""
    return np.concatenate([c * Z + rng.choice(Z, size=n, p=cond[c]) for c in range(C)])


def emp_j(t_flat, p_idx, q_idx):
    return float(np.mean(a_plus(t_flat[p_idx])) + np.mean(a_minus(t_flat[q_idx])))


def fit_mlp(feat, fit, sel, loss, seed, steps=1500, every=50, lr=3e-3):
    """2-layer MLP critic on the grid features; full-batch Adam; keep the checkpoint with the best SELECT Ĵ (common readout T = tanh f)."""
    import torch
    torch.manual_seed(seed); torch.set_num_threads(1)
    X = torch.tensor(feat.reshape(-1, feat.shape[-1]), dtype=torch.float64)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pi, qi = torch.tensor(fit[0]), torch.tensor(fit[1])
    best = (-np.inf, None, 0)
    for step in range(1, steps + 1):
        f = net(X).squeeze(-1); fp, fq = f[pi], f[qi]
        if loss == "vcs":
            tp, tq = torch.tanh(fp), torch.tanh(fq); obj = -((tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean())
        else:                                                                          # balanced logistic on 2f: f* = ½ log(p/q), tanh f* = η
            obj = torch.nn.functional.softplus(-2 * fp).mean() + torch.nn.functional.softplus(2 * fq).mean()
        opt.zero_grad(); obj.backward(); opt.step()
        if step % every == 0 or step == 1:
            with torch.no_grad():
                t = torch.tanh(net(X).squeeze(-1)).numpy()
            s_j = emp_j(t, sel[0], sel[1])
            if s_j > best[0]:
                best = (s_j, t.copy(), step)
    return best[1].reshape(C, Z), {"selected_step": best[2], "select_j": best[0]}


def fit_kernel(feat, fit, sel, seed, D=256):
    """Same-target closed-form kernel critic: T = clip(wᵀψ, −1, 1), RFF ψ; w = (Ê_P ψψᵀ + Ê_Q ψψᵀ + λI)⁻¹ (Ê_P ψ − Ê_Q ψ) maximises the
    empirical J over the linear span; (bandwidth multiple of the median distance, ridge λ) chosen on SELECT Ĵ.  Clipping to [−1, 1] can only
    move T towards η ∈ [−1, 1]."""
    X = feat.reshape(-1, feat.shape[-1]); g = np.random.default_rng([SEED, 7, seed])
    sub = X[np.concatenate(fit)][:512]; med = float(np.median(np.linalg.norm(sub[:, None] - sub[None], axis=-1))) + 1e-9
    W0, b0 = g.normal(size=(X.shape[1], D)), g.uniform(0, 2 * np.pi, D)
    best = (-np.inf, None, None)
    for hm in (0.5, 1.0, 2.0):
        psi = np.sqrt(2.0 / D) * np.cos(X @ (W0 / (hm * med)) + b0)
        P, Q = psi[fit[0]], psi[fit[1]]; A = P.T @ P / len(P) + Q.T @ Q / len(Q); bvec = P.mean(0) - Q.mean(0)
        for lam in (1e-3, 1e-2, 1e-1):
            t = np.clip(psi @ np.linalg.solve(A + lam * np.eye(D), bvec), -1, 1)
            s_j = emp_j(t, sel[0], sel[1])
            if s_j > best[0]:
                best = (s_j, t.reshape(C, Z), {"bandwidth_x_median": hm, "ridge": lam, "select_j": s_j})
    return best[1], best[2]


def mmd2(xa, xb, h):
    """Unbiased RBF MMD² between two sample sets (rows)."""
    def k(a, b):
        return np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * h * h))
    na, nb = len(xa), len(xb)
    kaa, kbb, kab = k(xa, xa), k(xb, xb), k(xa, xb)
    return float((kaa.sum() - np.trace(kaa)) / (na * (na - 1)) + (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1)) - 2 * kab.mean())


def one_cell(job):
    kind, lam, n, draw, smoke = job
    gc, ez = embeddings(); feat = features(gc, ez); pc, qc = cond_law(), q_cond(kind, lam); pj, qj = joint(pc), joint(qc); eta, S = truth(pj, qj)
    rng = np.random.default_rng([SEED, list(LEVELS).index(kind), int(round(1000 * lam)), n, draw])
    n_eval = 64 if smoke else N_EVAL
    fit, sel, ev = [(sample(pc, m, rng), sample(qc, m, rng)) for m in (n, n, n_eval)]
    s_c, _ = per_condition(pc, qc, np.zeros((C, Z)))
    out = {"kind": kind, "lambda": lam, "n_per_condition": n, "draw": draw, "S_true": S, "S_c_true": s_c.tolist(), "estimators": {}}
    for name in ESTIMATORS:
        t0 = time.perf_counter()
        if name == "null":
            t, info = np.zeros((C, Z)), {}
        elif name == "oracle":
            t, info = eta.copy(), {}
        elif name == "kernel":
            t, info = fit_kernel(feat, fit, sel, draw)
        else:
            t, info = fit_mlp(feat, fit, sel, name, seed=1000 * draw + n, steps=300 if smoke else 1500)
        jp = pop_j(pj, qj, t); tf = t.reshape(-1); _, j_c = per_condition(pc, qc, t)
        ev_c = [emp_j(tf, ev[0][ev[0] // Z == c], ev[1][ev[1] // Z == c]) for c in range(C)]
        out["estimators"][name] = {"J_population": jp, "posterior_mse": S - jp, "J_eval": emp_j(tf, ev[0], ev[1]),
                                   "per_condition_posterior_mse_mean": float(np.mean(s_c - j_c)), "per_condition_posterior_mse_max": float(np.max(s_c - j_c)),
                                   "per_condition_abs_err_eval_mean": float(np.mean(np.abs(np.array(ev_c) - s_c))),
                                   "seconds": time.perf_counter() - t0, **info}
    # discrimination statistics on the FIT + SELECT samples (same budget), never subtracted from S
    xp, xq = np.concatenate([fit[0], sel[0]]), np.concatenate([fit[1], sel[1]]); zp, zq = xp % Z, xq % Z
    h = float(np.median(np.linalg.norm(ez[zp[:256]][:, None] - ez[zp[:256]][None], axis=-1))) + 1e-9
    cond_vals = [mmd2(ez[zp[xp // Z == c]], ez[zq[xq // Z == c]], h) for c in range(C)]
    sp, sq = rng.permutation(len(zp))[:1024], rng.permutation(len(zq))[:1024]
    out["discrimination"] = {"grouped_conditional_mmd2": float(np.mean(cond_vals)), "unconditional_mmd2": mmd2(ez[zp[sp]], ez[zq[sq]], h)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ns, draws = ((8,), 1) if a.smoke else (NS, DRAWS)
    jobs = [(k, lam, n, d, a.smoke) for k, lams in LEVELS.items() for lam in lams for n in ns for d in range(draws)]
    pm = joint(cond_law()).sum(0)
    design = {"conditions": C, "states": Z, "levels": LEVELS, "n_per_condition": list(ns), "draws": draws, "eval_per_condition": 64 if a.smoke else N_EVAL,
              "S_true": {f"{k}:{lam:.3f}": truth(joint(cond_law()), joint(q_cond(k, lam)))[1] for k, lams in LEVELS.items() for lam in lams},
              "z_marginal_L1_shift": {f"{k}:{lam:.3f}": float(np.abs(pm - joint(q_cond(k, lam)).sum(0)).sum()) for k, lams in LEVELS.items() for lam in lams}}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one_cell, jobs))
    (out / ("d3_finite_smoke.json" if a.smoke else "d3_finite.json")).write_text(json.dumps({"design": design, "cells": res, "seconds": time.time() - t0, "smoke": a.smoke}))
    print(f"[r7_d3] {len(res)} cells in {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
