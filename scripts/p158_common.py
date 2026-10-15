"""P158 shared pieces (project map 2026-10-14 §11 B.2 — nested increments with truth on the X11 A2 joint-vs-product setting): the Gaussian
pair generator with irrelevant coordinates, an exact oracle for every kept set of pairs, Monte-Carlo truths for the VCS (L²) and the matched-JS
(Bregman) increments, and the readout helpers shared by the runner, the aggregate and the tests.

Generator (named numpy streams).  K_SIG = 10 signal pairs (x_j, y_j) ~ N(0, [[1, rho], [rho, 1]]) under P (X11 anchor A2 with k = 10);
M_IRR = 10 irrelevant coordinates per side, independent N(0, 1) of everything (v2 package §5.1: X = (X_s, N_x), Y = (Y_s, N_y)).
Layout x = [x_sig (10), x_irr (10)], y = [y_sig (10), y_irr (10)]  (D = 20 per side).  Q = product of the marginals: every coordinate independent
N(0, 1) (both marginals are N(0, I_20)); the runner realises Q pairs as (x_i, y_{pi(i)}) with a non-zero cyclic shift pi of the same draw.
Chain (deterministic coordinate deletion):  B = all 20 + 20  ⊇  M = the 10 signal pairs (drops only irrelevant coordinates: S_B = S_M exactly,
the irrelevant factors cancel in the density ratio)  ⊇  A = the first 5 signal pairs (S_A < S_M).
Oracle for a kept set of the first k signal pairs:  log r_k(x, y) = sum_{j<k} lg_rho(x_j, y_j),
    lg_rho(u, v) = -1/2 log(1 - rho^2) - (u^2 - 2 rho u v + v^2) / (2 (1 - rho^2)) + (u^2 + v^2) / 2    (X11 x11_anchors.log_ratio),
so the oracle logit is f*_k = log r_k / 2, eta_k = tanh f*_k, q*_k = (1 + eta_k) / 2 = sigma(2 f*_k).  The quadratic-feature class
(features x, y, x^2, y^2, x*y per coordinate) contains f*_k exactly.
Truths by Monte-Carlo on n_truth P units and n_truth Q units (SE recorded), E_M = 1/2 E_P + 1/2 E_Q:
    S_k = E_M eta_k^2                                       (VCS value; J(T*) = S)
    G_k = E_P log(2 q*_k) + E_Q log(2 (1 - q*_k))           (JS value at its optimum; = 2 JSD(P, Q) restricted to the kept set)
    L2 increment  (B ⊇ A):  E_M (eta_B - eta_A)^2  = S_B - S_A at the oracle (checked, not assumed)
    Bregman increment:      2 E_M KL(Ber(q*_B) || Ber(q*_A)) = G_B - G_A at the oracle (checked, not assumed)
Nothing here clips, monotonises or orthogonalises any readout; all log-probabilities are computed from logits (softplus), never from clipped q.
"""
from __future__ import annotations

import numpy as np

K_SIG, M_IRR, KEEP_A = 10, 10, 5
D = K_SIG + M_IRR                                   # per side
KSET = {"B": K_SIG, "M": K_SIG, "A": KEEP_A}        # signal pairs that enter the oracle of each set (B's irrelevant coordinates cancel)
DSET = {"B": D, "M": K_SIG, "A": KEEP_A}            # coordinates per side that the critic of each set sees
CHAIN = ["B", "M", "A"]
STEPS = (("B", "M"), ("M", "A"), ("B", "A"))


# ---------------------------------------------------------------------------------------------------------------------- generator + oracle
def sample_p(n: int, rho: float, rng) -> tuple[np.ndarray, np.ndarray]:
    z1, z2 = rng.normal(size=(n, K_SIG)), rng.normal(size=(n, K_SIG))
    x = np.concatenate([z1, rng.normal(size=(n, M_IRR))], 1)
    y = np.concatenate([rho * z1 + np.sqrt(1.0 - rho ** 2) * z2, rng.normal(size=(n, M_IRR))], 1)
    return x, y


def sample_q(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    return rng.normal(size=(n, D)), rng.normal(size=(n, D))


def shift(n: int, rng) -> np.ndarray:
    """Q pairs of a draw: partner index pi(i) = (i + s) mod n with a non-zero cyclic shift s (exact product-marginal law for i.i.d. units)."""
    s = int(rng.integers(1, n))
    return (np.arange(n) + s) % n


def feats(x: np.ndarray, y: np.ndarray) -> dict:
    return {"B": (x, y), "M": (x[:, :K_SIG], y[:, :K_SIG]), "A": (x[:, :KEEP_A], y[:, :KEEP_A])}


def log_ratio(x: np.ndarray, y: np.ndarray, rho: float, k: int) -> np.ndarray:
    """Analytic log p_k(x, y) / (p_k(x) p_k(y)) of the first k signal pairs (X11 A2 form); zero for rho = 0 or k = 0."""
    if rho == 0.0 or k == 0:
        return np.zeros(len(x))
    u, v = x[:, :k], y[:, :k]
    return (-0.5 * np.log(1 - rho ** 2) - (u ** 2 - 2 * rho * u * v + v ** 2) / (2 * (1 - rho ** 2)) + (u ** 2 + v ** 2) / 2).sum(1)


def oracle_logit(x, y, rho, k):
    return 0.5 * log_ratio(x, y, rho, k)


def eta(x, y, rho, k):
    return np.tanh(oracle_logit(x, y, rho, k))


# ---------------------------------------------------------------------------------------------------------------------- readout helpers
def softplus(x):
    return np.logaddexp(0.0, x)


def g_value(fp: np.ndarray, fq: np.ndarray) -> float:
    """JS value of a critic from its logits, q = sigma(2 f):  E_P log(2 q) + E_Q log(2 (1 - q)) = 2 log 2 - [E_P softplus(-2f) + E_Q softplus(2f)]."""
    return float(2 * np.log(2.0) - np.mean(softplus(-2.0 * fp)) - np.mean(softplus(2.0 * fq)))


def kl_bern(fb: np.ndarray, fa: np.ndarray) -> np.ndarray:
    """KL(Ber(q_b) || Ber(q_a)) per unit from logits (q = sigma(2 f)); log q = -softplus(-2f), log(1 - q) = -softplus(2f)."""
    qb = 1.0 / (1.0 + np.exp(-2.0 * fb))
    return qb * (softplus(-2.0 * fa) - softplus(-2.0 * fb)) + (1 - qb) * (softplus(2.0 * fa) - softplus(2.0 * fb))


def d_js(fbp, fbq, fap, faq) -> float:
    """Bregman step 2 E_M KL(Ber(q_B) || Ber(q_A)) = mean_P KL + mean_Q KL."""
    return float(np.mean(kl_bern(fbp, fap)) + np.mean(kl_bern(fbq, faq)))


def js_chain_summary(chain_fp: list, chain_fq: list, names: list) -> dict:
    """JS analogue of vcs_estim.increments.chain_summary on logits: G per set, steps (delta_G, D_JS, r = delta_G - D_JS), R_orth_js."""
    G = {nm: g_value(p, q) for nm, p, q in zip(names, chain_fp, chain_fq)}; steps = {}
    for l in range(len(names) - 1):
        d = d_js(chain_fp[l], chain_fq[l], chain_fp[l + 1], chain_fq[l + 1]); dg = G[names[l]] - G[names[l + 1]]
        steps[f"{names[l]}->{names[l + 1]}"] = {"G_B": G[names[l]], "G_A": G[names[l + 1]], "delta_G": dg, "D_JS": d, "r_BA": dg - d}
    total = d_js(chain_fp[0], chain_fq[0], chain_fp[-1], chain_fq[-1])
    return {"G": G, "steps": steps, "R_orth": float(sum(s["D_JS"] for s in steps.values()) - total),
            "ordering_violations": int(sum(G[names[i]] < G[names[i + 1]] for i in range(len(names) - 1)))}


def posterior_mse(tp: np.ndarray, tq: np.ndarray, ep: np.ndarray, eq: np.ndarray) -> float:
    """1/2 E_P (T - eta)^2 + 1/2 E_Q (T - eta)^2 on the sampled P / Q pairs (T scale; the JS posterior error is the same number / 4)."""
    return float(0.5 * np.mean((tp - ep) ** 2) + 0.5 * np.mean((tq - eq) ** 2))


# ---------------------------------------------------------------------------------------------------------------------- truths
def oracle(rho: float, n: int = 400_000, seed: int = 1580) -> dict:
    """Truth sample: n P units and n independent Q units; per set S, G (with MC SEs); per step the L2 and Bregman increments and the
    identities S_B - S_A = E_M (eta_B - eta_A)^2 and G_B - G_A = 2 E_M KL (reported as gaps, not assumed)."""
    rng = np.random.default_rng([seed, int(round(rho * 1000))]); xp, yp = sample_p(n, rho, rng); xq, yq = sample_q(n, rng)
    f = {k: (oracle_logit(xp, yp, rho, KSET[k]), oracle_logit(xq, yq, rho, KSET[k])) for k in CHAIN}
    out = {"n_truth": n, "rho": rho, "sets": {}, "steps": {}}
    for k in CHAIN:
        ep, eq = np.tanh(f[k][0]), np.tanh(f[k][1]); sp, sq = ep ** 2, eq ** 2
        gp, gq = 2 * np.log(2.0) - 2 * softplus(-2.0 * f[k][0]), 2 * np.log(2.0) - 2 * softplus(2.0 * f[k][1])  # per-unit 2 log(2 q*), 2 log(2 (1 - q*)); E_M of these = G
        out["sets"][k] = {"S": float(0.5 * sp.mean() + 0.5 * sq.mean()), "S_se": float(np.sqrt(0.25 * sp.var() / n + 0.25 * sq.var() / n)),
                          "G": float(0.5 * gp.mean() + 0.5 * gq.mean()), "G_se": float(np.sqrt(0.25 * gp.var() / n + 0.25 * gq.var() / n))}
    for b, a in STEPS:
        l2p, l2q = (np.tanh(f[b][0]) - np.tanh(f[a][0])) ** 2, (np.tanh(f[b][1]) - np.tanh(f[a][1])) ** 2
        klp, klq = kl_bern(f[b][0], f[a][0]), kl_bern(f[b][1], f[a][1])
        l2, br = float(0.5 * l2p.mean() + 0.5 * l2q.mean()), float(klp.mean() + klq.mean())
        out["steps"][f"{b}->{a}"] = {"S_diff": out["sets"][b]["S"] - out["sets"][a]["S"], "L2_increment": l2,
                                     "L2_se": float(np.sqrt(0.25 * l2p.var() / n + 0.25 * l2q.var() / n)),
                                     "G_diff": out["sets"][b]["G"] - out["sets"][a]["G"], "Bregman_increment": br,
                                     "Bregman_se": float(np.sqrt(klp.var() / n + klq.var() / n))}
        out["steps"][f"{b}->{a}"]["identity_gap_L2"] = out["steps"][f"{b}->{a}"]["S_diff"] - l2
        out["steps"][f"{b}->{a}"]["identity_gap_Bregman"] = out["steps"][f"{b}->{a}"]["G_diff"] - br
    return out


def true_increments(orc: dict) -> dict:
    """The truths the fitted increments are scored against: VCS scale = S differences, JS scale = G differences."""
    return {"vcs": {s: orc["steps"][s]["S_diff"] for s in orc["steps"]}, "js": {s: orc["steps"][s]["G_diff"] for s in orc["steps"]}}
