"""R4-T2 — fixed-critic paired-sampling validation (reports/R4/R4_T2_SAMPLING_PROTOCOL_FROZEN_20261011.yaml; object and hypotheses in
reports/R4/R4_T1_STATISTICAL_OBJECT_20261011.md; package r4 docs/03 §4–6).

Stage 1a  finite-state cross-pair: W_i = (X_i, Y_i) iid from P_ρ on a 4 × 5 grid; negatives (X_i, Y_j), i ≠ j, follow Q = P_X ⊗ P_Y; exact
          J(T), S, ζ₁ = E h₁², E h₂² and Var Ĵ = 4ζ₁/n + 2E h₂²/(n(n − 1)) by enumeration.
Stage 1b  finite-state image-local: images of r ~ U{2..6} iid pairs; P_G / Q_G uniform on matched / all r × r index pairs; J_G exact per image;
          targets μ_G = E_G J_G (exact) and E_G S_G = E[(r − 1)/(r + 1)].
Stage 2   existing generators (src/vcs_estim/data.py: gaussian d 20, xor_mixture 10 pairs) at I ∈ {0, 0.1, 2} nats, cross-pair; J(T), S by
          Monte Carlo on 2 × 10⁶ TRUTH pairs per side.
Stage 3   learned P85 JointMLP critics (VCS: T = tanh f; JS: T = tanh(f / 2)) on gaussian d 20, I = 2 nats, training seeds 0–4; run only after
          the stage 1–2 implementation checks pass (scripts/r4_t2_aggregate.py reports the gate).
Each cell writes <out>/<stage>_..._n<n>.json with per-draw readouts; scripts/r4_t2_aggregate.py computes coverage, widths and the checks.
    python scripts/r4_t2_sampling.py --stage 1a --out /home/infres/yinwang/CS_QMI/outputs/R4_T2 [--draws 500] [--workers 8] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEED, NS, KS, DRAWS, BOOT_IMG, RESAMPLE_CHECK, CHECK_DRAWS = 20261011, (32, 128, 512), (1, 4, 16), 500, 1000, 2000, 20
GRID, R_IMG, N_TRUTH, CHUNK = (4, 5), (2, 3, 4, 5, 6), 2_000_000, 200_000
LEVELS_1 = {"independent": 0.0, "weak": 0.05, "strong": 0.40}          # target S (ρ by bisection)
LEVELS_2 = {"independent": 0.0, "weak": 0.1, "strong": 2.0}            # nats
GENS_2 = ("gaussian", "xor_mixture")
CRITICS_1 = ("zero", "optimum", "prescribed_error", "fixed_random"); CRITICS_2 = ("zero", "optimum", "prescribed_error")
STAGE3 = {"setting": "gaussian", "mi": 2.0, "d": 20, "N_fit": 4096, "batch": 256, "updates": 2000, "lr": 5e-4, "variant": "product",
          "kinds": ("vcs", "js"), "seeds": (0, 1, 2, 3, 4), "ns": (128, 512), "n_truth": 400_000}


def a_plus(t):
    return t - 0.5 * t * t


def a_minus(t):
    return -t - 0.5 * t * t


def rng_for(*key):
    return np.random.default_rng(np.random.SeedSequence([SEED, *key]))


def critic_fn(name, eta):
    """The fixed critics as numpy functions of the oracle η (fixed_random is stage-1 only and handled on the grid)."""
    if name == "zero":
        return eta * 0.0
    if name == "optimum":
        return eta
    if name == "prescribed_error":
        return np.clip(0.7 * eta + 0.1, -1.0, 1.0)
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ finite-state laws (stage 1)
def base_table():
    return np.random.default_rng([SEED, 1]).dirichlet(np.full(GRID[0] * GRID[1], 0.3)).reshape(GRID)


def mixture(P1, rho):
    """P_ρ = (1 − ρ) P_X ⊗ P_Y + ρ P1 (same marginals as P1) and Q = P_X ⊗ P_Y."""
    Q = np.outer(P1.sum(1), P1.sum(0)); return (1 - rho) * Q + rho * P1, Q


def s_eta(P, Q):
    eta = (P - Q) / (P + Q); return float(((P + Q) / 2 * eta * eta).sum()), eta


def rho_for(P1, target):
    """ρ with S(P_ρ) = target (S increases with ρ); capped at 0.95."""
    if target <= 0:
        return 0.0
    lo, hi = 0.0, 0.95
    if s_eta(*mixture(P1, hi))[0] < target:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if s_eta(*mixture(P1, mid))[0] < target else (lo, mid)
    return 0.5 * (lo + hi)


def critics_grid(eta):
    z = np.random.default_rng([SEED, 2]).normal(0.0, 0.5, eta.shape)
    return {"zero": np.zeros_like(eta), "optimum": eta.copy(), "prescribed_error": np.clip(0.7 * eta + 0.1, -1, 1), "fixed_random": np.tanh(z)}


def grid_states(P):
    nx, ny = P.shape; return P.ravel(), np.repeat(np.arange(nx), ny), np.tile(np.arange(ny), nx)


def exact_hoeffding(P, T):
    """μ = E k_T, ζ₁ = E h₁² and E h₂² of the cross-pair kernel, by enumeration of the grid states (W, W′ iid ~ P)."""
    w, xs, ys = grid_states(P)
    a = a_plus(T[xs, ys]); b = a_minus(T[xs[:, None], ys[None, :]])                        # b(w, w′) = a₋(T(x_w, y_w′))
    k = 0.5 * (a[:, None] + a[None, :]) + 0.5 * (b + b.T); mu = float(w @ k @ w); h1 = k @ w - mu
    h2 = k - mu - h1[:, None] - h1[None, :]
    return mu, float(w @ h1 ** 2), float(w @ h2 ** 2 @ w)


def var_exact(z1, eh2, n):
    return 4 * z1 / n + 2 * eh2 / (n * (n - 1))


# ------------------------------------------------------------------------------------------------ readouts
def j_hat(Tm):
    """Ĵ = (1/n) Σ a₊(T(X_i, Y_i)) + (1/(n(n − 1))) Σ_{i≠j} a₋(T(X_i, Y_j)) from the n × n critic matrix."""
    n = Tm.shape[0]; B = a_minus(Tm); np.fill_diagonal(B, 0.0); return float(a_plus(np.diag(Tm)).sum() / n + B.sum() / (n * (n - 1)))


def delete_one(a, B):
    """U-statistics with observation i removed (positives a, negatives B with a zero diagonal), all i in O(n²)."""
    n = len(a); sa, sb = a.sum(), B.sum(); return (sa - a) / (n - 1) + (sb - B.sum(1) - B.sum(0)) / ((n - 1) * (n - 2))


def incomplete_delete_one(a, V, S):
    """Sparse-shift estimates with observation i removed: drops a_i and the 2K selected ordered pairs (i, i + l), (i − l, i); V[l − 1, i] = b(i, i + l)."""
    n, K = len(a), len(S); idx = np.arange(n); sel = V[S]
    rem = sel.sum(0) + np.stack([sel[s, (idx - (S[s] + 1)) % n] for s in range(K)]).sum(0)
    return (a.sum() - a) / (n - 1) + (sel.sum() - rem) / (K * n - 2 * K)


def mu_image_local(P, Q, T):
    """E_G J_G(T) for images of r ~ U(R_IMG) iid pairs with P_G / Q_G uniform on matched / all r × r index pairs (exact)."""
    inv_r = float(np.mean([1.0 / r for r in R_IMG]))
    return float((P * a_plus(T)).sum() + inv_r * (P * a_minus(T)).sum() + (1 - inv_r) * (Q * a_minus(T)).sum())


def cross_readouts(Tm, rng, check=False):
    """Every cross-pair readout from the n × n critic matrix Tm (entry (i, j) = T(X_i, Y_j); the diagonal holds the positive pairs)."""
    n = Tm.shape[0]; t0 = time.perf_counter()
    a = a_plus(np.diag(Tm).copy()); B = a_minus(Tm); np.fill_diagonal(B, 0.0)
    sa, sb, nn = a.sum(), B.sum(), n * (n - 1); J = sa / n + sb / nn
    vb = max(0.0, ((B * B).sum() - sb * sb / nn) / (nn - 1))                              # pair-iid (negative control)
    se_naive = math.sqrt(a.var(ddof=1) / n + vb / nn)
    U = delete_one(a, B)                                                                   # delete-one U-statistics
    se_jk = math.sqrt((n - 1) / n * ((U - U.mean()) ** 2).sum())
    i1 = np.arange(0, n - 1, 2); i2 = i1 + 1; kb = 0.5 * (a[i1] + a[i2]) + 0.5 * (B[i1, i2] + B[i2, i1])   # disjoint two-observation blocks
    out = {"J": float(J), "se_naive": se_naive, "se_jk": se_jk, "J_blk": float(kb.mean()), "sd_blk": float(kb.std(ddof=1)), "m_blk": int(len(kb)),
           "t_readouts_full": time.perf_counter() - t0}
    idx = np.arange(n); V = np.stack([B[idx, (idx + l) % n] for l in range(1, n)])           # row l − 1: the n pairs (i, i + l) of shift l
    v = V.mean(1); sv2 = float(v.var(ddof=1)); out["var_shift_means"] = sv2
    for K in KS:
        if K > n - 1:
            continue
        S = rng.choice(n - 1, K, replace=False); JK = sa / n + v[S].mean(); cv = (1 - K / (n - 1)) * sv2 / K
        UK = incomplete_delete_one(a, V, S)
        r = {"J": float(JK), "cond_var": cv, "se_jk": math.sqrt((n - 1) / n * ((UK - UK.mean()) ** 2).sum()), "evals": int(n + K * n)}
        if check:
            reps = np.array([v[rng.choice(n - 1, K, replace=False)].mean() for _ in range(RESAMPLE_CHECK)]); r["resampled_var"] = float(reps.var(ddof=1))
        out[f"K{K}"] = r
    out["evals_full"] = int(n * n)
    return out


def columns(rec):
    """List of per-draw dicts (one level of nesting) -> dict of per-draw lists."""
    flat = [{**{k: v for k, v in d.items() if not isinstance(v, dict)}, **{f"{k}.{kk}": vv for k, v in d.items() if isinstance(v, dict) for kk, vv in v.items()}} for d in rec]
    keys = sorted(set().union(*flat)); return {k: [f.get(k) for f in flat] for k in keys}


# ------------------------------------------------------------------------------------------------ stage cells
def stage1_law(level):
    P1 = base_table(); rho = rho_for(P1, LEVELS_1[level]); P, Q = mixture(P1, rho); S, eta = s_eta(P, Q); return P, Q, S, eta, rho


def cell_1a(job):
    level, critic, n, ci, draws = job
    P, Q, S, eta, rho = stage1_law(level); T = critics_grid(eta)[critic]
    J = float((P * a_plus(T)).sum() + (Q * a_minus(T)).sum()); mu, z1, eh2 = exact_hoeffding(P, T); w, xs, ys = grid_states(P); rec = []
    for d in range(draws):
        rng = rng_for(1, ci, d); s = rng.choice(len(w), n, p=w)
        t0 = time.perf_counter(); Tm = T[xs[s][:, None], ys[s][None, :]]; tm = time.perf_counter() - t0
        r = cross_readouts(Tm, rng, check=d < CHECK_DRAWS); r["t_critic_full"] = tm; rec.append(r)
    return {"stage": "1a", "level": level, "critic": critic, "n": n, "draws_done": draws, "rho": rho, "S": S, "J": J, "J_se": 0.0, "S_se": 0.0,
            "mu_kernel": mu, "zeta1": z1, "E_h2": eh2, "var_exact": var_exact(z1, eh2, n), "constant_kernel": z1 + eh2 <= 1e-14, "degenerate": z1 <= 1e-14,
            "draws": columns(rec)}


def cell_1b(job):
    level, critic, N, ci, draws = job
    P, Q, S, eta, rho = stage1_law(level); T = critics_grid(eta)[critic]; w, xs, ys = grid_states(P)
    mu_G = mu_image_local(P, Q, T)
    S_G = float(np.mean([(r - 1) / (r + 1) for r in R_IMG])); rec = []
    for d in range(draws):
        rng = rng_for(2, ci, d); t0 = time.perf_counter(); JG = np.empty(N); pos, neg = [], []
        for g, r in enumerate(rng.choice(R_IMG, N)):
            s = rng.choice(len(w), r, p=w); Tg = T[xs[s][:, None], ys[s][None, :]]
            ap, am = a_plus(np.diag(Tg)), a_minus(Tg); JG[g] = ap.mean() + am.mean(); pos.append(ap); neg.append(am.ravel())
        pos, neg = np.concatenate(pos), np.concatenate(neg)
        boot = JG[rng.integers(0, N, (BOOT_IMG, N))].mean(1); lo, hi, lcb = np.quantile(boot, [0.025, 0.975, 0.05])
        rec.append({"J": float(JG.mean()), "sd_img": float(JG.std(ddof=1)), "se_pooled": math.sqrt(pos.var(ddof=1) / len(pos) + neg.var(ddof=1) / len(neg)),
                    "boot_lo": float(lo), "boot_hi": float(hi), "boot_lcb": float(lcb), "n_pos": int(len(pos)), "n_neg": int(len(neg)), "t_draw": time.perf_counter() - t0})
    return {"stage": "1b", "level": level, "critic": critic, "n": N, "draws_done": draws, "rho": rho, "S_population": S, "mu_G": mu_G, "S_G": S_G,
            "constant_kernel": bool(np.ptp(T) == 0), "draws": columns(rec)}


def setting2(gen, level):
    from vcs_estim.data import setting_from_mi
    return setting_from_mi(gen, LEVELS_2[level], d=20)


def truth2(job):
    """J(T) for every stage-2 critic and S by Monte Carlo on n_truth joint and n_truth product pairs (own seed), with standard errors."""
    import torch
    gen, level, n_truth = job; s = setting2(gen, level)
    g = torch.Generator().manual_seed(SEED + 7919 * (10 * GENS_2.index(gen) + list(LEVELS_2).index(level)) + 1)
    acc = {c: np.zeros(4) for c in CRITICS_2}; e2 = np.zeros(4); m = 0
    for _ in range(max(1, n_truth // CHUNK)):
        c_n = min(CHUNK, n_truth); xp, yp = s.sample(c_n, g); xq = s.sample(c_n, g)[0]; yq = s.sample(c_n, g)[1]
        ep = torch.tanh(s.pmi(xp.double(), yp.double()) / 2).numpy(); eq = torch.tanh(s.pmi(xq.double(), yq.double()) / 2).numpy()
        for c in CRITICS_2:
            u, v = a_plus(critic_fn(c, ep)), a_minus(critic_fn(c, eq)); acc[c] += [u.sum(), (u * u).sum(), v.sum(), (v * v).sum()]
        e2 += [(ep ** 2).sum(), (ep ** 4).sum(), (eq ** 2).sum(), (eq ** 4).sum()]; m += c_n
    se = lambda s1, s2: math.sqrt(max(0.0, s2 / m - (s1 / m) ** 2) / m)
    out = {"setting": gen, "level": level, "mi": LEVELS_2[level], "param": float(s.param), "n_truth_per_side": m,
           "S": 0.5 * (e2[0] + e2[2]) / m, "S_se": 0.5 * math.hypot(se(e2[0], e2[1]), se(e2[2], e2[3]))}
    out["J"] = {c: (acc[c][0] + acc[c][2]) / m for c in CRITICS_2}; out["J_se"] = {c: math.hypot(se(acc[c][0], acc[c][1]), se(acc[c][2], acc[c][3])) for c in CRITICS_2}
    return out


def cell_2(job):
    import torch
    torch.set_num_threads(1)
    gen, level, critic, n, ci, draws, tr = job; s = setting2(gen, level); rec = []
    for d in range(draws):
        rng = rng_for(3, ci, d); g = torch.Generator().manual_seed(int(rng.integers(2 ** 62))); x, y = s.sample(n, g); x, y = x.double(), y.double()
        t0 = time.perf_counter(); Tm = critic_fn(critic, torch.tanh(s.pmi(x[:, None, :], y[None, :, :]) / 2).numpy()); tm = time.perf_counter() - t0
        idx = torch.arange(n); t0 = time.perf_counter()
        for l in range(1, 17):                                                             # cost of evaluating only K = 16 shifts' pairs
            critic_fn(critic, torch.tanh(s.pmi(x, y[(idx + l) % n]) / 2).numpy())
        ts = time.perf_counter() - t0
        r = cross_readouts(Tm, rng, check=d < CHECK_DRAWS); r["t_critic_full"] = tm; r["t_critic_K16"] = ts; rec.append(r)
    return {"stage": "2", "setting": gen, "level": level, "critic": critic, "n": n, "draws_done": draws, "S": tr["S"], "S_se": tr["S_se"],
            "J": tr["J"][critic], "J_se": tr["J_se"][critic], "constant_kernel": critic == "zero" or LEVELS_2[level] == 0.0,
            "truth": {k: tr[k] for k in ("mi", "param", "n_truth_per_side")}, "draws": columns(rec)}


def cell_3(job):
    """One trained critic (kind, seed): J(T_s) by Monte Carlo, then EVAL draws at each n with the cross-pair readouts."""
    import torch
    from vcs_estim.benchmark import build_roles, train_neural
    kind, seed, draws, smoke, threads = job; torch.set_num_threads(threads); c = STAGE3
    s = setting2(c["setting"], "strong"); assert abs(s.mi - c["mi"]) < 1e-9
    roles, _ = build_roles(s, c["d"], c["d"], c["N_fit"], seed, smoke=smoke)
    critic, info = train_neural(kind, c["variant"], c["lr"], roles, c["d"], c["batch"], 50 if smoke else c["updates"], seed, torch.device("cpu"))
    tf = (lambda f: torch.tanh(f)) if kind == "vcs" else (lambda f: torch.tanh(f / 2))
    g = torch.Generator().manual_seed(SEED + 104729 * (1 + seed) + (0 if kind == "vcs" else 7)); n_t = 20_000 if smoke else c["n_truth"]; parts = []
    with torch.no_grad():
        for side in ("P", "Q"):
            vals = []
            for _ in range(max(1, n_t // 50_000)):
                k_n = min(50_000, n_t); x, y = s.sample(k_n, g)
                if side == "Q":
                    y = s.sample(k_n, g)[1]
                t = tf(critic.pairs(x.float(), y.float())).double().numpy(); vals.append(a_plus(t) if side == "P" else a_minus(t))
            parts.append(np.concatenate(vals))
        J_s = float(parts[0].mean() + parts[1].mean()); J_s_se = math.hypot(parts[0].std() / math.sqrt(len(parts[0])), parts[1].std() / math.sqrt(len(parts[1])))
        res = []
        for ni, n in enumerate((32,) if smoke else c["ns"]):
            rec = []
            for d in range(draws):
                rng = rng_for(4, 2 * seed + (kind == "js"), ni, d); gg = torch.Generator().manual_seed(int(rng.integers(2 ** 62))); x, y = s.sample(n, gg)
                t0 = time.perf_counter(); Tm = tf(critic.matrix(x.float(), y.float())).double().numpy(); tm = time.perf_counter() - t0
                r = cross_readouts(Tm, rng, check=False); r["t_critic_full"] = tm; rec.append(r)
            res.append({"stage": "3", "kind": kind, "seed": seed, "n": n, "draws_done": draws, "J": J_s, "J_se": J_s_se, "fit": {k: v for k, v in info.items() if k != "select_curve"},
                        "draws": columns(rec)})
    return res


# ------------------------------------------------------------------------------------------------ driver
def write(out, name, obj):
    f = out / f"{name}.json"; tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(obj)); tmp.replace(f)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["1a", "1b", "2", "3"]); ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=DRAWS); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); out = Path(a.out) / ("smoke" if a.smoke else ""); out.mkdir(parents=True, exist_ok=True)
    draws = 5 if a.smoke else a.draws; t0 = time.time()
    if a.stage in ("1a", "1b"):
        cells = [(lv, c, n) for lv in LEVELS_1 for c in CRITICS_1 for n in NS]
        jobs = [(lv, c, n, i, draws) for i, (lv, c, n) in enumerate(cells) if not a.smoke or n == 32]
        fn, names = (cell_1a if a.stage == "1a" else cell_1b), [f"{a.stage}_{lv}_{c}_n{n}" for lv, c, n, _, _ in jobs]
    elif a.stage == "2":
        with ProcessPoolExecutor(min(a.workers, 6)) as ex:
            trs = {(t["setting"], t["level"]): t for t in ex.map(truth2, [(g, lv, 20_000 if a.smoke else N_TRUTH) for g in GENS_2 for lv in LEVELS_2])}
        write(out, "2_truths", list(trs.values())); print(f"[r4_t2] stage 2 truths ({time.time() - t0:.0f}s): " + "; ".join(f"{g}/{lv} S {t['S']:.4f}±{t['S_se']:.4f}" for (g, lv), t in trs.items()), flush=True)
        cells = [(g, lv, c, n) for g in GENS_2 for lv in LEVELS_2 for c in CRITICS_2 for n in NS]
        jobs = [(g, lv, c, n, i, draws, trs[(g, lv)]) for i, (g, lv, c, n) in enumerate(cells) if not a.smoke or n == 32]
        fn, names = cell_2, [f"2_{g}_{lv}_{c}_n{n}" for g, lv, c, n, _, _, _ in jobs]
    else:
        c = STAGE3; jobs = [(k, s, draws, a.smoke, max(1, a.workers // 10) if not a.smoke else 1) for k in c["kinds"] for s in (c["seeds"][:1] if a.smoke else c["seeds"])]
        with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
            for res in ex.map(cell_3, jobs):
                for r in res:
                    write(out, f"3_{r['kind']}_s{r['seed']}_n{r['n']}", r); print(f"[r4_t2] 3 {r['kind']} s{r['seed']} n{r['n']}: J(T_s) {r['J']:.4f}±{r['J_se']:.4f} "
                                                                                     f"mean Ĵ {np.mean(r['draws']['J']):.4f} ({time.time() - t0:.0f}s)", flush=True)
        return 0
    todo = [(j, nm) for j, nm in zip(jobs, names) if a.smoke or not (out / f"{nm}.json").exists()]
    print(f"[r4_t2] stage {a.stage}: {len(todo)} of {len(jobs)} cells to run, {draws} draws each", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for (j, nm), r in zip(todo, ex.map(fn, [j for j, _ in todo])):
            write(out, nm, r); tgt = r.get("J", r.get("mu_G"))
            print(f"[r4_t2] {nm}: target {tgt:.4f} mean Ĵ {np.mean(r['draws']['J']):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
