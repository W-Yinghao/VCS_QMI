"""P121 — v6 V6-THEORY (VCS_Server_Tasks_v6_CN.md §3; VCS_Theory_to_Experiments_v6_CN.md §2–6).  CPU-only exact / finite checks:

1. finite latent model (K latent classes, L observed states, strictly positive emissions A_uy and prior pi_y): exact P, Q, eta, M, S;
   the posterior-kernel identity P_uv / (p_u p_v) = sum_y P(y|u) P(y|v) / pi_y;
2. nested deterministic compressions W -> Z -> s with a bounded scorer depending only on s: exact three-term squared decomposition
   S_W - J(T) = E(eta_W - eta_Z)^2 + E(eta_Z - eta_s)^2 + E(eta_s - T)^2 (float64, relative tolerance 1e-10);
3. Gram realisability: G* = kappa + log(P/Q) / (2a) diagnostics (symmetry, range, diagonal, negative eigenvalue mass, spectral tail), then direct
   optimisation of unit vectors z_u in R^d (d in {2, 4, 8}) with the full weighted population J — all random starts reported, no success threshold,
   NOT an optimality certificate; contexts: free per-pair oracle T = eta (J = S), constant zero (J = 0);
4. two-sided Lipschitz bound a^2 sech^4(B) ||G - G*||_M^2 <= S - J(G) <= a^2 ||G - G*||_M^2 on the realised fits;
5. gradient formulas: dA_c/ds = (c - T)(1 - T^2) f'(s), affine peaks s_pm = kappa -/+ log2/(2a) with value 32a/27, the tangent gradient
   grad_r A_c = (c - T)(1 - T^2) f'(s) (z_j - s z_i) / ||r_i||, the matched-JS single-pair derivative T - c vs VCS (T - c)(1 - T^2) — all against
   autograd / finite differences.

Everything here is an algebraic / numerical check of definitions on a finite model; nothing is a CIFAR or SSL result.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .geometry_evidence_core import (contribution_gradient, curve_logits, curve_slope, gram_diagnostics, j_population, posterior,
                                     projection_decomposition, push_joint)

TOL_REL = 1e-10
GRAM_SETTINGS = ((2.0, 0.5), (1.0, 0.5), (3.0, 0.5), (2.0, 0.25), (2.0, 0.75))
DIMS = (2, 4, 8)
STARTS = 3
STEPS = 2000


# ------------------------------------------------------------------------------------------------------------ finite latent model
def latent(n_states: int = 12, n_classes: int = 4, seed: int = 73, gamma: float = 1.0, floor: float = 0.02) -> dict:
    """Strictly positive emissions A_uy = P(U=u | Y=y) and prior pi_y.  gamma tempers the emission contrast (A ∝ U^gamma with U ~ Unif(floor, 1)),
    so larger gamma = stronger dependence between two views that share Y; gamma = 1 reproduces the v6 support core's latent_model(seed)."""
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(n_classes))
    a = rng.uniform(floor, 1.0, (n_states, n_classes)) ** gamma
    a /= a.sum(axis=0, keepdims=True)
    marg = a @ pi
    P = (a * pi) @ a.T
    Q = np.outer(marg, marg)
    eta, M, S = posterior(P, Q)
    return {"P": P, "Q": Q, "M": M, "eta": eta, "S": S, "marginal": marg, "emission": a, "prior": pi,
            "class_posterior": (a * pi) / marg[:, None], "gamma": gamma, "seed": seed}


CONDITIONS = {"core_seed73": dict(seed=73, gamma=1.0), "weak": dict(seed=11, gamma=0.5), "moderate": dict(seed=11, gamma=2.0),
              "strong": dict(seed=11, gamma=4.0)}


def kernel_identity(d: dict) -> dict:
    u = d["class_posterior"] / np.sqrt(d["prior"])
    lhs, rhs = d["P"] / d["Q"], u @ u.T
    rel = float(np.max(np.abs(lhs - rhs) / np.abs(lhs)))
    ratio = d["P"] / d["Q"]
    glob = float(np.sum(d["Q"] * ratio)); anchor = ratio @ d["marginal"]
    return {"max_rel_error": rel, "pass": rel <= TOL_REL, "global_normalisation": glob, "max_anchor_normalisation_error": float(np.max(np.abs(anchor - 1))),
            "log_kernel_min_eig": float(np.linalg.eigvalsh(np.log(ratio)).min()), "kernel_min_eig": float(np.linalg.eigvalsh(ratio).min())}


# ------------------------------------------------------------------------------------------------------------ nested decomposition
def random_nested_maps(n: int, rng, k_mid: int, k_low: int) -> tuple[np.ndarray, np.ndarray]:
    """W (n states per side) -> Z (k_mid) -> coarse (k_low); every level is a deterministic surjective map."""
    zmap = np.concatenate([np.arange(k_mid), rng.integers(0, k_mid, n - k_mid)]); rng.shuffle(zmap)
    cmap = np.concatenate([np.arange(k_low), rng.integers(0, k_low, k_mid - k_low)]); rng.shuffle(cmap)
    return zmap, cmap


def nested_check(d: dict, n_draws: int = 50, seed: int = 0) -> dict:
    """Pair-level labels: Z = (z(u), z(v)); s = a symmetric scalar of the coarse pair (sorted pair index); T = tanh(scorer(s)) arbitrary bounded."""
    n = d["P"].shape[0]; rng = np.random.default_rng(seed); rows = []
    for k in range(n_draws):
        k_mid = int(rng.integers(3, n)); k_low = int(rng.integers(2, k_mid + 1))
        zmap, cmap = random_nested_maps(n, rng, k_mid, k_low)
        zlab = zmap[:, None] * n + zmap[None, :]
        cl = cmap[zmap]; lo, hi = np.minimum(cl[:, None], cl[None, :]), np.maximum(cl[:, None], cl[None, :]); slab = lo * k_low + hi
        vals = rng.normal(size=slab.max() + 1); t = np.tanh(vals[slab])
        out = projection_decomposition(d["P"], d["Q"], zlab, slab, t)
        rows.append({"draw": k, "k_mid": k_mid, "k_low": k_low, **out, "rel_residual": abs(out["residual"]) / max(out["gap"], 1e-300)})
    worst = max(r["rel_residual"] for r in rows)
    # data-processing check on the pushed-forward joint: S(compressed) = S - E(eta_W - eta_Z)^2 exactly
    zmap, _ = random_nested_maps(n, np.random.default_rng(seed + 1), 6, 3)
    pp, qq = push_joint(d["P"], zmap), push_joint(d["Q"], zmap)
    S_small = posterior(pp, qq)[2]; eta_small = posterior(pp, qq)[0]
    lossZ = float(np.sum(d["M"] * (d["eta"] - eta_small[zmap[:, None], zmap[None, :]]) ** 2))
    return {"n_draws": n_draws, "max_rel_residual": worst, "pass": worst <= TOL_REL, "example": rows[0],
            "data_processing": {"S_W": d["S"], "S_Z": S_small, "S_W_minus_S_Z": d["S"] - S_small, "E_eta_W_minus_eta_Z_sq": lossZ,
                                "abs_diff": abs(d["S"] - S_small - lossZ)}}


# ------------------------------------------------------------------------------------------------------------ Gram realisability
def gram_target(d: dict, a: float, kappa: float) -> np.ndarray:
    return kappa + np.log(d["P"] / d["Q"]) / (2 * a)


def gram_target_diagnostics(G: np.ndarray, k_tail: int = 4) -> dict:
    base = gram_diagnostics(G)
    vals = np.sort(np.linalg.eigvalsh((G + G.T) / 2))[::-1]
    tot = float(np.abs(vals).sum())
    return {**base, "eig_desc": [float(v) for v in vals], "frac_negative_eigs": float((vals < 0).mean()),
            "tail_mass_after_top%d" % k_tail: float(np.abs(vals[k_tail:]).sum() / tot) if tot > 0 else 0.0,
            "min": float(G.min()), "max": float(G.max()), "diag_mean": float(np.diag(G).mean())}


def fit_unit_vectors(d: dict, a: float, kappa: float, dim: int, start: int, steps: int = STEPS, lr: float = 0.05, two_tower: bool = False) -> dict:
    """Maximise the full weighted population J over unit vectors z_u in R^dim (same z on both sides; symmetric P, Q), T_uv = tanh(a (z_u.z_v - kappa)).
    Adam on an unconstrained r_u with z = r / ||r||; float64; fixed steps; returns the final state (no early stopping, no success threshold)."""
    P, Q, M, eta = (torch.as_tensor(d[k]) for k in ("P", "Q", "M", "eta"))
    g = torch.Generator().manual_seed(1000 * dim + 97 * start + int(100 * a) + int(1000 * kappa))
    r = torch.randn(P.shape[0], dim, generator=g, dtype=torch.float64).requires_grad_(True)
    w = torch.randn(P.shape[0], dim, generator=g, dtype=torch.float64).requires_grad_(True)   # second tower (used only if two_tower)
    opt = torch.optim.Adam([r, w] if two_tower else [r], lr=lr); hist = []
    gram = lambda: (r / r.norm(dim=1, keepdim=True)) @ ((w / w.norm(dim=1, keepdim=True)) if two_tower else (r / r.norm(dim=1, keepdim=True))).T
    for step in range(steps + 1):
        G = gram(); T = torch.tanh(a * (G - kappa))
        J = (P * (T - 0.5 * T ** 2)).sum() + (Q * (-T - 0.5 * T ** 2)).sum()
        if step % 250 == 0 or step == steps:
            hist.append((step, float(J.detach())))
        if step == steps:
            break
        opt.zero_grad(); (-J).backward(); opt.step()
    with torch.no_grad():
        G = gram().numpy(); T = np.tanh(a * (G - kappa))
    Gs = gram_target(d, a, kappa); Mn = d["M"]
    Jv = j_population(d["P"], d["Q"], T); gap = d["S"] - Jv
    gdist2 = float(np.sum(Mn * (G - Gs) ** 2))
    fstar = 0.5 * np.log(d["P"] / d["Q"]); B = float(max(np.abs(fstar).max(), np.abs(a * (G - kappa)).max()))
    lower, upper = a ** 2 / math.cosh(B) ** 4 * gdist2, a ** 2 * gdist2
    return {"geometry": "two_tower_cross_gram" if two_tower else "shared_unit_gram", "a": a, "kappa": kappa, "dim": dim, "start": start, "steps": steps, "J": Jv, "S": d["S"], "gap_S_minus_J": gap, "J_over_S": Jv / d["S"],
            "posterior_mse": float(np.sum(Mn * (T - d["eta"]) ** 2)), "gram_dist2_M": gdist2, "B": B, "lipschitz_lower": lower, "lipschitz_upper": upper,
            "lipschitz_holds": bool(lower <= gap + 1e-12 and gap <= upper + 1e-12), "lower_over_gap": lower / gap if gap > 0 else float("nan"),
            "J_trace": hist}


def gram_block(name: str, d: dict, settings=GRAM_SETTINGS, dims=DIMS, starts: int = STARTS, steps: int = STEPS) -> tuple[list, list]:
    diag_rows, fit_rows = [], []
    for a, kappa in settings:
        Gs = gram_target(d, a, kappa); gd = gram_target_diagnostics(Gs)
        diag_rows.append({"condition": name, "a": a, "kappa": kappa, **{k: v for k, v in gd.items() if k != "eig_desc"}, "eig_desc": gd["eig_desc"]})
        for dim in dims:
            for s in range(starts):
                fit_rows.append({"condition": name, **fit_unit_vectors(d, a, kappa, dim, s, steps)})
                # context (v6 theory note §4: the unit diagonal G_uu = 1 is a finite-model constraint; a two-tower cross-Gram has no such coupling)
                fit_rows.append({"condition": name, **fit_unit_vectors(d, a, kappa, dim, s, steps, two_tower=True)})
    return diag_rows, fit_rows


# ------------------------------------------------------------------------------------------------------------ gradient formulas
def gradient_checks(seed: int = 5) -> dict:
    out = {}
    # (a) dA_c/ds analytic vs autograd, affine and both curvatures, both labels
    errs = []
    for curv in (-0.25, 0.0, 0.25):
        for a, kappa in ((1.0, 0.5), (2.0, 0.5), (3.0, 0.25)):
            s = torch.linspace(-0.999, 0.999, 401, dtype=torch.float64, requires_grad=True)
            f = a * s - a * kappa + a * curv * s * (1 - s); T = torch.tanh(f)
            for c in (1.0, -1.0):
                A = c * T - 0.5 * T ** 2; (g,) = torch.autograd.grad(A.sum(), s, retain_graph=True)
                ana = contribution_gradient(c, s.detach().numpy(), a, kappa, curv)
                errs.append(float(np.max(np.abs(g.numpy() - ana))))
    out["dA_ds_max_abs_err"] = max(errs)
    # (b) affine peaks and value 32a/27 (dense grid on s in R, then the restricted-domain maximum on [-1, 1])
    peaks = []
    for a, kappa in ((1.0, 0.5), (2.0, 0.5), (3.0, 0.5), (2.0, 0.25), (2.0, 0.75), (0.5, 0.5)):
        s = np.linspace(-4, 4, 2_000_001)
        for c, pred in ((1.0, kappa - math.log(2) / (2 * a)), (-1.0, kappa + math.log(2) / (2 * a))):
            g = np.abs(contribution_gradient_unbounded(c, s, a, kappa)); i = int(np.argmax(g))
            in_dom = -1 <= pred <= 1; sd = np.linspace(-1, 1, 200_001); gd = np.abs(contribution_gradient(c, sd, a, kappa, 0.0)); j = int(np.argmax(gd))
            peaks.append({"a": a, "kappa": kappa, "c": c, "pred_peak": pred, "grid_peak": float(s[i]), "pred_max": 32 * a / 27, "grid_max": float(g[i]),
                          "peak_in_domain": in_dom, "domain_argmax": float(sd[j]), "domain_max": float(gd[j])})
    out["affine_peaks"] = peaks
    out["peak_max_abs_err"] = max(abs(p["pred_peak"] - p["grid_peak"]) for p in peaks)
    out["peak_value_max_rel_err"] = max(abs(p["pred_max"] - p["grid_max"]) / p["pred_max"] for p in peaks)
    # (c) tangent gradient w.r.t. the unnormalised r_i
    g = torch.Generator().manual_seed(seed); terr = []
    for a, kappa, curv in ((2.0, 0.5, 0.0), (2.0, 0.5, 0.25), (3.0, 0.25, -0.25)):
        for c in (1.0, -1.0):
            ri = torch.randn(16, dtype=torch.float64, generator=g) * 3; zj = torch.randn(16, dtype=torch.float64, generator=g); zj = zj / zj.norm()
            r = ri.clone().requires_grad_(True); z = r / r.norm(); s = z @ zj; f = a * s - a * kappa + a * curv * s * (1 - s); T = torch.tanh(f)
            A = c * T - 0.5 * T ** 2; (gr,) = torch.autograd.grad(A, r)
            sv, Tv = float(s.detach()), float(T.detach()); fp = a * (1 + curv * (1 - 2 * sv))
            ana = (c - Tv) * (1 - Tv ** 2) * fp * (zj - sv * z.detach()) / ri.norm()
            terr.append(float((gr - ana).abs().max()))
    out["tangent_grad_max_abs_err"] = max(terr)
    # s = 1: tangent direction vanishes even though the scalar gate is non-zero
    z1 = torch.tensor([1.0, 0, 0], dtype=torch.float64); r = (2 * z1).clone().requires_grad_(True); s = (r / r.norm()) @ z1
    T = torch.tanh(2.0 * (s - 0.5)); (gr,) = torch.autograd.grad(1.0 * T - 0.5 * T ** 2, r)
    out["s_equals_1_grad_norm"] = float(gr.norm()); out["s_equals_1_scalar_gate"] = float(((1 - T) * (1 - T ** 2) * 2.0).detach())
    # (d) matched JS vs VCS single-pair derivative w.r.t. f (JS logistic logit 2f): JS = T - c, VCS = (T - c)(1 - T^2); equal at f = 0
    f = torch.linspace(-3, 3, 601, dtype=torch.float64, requires_grad=True); T = torch.tanh(f); jerr, verr = [], []
    for c in (1.0, -1.0):
        Ljs = torch.nn.functional.softplus(-2 * c * f); (gj,) = torch.autograd.grad(Ljs.sum(), f, retain_graph=True)
        Lv = -(c * T - 0.5 * T ** 2); (gv,) = torch.autograd.grad(Lv.sum(), f, retain_graph=True)
        jerr.append(float((gj - (T - c)).abs().max())); verr.append(float((gv - (T - c) * (1 - T ** 2)).abs().max()))
    i0 = 300
    out["js_dL_df_max_abs_err"] = max(jerr); out["vcs_dL_df_max_abs_err"] = max(verr)
    t0 = float(torch.tanh(f[i0]))  # f[300] = 0
    out["js_vs_vcs_at_f0"] = {"f": float(f[i0]), "c=+1": {"js": t0 - 1.0, "vcs": (t0 - 1.0) * (1 - t0 ** 2)},
                              "c=-1": {"js": t0 + 1.0, "vcs": (t0 + 1.0) * (1 - t0 ** 2)}}
    # (e) curve scorer: endpoints, derivative bounds, actual zero != kappa when lambda != 0
    from .geometry_evidence_core import actual_zero
    cur = []
    for curv in (-0.25, 0.0, 0.25):
        s = np.linspace(-1, 1, 100_001); sl = curve_slope(s, 2.0, curv)
        cur.append({"curvature": curv, "f0": float(curve_logits(0.0, 2.0, 0.5, curv)), "f1": float(curve_logits(1.0, 2.0, 0.5, curv)),
                    "min_slope_over_a": float(sl.min() / 2.0), "max_slope_over_a": float(sl.max() / 2.0), "actual_zero": actual_zero(2.0, 0.5, curv)})
    out["curve"] = cur
    out["pass"] = bool(out["dA_ds_max_abs_err"] < 1e-12 and out["peak_max_abs_err"] < 1e-5 and out["peak_value_max_rel_err"] < 1e-8
                       and out["tangent_grad_max_abs_err"] < 1e-12 and out["js_dL_df_max_abs_err"] < 1e-12 and out["vcs_dL_df_max_abs_err"] < 1e-12)
    return out


def contribution_gradient_unbounded(c, s, a, kappa):
    """Affine dA_c/ds on an unbounded s grid (to locate the analytic peaks even when they fall outside [-1, 1])."""
    t = np.tanh(a * (np.asarray(s) - kappa)); return (c - t) * (1 - t ** 2) * a
