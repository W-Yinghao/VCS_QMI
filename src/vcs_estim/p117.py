"""P117 — v5 NEXT-E-CAL: independent regression calibration of a FIXED fitted critic (P108 generators, oracles and critic class).

Question.  Is the finite-fit readout error of P108 mostly an output-calibration problem (fixable by a bounded monotone map of the score) or does
the score U = T0(W) already lose dependence?  For any bounded g and m(U) = E[C | U] = E_M[eta | U] (C = +1 on P units, −1 on Q units, balanced):

    S − J(g(U)) = E_M[(eta − m(U))²] + E_M[(g(U) − m(U))²]                                                   (orthogonality of E[·|U])
                  └ A: dependence lost by the score ┘  └ B: calibration error of g ┘

so no calibrator can push the J gap below A.  The squared calibration risk on balanced C equals 1 − J(g) on the same sample, so calibrators are
fitted and selected by squared risk (= by J) on an independent CAL sample only.

Data roles (P85 builder, unchanged): FIT N, SELECT N/4 (T0 early stopping / lr choice, as P108), CAL = the P85 'TUNE' role (N/4; independent stream,
not used by any neural fit in P108), EVAL 32 768 per side (readouts only), DIAG = a new independent 100 000-per-side role ('P117-DIAG') used only
for the oracle decomposition.  EVAL never selects anything.  End-to-end arm: the identity critic trained on FIT ∪ CAL (same SELECT), i.e. the same
total independent sample budget as "T0 on FIT + calibrator on CAL".

Calibrators (bounded in [−1, 1]):
    identity        g(U) = U
    latent_affine   g(U) = tanh(alpha · atanh(clip(U, 1 − EPS)) + beta), (alpha, beta) by L-BFGS on CAL squared risk (start 1, 0)
    bins            K equal-frequency bins of the pooled CAL U; per bin (p_hat − q_hat) / (p_hat + q_hat) with p_hat = n_P,bin / n_P (balanced
                    P/Q weights = the squared-regression mean of C); K ∈ BIN_COUNTS chosen by 5-fold CV inside CAL
Selection among {identity, latent_affine, bins(K*)} by the 5-fold CV squared risk inside CAL (bins' CV risk is the minimum over K — disclosed).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .benchmark import Role, _peak_mem, build_roles, eta_of, train_neural, truths
from .data import pad_sides, setting_from_mi
from .p108 import CONDITIONS, LRS, _eval_T, readouts
from .synthetic import _gen

PROTOCOL = "P117"
BUDGET = 1000
NS = (4096, 16384)
SEEDS = (0, 1, 2, 3, 4)
EPS = 1e-6
BIN_COUNTS = (4, 8, 16, 32)
CV_FOLDS = 5
DIAG_N = 100_000
M_BINS = (100, 400, 1600)
CANDS = ("identity", "latent_affine", "bins")


# ------------------------------------------------------------------------------------------------------------------- risks / readouts (numpy)
def sq_risk(gp: np.ndarray, gq: np.ndarray) -> float:
    """Balanced squared regression risk of C (+1 on P, −1 on Q); equals 1 − J_hat(g) on the same units."""
    return float(0.5 * np.mean((gp - 1.0) ** 2) + 0.5 * np.mean((gq + 1.0) ** 2))


def j_np(gp: np.ndarray, gq: np.ndarray) -> float:
    return float(np.mean(gp - 0.5 * gp ** 2) + np.mean(-gq - 0.5 * gq ** 2))


# ------------------------------------------------------------------------------------------------------------------- calibrators
class Identity:
    name = "identity"

    def fit(self, up, uq):
        return self

    def __call__(self, u):
        return np.asarray(u, dtype=np.float64)

    def info(self):
        return {}


class LatentAffine:
    name = "latent_affine"

    def __init__(self):
        self.alpha, self.beta, self.converged = 1.0, 0.0, None

    @staticmethod
    def _lat(u):
        return np.arctanh(np.clip(np.asarray(u, dtype=np.float64), -1 + EPS, 1 - EPS))

    def fit(self, up, uq):
        lp, lq = torch.as_tensor(self._lat(up)), torch.as_tensor(self._lat(uq))
        th = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=200, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); gp, gq = torch.tanh(th[0] * lp + th[1]), torch.tanh(th[0] * lq + th[1])
            loss = 0.5 * ((gp - 1) ** 2).mean() + 0.5 * ((gq + 1) ** 2).mean(); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.alpha, self.beta = float(th[0]), float(th[1])
        st = opt.state[opt._params[0]]; self.converged = {"n_iter": int(st.get("n_iter", -1)), "final_grad_abs_max": float(th.grad.abs().max()) if th.grad is not None else None}
        return self

    def __call__(self, u):
        return np.tanh(self.alpha * self._lat(u) + self.beta)

    def info(self):
        return {"alpha": self.alpha, "beta": self.beta, **(self.converged or {})}


class Bins:
    name = "bins"

    def __init__(self, k: int):
        self.k, self.edges, self.vals, self.k_eff, self.empty = k, None, None, None, 0

    def fit(self, up, uq):
        up, uq = np.asarray(up, np.float64), np.asarray(uq, np.float64); pooled = np.concatenate([up, uq])
        inner = np.unique(np.quantile(pooled, np.linspace(0, 1, self.k + 1)[1:-1]))       # ties (near-constant scores) merge bins
        self.edges = inner; kb = len(inner) + 1; self.k_eff = kb
        bp, bq = np.searchsorted(inner, up, side="right"), np.searchsorted(inner, uq, side="right")
        cp, cq = np.bincount(bp, minlength=kb) / len(up), np.bincount(bq, minlength=kb) / len(uq)
        den = cp + cq; self.empty = int((den == 0).sum())
        self.vals = np.where(den > 0, (cp - cq) / np.where(den > 0, den, 1.0), 0.0)
        return self

    def __call__(self, u):
        return self.vals[np.searchsorted(self.edges, np.asarray(u, np.float64), side="right")]

    def info(self):
        return {"k": self.k, "k_eff": self.k_eff, "empty_bins": self.empty}


def _folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def cv_select(up: np.ndarray, uq: np.ndarray, seed: int) -> dict:
    """5-fold CV squared risk inside CAL for identity, latent_affine and bins(K) (K ∈ BIN_COUNTS); P and Q units folded separately.
    Returns per-candidate CV risk, K*, the selected candidate and the calibrators refitted on the full CAL.  EVAL is not an argument."""
    fp_, fq_ = _folds(len(up), CV_FOLDS, seed), _folds(len(uq), CV_FOLDS, seed + 1)
    risks = {"identity": [], "latent_affine": [], **{f"bins_{k}": [] for k in BIN_COUNTS}}
    for i in range(CV_FOLDS):
        trp, trq = np.concatenate([f for j, f in enumerate(fp_) if j != i]), np.concatenate([f for j, f in enumerate(fq_) if j != i])
        vp, vq = up[fp_[i]], uq[fq_[i]]
        risks["identity"].append(sq_risk(vp, vq))
        la = LatentAffine().fit(up[trp], uq[trq]); risks["latent_affine"].append(sq_risk(la(vp), la(vq)))
        for k in BIN_COUNTS:
            b = Bins(k).fit(up[trp], uq[trq]); risks[f"bins_{k}"].append(sq_risk(b(vp), b(vq)))
    cvr = {k: float(np.mean(v)) for k, v in risks.items()}
    kstar = min(BIN_COUNTS, key=lambda k: cvr[f"bins_{k}"])
    cand_risk = {"identity": cvr["identity"], "latent_affine": cvr["latent_affine"], "bins": cvr[f"bins_{kstar}"]}
    sel = min(cand_risk, key=cand_risk.get)
    fitted = {"identity": Identity(), "latent_affine": LatentAffine().fit(up, uq), "bins": Bins(kstar).fit(up, uq)}
    return {"cv_risk_all": cvr, "cv_risk": cand_risk, "k_star": kstar, "selected": sel, "calibrators": fitted}


# ------------------------------------------------------------------------------------------------------------------- oracle decomposition
def m_hat(u: np.ndarray, eta: np.ndarray, bins: int) -> tuple[np.ndarray, int]:
    """E_M[eta | U] by equal-frequency bins of the pooled DIAG U (balanced: n_P = n_Q, so a plain mean over pooled units in a bin)."""
    inner = np.unique(np.quantile(u, np.linspace(0, 1, bins + 1)[1:-1])); b = np.searchsorted(inner, u, side="right"); kb = len(inner) + 1
    s = np.bincount(b, weights=eta, minlength=kb); c = np.bincount(b, minlength=kb)
    return (s / np.maximum(c, 1))[b], kb


def decomposition(u: np.ndarray, eta: np.ndarray, g_vals: dict) -> dict:
    """On pooled DIAG units (first half P, second half Q, equal sizes): A = E_M(eta − m)², B_g = E_M(g − m)², gap_g = E_M(eta − g)² (= S − J(g) in
    population), residual_g = gap − A − B = 2 E_M[(eta − m)(m − g)] → 0 as the m-bins refine (m estimated, not known)."""
    out = {}
    for nb in M_BINS:
        m, kb = m_hat(u, eta, nb); A = float(np.mean((eta - m) ** 2)); row = {"bins_effective": kb, "A_score_loss": A, "per_calibrator": {}}
        for name, g in g_vals.items():
            gap = float(np.mean((eta - g) ** 2)); B = float(np.mean((g - m) ** 2))
            row["per_calibrator"][name] = {"gap": gap, "B_calibration": B, "residual": gap - A - B}
        out[str(nb)] = row
    return out


# ------------------------------------------------------------------------------------------------------------------- data helpers
def extra_role(st, ds: int, dt: int, n: int, N: int, seed: int, tag: str) -> Role:
    """A further independent role built exactly like benchmark.build_roles (own named streams; padding as the other roles)."""
    key = (st.name, round(st.mi, 6), ds, dt, N, seed); k = dt - ds
    gp = _gen((tag + "-P",) + key); base = st.sample_base(n, gp); xp, yp = st.from_base(base, st.param); xp, yp = xp.double(), yp.double()
    gq = _gen((tag + "-Q",) + key)
    xq = st.from_base(st.sample_base(n, gq), st.param)[0].double(); yq = st.from_base(st.sample_base(n, gq), st.param)[1].double()
    if k:
        xp, yp = pad_sides(xp, yp, k, _gen((tag + "-PAD-P",) + key)); xq, yq = pad_sides(xq, yq, k, _gen((tag + "-PAD-Q",) + key))
    return Role(xp, yp, xq, yq, None, None)


def concat_roles(a: Role, b: Role) -> Role:
    return Role(torch.cat([a.xp, b.xp]), torch.cat([a.yp, b.yp]), torch.cat([a.xq, b.xq]), torch.cat([a.yq, b.yq]), None, None)


def assert_disjoint(a: Role, b: Role, name: str = "") -> None:
    """Continuous independent draws never share a row; a shared row means a role was reused (CAL / EVAL leakage)."""
    def keys(r):
        return {tuple(np.round(v, 12)) for v in torch.cat([r.xp[:, :2], r.yp[:, :2]], 1).numpy()} | \
               {tuple(np.round(v, 12)) for v in torch.cat([r.xq[:, :2], r.yq[:, :2]], 1).numpy()}
    inter = keys(a) & keys(b)
    if inter:
        raise AssertionError(f"roles share {len(inter)} rows {name}")


def _fit_t0(kind, roles, dt, B, budget, lrs, seed, device):
    group = []
    for lr in lrs:
        critic, info = train_neural(kind, "product", lr, roles, dt, B, budget, seed, device); group.append((info["select_risk"], lr, critic, info))
    best = min(group, key=lambda r: r[0])
    return best[2], {"lr": best[1], "select_risk": best[0], "selected_update": best[3]["selected_update"], "fit_seconds_chosen": best[3]["fit_seconds"],
                     "tuning_seconds_total": sum(g[3]["fit_seconds"] for g in group)}


def _metrics(gp, gq, ep, eq, S) -> dict:
    tp, tq = torch.as_tensor(gp), torch.as_tensor(gq); r = readouts(tp, tq)
    return {**r, "J_err": r["J"] - S, "S_plug_err": r["S_plug"] - S, "posterior_mse": float(0.5 * np.mean((gp - ep) ** 2) + 0.5 * np.mean((gq - eq) ** 2))}


# ------------------------------------------------------------------------------------------------------------------- one cell
def cal_cell(cond: str, N: int, seed: int, device, smoke: bool = False) -> dict:
    name, ds, dt, I = CONDITIONS[cond]; st = setting_from_mi(name, I, ds)
    if name == "xor_mixture":
        ds = dt = st.d
    t0 = time.time(); roles, sizes = build_roles(st, ds, dt, N, seed, smoke); tr = truths(st, ds, roles["TRUTH"]); S = tr["S"]
    CAL, E = roles["TUNE"], roles["EVAL"]
    D = extra_role(st, ds, dt, 5000 if smoke else DIAG_N, N, seed, "P117-DIAG")
    for a, b, nm in ((CAL, E, "CAL/EVAL"), (CAL, roles["FIT"], "CAL/FIT"), (CAL, roles["SELECT"], "CAL/SELECT"), (D, E, "DIAG/EVAL")):
        assert_disjoint(a, b, nm)
    budget = 100 if smoke else BUDGET; lrs = (5e-4,) if smoke else LRS; B = min(256, N)
    ep, eq = eta_of(st, E.xp, E.yp, ds).double().numpy(), eta_of(st, E.xq, E.yq, ds).double().numpy()
    dp, dq = eta_of(st, D.xp, D.yp, ds).double().numpy(), eta_of(st, D.xq, D.yq, ds).double().numpy()
    out = {"protocol": PROTOCOL, "condition": cond, "setting": name, "d_signal": ds, "d_total": dt, "I": I, "N": N, "seed": seed,
           "sizes": {**sizes, "CAL": sizes["TUNE"], "DIAG": D.n}, "truth": {k: tr[k] for k in ("S", "S_se")}, "per_kind": {}, "device": str(device)}
    for kind in ("vcs", "js"):
        half = 1.0 if kind == "vcs" else 0.5
        _peak_mem(device); critic, fi = _fit_t0(kind, roles, dt, B, budget, lrs, seed, device)
        tc = time.time(); up_c, uq_c = (t.numpy() for t in _eval_T(critic, CAL, device, half))
        sel = cv_select(up_c, uq_c, seed * 31 + (0 if kind == "vcs" else 1)); cal_seconds = time.time() - tc
        up_e, uq_e = (t.numpy() for t in _eval_T(critic, E, device, half)); up_d, uq_d = (t.numpy() for t in _eval_T(critic, D, device, half))
        mech = {}
        for c in CANDS:
            g = sel["calibrators"][c]; mech[c] = {**_metrics(g(up_e), g(uq_e), ep, eq, S), "info": g.info(), "cv_risk": sel["cv_risk"][c]}
        mech["selected"] = {**mech[sel["selected"]], "which": sel["selected"]}
        u_d = np.concatenate([up_d, uq_d]); eta_d = np.concatenate([dp, dq])
        dec = decomposition(u_d, eta_d, {c: sel["calibrators"][c](u_d) for c in CANDS})
        # end-to-end at equal total independent budget: identity critic trained on FIT ∪ CAL (same SELECT, same lr grid / budget)
        r2 = {**roles, "FIT": concat_roles(roles["FIT"], CAL)}
        critic2, fi2 = _fit_t0(kind, r2, dt, min(256, r2["FIT"].n), budget, lrs, seed, device)
        up2, uq2 = (t.numpy() for t in _eval_T(critic2, E, device, half))
        out["per_kind"][kind] = {"T0_fit": fi, "calibration_seconds": cal_seconds, "cv_risk_all": sel["cv_risk_all"], "k_star": sel["k_star"],
                                 "selected": sel["selected"], "mechanism_same_T0": mech, "decomposition_DIAG": dec,
                                 "end_to_end_equal_budget": {"identity_on_FIT_plus_CAL": {**_metrics(up2, uq2, ep, eq, S), "T0_fit": fi2},
                                                             "T0_on_FIT_plus_selected_calibrator_on_CAL": {**mech["selected"]}},
                                 "memory": _peak_mem(device)}
    out["wall_seconds"] = time.time() - t0
    return out


def cal_cells(Ns=NS, seeds=SEEDS) -> list[tuple]:
    return [(c, N, s) for c in CONDITIONS for N in Ns for s in seeds]
