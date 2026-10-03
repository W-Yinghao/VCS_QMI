"""P116 (v5 NEXT-E-SOLVE) — solver decomposition on the T1 linear critic class phi(z, n) = [z (2n − 1), 1].

Three algorithms, one shared explicit FIT / VAL split, one shared budget table:

  A1  ridge_tanh_calibrated   ridge solution of the RAW linear J (quadratic sub-problem, exact), output scale c chosen on VAL from 25 values,
                              T = tanh(c phi w).  Identical numerics to scripts/precheck_d_tests.py::closed_form_critic (asserted in tests).
  A2  ridge_then_bounded_J    start from A1's v0 = c w and continue to optimise the ORIGINAL bounded J of T = tanh(phi v) on FIT with L-BFGS
                              (strong Wolfe), with the ridge-grid penalty lam sc |v|^2 / 2 (lam on VAL, as A3) and VAL early stopping; A1 (step 0)
                              is a candidate.  Non-convex: no global-optimum claim.
  A3  js_lbfgs                matched JS (Deep-InfoMax logistic, f = phi v, loss E_P softplus(−f) + E_Q softplus(f) + lam sc |v|^2 / 2) on the
                              same phi, L-BFGS from zero per lam of the ridge grid, lam chosen on VAL JS; common squared-score map T = tanh(f / 2).

Budgets are counted in objective+gradient evaluations ("closures") on FIT.  A2 and A3 run once to the largest budget in chunks; the budget-B model
is the best-VAL checkpoint (A2) / the last checkpoint (A3, per lam, with B split evenly over the lam grid) reached within B closures — identical to
separate runs with budget B up to the last chunk boundary.  A1's cost (4 linear solves + 100 VAL evaluations) is reported separately and is part
of A2's cost.  Nothing is selected on EVAL.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

LAM_GRID = (1e-3, 1e-2, 1e-1, 1.0)
C_GRID_SPEC = (-1.0, 1.5, 25)  # torch.logspace(-1, 1.5, 25), as in closed_form_critic
BUDGETS = (50, 200, 800)


def phi(z: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return torch.cat((z * (2 * n.float() - 1)[:, None], torch.ones(len(z), 1, device=z.device)), 1).double()


def make_split(n_items: int, split_seed: int, val_frac: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """Data-split role (independent of any initialisation RNG): the historical 80 / 20 rule on a Generator seeded with `split_seed`."""
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_items, generator=g); nv = max(8, int(val_frac * n_items))
    return perm[nv:], perm[:nv]


def bounded_j(fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    tp, tn = torch.tanh(fp), torch.tanh(fn)
    return (tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean()


def js_loss(fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    return F.softplus(-fp).mean() + F.softplus(fn).mean()


def js_value(fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """Deep-InfoMax JS value E_P[−softplus(−f)] − E_Q[softplus(f)] + log 4 (the P105 / T1 statistic)."""
    return (-F.softplus(-fp)).mean() - F.softplus(fn).mean() + float(np.log(4.0))


@dataclass
class LinearCritic:
    v: torch.Tensor                 # float64, length d + 1
    kind: str                       # "vcs": T = tanh(f); "js": T = tanh(f / 2)
    meta: dict[str, Any] = field(default_factory=dict)

    def f(self, z: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return phi(z, n) @ self.v

    def T(self, z: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        f = self.f(z, n)
        return torch.tanh(f) if self.kind == "vcs" else torch.tanh(0.5 * f)


@dataclass
class Design:
    """FIT positives / negatives split into train (ti) and VAL (vi) parts, as float64 feature matrices."""
    pp: torch.Tensor
    pn: torch.Tensor
    vp: torch.Tensor
    vn: torch.Tensor

    @classmethod
    def build(cls, z, n, n_neg, ti, vi) -> "Design":
        return cls(phi(z[ti], n[ti]), phi(z[ti], n_neg[ti]), phi(z[vi], n[vi]), phi(z[vi], n_neg[vi]))

    @property
    def sc(self) -> float:
        A = 0.5 * (self.pp.T @ self.pp / len(self.pp) + self.pn.T @ self.pn / len(self.pn))
        return float(torch.diag(A).mean())


# ------------------------------------------------------------------------------------------------------------------- A1
def ridge_tanh_calibrated(D: Design) -> LinearCritic:
    t0 = time.perf_counter()
    pp, pn, vp, vn = D.pp, D.pn, D.vp, D.vn
    d = pp.mean(0) - pn.mean(0); A = 0.5 * (pp.T @ pp / len(pp) + pn.T @ pn / len(pn)); sc = float(torch.diag(A).mean())
    best = (-9, None, None, None)
    for lam in LAM_GRID:
        w = 0.5 * torch.linalg.solve(A + lam * sc * torch.eye(len(A), dtype=torch.float64, device=A.device), d)
        for c in torch.logspace(*C_GRID_SPEC[:2], int(C_GRID_SPEC[2])):
            tp, tn = torch.tanh(float(c) * (vp @ w)), torch.tanh(float(c) * (vn @ w))
            jv = float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean())
            if jv > best[0]:
                best = (jv, w, float(c), lam)
    jv, w, c, lam = best
    v = (c * w).clone()
    return LinearCritic(v, "vcs", {"algo": "A1_ridge_tanh_calibrated", "lam": lam, "c": c, "val_J": jv, "n_linear_solves": len(LAM_GRID),
                                   "n_val_evals": len(LAM_GRID) * int(C_GRID_SPEC[2]), "seconds": time.perf_counter() - t0, "sc": sc})


# ------------------------------------------------------------------------------------------------------------------- chunked L-BFGS
def _lbfgs_chunked(v0: torch.Tensor, fit_obj, val_score, budget: int, chunk: int = 10, tol_grad: float = 1e-10, tol_change: float = 1e-12):
    """Minimise fit_obj(v) with L-BFGS in chunks of <= `chunk` closures; after each chunk record (closures, seconds, val_score, v).
    Returns the checkpoint list (first entry = v0 at 0 closures) and the end-of-run termination record."""
    v = v0.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([v], lr=1.0, max_iter=chunk, max_eval=chunk, tolerance_grad=tol_grad, tolerance_change=tol_change,
                            line_search_fn="strong_wolfe")
    count = {"n": 0}

    def closure():
        opt.zero_grad(); loss = fit_obj(v); loss.backward(); count["n"] += 1
        return loss
    t0 = time.perf_counter()
    with torch.no_grad():
        ckpts = [{"closures": 0, "seconds": 0.0, "val": float(val_score(v)), "v": v.detach().clone()}]
    reason = "budget"
    while count["n"] < budget:
        before = count["n"]
        opt.param_groups[0]["max_eval"] = min(chunk, budget - count["n"])
        opt.param_groups[0]["max_iter"] = opt.param_groups[0]["max_eval"]
        opt.step(closure)
        with torch.no_grad():
            ckpts.append({"closures": count["n"], "seconds": time.perf_counter() - t0, "val": float(val_score(v)), "v": v.detach().clone()})
        if count["n"] == before:
            reason = "no_progress"; break
        g = v.grad
        if g is not None and float(g.abs().max()) <= tol_grad:
            reason = "tolerance_grad"; break
        if count["n"] - before < opt.param_groups[0]["max_eval"] and count["n"] < budget:
            reason = "tolerance_change"; break  # L-BFGS stopped early inside the chunk (change / curvature tolerance)
    return ckpts, {"termination": reason, "closures_used": count["n"]}


def _residuals(fit_obj, v: torch.Tensor) -> dict[str, float]:
    u = v.detach().clone().requires_grad_(True)
    f = fit_obj(u); (g,) = torch.autograd.grad(f, u)
    return {"fit_objective": float(f.detach()), "grad_maxabs": float(g.abs().max()), "grad_l2": float(g.norm())}


# ------------------------------------------------------------------------------------------------------------------- A2
def ridge_then_bounded_j(D: Design, a1: LinearCritic, budgets=BUDGETS, chunk: int = 10) -> dict[int, LinearCritic]:
    """Penalised bounded J: minimise −J_FIT(tanh(phi v)) + lam sc |v|^2 / 2 for every lam of the ridge grid, each warm-started at A1's v0, with
    B / len(grid) closures per lam (as A3); the model is the (lam, checkpoint) with the best VAL J, A1 itself (checkpoint 0) included.
    (Gate 1019934: the UNpenalised continuation over-fitted at once — FIT J 0.03 -> 0.23, VAL never above A1 — hence the penalty grid.)"""
    sc = D.sc; per_lam = {}
    val_score = lambda v: bounded_j(D.vp @ v, D.vn @ v)
    for lam in LAM_GRID:
        fit_obj = lambda v, lam=lam: -bounded_j(D.pp @ v, D.pn @ v) + 0.5 * lam * sc * (v ** 2).sum()
        ckpts, term = _lbfgs_chunked(a1.v, fit_obj, val_score, max(budgets) // len(LAM_GRID), chunk)
        per_lam[lam] = (ckpts, term, fit_obj)
    plain = lambda v: -bounded_j(D.pp @ v, D.pn @ v)
    out = {}
    for B in budgets:
        share = B // len(LAM_GRID); cands = []
        for lam, (ckpts, term, fit_obj) in per_lam.items():
            sub = [c for c in ckpts if c["closures"] <= share]
            for c in sub:
                cands.append((c["val"], lam, c, sub[-1], term if sub[-1] is ckpts[-1] else {"termination": "budget"}, fit_obj))
        val, lam, best, _, _, fit_obj = max(cands, key=lambda t: (t[0], -t[2]["closures"]))  # ties -> fewer closures (A1 if nothing beats it)
        lasts = {lm: [c for c in ck if c["closures"] <= share][-1] for lm, (ck, _, _) in per_lam.items()}
        out[B] = LinearCritic(best["v"], "vcs", {"algo": "A2_ridge_then_bounded_J", "budget": B, "lam": lam, "chosen_closures": best["closures"],
                                                  "val_J": val, "chose_A1": best["closures"] == 0,
                                                  "seconds_continuation": sum(c["seconds"] for c in lasts.values()),
                                                  "seconds_total": sum(c["seconds"] for c in lasts.values()) + a1.meta["seconds"],
                                                  "closures_used": sum(c["closures"] for c in lasts.values()),
                                                  "per_lam": [{"lam": lm, "termination": (per_lam[lm][1]["termination"] if lasts[lm] is per_lam[lm][0][-1] else "budget"),
                                                               "closures": lasts[lm]["closures"], "best_val_J": max(c["val"] for c in per_lam[lm][0] if c["closures"] <= share),
                                                               "residual_last": _residuals(per_lam[lm][2], lasts[lm]["v"]),
                                                               "plain_fit_last": float(plain(lasts[lm]["v"]).detach())} for lm in LAM_GRID],
                                                  "residual_chosen": _residuals(fit_obj, best["v"]), "fit_obj_step0": float(plain(a1.v).detach())})
        # A2L: no early stopping — the LAST checkpoint within the budget share, lam chosen on VAL among those last checkpoints (decomposition of
        # "continued optimisation" from "VAL early stopping"; nothing selected on EVAL)
        lam_l = max(LAM_GRID, key=lambda lm: lasts[lm]["val"]); cl = lasts[lam_l]
        out[("last", B)] = LinearCritic(cl["v"], "vcs", {"algo": "A2L_ridge_then_bounded_J_last", "budget": B, "lam": lam_l, "val_J": cl["val"],
                                                          "closures_used": out[B].meta["closures_used"], "seconds_total": out[B].meta["seconds_total"],
                                                          "residual_last": _residuals(per_lam[lam_l][2], cl["v"]), "plain_fit_last": float(plain(cl["v"]).detach())})
    return out


# ------------------------------------------------------------------------------------------------------------------- A3
def js_lbfgs(D: Design, budgets=BUDGETS, chunk: int = 10) -> dict[int, LinearCritic]:
    sc = D.sc; per_lam = {}
    for lam in LAM_GRID:
        fit_obj = lambda v, lam=lam: js_loss(D.pp @ v, D.pn @ v) + 0.5 * lam * sc * (v ** 2).sum()
        val_score = lambda v: js_value(D.vp @ v, D.vn @ v)
        v0 = torch.zeros(D.pp.shape[1], dtype=torch.float64, device=D.pp.device)
        ckpts, term = _lbfgs_chunked(v0, fit_obj, val_score, max(budgets) // len(LAM_GRID), chunk)
        per_lam[lam] = (ckpts, term, fit_obj)
    out = {}
    for B in budgets:
        share = B // len(LAM_GRID); rows = []
        for lam, (ckpts, term, fit_obj) in per_lam.items():
            last = [c for c in ckpts if c["closures"] <= share][-1]
            rows.append({"lam": lam, "val_JS": last["val"], "v": last["v"], "closures": last["closures"], "seconds": last["seconds"],
                         "termination": term["termination"] if last is ckpts[-1] else "budget", "residual": _residuals(fit_obj, last["v"])})
        best = max(rows, key=lambda r: r["val_JS"])
        out[B] = LinearCritic(best["v"], "js", {"algo": "A3_js_lbfgs", "budget": B, "lam": best["lam"], "val_JS": best["val_JS"],
                                                  "seconds_total": sum(r["seconds"] for r in rows), "closures_used": sum(r["closures"] for r in rows),
                                                  "per_lam": [{k: v for k, v in r.items() if k != "v"} for r in rows]})
    return out


# ------------------------------------------------------------------------------------------------------------------- risks and tests
def risks(cr: LinearCritic, D: Design) -> dict[str, float]:
    """Own-objective FIT / VAL risk (unpenalised) and the common bounded-J score of T on FIT / VAL."""
    v = cr.v
    with torch.no_grad():
        if cr.kind == "vcs":
            own_fit, own_val = -float(bounded_j(D.pp @ v, D.pn @ v)), -float(bounded_j(D.vp @ v, D.vn @ v))
            jc_fit, jc_val = -own_fit, -own_val
        else:
            own_fit, own_val = float(js_loss(D.pp @ v, D.pn @ v)), float(js_loss(D.vp @ v, D.vn @ v))
            jc_fit, jc_val = float(bounded_j(0.5 * (D.pp @ v), 0.5 * (D.pn @ v))), float(bounded_j(0.5 * (D.vp @ v), 0.5 * (D.vn @ v)))
    return {"own_risk_fit": own_fit, "own_risk_val": own_val, "Jcommon_fit": jc_fit, "Jcommon_val": jc_val}


def perm_tests(cr: LinearCritic, ze: torch.Tensor, ne: torch.Tensor, ne_neg: torch.Tensor, PN: torch.Tensor, delta: float) -> dict[str, Any]:
    """Within-class permutation tests on EVAL (shared permutation matrix PN: B x n of N values).  Own statistic (VCS: J of tanh f; JS: JS value
    of f) and the common squared score J(T) (T = tanh f or tanh(f / 2)).  Negatives (POOL N) are fixed, as in T1."""
    t0 = time.perf_counter()
    with torch.no_grad():
        w, b = cr.v[:-1], cr.v[-1]
        s = ze.double() @ w
        f_obs = s * (2 * ne.double() - 1) + b; f_neg = s * (2 * ne_neg.double() - 1) + b
        f_perm = s[None, :] * (2 * PN.double() - 1) + b
        scale = 1.0 if cr.kind == "vcs" else 0.5

        def jt(fp, fn):  # batched over rows of fp
            tp, tn = torch.tanh(scale * fp), torch.tanh(scale * fn)
            return (tp - 0.5 * tp ** 2).mean(-1) + float((-tn - 0.5 * tn ** 2).mean())

        def jsv(fp, fn):
            return (-F.softplus(-fp)).mean(-1) - float(F.softplus(fn).mean()) + float(np.log(4.0))
        out = {}
        stats = {"common": jt} if cr.kind == "vcs" else {"common": jt, "own": jsv}
        for name, fn_ in stats.items():
            obs = float(fn_(f_obs[None, :], f_neg)[0]); null = fn_(f_perm, f_neg).cpu().numpy()
            p = float((1 + (null >= obs).sum()) / (1 + len(null)))
            out[name] = {"stat": obs, "p": p, "reject": p <= delta}
        if cr.kind == "vcs":
            out["own"] = dict(out["common"])  # VCS's own statistic IS the common bounded-J score
    out["seconds"] = time.perf_counter() - t0
    return out
