"""V9-N1 quantification methods (package §04.4) on one interface.

Scores.  Every method reads the same 1-D pairing similarity s (the common teacher score of §04.2) and produces a bounded read-out
T(s) ∈ [-1, 1] (and, where the method defines one, a component posterior p(P | s) under the balanced reference prior).  The batch
proportion follows the fixed-score moment identity of §03.B,  rho = (a − c) / (a − b)  with a = E_P T, b = E_Q T, c = E_R T, evaluated
on the reference-moments role and the batch; EM variants use the label-shift likelihood (Saerens et al. 2002; Alexandari et al. 2020 for
the bias-corrected calibration); BBSE uses the confusion / mean matrices (Lipton et al. 2018); the kernel-MPE column (Ramaswamy et al.
2016) reads P + R only and is reported as a conditional column.  Weights are the clean-component posterior of §03.D,
w = (1 − rho)(1 + T) / ((1 − rho)(1 + T) + rho (1 − T)), computed by the package reference (`reference_core.clean_posterior`).

Nothing here infers independence: the finite-sample confidence set is produced only through `confidence_set` after
`reference_core.role_disjointness` has passed on real cluster ids and the caller has asserted the fixed-score / specified-mixture
conditions (both flags are recorded in the output, never inferred).

Fitting is deterministic: scalar critics by multistart L-BFGS-B (scipy) on the balanced objective; the Adam path is a sensitivity
(`fit_scalar_adam`).  Kernel methods select (bandwidth, ridge, centres) on the critic-select role; the primary selection rule is the common
squared score J, native criteria are kept as sensitivities.  No EVAL / batch label is used anywhere in fitting or selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from . import reference_core as RC

EPS = 1e-12
LOG2 = math.log(2.0)


# ----------------------------------------------------------------------------------------------------------------------------- objectives
def j_common(tp: np.ndarray, tq: np.ndarray) -> float:
    """Common squared score J = E_P[T − T²/2] + E_Q[−T − T²/2] (separate means; equal P / Q weight)."""
    return float(np.mean(tp - 0.5 * tp * tp) + np.mean(-tq - 0.5 * tq * tq))


def balanced_logloss(ell_p: np.ndarray, ell_q: np.ndarray) -> float:
    """Balanced class log loss of the natural logit ell (P labelled 1, Q labelled 0)."""
    return float(-0.5 * np.mean(log_expit(ell_p)) - 0.5 * np.mean(log_expit(-ell_q)))


def squared_risk(tp: np.ndarray, tq: np.ndarray) -> float:
    """Balanced squared risk with targets ±1: R = ½E_P(1 − T)² + ½E_Q(−1 − T)²  (= 1 − J)."""
    return float(0.5 * np.mean((1 - tp) ** 2) + 0.5 * np.mean((1 + tq) ** 2))


# ------------------------------------------------------------------------------------------------------------------------- scalar critic
@dataclass
class ScalarCritic:
    """cos-2 critic on the scalar score: T = tanh(a s + b), a = softplus(a_raw) > 0 (no score-direction flip); natural logit ell = 2(a s + b).
    `rule` ∈ {vcs, bce} is the fitting objective (1 − J or the balanced log loss on ell); both share the class and the read-out."""
    a_raw: float
    b: float
    rule: str
    info: dict = field(default_factory=dict)

    @property
    def a(self) -> float:
        return float(np.logaddexp(0.0, self.a_raw))

    def logit(self, s: np.ndarray) -> np.ndarray:
        return 2.0 * (self.a * np.asarray(s, float) + self.b)

    def T(self, s: np.ndarray) -> np.ndarray:
        return np.tanh(0.5 * self.logit(s))

    def prob(self, s: np.ndarray) -> np.ndarray:
        """p(P | s) under the balanced reference prior: σ(ell) = (1 + T) / 2."""
        return expit(self.logit(s))


def _scalar_objective(theta, sp, sq, rule):
    a = np.logaddexp(0.0, theta[0]); ellp = 2 * (a * sp + theta[1]); ellq = 2 * (a * sq + theta[1])
    if rule == "vcs":
        tp, tq = np.tanh(0.5 * ellp), np.tanh(0.5 * ellq)
        return squared_risk(tp, tq)
    if rule == "bce":
        return balanced_logloss(ellp, ellq)
    raise ValueError(rule)


STARTS = [(ar, b) for ar in (-2.0, 0.0, 2.0) for b in (-1.0, 0.0, 1.0)]


def fit_scalar(sp: np.ndarray, sq: np.ndarray, rule: str, sp_sel: np.ndarray | None = None, sq_sel: np.ndarray | None = None,
               selection: str = "common_j") -> ScalarCritic:
    """Deterministic multistart L-BFGS-B on the balanced objective.  With a select role the start giving the best selection criterion
    (common J primary; native = the fitting objective) is kept; otherwise the lowest training objective."""
    sp, sq = np.asarray(sp, float), np.asarray(sq, float)
    cands = []
    for th0 in STARTS:
        r = minimize(_scalar_objective, np.array(th0), args=(sp, sq, rule), method="L-BFGS-B")
        c = ScalarCritic(float(r.x[0]), float(r.x[1]), rule)
        if sp_sel is not None:
            crit = -j_common(c.T(sp_sel), c.T(sq_sel)) if selection == "common_j" else _scalar_objective(r.x, sp_sel, sq_sel, rule)
        else:
            crit = float(r.fun)
        cands.append((crit, th0, float(r.fun), r.nit, c))
    cands.sort(key=lambda z: z[0])
    best = cands[0][4]
    best.info = {"starts": len(STARTS), "selection": selection if sp_sel is not None else "train_objective", "train_objective": cands[0][2],
                 "selection_value": cands[0][0], "start": cands[0][1], "iterations": int(cands[0][3]),
                 "all_train_objectives": sorted(round(z[2], 6) for z in cands)}
    return best


def fit_scalar_adam(sp, sq, rule, sp_sel, sq_sel, lrs=(1e-2, 3e-2, 1e-1), steps=500, seed=0) -> ScalarCritic:
    """Sensitivity: the same class fitted by Adam (torch) with an lr grid selected on the select role by the common J."""
    import torch
    sp_t, sq_t = torch.tensor(np.asarray(sp, float)), torch.tensor(np.asarray(sq, float)); best = None
    for lr in lrs:
        th = torch.zeros(2, dtype=torch.float64, requires_grad=True); opt = torch.optim.Adam([th], lr=lr)
        for _ in range(steps):
            opt.zero_grad(); a = torch.nn.functional.softplus(th[0]); ellp = 2 * (a * sp_t + th[1]); ellq = 2 * (a * sq_t + th[1])
            if rule == "vcs":
                tp, tq = torch.tanh(ellp / 2), torch.tanh(ellq / 2); loss = 0.5 * ((1 - tp) ** 2).mean() + 0.5 * ((1 + tq) ** 2).mean()
            else:
                loss = 0.5 * torch.nn.functional.softplus(-ellp).mean() + 0.5 * torch.nn.functional.softplus(ellq).mean()
            loss.backward(); opt.step()
        c = ScalarCritic(float(th[0].item()), float(th[1].item()), rule, {"optimiser": "adam", "lr": lr, "steps": steps})
        crit = -j_common(c.T(sp_sel), c.T(sq_sel))
        if best is None or crit < best[0]:
            best = (crit, c)
    best[1].info["selection"] = "common_j"; best[1].info["selection_value"] = best[0]
    return best[1]


# ------------------------------------------------------------------------------------------------------------------------- calibration
class Calibrator:
    """Post-hoc calibration of the natural logit on the critic-select role (reference budget): temperature (TS), bias-corrected temperature /
    Platt (BCTS: α ell + β) and isotonic (PAVA on σ(ell)).  All minimise the balanced log loss; `prob(ell)` = p(P | s) under the balanced prior."""

    def __init__(self, kind: str):
        if kind not in ("none", "ts", "bcts", "isotonic"):
            raise ValueError(kind)
        self.kind, self.params = kind, {}

    def fit(self, ell_p: np.ndarray, ell_q: np.ndarray) -> "Calibrator":
        ell_p, ell_q = np.asarray(ell_p, float), np.asarray(ell_q, float)
        if self.kind == "none":
            return self
        if self.kind in ("ts", "bcts"):
            def obj(th):
                al = np.exp(th[0]); be = th[1] if self.kind == "bcts" else 0.0
                return balanced_logloss(al * ell_p + be, al * ell_q + be)
            r = minimize(obj, np.zeros(2), method="L-BFGS-B"); self.params = {"alpha": float(np.exp(r.x[0])), "beta": float(r.x[1]) if self.kind == "bcts" else 0.0}
        else:
            x = np.concatenate([expit(ell_p), expit(ell_q)]); y = np.concatenate([np.ones_like(ell_p), np.zeros_like(ell_q)])
            w = np.concatenate([np.full(len(ell_p), 0.5 / len(ell_p)), np.full(len(ell_q), 0.5 / len(ell_q))])
            o = np.argsort(x, kind="stable"); xs, ys, ws = x[o], y[o], w[o]
            self.params = {"x": xs.tolist(), "p": pava(ys, ws).tolist()}
        return self

    def prob(self, ell: np.ndarray) -> np.ndarray:
        ell = np.asarray(ell, float)
        if self.kind == "none":
            return expit(ell)
        if self.kind in ("ts", "bcts"):
            return expit(self.params["alpha"] * ell + self.params["beta"])
        x, p = np.asarray(self.params["x"]), np.asarray(self.params["p"])
        return np.clip(np.interp(expit(ell), x, p), 1e-6, 1 - 1e-6)


def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators (weighted, non-decreasing)."""
    y, w = np.asarray(y, float), np.asarray(w, float); n = len(y)
    vals, wts, cnt = [], [], []
    for i in range(n):
        vals.append(y[i]); wts.append(w[i]); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1]); c = cnt[-2] + cnt[-1]; ww = wts[-2] + wts[-1]
            vals.pop(); wts.pop(); cnt.pop(); vals[-1], wts[-1], cnt[-1] = v, ww, c
    return np.repeat(np.array(vals), np.array(cnt))


# ------------------------------------------------------------------------------------------------------------------------- proportion
def rho_moment(T_ref_p: np.ndarray, T_ref_q: np.ndarray, T_batch: np.ndarray) -> dict:
    """Fixed-score moment estimate (reference `rho_from_means`); raw, projected and the identification status are all kept."""
    a, b, c = float(np.mean(T_ref_p)), float(np.mean(T_ref_q)), float(np.mean(T_batch))
    r = RC.rho_from_means(a, b, c)
    return {"a": a, "b": b, "c": c, "rho_raw": r.raw, "rho_clipped": r.projected, "gap": r.denominator, "status": r.status}


def rho_em(p_batch: np.ndarray, rho0: float = 0.5, iters: int = 200, tol: float = 1e-9) -> dict:
    """Label-shift EM / MLE of the Q proportion from p(P | s) under the balanced reference prior (Saerens et al. 2002): the posterior under
    prior (1 − rho, rho) is p(1 − rho) / (p(1 − rho) + (1 − p) rho) and rho ← 1 − mean posterior."""
    p = np.clip(np.asarray(p_batch, float), 1e-9, 1 - 1e-9); rho = float(rho0); hist = []
    for it in range(iters):
        post = p * (1 - rho) / (p * (1 - rho) + (1 - p) * rho); new = float(1 - post.mean()); hist.append(new)
        if abs(new - rho) < tol:
            rho = new; break
        rho = new
    return {"rho_raw": rho, "rho_clipped": min(1.0, max(0.0, rho)), "status": "converged" if len(hist) < iters else "max_iterations",
            "iterations": len(hist)}


def rho_bbse(pred_ref_p: np.ndarray, pred_ref_q: np.ndarray, pred_batch: np.ndarray, hard: bool, tol: float = 1e-6) -> dict:
    """BBSE (Lipton et al. 2018) for two classes with the black-box output p(P | s): hard = thresholded prediction, soft = the probability.
    Confusion / mean matrix C[pred, true] from the reference roles, mu = batch prediction mean, pi = C⁻¹ mu, rho = pi_Q.  For the binary soft
    case this is algebraically the moment formula on the probability scale (affine in T) — recorded, not hidden."""
    f = (lambda z: (np.asarray(z, float) >= 0.5).astype(float)) if hard else (lambda z: np.asarray(z, float))
    mp, mq, mb = f(pred_ref_p).mean(), f(pred_ref_q).mean(), f(pred_batch).mean()
    C = np.array([[mp, mq], [1 - mp, 1 - mq]]); det = float(np.linalg.det(C))
    if abs(det) < tol:
        return {"rho_raw": None, "rho_clipped": None, "status": "confusion_matrix_singular", "det": det}
    pi = np.linalg.solve(C, np.array([mb, 1 - mb])); rho = float(pi[1])
    return {"rho_raw": rho, "rho_clipped": min(1.0, max(0.0, rho)), "status": "point_estimate" if 0 <= rho <= 1 else "outside_parameter_space", "det": det}


# ------------------------------------------------------------------------------------------------------------------------- kernel methods
def silverman(x: np.ndarray) -> float:
    x = np.asarray(x, float); n = len(x); sd = float(np.std(x, ddof=1)) if n > 1 else 1.0
    iqr = float(np.subtract(*np.percentile(x, [75, 25]))); s = min(sd, iqr / 1.349) if iqr > 0 else sd
    return float(max(0.9 * s * n ** (-0.2), 1e-6))


def _kde(x_eval: np.ndarray, x_fit: np.ndarray, h: float) -> np.ndarray:
    z = (np.asarray(x_eval, float)[:, None] - np.asarray(x_fit, float)[None, :]) / h
    return np.exp(-0.5 * z * z).mean(1) / (h * math.sqrt(2 * math.pi))


@dataclass
class KDEPosterior:
    """Separate 1-D Gaussian KDEs of s under P and Q; T = (p̂ − q̂) / (p̂ + q̂), p(P | s) = p̂ / (p̂ + q̂).  Bandwidth multiplier of Silverman's rule
    selected on the select role by the common J (primary) or the held-out log-likelihood (native)."""
    sp_fit: np.ndarray
    sq_fit: np.ndarray
    h_p: float
    h_q: float
    info: dict = field(default_factory=dict)

    def densities(self, s):
        return np.maximum(_kde(s, self.sp_fit, self.h_p), 1e-300), np.maximum(_kde(s, self.sq_fit, self.h_q), 1e-300)

    def T(self, s):
        p, q = self.densities(s); return (p - q) / (p + q)

    def prob(self, s):
        p, q = self.densities(s); return p / (p + q)


def fit_kde(sp, sq, sp_sel, sq_sel, mults=(0.25, 0.5, 1.0, 2.0, 4.0), selection="common_j") -> KDEPosterior:
    sp, sq = np.asarray(sp, float), np.asarray(sq, float); hp0, hq0 = silverman(sp), silverman(sq); best = None; grid = []
    for m in mults:
        k = KDEPosterior(sp, sq, m * hp0, m * hq0)
        if selection == "common_j":
            crit = -j_common(k.T(sp_sel), k.T(sq_sel))
        else:
            p1, _ = k.densities(sp_sel); _, q1 = k.densities(sq_sel); crit = -(np.mean(np.log(p1)) + np.mean(np.log(q1)))
        grid.append((m, crit))
        if best is None or crit < best[0]:
            best = (crit, k, m)
    best[1].info = {"silverman": [hp0, hq0], "multiplier": best[2], "selection": selection, "grid": grid, "at_edge": best[2] in (mults[0], mults[-1])}
    return best[1]


def _gauss_basis(s: np.ndarray, centres: np.ndarray, sigma: float) -> np.ndarray:
    z = (np.asarray(s, float)[:, None] - centres[None, :]) / sigma
    return np.exp(-0.5 * z * z)


@dataclass
class RuLSIF:
    """Relative density-ratio (Yamada et al. 2011) with α = ½: r(s) = p / (½ p + ½ q) = 1 + η, fitted by least squares on Gaussian centres;
    T = r − 1 (raw and clipped to [−1, 1] both stored); p(P | s) from the clipped T."""
    centres: np.ndarray
    sigma: float
    lam: float
    theta: np.ndarray
    info: dict = field(default_factory=dict)

    def r(self, s):
        return _gauss_basis(s, self.centres, self.sigma) @ self.theta

    def T_raw(self, s):
        return self.r(s) - 1.0

    def T(self, s):
        return np.clip(self.T_raw(s), -1.0, 1.0)

    def prob(self, s):
        return (1 + self.T(s)) / 2


def _rulsif_solve(sp, sq, centres, sigma, lam, alpha=0.5):
    Pp, Pq = _gauss_basis(sp, centres, sigma), _gauss_basis(sq, centres, sigma)
    H = alpha * Pp.T @ Pp / len(sp) + (1 - alpha) * Pq.T @ Pq / len(sq); h = Pp.mean(0)
    return np.linalg.solve(H + lam * np.eye(len(centres)), h)


def _rulsif_objective(theta, sp, sq, centres, sigma, alpha=0.5):
    rp, rq = _gauss_basis(sp, centres, sigma) @ theta, _gauss_basis(sq, centres, sigma) @ theta
    return float(0.5 * (alpha * np.mean(rp * rp) + (1 - alpha) * np.mean(rq * rq)) - np.mean(rp))


def fit_rulsif(sp, sq, sp_sel, sq_sel, n_centres=100, sig_mults=(0.5, 1.0, 2.0, 4.0), lams=(1e-3, 1e-2, 1e-1, 1.0), selection="common_j", seed=0) -> RuLSIF:
    sp, sq = np.asarray(sp, float), np.asarray(sq, float); rng = np.random.default_rng(seed)
    centres = np.sort(rng.choice(sp, size=min(n_centres, len(sp)), replace=False)); s0 = silverman(np.concatenate([sp, sq])); best = None; grid = []
    for m in sig_mults:
        for lam in lams:
            th = _rulsif_solve(sp, sq, centres, m * s0, lam); k = RuLSIF(centres, m * s0, lam, th)
            crit = -j_common(k.T(sp_sel), k.T(sq_sel)) if selection == "common_j" else _rulsif_objective(th, sp_sel, sq_sel, centres, m * s0)
            grid.append((m, lam, crit))
            if best is None or crit < best[0]:
                best = (crit, k, m, lam)
    best[1].info = {"silverman_pooled": s0, "sigma_mult": best[2], "lambda": best[3], "selection": selection, "n_centres": int(len(centres)),
                    "at_edge": best[2] in (sig_mults[0], sig_mults[-1]) or best[3] in (lams[0], lams[-1]), "grid": grid}
    return best[1]


@dataclass
class NystromRegressor:
    """Nyström kernel ridge regression of the balanced label C ∈ {+1, −1} on s (squared loss, P / Q weighted ½ each); T = clip(f, −1, 1)."""
    centres: np.ndarray
    sigma: float
    lam: float
    beta: np.ndarray
    info: dict = field(default_factory=dict)

    def f(self, s):
        return _gauss_basis(s, self.centres, self.sigma) @ self.beta

    def T(self, s):
        return np.clip(self.f(s), -1.0, 1.0)

    def prob(self, s):
        return (1 + self.T(s)) / 2


def _nystrom_solve(sp, sq, centres, sigma, lam):
    Kp, Kq = _gauss_basis(sp, centres, sigma), _gauss_basis(sq, centres, sigma); Kmm = _gauss_basis(centres, centres, sigma)
    A = Kp.T @ Kp / (2 * len(sp)) + Kq.T @ Kq / (2 * len(sq)) + lam * Kmm; b = Kp.mean(0) / 2 - Kq.mean(0) / 2
    return np.linalg.solve(A + 1e-10 * np.eye(len(centres)), b)


def fit_nystrom(sp, sq, sp_sel, sq_sel, n_centres=100, sig_mults=(0.5, 1.0, 2.0, 4.0), lams=(1e-4, 1e-3, 1e-2, 1e-1), selection="common_j", seed=0) -> NystromRegressor:
    sp, sq = np.asarray(sp, float), np.asarray(sq, float); rng = np.random.default_rng(seed); pool = np.concatenate([sp, sq])
    centres = np.sort(rng.choice(pool, size=min(n_centres, len(pool)), replace=False)); s0 = silverman(pool); best = None; grid = []
    for m in sig_mults:
        for lam in lams:
            be = _nystrom_solve(sp, sq, centres, m * s0, lam); k = NystromRegressor(centres, m * s0, lam, be)
            crit = -j_common(k.T(sp_sel), k.T(sq_sel)) if selection == "common_j" else squared_risk(k.f(sp_sel), k.f(sq_sel))
            grid.append((m, lam, crit))
            if best is None or crit < best[0]:
                best = (crit, k, m, lam)
    best[1].info = {"silverman_pooled": s0, "sigma_mult": best[2], "lambda": best[3], "selection": selection, "n_centres": int(len(centres)),
                    "at_edge": best[2] in (sig_mults[0], sig_mults[-1]) or best[3] in (lams[0], lams[-1]), "grid": grid}
    return best[1]


def kernel_mpe(s_p: np.ndarray, s_r: np.ndarray, lambdas=None, sigma: float | None = None, fw_iters: int = 200, max_pool: int = 512, seed: int = 0) -> dict:
    """Kernel mixture-proportion estimation from the KNOWN component P and the mixture R only (Ramaswamy et al. 2016, KM idea): for λ ≥ 1 the
    RKHS distance d(λ) between λ μ_R − (λ − 1) μ_P and the convex hull of the sample feature maps is ≈ 0 while λ ≤ λ* = 1 / (1 − rho) and grows
    beyond it.  λ* is read off the knee of d(λ) by a two-segment (flat, then linear) least-squares fit; rho = 1 − 1 / λ*.  Our implementation
    (Frank–Wolfe on the simplex, 1-D Gaussian kernel, Silverman bandwidth of the pooled sample) — a conditional column with a different
    information budget (no Q reference), not the authors' KM1 / KM2 thresholds."""
    s_p, s_r = np.asarray(s_p, float), np.asarray(s_r, float); rng = np.random.default_rng(seed); sub = {}
    for name, arr in (("P", s_p), ("R", s_r)):                      # cost cap: at most max_pool points per role (recorded)
        if len(arr) > max_pool:
            arr = rng.choice(arr, size=max_pool, replace=False); sub[name] = max_pool
        if name == "P":
            s_p = arr
        else:
            s_r = arr
    pool = np.concatenate([s_r, s_p]); sig = sigma or silverman(pool)
    lambdas = np.asarray(lambdas if lambdas is not None else np.arange(1.0, 3.01, 0.1))
    K = np.exp(-0.5 * ((pool[:, None] - pool[None, :]) / sig) ** 2); n_r, n_p = len(s_r), len(s_p)
    u_r = np.concatenate([np.full(n_r, 1 / n_r), np.zeros(n_p)]); u_p = np.concatenate([np.zeros(n_r), np.full(n_p, 1 / n_p)])
    d = []
    for lam in lambdas:
        v = lam * u_r - (lam - 1) * u_p; w = np.full(len(pool), 1 / len(pool))
        for t in range(fw_iters):                                   # minimise ||Φw − Φv||² over the simplex: gradient = 2K(w − v)
            g = K @ (w - v); i = int(np.argmin(g)); e = np.zeros_like(w); e[i] = 1.0; step = 2.0 / (t + 2); w = (1 - step) * w + step * e
        diff = w - v; d.append(float(np.sqrt(max(diff @ K @ diff, 0.0))))
    d = np.array(d); best = None
    for k in range(1, len(lambdas) - 1):                               # two-segment fit: d ≈ c0 on [0, k], then linear on [k, end]
        c0 = d[:k + 1].mean(); x, y = lambdas[k:], d[k:]; A = np.vstack([x - x[0], np.ones_like(x)]).T; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((d[:k + 1] - c0) ** 2).sum() + ((A @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, k, float(coef[0]))
    lam_star = float(lambdas[best[1]]); rho = 1 - 1 / lam_star
    return {"rho_raw": rho, "rho_clipped": min(1.0, max(0.0, rho)), "status": "knee_estimate" if best[2] > 1e-6 else "no_rise_detected",
            "lambda_star": lam_star, "slope_after_knee": best[2], "d_curve": d.tolist(), "lambdas": lambdas.tolist(), "sigma": sig, "subsampled": sub,
            "n_used": {"P": int(len(s_p)), "R": int(len(s_r))}}


# ------------------------------------------------------------------------------------------------------------------------- weights
def weights(T_batch: np.ndarray, rho: float | None) -> dict:
    """Clean-component posterior weights of §03.D (reference `clean_posterior`) with ESS and clipping diagnostics; rho = None → no weights."""
    T = np.clip(np.asarray(T_batch, float), -1 + 1e-9, 1 - 1e-9); clip_rate = float(np.mean(np.abs(np.asarray(T_batch, float)) >= 1 - 1e-6))
    if rho is None:
        return {"w": None, "ess": None, "clip_rate_T": clip_rate, "status": "no_rho"}
    r = min(1.0 - 1e-9, max(1e-9, float(rho)))
    w = RC.clean_posterior(T, r)
    return {"w": w, "ess": RC.effective_sample_size(w), "ess_fraction": RC.effective_sample_size(w) / len(w), "mean_w": float(w.mean()),
            "clip_rate_T": clip_rate, "rho_used": r, "status": "ok" if 0 < rho < 1 else "rho_at_boundary"}


# ------------------------------------------------------------------------------------------------------------------------- confidence set
def confidence_set(rows_p: list, rows_q: list, rows_r: list, alpha: float = 0.05, *, fixed_score_outside_eval: bool,
                   specified_mixture: bool) -> dict:
    """Finite-sample set for rho from independent cluster means in [−1, 1].  Rows: {role, cluster_id, units, score}.  The reference
    `role_disjointness` must pass on the real unit ids; the two remaining conditions are the caller's assertions and are recorded as such."""
    rows = list(rows_p) + list(rows_q) + list(rows_r)
    try:
        RC.role_disjointness(rows); disjoint = True; err = None
    except ValueError as e:
        disjoint, err = False, str(e)
    out = {"n_clusters": {"P": len(rows_p), "Q": len(rows_q), "R": len(rows_r)}, "alpha": alpha, "disjoint_clusters": disjoint,
           "disjointness_error": err, "fixed_score_outside_eval": bool(fixed_score_outside_eval), "specified_mixture": bool(specified_mixture)}
    if not (disjoint and fixed_score_outside_eval and specified_mixture):
        out.update({"lower": None, "upper": None, "status": "not_verified"}); return out
    z = RC.rho_confidence_set([r["score"] for r in rows_p], [r["score"] for r in rows_q], [r["score"] for r in rows_r], alpha, assumptions_verified=True)
    out.update({"lower": z.lower, "upper": z.upper, "status": z.status}); return out


# ------------------------------------------------------------------------------------------------------------------------- orchestration
METHODS = ("raw_moment", "vcs_moment", "bce_moment", "bbse_hard", "bbse_soft", "em_uncal", "em_ts", "em_bcts", "em_isotonic",
           "kde_moment", "kde_em", "rulsif_moment", "rulsif_em", "nystrom_moment", "nystrom_em", "kernel_mpe")


@dataclass
class FittedPanel:
    """Everything fitted on the critic-fit / critic-select roles (no batch, no reference-moments data): the scalar critics, calibrators and
    kernel models.  `readouts(s)` gives every method's T and p(P | s)."""
    vcs: ScalarCritic
    bce: ScalarCritic
    cal: dict
    kde: KDEPosterior
    rulsif: RuLSIF
    nystrom: NystromRegressor
    adam: dict
    info: dict

    def readouts(self, s: np.ndarray) -> dict:
        s = np.asarray(s, float); ell = self.bce.logit(s)
        return {"raw": {"T": np.clip(s, -1, 1)}, "vcs": {"T": self.vcs.T(s), "p": self.vcs.prob(s)}, "bce": {"T": self.bce.T(s), "p": self.bce.prob(s)},
                "cal_ts": {"p": self.cal["ts"].prob(ell)}, "cal_bcts": {"p": self.cal["bcts"].prob(ell)}, "cal_iso": {"p": self.cal["isotonic"].prob(ell)},
                "kde": {"T": self.kde.T(s), "p": self.kde.prob(s)}, "rulsif": {"T": self.rulsif.T(s), "T_raw": self.rulsif.T_raw(s), "p": self.rulsif.prob(s)},
                "nystrom": {"T": self.nystrom.T(s), "p": self.nystrom.prob(s)}}


def fit_panel(sp_fit, sq_fit, sp_sel, sq_sel, seed: int = 0, selection: str = "common_j", adam_sensitivity: bool = True) -> FittedPanel:
    vcs = fit_scalar(sp_fit, sq_fit, "vcs", sp_sel, sq_sel, selection); bce = fit_scalar(sp_fit, sq_fit, "bce", sp_sel, sq_sel, selection)
    cal = {k: Calibrator(k).fit(bce.logit(sp_sel), bce.logit(sq_sel)) for k in ("ts", "bcts", "isotonic")}
    kde = fit_kde(sp_fit, sq_fit, sp_sel, sq_sel, selection=selection); rul = fit_rulsif(sp_fit, sq_fit, sp_sel, sq_sel, selection=selection, seed=seed)
    nys = fit_nystrom(sp_fit, sq_fit, sp_sel, sq_sel, selection=selection, seed=seed)
    adam = {}
    if adam_sensitivity:
        for rule in ("vcs", "bce"):
            c = fit_scalar_adam(sp_fit, sq_fit, rule, sp_sel, sq_sel, seed=seed); adam[rule] = {"a_raw": c.a_raw, "b": c.b, "info": c.info,
                                                                                                "J_select": j_common(c.T(sp_sel), c.T(sq_sel))}
    info = {"selection": selection, "n_fit": [int(len(sp_fit)), int(len(sq_fit))], "n_select": [int(len(sp_sel)), int(len(sq_sel))],
            "vcs": {"a": vcs.a, "b": vcs.b, **vcs.info}, "bce": {"a": bce.a, "b": bce.b, **bce.info}, "cal": {k: v.params if k != "isotonic" else {"n_knots": len(v.params["x"])} for k, v in cal.items()},
            "kde": kde.info, "rulsif": rul.info, "nystrom": nys.info, "J_select": {"vcs": j_common(vcs.T(sp_sel), vcs.T(sq_sel)), "bce": j_common(bce.T(sp_sel), bce.T(sq_sel)),
            "kde": j_common(kde.T(sp_sel), kde.T(sq_sel)), "rulsif": j_common(rul.T(sp_sel), rul.T(sq_sel)), "nystrom": j_common(nys.T(sp_sel), nys.T(sq_sel))}}
    return FittedPanel(vcs, bce, cal, kde, rul, nys, adam, info)


def quantify_batch(panel: FittedPanel, s_ref_p: np.ndarray, s_ref_q: np.ndarray, s_batch: np.ndarray, truth: dict | None = None) -> dict:
    """Every method's rho (raw / clipped / status) for one batch, plus the weights' ESS and, when `truth` carries the batch component labels
    (offline scorer only), the posterior Brier / log loss / AUROC of the clean-component weights.  `truth` = {"is_p": bool array, "rho": float}."""
    Rp, Rq, Rb = panel.readouts(s_ref_p), panel.readouts(s_ref_q), panel.readouts(s_batch); out = {}
    def T_based(name, key):
        m = rho_moment(Rp[key]["T"], Rq[key]["T"], Rb[key]["T"]); out[name] = {**m, "kind": "moment"}
        if "p" in Rb[key]:
            out[name]["weights"] = _weights_summary(Rb[key]["T"], m["rho_clipped"], truth)
    T_based("raw_moment", "raw"); T_based("vcs_moment", "vcs"); T_based("bce_moment", "bce"); T_based("kde_moment", "kde"); T_based("rulsif_moment", "rulsif"); T_based("nystrom_moment", "nystrom")
    out["raw_moment"]["note"] = "raw score is not a posterior: no weights"
    for name, hard in (("bbse_hard", True), ("bbse_soft", False)):
        out[name] = {**rho_bbse(Rp["bce"]["p"], Rq["bce"]["p"], Rb["bce"]["p"], hard), "kind": "bbse", "weights": None}
        out[name]["weights"] = _weights_summary(Rb["bce"]["T"], out[name]["rho_clipped"], truth)
    for name, key in (("em_uncal", "bce"), ("em_ts", "cal_ts"), ("em_bcts", "cal_bcts"), ("em_isotonic", "cal_iso"), ("kde_em", "kde"), ("rulsif_em", "rulsif"), ("nystrom_em", "nystrom")):
        e = rho_em(Rb[key]["p"]); out[name] = {**e, "kind": "em"}; T = 2 * Rb[key]["p"] - 1
        out[name]["weights"] = _weights_summary(T, e["rho_clipped"], truth)
    mpe = kernel_mpe(s_ref_p, s_batch); out["kernel_mpe"] = {k: v for k, v in mpe.items() if k != "d_curve"}; out["kernel_mpe"]["kind"] = "mpe_P_plus_R_only"
    out["kernel_mpe"]["weights"] = _weights_summary(Rb["vcs"]["T"], mpe["rho_clipped"], truth) if mpe["rho_clipped"] is not None else None
    out["_J_batch"] = {k: j_common(Rp[k]["T"], Rq[k]["T"]) for k in ("vcs", "bce", "kde", "rulsif", "nystrom")}
    if truth is not None:
        out["_truth"] = {"rho": truth["rho"], "rho_realized": float(1 - np.mean(truth["is_p"]))}
    return out


def _weights_summary(T, rho, truth):
    w = weights(T, rho)
    if w["w"] is None:
        return {"status": w["status"], "ess": None, "clip_rate_T": w["clip_rate_T"]}
    s = {"ess": w["ess"], "ess_fraction": w["ess_fraction"], "mean_w": w["mean_w"], "clip_rate_T": w["clip_rate_T"], "status": w["status"]}
    if truth is not None:
        y = np.asarray(truth["is_p"], float); s["brier"] = RC.brier(y, w["w"]); s["logloss"] = RC.logloss(y, w["w"]); s["auroc"] = auroc(y, w["w"])
    return s


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    y, score = np.asarray(y, float), np.asarray(score, float); pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # tie-averaged ranks (Mann–Whitney)
    allv = np.concatenate([pos, neg]); order = np.argsort(allv, kind="stable"); ranks = np.empty(len(allv)); sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def panel_to_json(panel: FittedPanel) -> dict:
    return {"vcs": {"a_raw": panel.vcs.a_raw, "b": panel.vcs.b}, "bce": {"a_raw": panel.bce.a_raw, "b": panel.bce.b}, "adam": panel.adam, "info": panel.info}
