"""Estimator package v2 (server spec §3): the direct CS-estimator controls that share the pair data with the neural VCS critic.

CS-K-native  classical kernel CS-QMI plug-in (fixed Lebesgue reference measure, Gaussian kernels) on pair representations:
             A = mean_ij K_ij L_ij,  B_q = mean(K) mean(L),  C = (1/n^3) sum_i (sum_j K_ij)(sum_j L_ij),  D_CS = log A + log B_q - 2 log C.
             Kernel convention is recorded: `effective=True` uses the integrated KDE-product kernel (bandwidth sqrt(2) h), so that A, B_q, C are
             the exact L2 inner products of the Gaussian KDE densities; `effective=False` uses the raw KDE kernel h.  Its own truth for the
             Gaussian is `cs_lebesgue_gaussian`; it is NOT the mixture-reference S and never enters an S column.
S-KDE        product Gaussian KDE on an independent support: p_hat(x, y) = mean_i k_h(x - x_i) l_b(y - y_i), q_hat(x, y) = p_hat_x(x) p_hat_y(y),
             eta_hat = tanh(1/2 (log p_hat - log q_hat)) on independent queries; two readouts kept apart: S_plug = E_M eta_hat^2 and
             J_kernel = J_hat(eta_hat) (+ posterior MSE when eta is known).  Bandwidth by the common quadratic risk on non-EVAL data; Scott's
             rule kept as the native sensitivity setting.
S-Kernel     random-Fourier features phi(w) = sqrt(2/m) cos(Omega w + b) on w = [x; y] (Omega, b fixed and seed-recorded), f = theta'[phi; 1],
             T = tanh f, theta trained on the original J (the package's `fitting.train`); a Nystrom exact-kernel feature map from FIT centres is
             the small-sample reference.
rLS          RuLSIF (alpha = 1/2) closed-form ridge in a feature class (Cholesky solve, intercept-penalty flag recorded); the raw linear T is
             unbounded and is a diagnostic only.  With g = 1 + T: 1/2 E_M g^2 - E_P g = -1/2 - J(T)/2.
rLS-tanh     tanh(c T_raw + b0) with the two scalars fitted on selection data by numerical maximisation of J_hat (not a closed form).
All exact algebra in float64; chunked Gram computations; no explicit matrix inverse.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn

from .objectives import j_hat, posterior_mse

LOG2PI = math.log(2.0 * math.pi)


# ------------------------------------------------------------------------------------------------------------ helpers
def median_distance(Z: torch.Tensor, n_sub: int = 1000, seed: int = 0) -> float:
    """Median pairwise Euclidean distance on a seeded subsample (the FIT statistic every bandwidth multiple refers to)."""
    g = torch.Generator().manual_seed(seed); Z = Z[torch.randperm(len(Z), generator=g)[:n_sub]].double()
    d2 = torch.cdist(Z, Z) ** 2; iu = torch.triu_indices(len(Z), len(Z), 1)
    return float(d2[iu[0], iu[1]].clamp_min(0).sqrt().median())


def scott_bandwidth(Z: torch.Tensor) -> float:
    """Scott's rule (isotropic): n^{-1/(d+4)} times the mean per-coordinate sd."""
    n, d = Z.shape
    return float(Z.double().std(0).mean() * n ** (-1.0 / (d + 4)))


def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T).clamp_min(0.0)


# ------------------------------------------------------------------------------------------------------------ CS-K-native
def kernel_cs_from_grams(K: torch.Tensor, L: torch.Tensor, floor: float = 1e-300) -> dict:
    """D_CS from two Gram matrices (float64 recommended).  Autodiff passes through A, B_q, C."""
    n = K.shape[0]
    A = (K * L).sum() / n ** 2
    Bq = (K.sum() / n ** 2) * (L.sum() / n ** 2)
    C = (K.sum(1) * L.sum(1)).sum() / n ** 3
    guard = int(sum(float(v.detach()) < floor for v in (A, Bq, C)))
    A, Bq, C = (v.clamp_min(floor) for v in (A, Bq, C))
    return {"D_CS": torch.log(A) + torch.log(Bq) - 2.0 * torch.log(C), "A": A, "B_q": Bq, "C": C, "guard_activations": guard}


def kernel_cs_native(x: torch.Tensor, y: torch.Tensor, sigma_x: float, sigma_y: float, *, effective: bool = True, chunk: int = 2048,
                     floor: float = 1e-300) -> dict:
    """Chunked plug-in on n paired representations (x_i, y_i); gradients flow to x and y.  Returns tensors (D_CS differentiable)."""
    n = x.shape[0]; sx = sigma_x * (math.sqrt(2.0) if effective else 1.0); sy = sigma_y * (math.sqrt(2.0) if effective else 1.0)
    A = x.new_zeros(()); Kr, Lr = [], []
    for s in range(0, n, chunk):
        K = torch.exp(-_sqdist(x[s: s + chunk], x) / (2.0 * sx * sx)); L = torch.exp(-_sqdist(y[s: s + chunk], y) / (2.0 * sy * sy))
        A = A + (K * L).sum(); Kr.append(K.sum(1)); Lr.append(L.sum(1))
    Kr, Lr = torch.cat(Kr), torch.cat(Lr)
    A = A / n ** 2; Bq = (Kr.sum() / n ** 2) * (Lr.sum() / n ** 2); C = (Kr * Lr).sum() / n ** 3
    guard = int(sum(float(v.detach()) < floor for v in (A, Bq, C)))
    A, Bq, C = (v.clamp_min(floor) for v in (A, Bq, C))
    return {"D_CS": torch.log(A) + torch.log(Bq) - 2.0 * torch.log(C), "A": A, "B_q": Bq, "C": C, "guard_activations": guard,
            "sigma_x_gram": sx, "sigma_y_gram": sy, "effective_kernel": effective, "n_pairs": n}


def cs_lebesgue_gaussian(rho: float, d: int) -> float:
    """Classical (Lebesgue-reference) CS divergence between d independent bivariate normals (unit variances, correlation rho) and the
    product of their marginals: d log(1 - rho^2/4) - d/2 log(1 - rho^2).  Independent padded coordinates cancel (they multiply the three
    integrals by the same constant), so the value is that of the signal coordinates alone."""
    return d * math.log(1.0 - rho * rho / 4.0) - 0.5 * d * math.log(1.0 - rho * rho)


def cs_lebesgue_gaussian_quadrature(rho: float, grid: int = 1201, lim: float = 8.0) -> float:
    """The same quantity for d = 1 by numerical integration of the three L2 inner products (independent check of the closed form)."""
    t = np.linspace(-lim, lim, grid); X, Y = np.meshgrid(t, t, indexing="ij"); w = (t[1] - t[0]) ** 2; a = 1.0 - rho * rho
    p = np.exp(-(X * X - 2 * rho * X * Y + Y * Y) / (2 * a)) / (2 * np.pi * np.sqrt(a)); q = np.exp(-(X * X + Y * Y) / 2) / (2 * np.pi)
    return float(-np.log((p * q).sum() * w) * 2 + np.log((p * p).sum() * w) + np.log((q * q).sum() * w))


# ------------------------------------------------------------------------------------------------------------ S-KDE
def kde_log_density(query: torch.Tensor, support: torch.Tensor, h: float, chunk: int = 2048) -> torch.Tensor:
    """log of the isotropic Gaussian KDE (normalised, bandwidth h) of `support` at `query` rows; stable log-sum-exp; float64."""
    q, s = query.double(), support.double(); d = s.shape[1]; out = []
    for i in range(0, len(q), chunk):
        lk = -_sqdist(q[i: i + chunk], s) / (2.0 * h * h) - 0.5 * d * (LOG2PI + 2.0 * math.log(h))
        out.append(torch.logsumexp(lk, 1) - math.log(len(s)))
    return torch.cat(out)


def kde_joint_log_density(qx, qy, sx, sy, h: float, b: float, chunk: int = 2048) -> torch.Tensor:
    """log p_hat(x, y) of the product-kernel KDE built from paired support (sx_i, sy_i)."""
    qx, qy, sx, sy = (t.double() for t in (qx, qy, sx, sy)); dx, dy = sx.shape[1], sy.shape[1]; out = []
    for i in range(0, len(qx), chunk):
        lk = -_sqdist(qx[i: i + chunk], sx) / (2.0 * h * h) - _sqdist(qy[i: i + chunk], sy) / (2.0 * b * b) \
            - 0.5 * dx * (LOG2PI + 2.0 * math.log(h)) - 0.5 * dy * (LOG2PI + 2.0 * math.log(b))
        out.append(torch.logsumexp(lk, 1) - math.log(len(sx)))
    return torch.cat(out)


def s_kde_eta(qx, qy, sx, sy, h: float, b: float, chunk: int = 2048) -> torch.Tensor:
    """eta_hat = tanh(1/2 (log p_hat(x, y) - log p_hat_x(x) - log p_hat_y(y))) on query pairs; support = FIT joint pairs (their marginals
    give q_hat; the spec's product-of-marginal construction)."""
    lj = kde_joint_log_density(qx, qy, sx, sy, h, b, chunk)
    lx = kde_log_density(qx, sx, h, chunk); ly = kde_log_density(qy, sy, b, chunk)
    return torch.tanh(0.5 * (lj - lx - ly))


def same_target_readouts(tp: torch.Tensor, tn: torch.Tensor, eta_p=None, eta_n=None, S: float | None = None) -> dict:
    """The two fields kept apart: S_plug = E_M T^2 (plug-in reading) and J_kernel = J_hat(T) (common quadratic evaluation); posterior MSE."""
    tp, tn = tp.double(), tn.double()
    out = {"S_plug": float(0.5 * (tp * tp).mean() + 0.5 * (tn * tn).mean()), "J_kernel": float(j_hat(tp, tn)),
           "n_positive_pairs": len(tp), "n_negative_pairs": len(tn)}
    if eta_p is not None:
        out["posterior_mse"] = float(posterior_mse(tp, tn, eta_p.double(), eta_n.double()))
    if S is not None:
        out["S_plug_error"] = out["S_plug"] - S; out["J_kernel_error"] = out["J_kernel"] - S
    return out


def select_kde_bandwidth(sx, sy, tx_p, ty_p, tx_q, ty_q, multipliers=(0.25, 0.5, 1.0, 2.0), base_x: float | None = None,
                         base_y: float | None = None, chunk: int = 2048) -> dict:
    """Pick (h, b) = multiples of the FIT median distances maximising J_kernel on the TUNE queries (common quadratic risk); returns the grid."""
    base_x = base_x if base_x is not None else median_distance(sx); base_y = base_y if base_y is not None else median_distance(sy)
    grid = []
    for mx in multipliers:
        for my in multipliers:
            h, b = mx * base_x, my * base_y
            e = same_target_readouts(s_kde_eta(tx_p, ty_p, sx, sy, h, b, chunk), s_kde_eta(tx_q, ty_q, sx, sy, h, b, chunk))
            grid.append({"mult_x": mx, "mult_y": my, "h": h, "b": b, **e})
    best = max(grid, key=lambda r: r["J_kernel"])
    return {"h": best["h"], "b": best["b"], "mult_x": best["mult_x"], "mult_y": best["mult_y"], "base_x": base_x, "base_y": base_y,
            "tune_J_kernel": best["J_kernel"], "grid": grid, "rule": "max J_kernel on TUNE"}


# ------------------------------------------------------------------------------------------------------------ S-Kernel (RFF / Nystrom + tanh)
class RFFTanh(nn.Module):
    """f(x, y) = theta'[phi(w); 1], phi(w) = sqrt(2/m) cos(Omega w + b), w = [x; y]; Omega ~ N(0, I/sigma^2), b ~ U(0, 2 pi) fixed (seeded).
    theta starts at 0 (f = 0: the package's matched initial gradient scale).  T = tanh f is applied by the loss."""
    def __init__(self, dim: int, m: int, sigma: float, seed: int):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.register_buffer("Omega", torch.randn(dim, m, generator=g) / sigma); self.register_buffer("b", 2 * math.pi * torch.rand(m, generator=g))
        self.theta = nn.Parameter(torch.zeros(m + 1)); self.m, self.sigma, self.seed = m, float(sigma), seed

    def features(self, x, y):
        w = torch.cat([x, y], -1)
        return math.sqrt(2.0 / self.m) * torch.cos(w @ self.Omega + self.b)

    def forward(self, x, y):
        return self.features(x, y) @ self.theta[:-1] + self.theta[-1]


class NystromTanh(nn.Module):
    """Exact-kernel reference: phi(w) = K(w, C) R with R = (K(C, C) + eps I)^{-1/2} from m FIT centres, so phi phi' -> the Gaussian kernel."""
    def __init__(self, centres: torch.Tensor, sigma: float, eps: float = 1e-6):
        super().__init__(); C = centres.double(); Kcc = torch.exp(-_sqdist(C, C) / (2 * sigma * sigma)) + eps * torch.eye(len(C), dtype=C.dtype)
        ev, U = torch.linalg.eigh(Kcc); R = U @ torch.diag(ev.clamp_min(eps).rsqrt()) @ U.T
        self.register_buffer("C", C.float()); self.register_buffer("R", R.float()); self.sigma = float(sigma)
        self.theta = nn.Parameter(torch.zeros(len(C) + 1)); self.m = len(C)

    def features(self, x, y):
        w = torch.cat([x, y], -1)
        return torch.exp(-_sqdist(w, self.C) / (2 * self.sigma * self.sigma)) @ self.R

    def forward(self, x, y):
        return self.features(x, y) @ self.theta[:-1] + self.theta[-1]


# ------------------------------------------------------------------------------------------------------------ rLS (RuLSIF alpha = 1/2) and rLS-tanh
def _with_intercept(F: torch.Tensor) -> torch.Tensor:
    return torch.cat([F.double(), torch.ones(len(F), 1, dtype=torch.float64, device=F.device)], 1)


def rls_moments(Fp: torch.Tensor, Fq: torch.Tensor) -> dict:
    """Second moments of the intercept-augmented features: G = E_M phi phi', d = E_P phi - E_Q phi, E_P phi (float64, on the feature device)."""
    Ap, Aq = _with_intercept(Fp), _with_intercept(Fq)
    G = 0.5 * (Ap.T @ Ap / len(Ap) + Aq.T @ Aq / len(Aq)); d = Ap.mean(0) - Aq.mean(0)
    return {"G": G, "d": d, "mean_p": Ap.mean(0), "k": Ap.shape[1], "n_p": len(Ap), "n_q": len(Aq)}


def rls_from_moments(mom: dict, lam: float, penalise_intercept: bool = False) -> dict:
    """Closed-form ridge maximiser of J over the raw linear class T = theta'[phi; 1]: theta = 1/2 (G + lam P)^{-1} d (Cholesky solve), and the
    RuLSIF (alpha = 1/2) solution theta_g = (G + lam P)^{-1} E_P phi for g = 1 + T (identical up to the intercept when it is unpenalised)."""
    G, d, mp, k = mom["G"], mom["d"], mom["mean_p"], mom["k"]
    P = torch.eye(k, dtype=G.dtype, device=G.device)
    if not penalise_intercept:
        P[-1, -1] = 0.0
    Lc = torch.linalg.cholesky(G + lam * P)
    theta = 0.5 * torch.cholesky_solve(d[:, None], Lc)[:, 0]; theta_g = torch.cholesky_solve(mp[:, None], Lc)[:, 0]
    J_fit_raw = float(2.0 * theta @ d - theta @ G @ theta)
    return {"theta": theta, "theta_g": theta_g, "lam": lam, "penalise_intercept": penalise_intercept, "J_fit_raw": J_fit_raw, "k": k,
            "solver": "cholesky_solve (no explicit inverse)"}


def rls_fit(Fp: torch.Tensor, Fq: torch.Tensor, lam: float, penalise_intercept: bool = False) -> dict:
    """rls_from_moments on the full feature matrices (features are the un-augmented phi; the intercept is appended here)."""
    return rls_from_moments(rls_moments(Fp, Fq), lam, penalise_intercept)


def rls_apply(F: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return _with_intercept(F) @ theta


def rulsif_ls_objective(gp: torch.Tensor, gq: torch.Tensor) -> float:
    """1/2 E_M g^2 - E_P g (RuLSIF alpha = 1/2 least-squares objective on P / Q samples)."""
    gp, gq = gp.double(), gq.double()
    return float(0.5 * (0.5 * (gp * gp).mean() + 0.5 * (gq * gq).mean()) - gp.mean())


def fit_tanh_wrap(tp_raw: torch.Tensor, tn_raw: torch.Tensor, x0=(1.0, 0.0)) -> dict:
    """Scalars (c, b0) of T = tanh(c T_raw + b0) maximising J_hat on selection scores (Nelder-Mead from (1, 0), then from the best of a
    small c grid); numerical optimisation — not a closed form."""
    tp, tn = tp_raw.detach().double().cpu().numpy(), tn_raw.detach().double().cpu().numpy()

    def negJ(v):
        a, b = np.tanh(v[0] * tp + v[1]), np.tanh(v[0] * tn + v[1])
        return -(float((a - 0.5 * a * a).mean()) + float((-b - 0.5 * b * b).mean()))

    starts = [np.array(x0, dtype=float)] + [np.array([c, 0.0]) for c in (0.5, 2.0, 4.0)]
    best = min((minimize(negJ, s, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000}) for s in starts), key=lambda r: r.fun)
    return {"c": float(best.x[0]), "b0": float(best.x[1]), "J_select": float(-best.fun), "method": "Nelder-Mead on J_hat (numerical)"}


def tanh_wrap(t_raw: torch.Tensor, c: float, b0: float) -> torch.Tensor:
    return torch.tanh(c * t_raw.double() + b0)


# ------------------------------------------------------------------------------------------------------------ dictionary disagreement bound (spec §9.1)
def dictionary_disagreement(Tm: np.ndarray, w: np.ndarray, weights_M: np.ndarray | None = None) -> float:
    """D(w) = E_M[sum_j w_j T_j^2 - T_w^2] for member outputs Tm [n, m] under the empirical M (optional per-row weights)."""
    Tm = np.asarray(Tm, np.float64); w = np.asarray(w, np.float64); Tw = Tm @ w
    v = (Tm * Tm) @ w - Tw * Tw
    return float(v.mean() if weights_M is None else (weights_M * v).sum() / weights_M.sum())


def dictionary_upper_bound(Tm: np.ndarray, weights_M: np.ndarray | None = None) -> dict:
    """U_dict = min{ 1/4 E_M (max_j T_j - min_j T_j)^2,  1/2 (1 - 1/m) max_jk E_M (T_j - T_k)^2 }: a bound on D(w) over the whole simplex."""
    Tm = np.asarray(Tm, np.float64); m = Tm.shape[1]
    mean = (lambda v: float(v.mean())) if weights_M is None else (lambda v: float((weights_M * v).sum() / weights_M.sum()))
    ua = 0.25 * mean((Tm.max(1) - Tm.min(1)) ** 2)
    ub = 0.5 * (1 - 1.0 / m) * max(mean((Tm[:, j] - Tm[:, k]) ** 2) for j in range(m) for k in range(m))
    return {"U_dict": min(ua, ub), "range_bound": ua, "pair_bound": ub, "m": m}
