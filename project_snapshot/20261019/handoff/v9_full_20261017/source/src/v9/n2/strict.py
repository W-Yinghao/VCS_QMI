"""V9-N2 strict-interface check (package §03.E and §05.5, last paragraph): a small but complete experiment on a FIXED teacher representation
Z = s with an independent negative pool, where the per-pair loss depends only on what the critic sees.  Under the clean-component weights
w = E[C | Z] the identity E_R[w ell(Z)] = (1 - rho) E_P[ell(Z)] holds; with the fitted cos-2 posterior it holds up to estimation error; and a loss
that reads an extra variable correlated with C given Z (the student's richer input) leaves a covariance term that does NOT vanish — the
package's own counter-example, reproduced on the same pipeline.  Score model: s | P ~ N(mu_P, sd^2), s | Q ~ N(mu_Q, sd^2), so the true
log-ratio is linear in s and the cos-2 class contains T* exactly.
"""
from __future__ import annotations

import numpy as np

from .weights import REF, fit_scalar_critic, posterior_weights


def run_strict_check(rho: float = 0.4, n: int = 4000, mu_p: float = 0.7, mu_q: float = 0.3, sd: float = 0.15, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    def draw(m, k):
        return np.clip(rng.normal(m, sd, size=k), 0, 1)
    # oracle eta on the clipped-score model is approximated by the unclipped Gaussian ratio (clipping mass is negligible at these settings)
    def eta(s):
        lr = ((s - mu_q) ** 2 - (s - mu_p) ** 2) / (2 * sd ** 2)
        return np.tanh(lr / 2)
    sP_fit, sQ_fit, sP_sel, sQ_sel = draw(mu_p, n), draw(mu_q, n), draw(mu_p, n // 2), draw(mu_q, n // 2)
    C = rng.random(n) < (1 - rho); sR = np.where(C, draw(mu_p, n), draw(mu_q, n))
    sP_ref, sQ_ref = draw(mu_p, n), draw(mu_q, n)
    # per-pair loss on the critic's representation only
    ell = lambda s: (1.0 - s) ** 2
    target = (1 - rho) * float(np.mean(ell(sP_ref)))
    # oracle weights
    w_or = REF.clean_posterior(eta(sR), rho); lhs_oracle = float(np.mean(w_or * ell(sR)))
    # fitted cos-2 posterior (VCS rule), rho estimated from the three means
    fit = fit_scalar_critic(sP_fit, sQ_fit, sP_sel, sQ_sel, kind="vcs", steps=300, seed=seed); T = fit["critic"].T
    a, b, c = float(np.mean(T(sP_ref))), float(np.mean(T(sQ_ref))), float(np.mean(T(sR))); est = REF.rho_from_means(a, b, c)
    rho_hat = est.projected if est.projected is not None else 0.0
    w_fit = posterior_weights(T(sR), rho_hat); lhs_fit = float(np.mean(w_fit * ell(sR)))
    # counter-example: the student reads W = (s, g) where g is informative about C beyond s; loss depends on g
    g = np.where(C, rng.normal(1.0, 0.5, n), rng.normal(-1.0, 0.5, n)); ell_w = ell(sR) * (1.0 + 0.5 * np.tanh(g))
    gP = rng.normal(1.0, 0.5, n); target_w = (1 - rho) * float(np.mean(ell(sP_ref) * (1.0 + 0.5 * np.tanh(gP))))
    lhs_w_oracle_z = float(np.mean(w_or * ell_w))
    return {"rho": rho, "rho_hat": rho_hat, "rho_status": est.status, "critic": fit["selected"],
            "z_only": {"target": target, "oracle_w": lhs_oracle, "fitted_w": lhs_fit, "rel_err_oracle": abs(lhs_oracle - target) / target, "rel_err_fitted": abs(lhs_fit - target) / target},
            "richer_student": {"target": target_w, "oracle_w_on_Z": lhs_w_oracle_z, "rel_err": abs(lhs_w_oracle_z - target_w) / target_w,
                               "reading": "covariance term E_R Cov(C, ell | Z) does not vanish: Z-posterior weights do not recover the richer student's risk"},
            "n": n, "seed": seed}
