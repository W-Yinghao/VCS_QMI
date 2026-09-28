"""Estimator package v1 (§3.3, §6.4): per-estimator metrics on a fixed model and the channel-parameter gradient check."""
from __future__ import annotations

import math

import numpy as np
import torch

from .objectives import j_hat, posterior_mse, risk_hat, score_diagnostics
from .synthetic import eta


def metrics(tp, tn, eta_p=None, eta_n=None, S=None, prefix="") -> dict:
    """tp / tn: bounded scores T (float64).  J, R, and, when the oracle is known, posterior MSE and excess-to-Bayes-risk."""
    tp, tn = torch.as_tensor(tp, dtype=torch.float64), torch.as_tensor(tn, dtype=torch.float64)
    jp, jn = tp - 0.5 * tp ** 2, -tn - 0.5 * tn ** 2
    out = {"J": float(j_hat(tp, tn)), "J_se": math.sqrt(float(jp.var()) / len(tp) + float(jn.var()) / len(tn)), "R": float(risk_hat(tp, tn)),
           "n_positive_pairs": len(tp), "n_negative_pairs": len(tn)}
    if eta_p is not None:
        out["posterior_mse"] = float(posterior_mse(tp, tn, eta_p, eta_n))
        out["J_oracle_same_pairs"] = float(j_hat(eta_p, eta_n))
    if S is not None:
        out["signed_value_error"] = out["J"] - S
        out["excess_to_bayes_risk"] = out["posterior_mse"] / (1 - S) if (eta_p is not None and 1 - S > 1e-3) else None
    return {prefix + k: v for k, v in out.items()}


def full_row(tp, tn, eta_p, eta_n, S) -> dict:
    row = metrics(tp, tn, eta_p, eta_n, S); row["J_eval"] = row.pop("J"); row["score_diagnostics"] = score_diagnostics(tp.numpy(), tn.numpy())
    return row


def drho_learned(T_fn, x, e, rho, device="cpu", chunk=100000) -> dict:
    """d/d rho of E_{P_rho}[T - T^2/2] with the critic fixed; the gradient flows only through Y_rho = rho X + sqrt(1 - rho^2) E.
    Q does not depend on rho in this generator.  Per-sample derivatives give the MC standard error."""
    gs = []
    for s in range(0, len(x), chunk):
        xb, eb = x[s: s + chunk].to(device), e[s: s + chunk].to(device)
        r = torch.full((len(xb), 1), float(rho), dtype=xb.dtype, device=device, requires_grad=True)
        y = r * xb + torch.sqrt(1 - r * r) * eb
        t = T_fn(xb, y); h = (t - 0.5 * t * t).sum()
        (g,) = torch.autograd.grad(h, r); gs.append(g.detach().double().cpu().squeeze(1))
    g = torch.cat(gs)
    return {"dJ_drho": float(g.mean()), "se": float(g.std() / math.sqrt(len(g)))}


def drho_oracle_fd(x, e, xq, yq, rho, deltas=(1e-2, 3e-3, 1e-3)) -> list:
    """dS/d rho by common-random-number central differences: S(rho) = 1/2 E_P eta_rho^2 + 1/2 E_Q eta_rho^2 with P re-generated at rho +- delta."""
    out = []
    for dl in deltas:
        vals = []
        for r in (rho + dl, rho - dl):
            y = r * x + math.sqrt(1 - r * r) * e
            vals.append(0.5 * eta(x, y, r) ** 2 + 0.5 * eta(xq, yq, r) ** 2)
        diff = (vals[0] - vals[1]) / (2 * dl)
        out.append({"delta": dl, "dS_drho": float(diff.mean()), "se": float(diff.std() / math.sqrt(len(diff)))})
    return out


def drho_oracle_envelope(x, e, rho0) -> dict:
    """Envelope form: the oracle critic eta_{rho0} held fixed, derivative through the data only (equals dS/d rho at rho0)."""
    return drho_learned(lambda a, b: eta(a, b, rho0), x, e, rho0)
