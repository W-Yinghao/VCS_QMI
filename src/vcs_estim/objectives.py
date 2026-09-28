"""Estimator package v1 (§3): the sample objectives shared by every implementation.

VCS:   J = mean(t+ - t+^2/2) + mean(-t- - t-^2/2) = 1 - R,  R = 1/2 mean(1-t+)^2 + 1/2 mean(-1-t-)^2   (P and Q averaged separately).
JS control with the matched posterior parameterisation (T = tanh f, q = sigmoid(2f) = (1+T)/2):
       L_JS,match = mean softplus(-2 f+) + mean softplus(2 f-),   JS_hat = log 2 - L_JS,match / 2.
Both have logit gradient -1 / +1 per (positive / negative) pair at f = 0, divided by the pair count of the side.
Functions accept torch tensors or numpy arrays (numpy in float64 for the exact algebra tests).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _m(a):
    return a.mean()


def j_hat(tp, tn):
    return _m(tp - 0.5 * tp * tp) + _m(-tn - 0.5 * tn * tn)


def risk_hat(tp, tn):
    return 0.5 * _m((1 - tp) ** 2) + 0.5 * _m((-1 - tn) ** 2)


def vcs_loss(fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    return -j_hat(torch.tanh(fp), torch.tanh(fn))


def js_match_loss(fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    return F.softplus(-2 * fp).mean() + F.softplus(2 * fn).mean()


def js_native(loss_match: float) -> float:
    return math.log(2.0) - 0.5 * float(loss_match)


def loss_for(kind: str):
    return {"vcs": vcs_loss, "js": js_match_loss}[kind]


def posterior_mse(tp, tn, eta_p, eta_n):
    """1/2 E_P (T - eta)^2 + 1/2 E_Q (T - eta)^2  (= E_M (T - eta)^2)."""
    return 0.5 * _m((tp - eta_p) ** 2) + 0.5 * _m((tn - eta_n) ** 2)


def gate_stats(t) -> dict:
    """1 - T^2 (the tanh gate): mean and quantiles."""
    g = 1 - np.asarray(t, dtype=np.float64) ** 2
    q = np.quantile(g, [0.01, 0.1, 0.5, 0.9, 0.99])
    return {"mean": float(g.mean()), "q01": float(q[0]), "q10": float(q[1]), "q50": float(q[2]), "q90": float(q[3]), "q99": float(q[4])}


def score_diagnostics(tp, tn) -> dict:
    """Per-side score quantiles and saturation on the right / wrong end (|T| > 0.95)."""
    tp, tn = np.asarray(tp, dtype=np.float64), np.asarray(tn, dtype=np.float64)
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    return {"pos_quantiles": [float(v) for v in np.quantile(tp, qs)], "neg_quantiles": [float(v) for v in np.quantile(tn, qs)],
            "pos_sat_right": float((tp > 0.95).mean()), "pos_sat_wrong": float((tp < -0.95).mean()),
            "neg_sat_right": float((tn < -0.95).mean()), "neg_sat_wrong": float((tn > 0.95).mean()),
            "gate_pos": gate_stats(tp), "gate_neg": gate_stats(tn)}
