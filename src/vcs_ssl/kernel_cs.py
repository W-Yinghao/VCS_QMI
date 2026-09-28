"""CS-K-native: the classical kernel (plug-in) Cauchy–Schwarz QMI on paired representations (Server Spec v2 §3.2, §6.1).

For a view pair with L2-normalised projector outputs z1, z2 ∈ R^{B×D}, Gaussian kernels K_ij = k(z1_i, z1_j) = exp(−‖z1_i − z1_j‖² / 2σ²) and
L_ij = l(z2_i, z2_j) with one shared bandwidth σ (a fixed multiple of the FIT median pairwise distance measured once at the start of the run),

    A   = (1/B²) Σ_ij K_ij L_ij,
    B_q = ((1/B²) Σ_ij K_ij) ((1/B²) Σ_ij L_ij),
    C   = (1/B³) Σ_i (Σ_j K_ij)(Σ_j L_ij),
    D_CS = log A + log B_q − 2 log C.

Fixed reference measure (Lebesgue / KDE), not the mixture-reference S of VCS: the two are different quantities and never share an RMSE
column.  Everything is computed in the log domain (logsumexp over log K + log L), so the products cannot underflow; the fraction of kernel
entries whose *naive* float32 product K_ij·L_ij would underflow (< 1e-38) is counted and reported as ``underflow_frac`` (the guard's
"activation frequency").  Native autodiff through both z1 and z2; there is no detach and no "negative branch".  Extremes (tested):
K = L = 11ᵀ gives 0; K = L = I gives log B (the plug-in diagonal effect, not dependence).  Row chunking gives identical values.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

FLOAT32_UNDERFLOW_LOG = -87.3  # log(1e-38): below this a float32 product K_ij * L_ij is denormal / zero


def log_gaussian_gram(z: Tensor, sigma: float) -> Tensor:
    """log K_ij = −‖z_i − z_j‖² / (2σ²), computed from the exact squared distance (clamped at 0)."""
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("need a [B, D] matrix with B >= 2")
    if not (sigma > 0):
        raise ValueError("bandwidth must be > 0")
    sq = (z * z).sum(-1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)).clamp_min(0.0)
    return -d2 / (2.0 * sigma * sigma)


def kernel_cs_from_log_grams(logK: Tensor, logL: Tensor, *, chunk: int = 0) -> dict[str, Tensor]:
    """D_CS and its three log terms from log-kernel matrices [B, B] (log domain throughout; optional row chunking)."""
    if logK.shape != logL.shape or logK.ndim != 2 or logK.shape[0] != logK.shape[1]:
        raise ValueError("log kernel matrices must be square and equal-shaped")
    B = logK.shape[0]
    logB = torch.log(torch.tensor(float(B), dtype=logK.dtype, device=logK.device))
    if chunk and 0 < chunk < B:
        parts_A, parts_K, parts_L, parts_C = [], [], [], []
        for s in range(0, B, chunk):
            k, l = logK[s:s + chunk], logL[s:s + chunk]
            parts_A.append(torch.logsumexp((k + l).reshape(-1), dim=0))
            parts_K.append(torch.logsumexp(k.reshape(-1), dim=0))
            parts_L.append(torch.logsumexp(l.reshape(-1), dim=0))
            parts_C.append(torch.logsumexp(torch.logsumexp(k, dim=1) + torch.logsumexp(l, dim=1), dim=0))
        lse_A = torch.logsumexp(torch.stack(parts_A), dim=0)
        lse_K = torch.logsumexp(torch.stack(parts_K), dim=0)
        lse_L = torch.logsumexp(torch.stack(parts_L), dim=0)
        lse_C = torch.logsumexp(torch.stack(parts_C), dim=0)
    else:
        lse_A = torch.logsumexp((logK + logL).reshape(-1), dim=0)
        lse_K = torch.logsumexp(logK.reshape(-1), dim=0)
        lse_L = torch.logsumexp(logL.reshape(-1), dim=0)
        lse_C = torch.logsumexp(torch.logsumexp(logK, dim=1) + torch.logsumexp(logL, dim=1), dim=0)
    log_A = lse_A - 2.0 * logB
    log_Bq = lse_K + lse_L - 4.0 * logB
    log_C = lse_C - 3.0 * logB
    d_cs = log_A + log_Bq - 2.0 * log_C
    with torch.no_grad():
        underflow = ((logK + logL) < FLOAT32_UNDERFLOW_LOG).float().mean()
    return {"D_CS": d_cs, "log_A": log_A, "log_Bq": log_Bq, "log_C": log_C, "underflow_frac": underflow}


def kernel_cs_pair_loss(z1: Tensor, z2: Tensor, *, sigma: float, chunk: int = 0) -> dict[str, Any]:
    """loss = −D_CS for one view pair (both sides differentiable).  Returns loss and float diagnostics."""
    if z1.ndim != 2 or z1.shape != z2.shape or len(z1) < 2:
        raise ValueError("z1 and z2 must have equal [B, D] shapes with B >= 2")
    out = kernel_cs_from_log_grams(log_gaussian_gram(z1, sigma), log_gaussian_gram(z2, sigma), chunk=chunk)
    return {"loss": -out["D_CS"], "kernel_cs": out["D_CS"], "kcs_log_A": out["log_A"], "kcs_log_Bq": out["log_Bq"], "kcs_log_C": out["log_C"],
            "kcs_underflow_frac": out["underflow_frac"]}


@torch.no_grad()
def median_pairwise_distance(z: Tensor, max_n: int = 4096) -> float:
    """Median of ‖z_i − z_j‖ over i < j (first ``max_n`` rows); the bandwidth calibration statistic."""
    z = z[:max_n].double()
    d = torch.cdist(z, z)
    iu = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    return float(d[iu[0], iu[1]].median())


KERNEL_CS_STAT_KEYS = ("kernel_cs", "kcs_log_A", "kcs_log_Bq", "kcs_log_C", "kcs_underflow_frac")
