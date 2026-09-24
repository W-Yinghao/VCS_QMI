"""Dispatch to the reference objectives.  No loss is re-implemented here (spec §0.1, §3, §9)."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from reference.ssl_core import simclr_nt_xent, vcs_pair_loss, vicreg_loss

VCS_STAT_KEYS = ("J_raw", "R_binary", "t_pos_mean", "t_neg_mean", "t_pos_second", "t_neg_second", "sat_pos_frac", "sat_neg_frac")


def forward_features(encoder, projector, x1: Tensor, x2: Tensor, eps: float) -> dict[str, Tensor]:
    """One concatenated 2B forward so BN policy is identical for all methods (spec §5.2)."""
    if x1.shape != x2.shape or len(x1) < 2:
        raise ValueError("require equal two-view minibatches with B >= 2")
    h = encoder(torch.cat((x1, x2), dim=0))
    p = projector(h)
    z = F.normalize(p, dim=1, eps=eps)
    return {"h": h, "p_raw": p, "z_l2": z}


def critic_input_key(cfg: dict[str, Any]) -> str:
    """'z_l2' (frozen default) or 'p_raw' when model.normalization.vcs_and_simclr == 'none' (named variant)."""
    return "p_raw" if cfg["model"]["normalization"]["vcs_and_simclr"] == "none" else "z_l2"


def compute_objective(method: str, feats: dict[str, Tensor], *, cfg: dict[str, Any], critic=None,
                      pair_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Return ``{"loss": Tensor, "stats": {...floats/None}, "shift": int|None, "n_pos": int, "n_neg": int}``."""
    ocfg = cfg["objective"]
    b = feats["p_raw"].shape[0] // 2
    stats: dict[str, Any] = {k: None for k in VCS_STAT_KEYS}
    stats.update({"nt_xent": None, "vicreg_invariance": None, "vicreg_variance": None, "vicreg_covariance": None})
    if method == "vcs_qmi":
        if critic is None:
            raise ValueError("vcs_qmi requires a critic")
        key = critic_input_key(cfg)
        z1, z2 = feats[key].chunk(2, dim=0)
        s, shifts = vcs_pair_loss(z1, z2, critic, k=cfg["pairing"]["k"], generator=pair_generator)
        loss = s["loss"]
        for k in VCS_STAT_KEYS:
            stats[k] = float(s[k].detach())
        return {"loss": loss, "stats": stats, "shift": int(shifts[0]) if len(shifts) == 1 else [int(v) for v in shifts],
                "n_pos": b, "n_neg": b * cfg["pairing"]["k"]}
    if method == "simclr_matched":
        z1, z2 = feats["z_l2"].chunk(2, dim=0)  # normalized; the reference re-normalizes (idempotent)
        loss = simclr_nt_xent(z1, z2, temperature=ocfg["simclr_temperature"])
        stats["nt_xent"] = float(loss.detach())
        return {"loss": loss, "stats": stats, "shift": None, "n_pos": 2 * b, "n_neg": 2 * b * (2 * b - 2)}
    if method == "vicreg_matched_128":
        p1, p2 = feats["p_raw"].chunk(2, dim=0)  # RAW projector output, never L2-normalized
        w = ocfg["vicreg_weights"]
        v = vicreg_loss(p1, p2, inv_weight=w["invariance"], var_weight=w["variance"], cov_weight=w["covariance"],
                        eps=ocfg["vicreg_variance_eps"])
        stats["vicreg_invariance"] = float(v["invariance"].detach())
        stats["vicreg_variance"] = float(v["variance"].detach())
        stats["vicreg_covariance"] = float(v["covariance"].detach())
        return {"loss": v["loss"], "stats": stats, "shift": None, "n_pos": b, "n_neg": 0}
    raise ValueError(f"unknown method {method!r}")
