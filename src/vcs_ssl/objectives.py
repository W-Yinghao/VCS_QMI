"""Dispatch to the reference objectives.  No loss is re-implemented here (spec §0.1, §3, §9)."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from reference.ssl_core import cyclic_negative_indices, simclr_nt_xent, vcs_from_scores, vcs_pair_loss, vicreg_loss

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


def pair_symmetric(cfg: dict[str, Any]) -> bool:
    return cfg["pairing"]["sampler"] == "random_nonzero_cyclic_shift_symmetric"


def critic_steps(cfg: dict[str, Any]) -> int:
    """Number of critic updates per batch: 1 for 'joint'; N for 'joint_critic_steps_N' (N-1 critic-only steps on detached features)."""
    mode = cfg["train"]["mode"]
    return 1 if mode == "joint" else int(mode.rsplit("_", 1)[1])


def vcs_pair_loss_symmetric(z1: Tensor, z2: Tensor, critic, *, k: int, generator: torch.Generator | None):
    """Named variant: score both orders. Positives (z1[i],z2[i]) and (z2[i],z1[i]); negatives (z1[i],z2[pi(i)]) and (z2[i],z1[pi(i)])
    with the same K shifts. Same averaging rule as the reference (each distribution averaged separately)."""
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have equal [B,D] shapes")
    indices, shifts = cyclic_negative_indices(len(z1), k, generator=generator, device=z1.device)
    t_pos = torch.cat((critic(z1, z2), critic(z2, z1)))
    l1 = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1])
    l2 = z2.unsqueeze(0).expand(k, -1, -1).reshape(-1, z2.shape[1])
    t_neg = torch.cat((critic(l1, z2[indices].reshape(-1, z2.shape[1])), critic(l2, z1[indices].reshape(-1, z1.shape[1]))))
    return vcs_from_scores(t_pos, t_neg), shifts


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
        sym = pair_symmetric(cfg)
        fn = vcs_pair_loss_symmetric if sym else vcs_pair_loss
        s, shifts = fn(z1, z2, critic, k=cfg["pairing"]["k"], generator=pair_generator)
        loss = s["loss"]
        for k in VCS_STAT_KEYS:
            stats[k] = float(s[k].detach())
        mult = 2 if sym else 1
        return {"loss": loss, "stats": stats, "shift": int(shifts[0]) if len(shifts) == 1 else [int(v) for v in shifts],
                "n_pos": b * mult, "n_neg": b * cfg["pairing"]["k"] * mult}
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
