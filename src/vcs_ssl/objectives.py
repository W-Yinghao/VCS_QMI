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
    return {"h": h, "p_raw": p, "z_l2": z, "h_l2": F.normalize(h, dim=1, eps=eps)}


def critic_input_key(cfg: dict[str, Any]) -> str:
    """'z_l2' (frozen default), 'p_raw' (normalization 'none'), or 'h_l2' (critic reads the L2-normalized encoder output; named variant)."""
    if cfg["model"]["critic"].get("feature_source", "z") == "h_l2":
        return "h_l2"
    return "p_raw" if cfg["model"]["normalization"]["vcs_and_simclr"] == "none" else "z_l2"


def critic_feature_dim(cfg: dict[str, Any]) -> int:
    return int(cfg["model"]["h_dim"]) if cfg["model"]["critic"].get("feature_source", "z") == "h_l2" else int(cfg["model"]["projector"]["output_dim"])


def pair_symmetric(cfg: dict[str, Any]) -> bool:
    return cfg["pairing"]["sampler"] == "random_nonzero_cyclic_shift_symmetric"


def critic_steps(cfg: dict[str, Any]) -> int:
    """Number of critic updates per batch: 1 for 'joint'; N for 'joint_critic_steps_N' (N-1 critic-only steps on detached features)."""
    mode = cfg["train"]["mode"]
    return 1 if mode == "joint" else int(mode.rsplit("_", 1)[1])


def vcs_pair_loss_negdetach(z1: Tensor, z2: Tensor, critic, *, k: int, generator: torch.Generator | None):
    """Named variant: the shifted partner z2[pi(i)] in negative pairs is detached (no gradient into the encoder through negatives'
    second view); positives unchanged. Same averaging rule as the reference."""
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have equal [B,D] shapes")
    indices, shifts = cyclic_negative_indices(len(z1), k, generator=generator, device=z1.device)
    t_pos = critic(z1, z2)
    left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1])
    right = z2.detach()[indices].reshape(-1, z2.shape[1])
    t_neg = critic(left, right)
    return vcs_from_scores(t_pos, t_neg), shifts


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
        if sym and cfg["pairing"]["negative_detach"]:
            raise ValueError("symmetric pairing and negative_detach are not combined")
        fn = vcs_pair_loss_symmetric if sym else (vcs_pair_loss_negdetach if cfg["pairing"]["negative_detach"] else vcs_pair_loss)
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


def forward_features_target(encoder, projector, x1: Tensor, x2: Tensor, eps: float, *, target_branch: str, teacher=None, predictor=None) -> dict[str, Tensor]:
    """Named variants of the two-view wiring. Returns the same keys as forward_features plus 'tgt_<key>' tensors for the target branch.

    shared  : identical to forward_features (both views through the student; no target tensors).
    stopgrad: target = detached student features of the *other* view (SimSiam-style).
    ema_tau : target = EMA teacher features (no grad).
    predictor (optional): the student side that meets the critic is predictor(student projector output), L2-normalized.
    Pairing used by compute_objective_target: positives (s1, t2) and (s2, t1) (symmetric); negatives with the same shifts.
    """
    feats = forward_features(encoder, projector, x1, x2, eps)
    if target_branch == "shared":
        return feats
    if target_branch == "stopgrad":
        t_h, t_p = feats["h"].detach(), feats["p_raw"].detach()
    else:
        with torch.no_grad():
            t_h = teacher["encoder"](torch.cat((x1, x2), dim=0))
            t_p = teacher["projector"](t_h)
    feats["tgt_p_raw"] = t_p
    feats["tgt_z_l2"] = F.normalize(t_p, dim=1, eps=eps)
    feats["tgt_h_l2"] = F.normalize(t_h, dim=1, eps=eps)
    if predictor is not None:
        q = predictor(feats["p_raw"])
        feats["p_raw"] = q
        feats["z_l2"] = F.normalize(q, dim=1, eps=eps)
    return feats


def compute_objective_target(feats: dict[str, Tensor], *, cfg: dict[str, Any], critic, pair_generator: torch.Generator | None) -> dict[str, Any]:
    """VCS objective with a separate target branch: symmetric pairs (student view a, target view b), b != a."""
    key = critic_input_key(cfg)
    s1, s2 = feats[key].chunk(2, dim=0)
    t1, t2 = feats["tgt_" + key].chunk(2, dim=0)
    k = cfg["pairing"]["k"]
    indices, shifts = cyclic_negative_indices(len(s1), k, generator=pair_generator, device=s1.device)
    t_pos = torch.cat((critic(s1, t2), critic(s2, t1)))
    l1 = s1.unsqueeze(0).expand(k, -1, -1).reshape(-1, s1.shape[1])
    l2 = s2.unsqueeze(0).expand(k, -1, -1).reshape(-1, s2.shape[1])
    t_neg = torch.cat((critic(l1, t2[indices].reshape(-1, t2.shape[1])), critic(l2, t1[indices].reshape(-1, t1.shape[1]))))
    st = vcs_from_scores(t_pos, t_neg)
    stats: dict[str, Any] = {k_: None for k_ in VCS_STAT_KEYS}
    stats.update({"nt_xent": None, "vicreg_invariance": None, "vicreg_variance": None, "vicreg_covariance": None})
    for k_ in VCS_STAT_KEYS:
        stats[k_] = float(st[k_].detach())
    b = s1.shape[0]
    return {"loss": st["loss"], "stats": stats, "shift": int(shifts[0]) if len(shifts) == 1 else [int(v) for v in shifts], "n_pos": 2 * b, "n_neg": 2 * b * k}
