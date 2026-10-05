"""Dispatch to the reference objectives.  No loss is re-implemented here (spec §0.1, §3, §9)."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from reference.ssl_core import cyclic_negative_indices, simclr_nt_xent, vcs_from_scores, vcs_pair_loss, vicreg_loss

from .kernel_cs import KERNEL_CS_STAT_KEYS, kernel_cs_pair_loss

VCS_STAT_KEYS = ("J_raw", "R_binary", "t_pos_mean", "t_neg_mean", "t_pos_second", "t_neg_second", "sat_pos_frac", "sat_neg_frac")


def _empty_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {k: None for k in VCS_STAT_KEYS}
    stats.update({"nt_xent": None, "vicreg_invariance": None, "vicreg_variance": None, "vicreg_covariance": None})
    stats.update({k: None for k in KERNEL_CS_STAT_KEYS})
    return stats


def _kernel_cs_objective(pairs: list[tuple[Tensor, Tensor]], *, cfg: dict[str, Any], kernel_sigma: float | None) -> dict[str, Any]:
    """CS-K-native (Server Spec v2 §3.2): −D_CS averaged over the given view pairs; kernels on z_l2 with the calibrated bandwidth."""
    if kernel_sigma is None or not (kernel_sigma > 0):
        raise ValueError("cs_kernel_native needs the calibrated bandwidth (kernel_sigma > 0) from the trainer")
    chunk = int(cfg["objective"].get("kernel_cs_chunk", 0))
    outs = [kernel_cs_pair_loss(a, b_, sigma=float(kernel_sigma), chunk=chunk) for a, b_ in pairs]
    loss = torch.stack([o["loss"] for o in outs]).mean()
    stats = _empty_stats()
    for k in KERNEL_CS_STAT_KEYS:
        stats[k] = float(torch.stack([o[k].detach() for o in outs]).mean())
    stats["kcs_sigma"] = float(kernel_sigma)
    B = pairs[0][0].shape[0]
    return {"loss": loss, "stats": stats, "shift": None, "n_pos": B * len(pairs), "n_neg": B * (B - 1) * len(pairs)}


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


def pair_all_matrix(cfg: dict[str, Any]) -> bool:
    return cfg["pairing"]["sampler"] == "all_pairs_matrix"


def vcs_pair_loss_all_matrix(z1: Tensor, z2: Tensor, critic, *, negative_detach: bool = False):
    """All B(B-1) off-diagonal pairs at once (equivalent to K = B-1 distinct nonzero shifts; same separate averaging as the reference).
    Requires a similarity-type critic exposing embed(z) and score_matrix(C). Returns (stats, shifts=None)."""
    if z1.ndim != 2 or z1.shape != z2.shape or len(z1) < 2:
        raise ValueError("z1 and z2 must have equal [B,D] shapes with B >= 2")
    e1, e2 = critic.embed(z1), critic.embed(z2)
    e2n = e2.detach() if negative_detach else e2
    t_pos = critic.score_matrix((e1 * e2).sum(-1))
    T = critic.score_matrix(e1 @ e2n.T)
    mask = ~torch.eye(len(z1), dtype=torch.bool, device=z1.device)
    t_neg = T[mask]
    return vcs_from_scores(t_pos, t_neg), None


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


class NegativeQueue:
    """Named variant (wave-2 C-T): FIFO ring of *detached* features from previous optimizer steps, used as the product-of-marginals
    sample for VCS (K partners per anchor, drawn without replacement per anchor with the dedicated CPU pair generator) or as
    MoCo-style negatives for SimCLR.  Same encoder, no momentum branch; both methods get the identical queue treatment."""

    def __init__(self, size: int, dim: int, device: torch.device | str = "cpu") -> None:
        if size < 2 or dim < 1:
            raise ValueError("queue needs size >= 2 and dim >= 1")
        self.size, self.dim = int(size), int(dim)
        self.buffer = torch.zeros(self.size, self.dim, device=device)
        self.ptr, self.count = 0, 0

    @property
    def filled(self) -> int:
        return self.count

    def ready(self, k: int) -> bool:
        return self.count >= k

    def features(self) -> Tensor:
        """The filled part in FIFO order (oldest first); detached, no grad."""
        if self.count < self.size:
            return self.buffer[: self.count]
        return torch.cat((self.buffer[self.ptr:], self.buffer[: self.ptr]), dim=0)

    @torch.no_grad()
    def enqueue(self, feats: Tensor) -> None:
        f = feats.detach().to(self.buffer.device, self.buffer.dtype)
        if f.ndim != 2 or f.shape[1] != self.dim:
            raise ValueError("queue features must be [n, dim]")
        if len(f) >= self.size:  # only the newest `size` entries survive
            f = f[-self.size:]
            self.buffer.copy_(f); self.ptr, self.count = 0, self.size
            return
        n = len(f)
        end = self.ptr + n
        if end <= self.size:
            self.buffer[self.ptr:end] = f
        else:
            first = self.size - self.ptr
            self.buffer[self.ptr:] = f[:first]; self.buffer[: n - first] = f[first:]
        self.ptr = end % self.size
        self.count = min(self.size, self.count + n)

    def sample_indices(self, n_anchor: int, k: int, generator: torch.Generator | None) -> Tensor:
        """[K, n_anchor] indices into `features()`: per anchor K distinct queue entries (uniform without replacement)."""
        if not self.ready(k):
            raise ValueError("queue holds fewer entries than K")
        r = torch.rand(n_anchor, self.count, generator=generator)  # CPU generator (dedicated pairing stream)
        return r.argsort(dim=1)[:, :k].T.contiguous().to(self.buffer.device)

    def sample_indices_fast(self, n_anchor: int, k: int, generator: torch.Generator | None) -> Tensor:
        """P100: [K, n_anchor] indices into `features()` drawn uniformly *with* replacement (dedicated CPU pair generator).  The
        without-replacement sampler above costs an argsort of n_anchor x Q per call (6 view pairs per step at Q = 4096); with replacement a
        duplicate partner occurs for ~K(K-1)/(2Q) = 0.7 % of anchors at K = 8, Q = 4096 (disclosed)."""
        if not self.ready(k):
            raise ValueError("queue holds fewer entries than K")
        return torch.randint(0, self.count, (k, n_anchor), generator=generator).to(self.buffer.device)

    def state_dict(self) -> dict[str, Any]:
        return {"buffer": self.buffer.detach().cpu().clone(), "ptr": self.ptr, "count": self.count, "size": self.size, "dim": self.dim}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        if int(st["size"]) != self.size or int(st["dim"]) != self.dim:
            raise ValueError("queue state shape mismatch")
        self.buffer.copy_(st["buffer"].to(self.buffer.device)); self.ptr, self.count = int(st["ptr"]), int(st["count"])


def vcs_pair_loss_queue(z1: Tensor, z2: Tensor, critic, *, k: int, queue: NegativeQueue, generator: torch.Generator | None, indices: Tensor | None = None):
    """Named variant: positives (z1[i], z2[i]) as in the reference; the K partners of anchor z1[i] in the negative (product) term come
    from the queue of detached features of previous steps (no gradient through partners — negative_detach semantics). Same separate
    averaging as the reference. `indices` [K,B] overrides the sampled partner indices (tests)."""
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have equal [B,D] shapes")
    q = queue.features()
    idx = queue.sample_indices(len(z1), k, generator) if indices is None else indices.to(q.device)
    t_pos = critic(z1, z2)
    left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1])
    right = q[idx].reshape(-1, q.shape[1])
    t_neg = critic(left, right)
    return vcs_from_scores(t_pos, t_neg), None


def simclr_nt_xent_queue(p1: Tensor, p2: Tensor, queue_feats: Tensor, temperature: float) -> Tensor:
    """MoCo-style NT-Xent with queue negatives and no momentum encoder: for each of the 2B anchors the logits are
    [sim(anchor, its positive = the other view of the same image), sim(anchor, queue_1..Q)] / tau, target index 0."""
    if p1.ndim != 2 or p1.shape != p2.shape or len(p1) < 1 or queue_feats.ndim != 2 or queue_feats.shape[1] != p1.shape[1]:
        raise ValueError("require matching [B,D] views and a [Q,D] queue")
    x = F.normalize(torch.cat((p1, p2), dim=0).float(), dim=-1, eps=1e-8)
    pos = F.normalize(torch.cat((p2, p1), dim=0).float(), dim=-1, eps=1e-8)
    q = F.normalize(queue_feats.detach().float(), dim=-1, eps=1e-8)
    l_pos = (x * pos).sum(-1, keepdim=True)
    l_neg = x @ q.T
    logits = torch.cat((l_pos, l_neg), dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(x), dtype=torch.long, device=x.device))


def simclr_nt_xent_plus_queue(p1: Tensor, p2: Tensor, queue_feats: Tensor, temperature: float) -> Tensor:
    """P100: the reference NT-Xent (2B anchors, self masked, the other 2B-2 in-batch samples as negatives) with the Q momentum keys of the
    queue appended as *additional* negatives (no gradient through the queue).  With an empty queue this equals `simclr_nt_xent`."""
    if p1.ndim != 2 or p1.shape != p2.shape or len(p1) < 2 or queue_feats.ndim != 2 or queue_feats.shape[1] != p1.shape[1]:
        raise ValueError("require matching [B,D] views and a [Q,D] queue")
    x = F.normalize(torch.cat((p1, p2), dim=0).float(), dim=-1, eps=1e-8)
    q = F.normalize(queue_feats.detach().float(), dim=-1, eps=1e-8)
    n, b = len(x), len(p1)
    inb = (x @ x.T).masked_fill(torch.eye(n, dtype=torch.bool, device=x.device), -torch.inf)
    logits = torch.cat((inb, x @ q.T), dim=1) / temperature
    targets = (torch.arange(n, device=x.device) + b) % n
    return F.cross_entropy(logits, targets)


def vcs_pair_loss_momentum_queue(z1: Tensor, z2: Tensor, critic, *, k: int, queue: "NegativeQueue", generator: torch.Generator | None):
    """P100: positives (z1[i], z2[i]) as in the recipe; the K partners of anchor z1[i] in the product term are momentum keys of *previous*
    steps drawn from the queue (other images; no gradient by construction = the recipe's negative-detach rule).  Same separate averaging."""
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have equal [B,D] shapes")
    q = queue.features()
    idx = queue.sample_indices_fast(len(z1), k, generator)
    t_pos = critic(z1, z2)
    left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1])
    t_neg = critic(left, q[idx].reshape(-1, q.shape[1]))
    return vcs_from_scores(t_pos, t_neg)


def js_matched_pair_loss_negdetach(z1: Tensor, z2: Tensor, critic, *, k: int, generator: torch.Generator | None, negative_detach: bool = True):
    """P95 control (Server Spec v2 §2.2): the balanced logistic (JS) loss on the *same* cosine critic logit f = a⟨z1, z2⟩ + b, the same
    K cyclic-shift negatives from the same generator draw and the same detach of the shifted partner:
        L = mean_P softplus(−2f) + mean_Q softplus(2f)        (P and Q averaged separately = 1:1 total weight).
    Its f-gradient at f = 0 equals that of −J (both ∓1), so the two losses start at the same step size.  The posterior is
    q = σ(2f) = (1 + tanh f)/2; J and the VCS statistics are computed from T = tanh(f) (detached) for comparison."""
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have equal [B,D] shapes")
    if not (hasattr(critic, "scale") and hasattr(critic, "bias")):
        raise ValueError("js_matched_logistic needs the cosine critic (scale, bias)")
    indices, shifts = cyclic_negative_indices(len(z1), k, generator=generator, device=z1.device)
    noisy = getattr(critic, "is_noisy", False)  # P104 N2: the matched-JS control on the same *noisy* logit (fresh noise per call and side)
    # P126: the fixed curved scorer's logit is non-affine in s -> read it through critic.logits (never re-hard-code a*s + b)
    via_logits = noisy or getattr(critic, "is_curved", False)
    f_pos = critic.logits(z1, z2) if via_logits else critic.scale * (z1 * z2).sum(-1) + critic.bias
    left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1])
    partner = z2.detach() if negative_detach else z2
    right = partner[indices].reshape(-1, z2.shape[1])
    f_neg = critic.logits(left, right) if via_logits else critic.scale * (left * right).sum(-1) + critic.bias
    loss = F.softplus(-2.0 * f_pos).mean() + F.softplus(2.0 * f_neg).mean()
    with torch.no_grad():
        st = vcs_from_scores(torch.tanh(f_pos), torch.tanh(f_neg))
    st = dict(st)
    st["loss"] = loss
    st["js_loss"] = loss.detach()
    return st, shifts


def uses_queue(cfg: dict[str, Any]) -> bool:
    return cfg["pairing"].get("negative_source", "cyclic") == "queue"


def compute_objective(method: str, feats: dict[str, Tensor], *, cfg: dict[str, Any], critic=None,
                      pair_generator: torch.Generator | None = None, queue: NegativeQueue | None = None,
                      kernel_sigma: float | None = None) -> dict[str, Any]:
    """Return ``{"loss": Tensor, "stats": {...floats/None}, "shift": int|None, "n_pos": int, "n_neg": int}``.
    With pairing.negative_source == 'queue' a NegativeQueue must be passed; while it holds fewer than K entries (first step) the
    cyclic-shift negatives are used and ``stats['queue_fallback']`` is 1.0.  ``kernel_sigma`` is the calibrated bandwidth of cs_kernel_native."""
    ocfg = cfg["objective"]
    b = feats["p_raw"].shape[0] // 2
    stats = _empty_stats()
    if method == "cs_kernel_native":
        z1, z2 = feats["z_l2"].chunk(2, dim=0)
        return _kernel_cs_objective([(z1, z2)], cfg=cfg, kernel_sigma=kernel_sigma)
    if uses_queue(cfg):
        if queue is None:
            raise ValueError("pairing.negative_source == 'queue' requires a NegativeQueue")
        k = int(cfg["pairing"]["k"])
        stats["queue_fill"] = float(queue.filled)
        if queue.ready(k):
            stats["queue_fallback"] = 0.0
            if method == "vcs_qmi":
                if critic is None:
                    raise ValueError("vcs_qmi requires a critic")
                z1, z2 = feats[critic_input_key(cfg)].chunk(2, dim=0)
                s, _ = vcs_pair_loss_queue(z1, z2, critic, k=k, queue=queue, generator=pair_generator)
                for kk in VCS_STAT_KEYS:
                    stats[kk] = float(s[kk].detach())
                return {"loss": s["loss"], "stats": stats, "shift": None, "n_pos": b, "n_neg": b * k}
            if method == "simclr_matched":
                z1, z2 = feats["z_l2"].chunk(2, dim=0)
                loss = simclr_nt_xent_queue(z1, z2, queue.features(), temperature=ocfg["simclr_temperature"])
                stats["nt_xent"] = float(loss.detach())
                return {"loss": loss, "stats": stats, "shift": None, "n_pos": 2 * b, "n_neg": 2 * b * queue.filled}
            raise ValueError(f"queue negatives are not defined for {method!r}")
        stats["queue_fallback"] = 1.0  # queue not yet filled to K: this step uses the frozen in-batch negatives
    if method == "vcs_qmi":
        if critic is None:
            raise ValueError("vcs_qmi requires a critic")
        key = critic_input_key(cfg)
        z1, z2 = feats[key].chunk(2, dim=0)
        sym = pair_symmetric(cfg)
        if sym and cfg["pairing"]["negative_detach"]:
            raise ValueError("symmetric pairing and negative_detach are not combined")
        if pair_all_matrix(cfg):
            s, shifts = vcs_pair_loss_all_matrix(z1, z2, critic, negative_detach=cfg["pairing"]["negative_detach"])
            for k in VCS_STAT_KEYS:
                stats[k] = float(s[k].detach())
            return {"loss": s["loss"], "stats": stats, "shift": None, "n_pos": b, "n_neg": b * (b - 1)}
        if ocfg["loss"] == "js_matched_logistic":  # P95 control
            s, shifts = js_matched_pair_loss_negdetach(z1, z2, critic, k=cfg["pairing"]["k"], generator=pair_generator,
                                                       negative_detach=cfg["pairing"]["negative_detach"])
            stats["js_loss"] = float(s["js_loss"])
        else:
            fn = vcs_pair_loss_symmetric if sym else (vcs_pair_loss_negdetach if cfg["pairing"]["negative_detach"] else vcs_pair_loss)
            s, shifts = fn(z1, z2, critic, k=cfg["pairing"]["k"], generator=pair_generator)
        loss = s["loss"]
        for k in VCS_STAT_KEYS:
            stats[k] = float(s[k].detach())
        if hasattr(critic, "pop_stats"):  # P95 critic variants
            stats.update(critic.pop_stats())
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
    stats = _empty_stats()
    for k_ in VCS_STAT_KEYS:
        stats[k_] = float(st[k_].detach())
    b = s1.shape[0]
    return {"loss": st["loss"], "stats": stats, "shift": int(shifts[0]) if len(shifts) == 1 else [int(v) for v in shifts], "n_pos": 2 * b, "n_neg": 2 * b * k}


def forward_features_views(encoder, projector, views: list[Tensor], eps: float) -> dict[str, Any]:
    """n >= 2 views of the same B images: one concatenated forward (BN policy shared), returns per-view lists."""
    if len(views) < 2 or any(v.shape != views[0].shape for v in views):
        raise ValueError("views must be a list of >= 2 equal-shape tensors")
    h = encoder(torch.cat(views, dim=0))
    p = projector(h)
    z = F.normalize(p, dim=1, eps=eps)
    n = len(views)
    return {"h": h, "p_raw": p, "z_l2": z, "h_l2": F.normalize(h, dim=1, eps=eps),
            "views_z": list(z.chunk(n, dim=0)), "views_p": list(p.chunk(n, dim=0)), "views_h": list(F.normalize(h, dim=1, eps=eps).chunk(n, dim=0))}


def compute_objective_views(feats: dict[str, Any], *, cfg: dict[str, Any], critic, pair_generator: torch.Generator | None,
                            kernel_sigma: float | None = None, queue: "NegativeQueue | None" = None) -> dict[str, Any]:
    """Named variant (views.count = 4): the same J averaged over all view pairs (a < b) of the same images; each pair draws its own shifts.
    Q still comes from different UIDs (cyclic shifts). Not a new loss: more Monte-Carlo coverage of the same P and Q."""
    method = cfg["run"]["method"]
    ocfg = cfg["objective"]
    if queue is not None:
        return _views_momentum_queue(feats, cfg=cfg, critic=critic, pair_generator=pair_generator, queue=queue)
    if method == "vcs_qmi":  # P104 paths (inactive unless the new optional fields are set)
        p104 = compute_objective_views_p104(feats, cfg=cfg, critic=critic, pair_generator=pair_generator)
        if p104 is not None:
            return p104
    if method == "cs_kernel_native":
        vs = feats["views_z"]
        return _kernel_cs_objective([(vs[a], vs[b_]) for a in range(len(vs)) for b_ in range(a + 1, len(vs))], cfg=cfg, kernel_sigma=kernel_sigma)
    if method != "vcs_qmi":
        # control-tuning named variant: the control's own pairwise loss averaged over all view pairs (a < b); nothing else changes
        vs = feats["views_z"] if method == "simclr_matched" else feats["views_p"]
        pairs = [(a, b_) for a in range(len(vs)) for b_ in range(a + 1, len(vs))]
        stats = _empty_stats()
        B = vs[0].shape[0]
        if method == "simclr_matched":
            losses = [simclr_nt_xent(vs[a], vs[b_], temperature=ocfg["simclr_temperature"]) for a, b_ in pairs]
            loss = torch.stack(losses).mean()
            stats["nt_xent"] = float(loss.detach())
            return {"loss": loss, "stats": stats, "shift": None, "n_pos": 2 * B * len(pairs), "n_neg": 2 * B * (2 * B - 2) * len(pairs)}
        if method == "vicreg_matched_128":
            w = ocfg["vicreg_weights"]
            outs = [vicreg_loss(vs[a], vs[b_], inv_weight=w["invariance"], var_weight=w["variance"], cov_weight=w["covariance"], eps=ocfg["vicreg_variance_eps"])
                    for a, b_ in pairs]
            loss = torch.stack([o["loss"] for o in outs]).mean()
            for name in ("invariance", "variance", "covariance"):
                stats[f"vicreg_{name}"] = float(torch.stack([o[name] for o in outs]).mean().detach())
            return {"loss": loss, "stats": stats, "shift": None, "n_pos": B * len(pairs), "n_neg": 0}
        raise ValueError(f"unknown method {method!r}")
    key = {"z_l2": "views_z", "p_raw": "views_p", "h_l2": "views_h"}[critic_input_key(cfg)]
    vs = feats[key]
    k = cfg["pairing"]["k"]
    nd = cfg["pairing"]["negative_detach"]
    js = ocfg["loss"] == "js_matched_logistic"  # P95 control: same critic, pairs, shifts and detach; balanced logistic loss
    fn = vcs_pair_loss_negdetach if nd else vcs_pair_loss
    losses, acc, shifts = [], {kk: 0.0 for kk in VCS_STAT_KEYS}, []
    pairs = [(a, b_) for a in range(len(vs)) for b_ in range(a + 1, len(vs))]
    for a, b_ in pairs:
        if js:
            s, sh = js_matched_pair_loss_negdetach(vs[a], vs[b_], critic, k=k, generator=pair_generator, negative_detach=nd)
        else:
            s, sh = fn(vs[a], vs[b_], critic, k=k, generator=pair_generator)
        losses.append(s["loss"]); shifts.append(int(sh[0]))
        for kk in VCS_STAT_KEYS:
            acc[kk] += float(s[kk].detach()) / len(pairs)
    loss = torch.stack(losses).mean()
    stats = _empty_stats()
    stats.update(acc)
    if js:
        stats["js_loss"] = float(loss.detach())
    if hasattr(critic, "pop_stats"):  # P95 critic variants (call-averaged over the 2 x n_pairs critic calls = E_M)
        stats.update(critic.pop_stats())
    B = vs[0].shape[0]
    return {"loss": loss, "stats": stats, "shift": shifts, "n_pos": B * len(pairs), "n_neg": B * k * len(pairs)}


def _views_momentum_queue(feats: dict[str, Any], *, cfg: dict[str, Any], critic, pair_generator: torch.Generator | None, queue: "NegativeQueue") -> dict[str, Any]:
    """P100 (named variant; views.count = 2 or 4): the recipe's loss averaged over all view pairs (a < b) with negatives from the
    momentum-key queue.  VCS: K = 8 queue keys replace the K cyclic-shift partners (same J, same critic).  SimCLR: the queue keys are
    appended to the in-batch negatives (temperature unchanged).  While the queue holds fewer than K keys (first step) the recipe's own
    loss is used and stats['queue_fallback'] = 1.0."""
    method = cfg["run"]["method"]
    k = int(cfg["pairing"]["k"])
    stats = _empty_stats()
    stats["queue_fill"] = float(queue.filled)
    if not queue.ready(k):
        out = compute_objective_views(feats, cfg=cfg, critic=critic, pair_generator=pair_generator, queue=None)
        out["stats"]["queue_fill"] = float(queue.filled); out["stats"]["queue_fallback"] = 1.0
        return out
    stats["queue_fallback"] = 0.0
    if method == "simclr_matched":
        vs = feats["views_z"]
        pairs = [(a, b_) for a in range(len(vs)) for b_ in range(a + 1, len(vs))]
        qf = queue.features()
        loss = torch.stack([simclr_nt_xent_plus_queue(vs[a], vs[b_], qf, temperature=cfg["objective"]["simclr_temperature"]) for a, b_ in pairs]).mean()
        stats["nt_xent"] = float(loss.detach())
        B = vs[0].shape[0]
        return {"loss": loss, "stats": stats, "shift": None, "n_pos": 2 * B * len(pairs), "n_neg": 2 * B * (2 * B - 2 + queue.filled) * len(pairs)}
    if method != "vcs_qmi":
        raise ValueError(f"momentum-queue negatives are not defined for {method!r}")
    key = {"z_l2": "views_z", "p_raw": "views_p", "h_l2": "views_h"}[critic_input_key(cfg)]
    vs = feats[key]
    pairs = [(a, b_) for a in range(len(vs)) for b_ in range(a + 1, len(vs))]
    losses, acc = [], {kk: 0.0 for kk in VCS_STAT_KEYS}
    for a, b_ in pairs:
        s = vcs_pair_loss_momentum_queue(vs[a], vs[b_], critic, k=k, queue=queue, generator=pair_generator)
        losses.append(s["loss"])
        for kk in VCS_STAT_KEYS:
            acc[kk] += float(s[kk].detach()) / len(pairs)
    stats.update(acc)
    B = vs[0].shape[0]
    return {"loss": torch.stack(losses).mean(), "stats": stats, "shift": None, "n_pos": B * len(pairs), "n_neg": B * k * len(pairs)}


# ----------------------------------------------------------------------------------------------------------------------
# P104 (package v3, owner 2026-09-30): N line (noise-draw average of losses) and U line (all view tokens).  The frozen paths above are
# unchanged; these functions are reached only through the new optional config fields (critic.noise_repeats, pairing.pair_scope).
def pair_scope(cfg: dict[str, Any]) -> str:
    return cfg["pairing"].get("pair_scope", "cross_view_k")


def noise_repeats(cfg: dict[str, Any]) -> int:
    return int(cfg["model"]["critic"].get("noise_repeats", 1))


def noisy_pair_loss_repeats(z1: Tensor, z2: Tensor, critic, *, k: int, generator: torch.Generator | None, negative_detach: bool,
                            repeats: int, objective: str = "vcs") -> tuple[dict[str, Any], Tensor]:
    """P104 N1 / N2 (package v3 §7.1): ONE draw of the K cyclic shifts (same pair stream as the R = 1 path), then R independent noise
    draws on the same encoded features and the same pairs; each draw scores positives and negatives with fresh noise per pair
    occurrence and side (the critic's counter-based, checkpointed stream); the returned loss is the MEAN OF THE PER-DRAW LOSSES
    (never the loss of the mean score).  objective 'vcs': −J per draw; 'js': the balanced logistic loss on the noisy logit.  The
    negative partner is detached before the noise is added (noise does not depend on parameters).  Stats: per-draw means of the VCS
    statistics computed from T = tanh(f)."""
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have equal [B,D] shapes")
    if repeats < 1 or objective not in ("vcs", "js"):
        raise ValueError("repeats >= 1 and objective in {vcs, js}")
    indices, shifts = cyclic_negative_indices(len(z1), k, generator=generator, device=z1.device)
    left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1])
    partner = z2.detach() if negative_detach else z2
    right = partner[indices].reshape(-1, z2.shape[1])
    total = None
    acc: dict[str, float] = {}
    for _ in range(repeats):
        f_pos, f_neg = critic.logits(z1, z2), critic.logits(left, right)
        if objective == "vcs":
            st = vcs_from_scores(torch.tanh(f_pos), torch.tanh(f_neg))
            term = st["loss"]
        else:
            term = F.softplus(-2.0 * f_pos).mean() + F.softplus(2.0 * f_neg).mean()
            with torch.no_grad():
                st = vcs_from_scores(torch.tanh(f_pos), torch.tanh(f_neg))
        total = term if total is None else total + term
        for kk in VCS_STAT_KEYS:
            acc[kk] = acc.get(kk, 0.0) + float(st[kk].detach()) / repeats
    out: dict[str, Any] = {kk: torch.tensor(v) for kk, v in acc.items()}
    out["loss"] = total / repeats
    return out, shifts


def all_view_tokens_loss(views_z: list[Tensor], critic, *, negative_detach: bool, chunk_size: int = 256, objective: str = "vcs") -> dict[str, Any]:
    """P104 U line (package v3 §6): the original J over ALL view tokens of the batch.  Tokens are the flat view-major [V·B, D] matrix
    (flat index = view·B + image); P = ordered pairs of distinct tokens of the same base image, N_P = VB(V−1); Q = ordered pairs of
    different base images (same view index allowed), N_Q = V·B·V(B−1); self and same-image pairs never enter Q.  P and Q are averaged
    separately over their GLOBAL counts: row chunks accumulate SUMS, divided once at the end.  Negative right detach: Q scores use
    flat @ flat.detach().T (critic parameters keep their full gradient).  Deterministic scorer only (no noise hook; package v3 §6.4).
    objective "js" (P114, v5 NEXT-A-JS-AP3): the matched balanced-logistic loss on the SAME logit matrix f = a·C + b, the SAME tokens,
    masks, chunks and separate P / Q means:  L = mean_P softplus(−2f) + mean_Q softplus(2f)  (its f-gradient at f = 0 equals −J's per
    pair).  The VCS statistics are then computed from T = tanh(f) under no_grad, for comparison only."""
    if getattr(critic, "is_noisy", False):
        raise TypeError("the all-view-token path has no noise model (package v3 §6.4)")
    if not hasattr(critic, "score_matrix"):
        raise TypeError("all_view_tokens needs a similarity critic exposing score_matrix (cosine: learned or fixed)")
    if objective not in ("vcs", "js"):
        raise ValueError("objective must be 'vcs' or 'js'")
    if objective == "js" and not (hasattr(critic, "scale") and hasattr(critic, "bias")):
        raise TypeError("the all-view matched-JS loss needs the angular critic logit f = a·C + b (scale, bias)")
    V = len(views_z)
    if V < 2 or chunk_size < 1:
        raise ValueError("need >= 2 views and chunk_size >= 1")
    B = views_z[0].shape[0]
    flat = torch.cat(views_z, dim=0)  # view-major
    n = flat.shape[0]
    ids = torch.arange(n, device=flat.device) % B
    idx = torch.arange(n, device=flat.device)
    keys = flat.detach() if negative_detach else flat
    zero = flat.new_zeros(())
    s_p = s_q = s_p2 = s_q2 = sat_p = sat_q = zero
    l_p = l_q = zero  # P114: summed softplus terms of the matched-JS loss
    tp_first = tq_first = zero
    n_p = n_q = 0
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        same = ids[lo:hi, None] == ids[None, :]
        diag = idx[lo:hi, None] == idx[None, :]
        mask_p, mask_q = same & ~diag, ~same
        if objective == "js" and getattr(critic, "is_curved", False):  # P126: same non-linear logit matrix as score_matrix uses
            fp = critic.logits_from_similarity(flat[lo:hi] @ flat.T)[mask_p]
            fq = critic.logits_from_similarity(flat[lo:hi] @ keys.T)[mask_q]
            l_p = l_p + F.softplus(-2.0 * fp).sum(); l_q = l_q + F.softplus(2.0 * fq).sum()
            with torch.no_grad():
                tp, tq = torch.tanh(fp), torch.tanh(fq)
        elif objective == "js":
            fp = (critic.scale * (flat[lo:hi] @ flat.T) + critic.bias)[mask_p]
            fq = (critic.scale * (flat[lo:hi] @ keys.T) + critic.bias)[mask_q]
            l_p = l_p + F.softplus(-2.0 * fp).sum(); l_q = l_q + F.softplus(2.0 * fq).sum()
            with torch.no_grad():
                tp, tq = torch.tanh(fp), torch.tanh(fq)
        else:
            tp = critic.score_matrix(flat[lo:hi] @ flat.T)[mask_p]
            tq = critic.score_matrix(flat[lo:hi] @ keys.T)[mask_q]
        s_p = s_p + tp.sum(); s_q = s_q + tq.sum()
        s_p2 = s_p2 + tp.square().sum(); s_q2 = s_q2 + tq.square().sum()
        sat_p = sat_p + (tp.detach().abs() > 0.95).sum(); sat_q = sat_q + (tq.detach().abs() > 0.95).sum()
        n_p += int(mask_p.sum()); n_q += int(mask_q.sum())
    if n_p != V * B * (V - 1) or n_q != V * B * V * (B - 1):
        raise RuntimeError("all-view-token pair counting failed")
    mp, mq, sp, sq = s_p / n_p, s_q / n_q, s_p2 / n_p, s_q2 / n_q
    j = mp - mq - 0.5 * sp - 0.5 * sq
    risk = 0.5 * (1.0 - 2.0 * mp + sp) + 0.5 * (1.0 + 2.0 * mq + sq)
    loss = (l_p / n_p + l_q / n_q) if objective == "js" else -j
    return {"loss": loss, "js_loss": loss.detach() if objective == "js" else None, "J_raw": j, "R_binary": risk, "t_pos_mean": mp, "t_neg_mean": mq, "t_pos_second": sp, "t_neg_second": sq,
            "sat_pos_frac": sat_p.float() / n_p, "sat_neg_frac": sat_q.float() / n_q, "n_pos": n_p, "n_neg": n_q, "n_views": V, "n_base_images": B}


def compute_objective_views_p104(feats: dict[str, Any], *, cfg: dict[str, Any], critic, pair_generator: torch.Generator | None) -> dict[str, Any] | None:
    """Dispatcher for the P104 paths; returns None when the config uses none of them (the caller then runs the frozen path).
    Adds 'critic_pair_evals' (pair scorings actually computed, incl. noise repeats) to the returned dict."""
    scope = pair_scope(cfg)
    R = noise_repeats(cfg)
    js = cfg["objective"]["loss"] == "js_matched_logistic"
    noisy = getattr(critic, "is_noisy", False)
    key = {"z_l2": "views_z", "p_raw": "views_p", "h_l2": "views_h"}[critic_input_key(cfg)]
    vs = feats[key]
    nd = cfg["pairing"]["negative_detach"]
    stats = _empty_stats()
    if scope == "all_view_tokens":
        s = all_view_tokens_loss(vs, critic, negative_detach=nd, chunk_size=int(cfg["pairing"].get("all_view_chunk", 256)), objective="js" if js else "vcs")
        for kk in VCS_STAT_KEYS:
            stats[kk] = float(s[kk].detach())
        if js:
            stats["js_loss"] = float(s["js_loss"])
        return {"loss": s["loss"], "stats": stats, "shift": None, "n_pos": s["n_pos"], "n_neg": s["n_neg"], "critic_pair_evals": s["n_pos"] + s["n_neg"]}
    if not (R > 1 or (js and noisy)):
        return None
    k = cfg["pairing"]["k"]
    pairs = [(a, b_) for a in range(len(vs)) for b_ in range(a + 1, len(vs))]
    losses, acc, shifts = [], {kk: 0.0 for kk in VCS_STAT_KEYS}, []
    for a, b_ in pairs:
        s, sh = noisy_pair_loss_repeats(vs[a], vs[b_], critic, k=k, generator=pair_generator, negative_detach=nd, repeats=R, objective="js" if js else "vcs")
        losses.append(s["loss"]); shifts.append(int(sh[0]))
        for kk in VCS_STAT_KEYS:
            acc[kk] += float(s[kk]) / len(pairs)
    loss = torch.stack(losses).mean()
    stats.update(acc)
    stats["noise_repeats"] = float(R)
    if js:
        stats["js_loss"] = float(loss.detach())
    if hasattr(critic, "pop_stats"):
        stats.update(critic.pop_stats())
    B = vs[0].shape[0]
    return {"loss": loss, "stats": stats, "shift": shifts, "n_pos": B * len(pairs), "n_neg": B * k * len(pairs),
            "critic_pair_evals": R * (B + B * k) * len(pairs)}


# ----------------------------------------------------------------------------------------------------------------------
# P133 (owner 2026-10-04, "全部提交"; the follow-up proposed in the P100 report §3): MoCo-CONSISTENT momentum-key queue.  Reached only
# through the optional marker pairing.moco_consistent (not filled when absent); the frozen paths above and the P100 path are unchanged.
# Every pair is (online query, momentum key):  P = (q_i(x), k_j(x)), i != j (all ordered different-view pairs of one image);
# Q = (q_i(x), k) for every key k of ANOTHER image: the current batch's momentum keys of the other B - 1 images (all views) and the queue of
# momentum keys of previous steps, masked by image uid (an image re-drawn across an epoch boundary never enters its own Q).  P and Q
# therefore differ only in image identity (and, for queue keys, by <= queue_size / keys_per_step steps of key-encoder staleness).
class KeyUidQueue:
    """FIFO ring of detached momentum keys together with the uid of the image each key came from (P133).  Composes the existing
    NegativeQueue for the keys (same state layout) and a parallel int64 uid ring; state_dict carries both."""

    def __init__(self, size: int, dim: int, device: torch.device | str = "cpu") -> None:
        self.keys = NegativeQueue(size, dim, device=device)
        self.uid_buffer = torch.full((int(size),), -1, dtype=torch.long, device=device)

    @property
    def size(self) -> int:
        return self.keys.size

    @property
    def filled(self) -> int:
        return self.keys.filled

    def features(self) -> Tensor:
        return self.keys.features()

    def uids(self) -> Tensor:
        """uids aligned with features() (oldest first)."""
        k = self.keys
        if k.count < k.size:
            return self.uid_buffer[: k.count]
        return torch.cat((self.uid_buffer[k.ptr:], self.uid_buffer[: k.ptr]), dim=0)

    @torch.no_grad()
    def enqueue(self, feats: Tensor, uids: Tensor) -> None:
        u = uids.detach().to(self.uid_buffer.device, torch.long).reshape(-1)
        if len(u) != len(feats):
            raise ValueError("one uid per key")
        ptr0, size = self.keys.ptr, self.keys.size
        if len(u) >= size:
            self.uid_buffer.copy_(u[-size:])
        else:
            idx = (ptr0 + torch.arange(len(u), device=u.device)) % size
            self.uid_buffer[idx] = u
        self.keys.enqueue(feats)

    def state_dict(self) -> dict[str, Any]:
        return {"keys": self.keys.state_dict(), "uids": self.uid_buffer.detach().cpu().clone()}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        self.keys.load_state_dict(st["keys"])
        if st["uids"].shape != self.uid_buffer.shape:
            raise ValueError("uid ring shape mismatch")
        self.uid_buffer.copy_(st["uids"].to(self.uid_buffer.device))


def moco_consistent_scores(views_q: list[Tensor], views_k: list[Tensor], batch_uids: Tensor, queue: "KeyUidQueue | None"):
    """Cosine matrices of the P133 design.  views_q: V tensors [B, D] (online, L2-normalised, with gradient); views_k: V tensors [B, D]
    (momentum keys, L2-normalised, no gradient).  Returns (s_pos [V, V-1, B]: s_pos[i, r, b] = <q_i(b), k_j(b)> for the r-th j != i,
    C_in [VB, VB] = Q_flat @ K_flat.T with mask_in (True = valid negative: different image), C_q [VB, Qn] with mask_q (uid differs))."""
    V = len(views_q)
    if V < 2 or len(views_k) != V or any(q.shape != views_q[0].shape or k.shape != views_q[0].shape for q, k in zip(views_q, views_k)):
        raise ValueError("need V >= 2 query views and V key views of equal [B, D] shape")
    B = views_q[0].shape[0]
    if batch_uids.numel() != B:
        raise ValueError("one uid per base image")
    kflat = torch.cat(views_k, dim=0).detach()
    qflat = torch.cat(views_q, dim=0)
    s_pos = torch.stack([torch.stack([(views_q[i] * views_k[j].detach()).sum(-1) for j in range(V) if j != i]) for i in range(V)])
    ids = torch.arange(V * B, device=qflat.device) % B
    C_in = qflat @ kflat.T
    mask_in = ids[:, None] != ids[None, :]
    if queue is not None and queue.filled > 0:
        qk = queue.features().detach().to(qflat.dtype)
        qu = queue.uids().to(qflat.device)
        bu = batch_uids.to(qflat.device, torch.long).reshape(-1)[ids]
        C_q = qflat @ qk.T
        mask_q = bu[:, None] != qu[None, :]
    else:
        C_q, mask_q = qflat.new_zeros((V * B, 0)), torch.zeros((V * B, 0), dtype=torch.bool, device=qflat.device)
    return s_pos, C_in, mask_in, C_q, mask_q


def moco_consistent_vcs(views_q: list[Tensor], views_k: list[Tensor], batch_uids: Tensor, queue: "KeyUidQueue | None", critic) -> dict[str, Any]:
    """P133 VCS: the original J with the scorer T = critic.score_matrix(cos) (A-P3: fixed tanh(2s - 1)); P and Q averaged separately over
    their global counts (balanced, as in every VCS run).  Keys carry no gradient (MoCo), queries carry the full gradient."""
    if not hasattr(critic, "score_matrix"):
        raise TypeError("P133 needs a similarity critic exposing score_matrix")
    s_pos, C_in, mask_in, C_q, mask_q = moco_consistent_scores(views_q, views_k, batch_uids, queue)
    tp = critic.score_matrix(s_pos).reshape(-1)
    tq = torch.cat((critic.score_matrix(C_in)[mask_in], critic.score_matrix(C_q)[mask_q]))
    out = vcs_from_scores(tp, tq)
    out["n_pos"], out["n_neg"] = int(tp.numel()), int(tq.numel())
    out["n_neg_queue"] = int(mask_q.sum()); out["n_queue_uid_masked"] = int((~mask_q).sum())
    out["s_pos_mean"] = float(s_pos.detach().mean()); out["s_neg_mean"] = float(torch.cat((C_in.detach()[mask_in], C_q.detach()[mask_q])).mean())
    return out


def moco_consistent_infonce(views_q: list[Tensor], views_k: list[Tensor], batch_uids: Tensor, queue: "KeyUidQueue | None", temperature: float) -> dict[str, Any]:
    """P133 SimCLR / MoCo-v2 counterpart: for every (image b, query view i, key view j != i) the InfoNCE term
    -log softmax over [<q_i(b), k_j(b)>, <q_i(b), k> for the SAME negative keys as the VCS cell] / tau, target = the positive; mean over
    the V(V-1)B terms.  The negative log-sum-exp of a query row is shared across its V - 1 positives."""
    s_pos, C_in, mask_in, C_q, mask_q = moco_consistent_scores(views_q, views_k, batch_uids, queue)
    neg = torch.cat((C_in.masked_fill(~mask_in, float("-inf")), C_q.masked_fill(~mask_q, float("-inf"))), dim=1) / temperature
    neg_lse = torch.logsumexp(neg, dim=1)  # [VB], row = view-major query token (view i, image b)
    V, B = len(views_q), views_q[0].shape[0]
    lpos = s_pos / temperature  # [V, V-1, B]
    nl = neg_lse.reshape(V, 1, B).expand_as(lpos)
    terms = torch.logaddexp(lpos, nl) - lpos
    loss = terms.mean()
    return {"loss": loss, "nt_xent": loss.detach(), "n_pos": int(lpos.numel()), "n_neg": int(mask_in.sum() + mask_q.sum()),
            "n_neg_queue": int(mask_q.sum()), "n_queue_uid_masked": int((~mask_q).sum()),
            "s_pos_mean": float(s_pos.detach().mean()), "s_neg_mean": float(torch.cat((C_in.detach()[mask_in], C_q.detach()[mask_q])).mean())}


def compute_objective_moco_consistent(feats: dict[str, Any], keys_views: list[Tensor], batch_uids: Tensor, *, cfg: dict[str, Any], critic,
                                      queue: "KeyUidQueue | None") -> dict[str, Any]:
    """P133 dispatcher (views.count = 4, shared branch, joint step).  VCS reads the critic-input views (z); SimCLR reads z."""
    method = cfg["run"]["method"]
    stats = _empty_stats()
    stats["queue_fill"] = float(0 if queue is None else queue.filled)
    if method == "vcs_qmi":
        vs = feats[{"z_l2": "views_z", "p_raw": "views_p", "h_l2": "views_h"}[critic_input_key(cfg)]]
        s = moco_consistent_vcs(vs, keys_views, batch_uids, queue, critic)
        for kk in VCS_STAT_KEYS:
            stats[kk] = float(s[kk].detach())
    elif method == "simclr_matched":
        s = moco_consistent_infonce(feats["views_z"], keys_views, batch_uids, queue, float(cfg["objective"]["simclr_temperature"]))
        stats["nt_xent"] = float(s["nt_xent"])
    else:
        raise ValueError(f"P133 is defined for vcs_qmi and simclr_matched only, got {method!r}")
    for kk in ("n_neg_queue", "n_queue_uid_masked", "s_pos_mean", "s_neg_mean"):
        stats[f"p133_{kk}"] = float(s[kk])
    return {"loss": s["loss"], "stats": stats, "shift": None, "n_pos": s["n_pos"], "n_neg": s["n_neg"]}
