"""Reference loss and gradient routing for the collaborator's VCS-QMI objective.

Primary source: Variational_CS_QMI_Research_Plan (1).pdf, Sections 4, 5, 7, 9.
The critic architecture and optimizer choices are engineering defaults, not theorems.
No latent smoothing, two-point gradients, EMA encoder, or auxiliary VCS loss.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _scores(x: Tensor, name: str) -> Tensor:
    if not isinstance(x, Tensor) or not x.is_floating_point():
        raise TypeError(f"{name} must be a floating-point Tensor")
    if x.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    # Keep float64 for mathematical tests; reductions are >= float32 under AMP.
    return x.reshape(-1).float() if x.dtype in (torch.float16, torch.bfloat16) else x.reshape(-1)


def vcs_from_scores(t_pos: Tensor, t_neg: Tensor) -> dict[str, Tensor]:
    """Each distribution is averaged separately, even if sample counts differ.

    Minimize loss = -J. R_binary = 1 - J, not 0.5 * (1 - J).
    J_raw is an empirical critic objective, not an oracle S or Shannon MI.
    """
    p, q = _scores(t_pos, "t_pos"), _scores(t_neg, "t_neg")
    if p.device != q.device:
        raise ValueError("positive and negative scores must be on the same device")
    mp, mq, sp, sq = p.mean(), q.mean(), p.square().mean(), q.square().mean()
    j = mp - mq - 0.5 * sp - 0.5 * sq
    risk = 0.5 * (1.0 - p).square().mean() + 0.5 * (-1.0 - q).square().mean()
    return {
        "loss": -j,
        "J_raw": j,
        "R_binary": risk,
        "t_pos_mean": mp,
        "t_neg_mean": mq,
        "t_pos_second": sp,
        "t_neg_second": sq,
        "sat_pos_frac": (p.abs() > 0.95).float().mean(),
        "sat_neg_frac": (q.abs() > 0.95).float().mean(),
    }


def rpc_score_reference(
    t_pos: Tensor, t_neg: Tensor,
    alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
) -> Tensor:
    """Independent algebraic reference, NOT a full reproduction of tuned RPC."""
    if not all(math.isfinite(v) for v in (alpha, beta, gamma)):
        raise ValueError("relative parameters must be finite")
    p, q = _scores(t_pos, "t_pos"), _scores(t_neg, "t_neg")
    return p.mean() - alpha * q.mean() - 0.5 * beta * p.square().mean() - 0.5 * gamma * q.square().mean()


def cyclic_negative_indices(
    batch_size: int, k: int = 1, *, generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Return [K,B] partner indices with K distinct, nonzero cyclic shifts.

    Pass a dedicated CPU generator. Rows and examples are not independent
    observations. This approximates the product-marginal construction from
    off-diagonal pairs; it is not B*K independent base samples.
    """
    if batch_size < 2:
        raise ValueError("VCS negative pairing requires batch_size >= 2")
    if not 1 <= k <= batch_size - 1:
        raise ValueError("require 1 <= k <= batch_size - 1")
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("use a dedicated CPU generator for pairing")
    shifts_cpu = torch.randperm(batch_size - 1, generator=generator, device="cpu")[:k] + 1
    shifts = shifts_cpu.to(device)
    base = torch.arange(batch_size, device=device)
    return (base[None, :] + shifts[:, None]) % batch_size, shifts


class PairCritic(nn.Module):
    """Pointwise ordered-pair MLP; no batch normalization or dropout."""
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("dimensions must be positive")
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Nonzero small initialization: exactly zero would block encoder gradients
        # on the first step. This is a disclosed engineering choice.
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape:
            raise ValueError("critic inputs must have matching [N,D] shapes")
        if left.shape[1] != self.feature_dim:
            raise ValueError("critic input dimension does not match configuration")
        logits = self.net(torch.cat((left, right), dim=-1)).squeeze(-1)
        return torch.tanh(logits)


def vcs_pair_loss(
    z1: Tensor, z2: Tensor, critic: PairCritic, *, k: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    """No detach on either positive or negative branches in end-to-end mode."""
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have equal [B,D] shapes")
    indices, shifts = cyclic_negative_indices(len(z1), k, generator=generator, device=z1.device)
    t_pos = critic(z1, z2)
    left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1])
    right = z2[indices].reshape(-1, z2.shape[1])
    t_neg = critic(left, right)
    return vcs_from_scores(t_pos, t_neg), shifts


def simclr_nt_xent(p1: Tensor, p2: Tensor, temperature: float = 0.2) -> Tensor:
    """2B anchors; mask self only, retain each positive in the denominator."""
    if p1.ndim != 2 or p1.shape != p2.shape or len(p1) < 2:
        raise ValueError("require matching [B,D] with B >= 2")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    x = torch.cat((p1, p2), dim=0)
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    x = F.normalize(x, dim=-1, eps=1e-8)
    logits = x @ x.T / temperature
    n, b = len(x), len(p1)
    diag = torch.eye(n, dtype=torch.bool, device=x.device)
    logits = logits.masked_fill(diag, -torch.inf)
    targets = (torch.arange(n, device=x.device) + b) % n
    return F.cross_entropy(logits, targets)


def vicreg_loss(
    p1: Tensor, p2: Tensor, *, inv_weight: float = 25.0,
    var_weight: float = 25.0, cov_weight: float = 1.0, eps: float = 1e-4,
) -> dict[str, Tensor]:
    """Matched-projector VICReg control: input RAW projector outputs, not L2 z.

    Unbiased covariance denominator B-1. Variance penalty averaged over views;
    off-diagonal covariance squared sums are divided by feature dimension.
    This is the loss, not a claim to reproduce the full native VICReg recipe.
    """
    if p1.ndim != 2 or p1.shape != p2.shape or len(p1) < 2:
        raise ValueError("require matching [B,D] with B >= 2")
    x = p1.float() if p1.dtype in (torch.float16, torch.bfloat16) else p1
    y = p2.float() if p2.dtype in (torch.float16, torch.bfloat16) else p2
    inv = F.mse_loss(x, y)
    sx = torch.sqrt(x.var(dim=0, unbiased=True) + eps)
    sy = torch.sqrt(y.var(dim=0, unbiased=True) + eps)
    var = 0.5 * (F.relu(1.0 - sx).mean() + F.relu(1.0 - sy).mean())
    xc, yc = x - x.mean(0), y - y.mean(0)
    cx, cy = xc.T @ xc / (len(x) - 1), yc.T @ yc / (len(y) - 1)
    mask = ~torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
    cov = (cx[mask].square().sum() + cy[mask].square().sum()) / x.shape[1]
    return {"loss": inv_weight * inv + var_weight * var + cov_weight * cov,
            "invariance": inv, "variance": var, "covariance": cov}


def joint_step_fp32(
    encoder: nn.Module, projector: nn.Module, critic: PairCritic,
    optimizer: torch.optim.Optimizer, x1: Tensor, x2: Tensor,
    *, generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Reference joint update. Module parameters must already be in optimizer.

    Uses one concatenated 2B forward so backbone/projector BN share the same
    batch policy. Stale graphs and detached negative branches are not used.
    Caller handles labels, logging, scheduler, checkpoints, and data splits.
    """
    encoder.train(); projector.train(); critic.train()
    optimizer.zero_grad(set_to_none=True)
    if x1.shape != x2.shape or len(x1) < 2:
        raise ValueError("require equal two-view minibatches with B >= 2")
    h = encoder(torch.cat((x1, x2), dim=0))
    p = projector(h)
    z = F.normalize(p, dim=-1, eps=1e-8)
    z1, z2 = z.chunk(2, dim=0)
    stats, _ = vcs_pair_loss(z1, z2, critic, generator=generator)
    if not torch.isfinite(stats["loss"]):
        raise FloatingPointError("nonfinite VCS loss; do not silently skip batch")
    stats["loss"].backward()
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError("nonfinite gradient; preserve failure context")
    optimizer.step()
    return {name: float(value.detach().cpu()) for name, value in stats.items()}
