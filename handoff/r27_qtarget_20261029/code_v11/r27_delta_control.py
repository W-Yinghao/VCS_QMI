"""R27 supplementary A-delta control; reference implementation, not a production patch.

The original VCS objective and production runs remain unchanged. This control
isolates background-mean subtraction from the full quadratic pseudo-target.
Inputs and diagnostics match r27_targets.generate sufficiently for acc_info.
Use in a NEW source snapshot only. See SERVER_ADDENDUM_CN.md for integration.
"""
from __future__ import annotations
from typing import Any
import torch

ZMIN = 1e-6  # Same fallback threshold as the frozen R27 generator.

@torch.no_grad()
def generate_centered_mean(
    A: torch.Tensor, Y: torch.Tensor, *, zmin: float = ZMIN
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return normalize(mu_i - mu_Q), with the existing mean fallback policy.

    A: [B,M], fixed row-stochastic weights; Y: [M,d], fixed unit atoms.
    This maximizes ONLY the linear component over the unit ball. It is NOT
    another solution of the full VCS objective. Jv_* are diagnostic readings,
    not optimality certificates for this control.
    """
    if A.ndim != 2 or Y.ndim != 2 or A.shape[1] != Y.shape[0]:
        raise ValueError('Expected A[B,M], Y[M,d].')
    if min(A.shape) < 1 or Y.shape[1] < 1:
        raise ValueError('Empty graphs are not valid generator inputs.')
    if A.device != Y.device or not (A.is_floating_point() and Y.is_floating_point()):
        raise ValueError('Inputs must be floating-point tensors on the same device.')
    if A.requires_grad or Y.requires_grad:
        raise ValueError('Graph and candidate support must be stop-gradient.')
    if zmin <= 0:
        raise ValueError('zmin must be positive.')
    a, y = A.double(), Y.double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(y).all()):
        raise ValueError('Inputs must be finite.')
    if bool((a < 0).any()) or not torch.allclose(a.sum(1), torch.ones(a.shape[0], device=a.device, dtype=a.dtype), atol=1e-5, rtol=0):
        raise ValueError('A must be nonnegative with unit row sums.')
    if not torch.allclose(y.norm(dim=1), torch.ones(y.shape[0], device=y.device, dtype=y.dtype), atol=1e-5, rtol=0):
        raise ValueError('Y must contain unit atoms.')
    c = a.mean(0)
    mu = a @ y
    delta = mu - c @ y
    mn, dn = mu.norm(dim=1), delta.norm(dim=1)
    mean_t = mu / mn.clamp(min=1e-300).unsqueeze(1)
    delta_t = delta / dn.clamp(min=1e-300).unsqueeze(1)
    mean_ok = torch.isfinite(mean_t).all(1) & (mn > zmin)
    delta_ok = torch.isfinite(delta_t).all(1) & (dn > zmin)
    fallback = ~delta_ok
    target = torch.where(delta_ok.unsqueeze(1), delta_t, mean_t)
    valid = (delta_ok | mean_ok) & torch.isfinite(target).all(1)
    target = torch.where(valid.unsqueeze(1), target, torch.zeros_like(target))

    def j_readout(z: torch.Tensor) -> torch.Tensor:
        score = z @ y.T
        return ((a-c)*score - 0.5*(a+c)*score.square()).sum(1)

    j_target = j_readout(target)
    info = dict(mu_norm=mn, delta_norm=dn, fallback=fallback,
                Jv_raw=j_target, Jv_norm=j_target, Jv_mean=j_readout(mean_t),
                cos_to_mean=(target*mean_t).sum(1),
                control='linear_background_subtracted_target_only',
                linear_objective=(delta*target).sum(1))
    return target.to(Y.dtype), valid, info
