"""Fixed mixture-reference VCS objective and explicit comparison losses.

S is NOT Shannon MI.  All VCS methods maximize the SAME quadratic J.
The MF implementation changes evaluation of empirical Q, not its target.
Main U mode excludes the matched index. No semantic/class exclusions occur.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mf_backend import (
    SceneBatch, SimplexBilinearCritic, QuadraticHead, solve_box_head,
    objective_terms, dense_objective_terms,
)


METHODS = ("vcs_exact", "vcs_dense", "vcs_sampled", "vcs_m1", "bce", "bce_logits", "infonce")


def _paired(x: Tensor, y: Tensor, q_mode: str) -> None:
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("Expected paired nonempty [batch,dimension] tensors")
    if x.dtype != y.dtype or x.device != y.device:
        raise ValueError("Both inputs must use the same dtype and device")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("Dependence branch requires float32 or float64; do not run it under AMP")
    if q_mode not in ("u", "v"):
        raise ValueError("q_mode must explicitly be u or v")
    if q_mode == "u" and len(x) < 2:
        raise ValueError("U-mode needs at least two base examples; merge or drop a singleton training tail")
    if torch.is_autocast_enabled(x.device.type):
        raise ValueError("Disable autocast around the dependence branch")


def q_values(scores: Tensor, q_mode: str) -> Tensor:
    """Flatten the EXACT finite empirical Q support, retaining its denominator."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Expected square pair score matrix")
    if q_mode == "v":
        return scores.reshape(-1)
    if q_mode == "u" and len(scores) > 1:
        return scores[~torch.eye(len(scores), dtype=torch.bool, device=scores.device)]
    raise ValueError("U needs n>1; unsupported Q mode")


def quadratic_from_scores(scores: Tensor, q_mode: str) -> tuple[Tensor, dict[str, Tensor]]:
    pos = scores.diagonal()
    neg = q_values(scores, q_mode)
    p1, p2, q1, q2 = pos.mean(), pos.square().mean(), neg.mean(), neg.square().mean()
    return p1 - q1 - 0.5 * (p2 + q2), dict(p1=p1, p2=p2, q1=q1, q2=q2)


def sampled_indices(n: int, draws: int, q_mode: str, *, device: torch.device,
                    generator: torch.Generator | None = None) -> Tensor:
    """Pair-marginal unbiased sampler for the specified empirical Q.

    U uses random nonzero cyclic shifts. This is NOT a uniform draw from all
    derangements; each off-diagonal (i,j) has exactly probability 1/(n-1).
    V uses ordinary uniform permutations including fixed points.
    Dependence across rows is retained; draws are NOT extra independent images.
    """
    if draws < 1 or n < 1:
        raise ValueError("draws and n must be positive")
    if q_mode == "u":
        if n < 2:
            raise ValueError("U needs n>1")
        shifts = torch.randint(1, n, (draws, 1), device=device, generator=generator)
        return (torch.arange(n, device=device)[None] + shifts) % n
    if q_mode == "v":
        return torch.stack([torch.randperm(n, device=device, generator=generator) for _ in range(draws)])
    raise ValueError("Unsupported Q")


def sampled_quadratic(a: Tensor, b: Tensor, w: Tensor, indices: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    aw = a @ w
    pos = (aw * b).sum(-1)
    neg = (aw[None] * b[indices]).sum(-1)
    p1, p2, q1, q2 = pos.mean(), pos.square().mean(), neg.mean(), neg.square().mean()
    return p1 - q1 - 0.5 * (p2 + q2), dict(p1=p1, p2=p2, q1=q1, q2=q2)


class DependenceRegularizer(nn.Module):
    """Return a loss to ADD to supervised/task training.

    M0: Adam/SGD updates simplex maps and boxed W, then caller calls project_().
    M1: prepare_step caches one set of graph-bearing simplex features, solves W
    on their detached values, then forward uses those same tensors. No implicit
    differentiation or recomputed augmentation is used. The solver is approximate
    if its cap is reached, and its gap only certifies inner_pre_outer_batch.

    bce uses exactly the same boxed h=a^T W b with D=(1+h)/2, balanced P/Q.
    Its returned loss is 2*(balanced_BCE-log(2)); near h=0 this equals -J
    through second order. The scale is analytic, fixed and never CAL-adapted.
    bce_logits is a separate stronger baseline with unbounded bilinear logits.
    infonce uses normalized linear projections and symmetric batch InfoNCE;
    it is a reference objective, not called original CRD or MIMKD.
    """

    def __init__(self, x_dim: int, y_dim: int, method: str = "vcs_exact", rank: int = 32,
                 q_mode: str = "u", sampled_draws: int = 8, temperature: float = 1.0,
                 solver_max_iter: int = 30, solver_relative_tol: float = 0.01,
                 nce_temperature: float = 0.1, dense_chunk_size: int | None = None,
                 dense_checkpoint: bool = False) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; supported: {METHODS}")
        if q_mode not in ("u", "v") or sampled_draws < 1:
            raise ValueError("Invalid Q mode or sample count")
        if not 0 < solver_relative_tol < 1 or solver_max_iter < 1:
            raise ValueError("M1 relative tolerance must be in (0,1); cap must be positive")
        if not math.isfinite(nce_temperature) or nce_temperature <= 0:
            raise ValueError("InfoNCE temperature must be positive")
        self.method, self.q_mode, self.sampled_draws = method, q_mode, int(sampled_draws)
        self.solver_max_iter, self.solver_relative_tol = int(solver_max_iter), float(solver_relative_tol)
        self.nce_temperature = float(nce_temperature)
        self.dense_chunk_size, self.dense_checkpoint = dense_chunk_size, bool(dense_checkpoint)
        self.head = SimplexBilinearCritic(x_dim, y_dim, rank, rank, temperature=temperature)
        if method in ("vcs_m1", "infonce"):
            self.head.W.requires_grad_(False)
        self.diagnostics: dict[str, Any] = {}
        self._prepared: tuple | None = None

    @property
    def W(self) -> nn.Parameter:
        return self.head.W

    def optimizer_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def project_(self) -> None:
        if self.method not in ("bce_logits", "infonce"):
            self.head.project_head_()

    def prepare_step(self, x: Tensor, y: Tensor) -> None:
        self._prepared = None
        if self.method != "vcs_m1" or not self.training:
            return
        _paired(x, y, self.q_mode)
        # Build exactly once WITH outer autograd; the quadratic snapshot detaches.
        a, b = self.head.features(x, y)
        scene = SceneBatch(a, b, q_mode=self.q_mode)
        quadratic = QuadraticHead([scene], weights=[1.0], ridge=0.0)
        zero = torch.zeros_like(self.W)
        gap0 = float(quadratic.fw_gap(zero))
        warm_gap = float(quadratic.fw_gap(self.W.detach()))
        if gap0 == 0.0:
            # Delta==0 makes W=0 a true optimum of this PSD quadratic.
            with torch.no_grad():
                self.W.zero_()
            sd = dict(solver_converged=True, solver_iterations=0, solver_gap=0.0,
                      solver_tolerance=0.0, solver_status="zero_reference_optimum")
        else:
            tol = self.solver_relative_tol * gap0
            result = solve_box_head(quadratic, initial=self.W.detach(), tolerance=tol,
                                    max_iterations=self.solver_max_iter, history_every=1)
            with torch.no_grad():
                self.W.copy_(result.w)
            sd = dict(solver_converged=result.converged, solver_iterations=result.iterations,
                      solver_gap=result.fw_gap, solver_tolerance=tol, solver_status=result.status)
        sd.update(solver_zero_gap=gap0, solver_warm_start_gap=warm_gap,
                  solver_relative_tol=self.solver_relative_tol,
                  solver_gap_scope="inner_pre_outer_batch", solver_cap=self.solver_max_iter)
        self.diagnostics = sd
        self._prepared = (x, y, x._version, y._version, a, b, self.W._version)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        _paired(x, y, self.q_mode)
        if x.shape[1] != self.head.left.in_features or y.shape[1] != self.head.right.in_features:
            raise ValueError("Feature dimensions differ from initialized critic")
        if self.method == "infonce":
            a = F.normalize(self.head.left(x), dim=-1)
            b = F.normalize(self.head.right(y), dim=-1)
            cosine = a @ b.T
            logits = cosine / self.nce_temperature
            target = torch.arange(len(x), device=x.device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            self.diagnostics = dict(method=self.method, q_mode=self.q_mode, batch=len(x),
                                    loss=float(loss.detach()), nce_temperature=self.nce_temperature)
            return loss
        if self.method == "vcs_m1" and self.training:
            if self._prepared is None:
                self.prepare_step(x, y)
            assert self._prepared is not None
            px, py, vx, vy, a, b, vw = self._prepared
            if px is not x or py is not y or vx != x._version or vy != y._version or vw != self.W._version:
                raise RuntimeError("M1 requires exactly the prepared tensors and unmodified solved W")
            self._prepared = None
        else:
            a, b = self.head.features(x, y)
            self.diagnostics = {}
        if self.method in ("vcs_exact", "vcs_m1", "vcs_dense"):
            scene = SceneBatch(a, b, q_mode=self.q_mode)
            if self.method == "vcs_dense":
                terms = dense_objective_terms([scene], self.W, weights=[1.0],
                         chunk_size=self.dense_chunk_size, checkpoint_blocks=self.dense_checkpoint)
            else:
                terms = objective_terms([scene], self.W, weights=[1.0])
            j = terms.j
            parts = {k: getattr(terms, k) for k in ("p1", "p2", "q1", "q2")}
            loss = -j
        elif self.method == "vcs_sampled":
            idx = sampled_indices(len(x), self.sampled_draws, self.q_mode, device=x.device)
            j, parts = sampled_quadratic(a, b, self.W, idx)
            loss = -j
        elif self.method in ("bce", "bce_logits"):
            scores = a @ self.W @ b.T
            pos, neg = scores.diagonal(), q_values(scores, self.q_mode)
            if self.method == "bce":
                eps = torch.finfo(scores.dtype).eps
                # Algebraically 2*(balanced_BCE-log2). log1p avoids
                # subtracting two ~log2 scalars when the head is near zero.
                loss = -(torch.log1p(pos.clamp_min(-1+2*eps)).mean()
                         + torch.log1p((-neg).clamp_min(-1+2*eps)).mean())
                raw_bce = math.log(2.0) + 0.5*loss
                self.diagnostics["probability_log_floor"] = eps
                self.diagnostics.update(raw_balanced_bce=float(raw_bce.detach()),
                                        objective_scale=2.0, pre_scale_constant_offset=-math.log(2.0))
            else:
                loss = 0.5 * (F.softplus(-pos).mean() + F.softplus(neg).mean())
            # Raw J here is only a same-score diagnostic for bounded BCE.
            if self.method == "bce":
                j, parts = quadratic_from_scores(scores, self.q_mode)
            else:
                j, parts = None, {}
        else:
            raise AssertionError("Exhaustive method branch violated")
        self.diagnostics.update(method=self.method, q_mode=self.q_mode, batch=len(x),
            loss=float(loss.detach()), head_abs_max=float(self.W.detach().abs().max()),
            simplex_left_max=float(a.detach().max()), simplex_right_max=float(b.detach().max()))
        self.diagnostics.update({k: float(v.detach()) for k, v in parts.items()})
        if j is not None:
            self.diagnostics["raw_j"] = float(j.detach())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite dependence loss; preserve run as FAILED")
        return loss


def parameter_groups(model: nn.Module, *, weight_decay: float) -> list[dict]:
    """Keep W free of optimizer weight decay; all other parameters share recipe.

    Includes parameters of wrapped regularizers in larger task modules. The
    task code may use equivalent explicit groups, but must not regularize W
    silently while claiming the fixed unregularized J.
    """
    boxed = {id(m.W) for m in model.modules() if isinstance(m, DependenceRegularizer)}
    ordinary, heads = [], []
    seen: set[int] = set()
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            (heads if id(p) in boxed else ordinary).append(p)
            seen.add(id(p))
    result = []
    if ordinary:
        result.append(dict(params=ordinary, weight_decay=float(weight_decay)))
    if heads:
        result.append(dict(params=heads, weight_decay=0.0))
    return result
