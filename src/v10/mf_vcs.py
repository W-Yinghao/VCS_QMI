"""Moment-factorized VCS, V10 execution candidate.

The statistical objective is unchanged:
    J = E_P[T] - E_Q[T] - (E_P[T**2] + E_Q[T**2]) / 2.
Only float32/float64, complete V/U products, independent marginal pools,
and fixed, nonnegative scene weights are implemented. Forbidden edges are
deliberately unsupported. See README_math_CN.md for the validation status.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Literal, Sequence

import torch
from torch import Tensor, nn


QMode = Literal["v", "u", "independent"]


def _features(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor) or x.ndim != 2 or min(x.shape) < 1:
        raise ValueError(f"{name} must be a nonempty [n, k] tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name}: only float32/float64 are supported; disable AMP here")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains nonfinite values")
    if bool((x < 0).any()):
        raise ValueError(f"{name} must be nonnegative simplex features")
    atol = 32 * torch.finfo(x.dtype).eps
    if not bool(torch.allclose(x.sum(-1), torch.ones_like(x[:, 0]), rtol=0, atol=atol)):
        raise ValueError(f"{name} rows must sum to one; no implicit normalization is performed")


def _same_type_device(xs: Sequence[Tensor]) -> None:
    if any(x.dtype != xs[0].dtype or x.device != xs[0].device for x in xs[1:]):
        raise ValueError("All feature tensors and W must share dtype and device")


def _matrix(w: Tensor, a: Tensor, b: Tensor, *, box: bool) -> None:
    if not isinstance(w, Tensor) or w.shape != (a.shape[1], b.shape[1]):
        raise ValueError("Expected W (or an HVP direction) with shape [r, s]")
    _same_type_device((a, b, w))
    if not bool(torch.isfinite(w).all()):
        raise ValueError("The head matrix contains nonfinite values")
    if box and bool((w.abs() > 1).any()):
        raise ValueError("W must lie in [-1, 1]; project after each M0 optimizer step")


def _ridge(value: float) -> float:
    if isinstance(value, (Tensor, bool)) or not isinstance(value, numbers.Real):
        raise ValueError("ridge must be a fixed, finite nonnegative Python number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("ridge must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class SceneBatch:
    """One local positive batch and its local Q; never pool different scenes.

    V and U use the positive rows as the two empirical marginal pools.
    Independent mode requires both separate pools and does not delete an
    index diagonal. Statistical independence is the caller's responsibility.
    Input validation is performed once on construction; do not mutate tensors.
    """

    a: Tensor
    b: Tensor
    q_mode: QMode = "v"
    a_marginal: Tensor | None = None
    b_marginal: Tensor | None = None
    scene_id: str = ""
    forbidden_edges: object | None = None

    def __post_init__(self) -> None:
        if self.forbidden_edges is not None:
            raise NotImplementedError(
                "Forbidden-edge/row-normalized Q is not implemented. "
                "Do not substitute V/U or silently pool eligible edges."
            )
        _features(self.a, "a")
        _features(self.b, "b")
        _same_type_device((self.a, self.b))
        if self.a.shape[0] != self.b.shape[0]:
            raise ValueError("Positive rows must be paired with the same length")
        if self.q_mode not in ("v", "u", "independent"):
            raise ValueError("q_mode must be 'v', 'u', or 'independent'")
        if self.q_mode == "u" and self.a.shape[0] < 2:
            raise ValueError("U mode needs at least two positive base pairs per scene")
        if self.q_mode == "independent":
            if self.a_marginal is None or self.b_marginal is None:
                raise ValueError("Independent Q requires both marginal pools")
            _features(self.a_marginal, "a_marginal")
            _features(self.b_marginal, "b_marginal")
            _same_type_device((self.a, self.b, self.a_marginal, self.b_marginal))
            if (self.a_marginal.shape[1], self.b_marginal.shape[1]) != (
                self.a.shape[1], self.b.shape[1]
            ):
                raise ValueError("Independent pools must use the same feature dimensions")
        elif self.a_marginal is not None or self.b_marginal is not None:
            raise ValueError("Separate marginal pools are only valid in independent mode")

    def q_pools(self) -> tuple[Tensor, Tensor]:
        if self.q_mode == "independent":
            assert self.a_marginal is not None and self.b_marginal is not None
            return self.a_marginal, self.b_marginal
        return self.a, self.b


def partition_scenes(a: Tensor, b: Tensor, scene_ids: Tensor, *, q_mode: QMode = "v") -> tuple[SceneBatch, ...]:
    """Build local V/U batches in sorted scene-id order, preserving autograd.

    The caller supplies weights in this returned order. For independent pools,
    construct SceneBatch objects explicitly with their own local marginal pools.
    """
    if q_mode not in ("v", "u"):
        raise ValueError("partition_scenes only builds paired V/U scenes")
    if not isinstance(scene_ids, Tensor) or scene_ids.ndim != 1:
        raise ValueError("scene_ids must be a one-dimensional integer tensor")
    if scene_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("scene_ids must have an integer dtype")
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or len(scene_ids) != len(a):
        raise ValueError("a, b and scene_ids must have matching positive-row counts")
    _same_type_device((a, b))
    result = []
    for sid in torch.unique(scene_ids, sorted=True):
        idx = torch.nonzero(scene_ids == sid, as_tuple=False).flatten().to(a.device)
        result.append(SceneBatch(a[idx], b[idx], q_mode=q_mode, scene_id=str(int(sid.item()))))
    if not result:
        raise ValueError("At least one scene is required")
    return tuple(result)


def _inputs(scenes: Sequence[SceneBatch], weights: Sequence[float]) -> tuple[tuple[SceneBatch, ...], tuple[float, ...]]:
    scenes = tuple(scenes)
    if not scenes or any(not isinstance(s, SceneBatch) for s in scenes):
        raise ValueError("Provide a nonempty sequence of SceneBatch objects")
    if isinstance(weights, Tensor):
        raise ValueError("Scene weights must be fixed Python numbers, not tensors")
    weights = tuple(weights)
    if len(weights) != len(scenes):
        raise ValueError("One explicitly specified weight is required for each scene")
    if any(isinstance(v, (Tensor, bool)) or not isinstance(v, numbers.Real) for v in weights):
        raise ValueError("Scene weights must be fixed real numbers")
    weights = tuple(float(v) for v in weights)
    if any(not math.isfinite(v) or v < 0 for v in weights):
        raise ValueError("Scene weights must be finite and nonnegative")
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Scene weights must sum to one; they are never silently renormalized")
    first = scenes[0]
    for s in scenes[1:]:
        if (s.a.shape[1], s.b.shape[1]) != (first.a.shape[1], first.b.shape[1]):
            raise ValueError("All scenes must share relation-head dimensions")
        _same_type_device((first.a, s.a, s.b))
    return scenes, weights


def _diag(a: Tensor, b: Tensor, w: Tensor) -> Tensor:
    return ((a @ w) * b).sum(-1)


def _product_moments(a: Tensor, b: Tensor, w: Tensor, off: bool) -> tuple[Tensor, Tensor]:
    first = a.sum(0) @ w @ b.sum(0)
    second = (w * ((a.T @ a) @ w @ (b.T @ b))).sum()
    denominator = a.shape[0] * b.shape[0]
    if off:
        diag = _diag(a, b, w)
        first = first - diag.sum()
        second = second - diag.square().sum()
        denominator -= a.shape[0]
    return first / denominator, second / denominator


def product_moments(a: Tensor, b: Tensor, w: Tensor, *, q_mode: Literal["v", "u"] = "v") -> tuple[Tensor, Tensor]:
    """Complete product moments. Rectangular pools are permitted in V mode."""
    _features(a, "a")
    _features(b, "b")
    _matrix(w, a, b, box=True)
    if q_mode not in ("v", "u"):
        raise ValueError("Use q_mode='v' or 'u'")
    if q_mode == "u" and (len(a) != len(b) or len(a) < 2):
        raise ValueError("U mode requires paired rows n=m>=2")
    return _product_moments(a, b, w, q_mode == "u")


@dataclass(frozen=True)
class ObjectiveTerms:
    p1: Tensor
    p2: Tensor
    q1: Tensor
    q2: Tensor
    j: Tensor
    ridge_penalty: Tensor
    f: Tensor


def _finish(p1: Tensor, p2: Tensor, q1: Tensor, q2: Tensor, w: Tensor, ridge: float) -> ObjectiveTerms:
    j = p1 - q1 - 0.5 * (p2 + q2)
    penalty = 0.5 * ridge * w.square().sum()
    return ObjectiveTerms(p1, p2, q1, q2, j, penalty, j - penalty)


def objective_terms(scenes: Sequence[SceneBatch], w: Tensor, *, weights: Sequence[float], ridge: float = 0.0, check_w: bool = True) -> ObjectiveTerms:
    """Exact moment J/F on local scenes; no cross-scene pair is constructed.

    check_w=False is only for already-validated benchmark/training inputs.
    It does not project or repair an invalid head.
    """
    scenes, weights = _inputs(scenes, weights)
    ridge = _ridge(ridge)
    if check_w:
        _matrix(w, scenes[0].a, scenes[0].b, box=True)
    p1 = p2 = q1 = q2 = w.new_zeros(())
    for s, weight in zip(scenes, weights):
        t = _diag(s.a, s.b, w)
        x, y = s.q_pools()
        local_q1, local_q2 = _product_moments(x, y, w, s.q_mode == "u")
        p1 = p1 + weight * t.mean()
        p2 = p2 + weight * t.square().mean()
        q1 = q1 + weight * local_q1
        q2 = q2 + weight * local_q2
    return _finish(p1, p2, q1, q2, w, ridge)


def _dense_block_sums(a: Tensor, b: Tensor, w: Tensor, start: int, off: bool) -> tuple[Tensor, Tensor]:
    scores = a @ w @ b.T
    if off:
        mask = torch.arange(start, start + len(a), device=a.device)[:, None] != torch.arange(len(b), device=a.device)[None, :]
        scores = scores.masked_select(mask)
    return scores.sum(), scores.square().sum()


def _dense_moments(a: Tensor, b: Tensor, w: Tensor, off: bool, chunk_size: int | None, checkpoint_blocks: bool) -> tuple[Tensor, Tensor]:
    n, m = len(a), len(b)
    block = n if chunk_size is None else chunk_size
    first = second = w.new_zeros(())
    for start in range(0, n, block):
        stop = min(start + block, n)
        if checkpoint_blocks and torch.is_grad_enabled() and any(x.requires_grad for x in (a, b, w)):
            from torch.utils.checkpoint import checkpoint
            # Pass start as an argument; a loop closure would capture the final
            # row offset during backward recomputation and corrupt U gradients.
            one, two = checkpoint(_dense_block_sums, a[start:stop], b, w, start, off, use_reentrant=False)
        else:
            one, two = _dense_block_sums(a[start:stop], b, w, start, off)
        first = first + one
        second = second + two
    denominator = n * m - (n if off else 0)
    return first / denominator, second / denominator


def dense_objective_terms(scenes: Sequence[SceneBatch], w: Tensor, *, weights: Sequence[float], ridge: float = 0.0, chunk_size: int | None = None, checkpoint_blocks: bool = False, check_w: bool = True) -> ObjectiveTerms:
    """Exact dense reference, optionally row-blocked.

    Without checkpoint_blocks, autograd can retain all blocks. With it, each
    block uses non-reentrant checkpointing and recomputes its scores on backward.
    Peak memory and time must still be measured on the actual PyTorch backend.
    """
    scenes, weights = _inputs(scenes, weights)
    ridge = _ridge(ridge)
    if chunk_size is not None and (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer or None")
    if check_w:
        _matrix(w, scenes[0].a, scenes[0].b, box=True)
    p1 = p2 = q1 = q2 = w.new_zeros(())
    for s, weight in zip(scenes, weights):
        t = _diag(s.a, s.b, w)
        x, y = s.q_pools()
        local_q1, local_q2 = _dense_moments(x, y, w, s.q_mode == "u", chunk_size, checkpoint_blocks)
        p1 = p1 + weight * t.mean()
        p2 = p2 + weight * t.square().mean()
        q1 = q1 + weight * local_q1
        q2 = q2 + weight * local_q2
    return _finish(p1, p2, q1, q2, w, ridge)


def draw_ordinary_permutations(scenes: Sequence[SceneBatch], draws: int, *, generator: torch.Generator | None = None) -> tuple[Tensor, ...]:
    """Ordinary permutations, including fixed points, independently per scene."""
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not scenes or any(s.q_mode != "v" for s in scenes):
        raise ValueError("Ordinary permutation is a comparator for V scenes only")
    return tuple(torch.stack([torch.randperm(len(s.a), device=s.a.device, generator=generator) for _ in range(draws)]) for s in scenes)


def sampled_permutation_terms(scenes: Sequence[SceneBatch], w: Tensor, *, weights: Sequence[float], permutations: Sequence[Tensor], ridge: float = 0.0, check_w: bool = True, check_permutations: bool = True) -> ObjectiveTerms:
    """Fixed ordinary-permutation comparator; conditional expectation equals V.

    Every element of permutations is [draws, n_scene]. The samples are passed in
    explicitly to allow identical sampling states across gradients/methods.
    U/derangement and independent-pool sampling are intentionally not aliases.
    """
    scenes, weights = _inputs(scenes, weights)
    ridge = _ridge(ridge)
    permutations = tuple(permutations)
    if len(permutations) != len(scenes) or any(s.q_mode != "v" for s in scenes):
        raise ValueError("Provide one ordinary-permutation tensor per V scene")
    if check_w:
        _matrix(w, scenes[0].a, scenes[0].b, box=True)
    p1 = p2 = q1 = q2 = w.new_zeros(())
    for s, weight, perm in zip(scenes, weights, permutations):
        n = len(s.a)
        if perm.ndim != 2 or perm.shape[0] < 1 or perm.shape[1] != n or perm.dtype != torch.long or perm.device != s.a.device:
            raise ValueError("Each permutation tensor must be int64 [draws,n] on the feature device")
        if check_permutations:
            ordered = perm.sort(dim=1).values
            expected = torch.arange(n, device=perm.device).expand_as(perm)
            if not bool(torch.equal(ordered, expected)):
                raise ValueError("Every row must be a full local permutation; fixed points are allowed")
        t = _diag(s.a, s.b, w)
        neg = ((s.a @ w)[None, :, :] * s.b[perm]).sum(-1)
        p1 = p1 + weight * t.mean()
        p2 = p2 + weight * t.square().mean()
        q1 = q1 + weight * neg.mean()
        q2 = q2 + weight * neg.square().mean()
    return _finish(p1, p2, q1, q2, w, ridge)


class SimplexBilinearCritic(nn.Module):
    """Minimal learned simplex maps and a directly projected box head.

    No output tanh, no implicit ridge, no PSD/symmetry constraint on W.
    All M0 optimizer steps must be followed by project_head_().
    """

    def __init__(self, x_dim: int, y_dim: int, r: int, s: int, *, temperature: float = 1.0) -> None:
        super().__init__()
        if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in (x_dim, y_dim, r, s)):
            raise ValueError("All input and simplex dimensions must be positive integers")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.left = nn.Linear(x_dim, r)
        self.right = nn.Linear(y_dim, s)
        self.W = nn.Parameter(torch.zeros(r, s))
        self.temperature = float(temperature)

    def features(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return torch.softmax(self.left(x) / self.temperature, -1), torch.softmax(self.right(y) / self.temperature, -1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        a, b = self.features(x, y)
        if len(a) != len(b):
            raise ValueError("forward scores paired rows; use features() for separate pools")
        _features(a, "a")
        _features(b, "b")
        _matrix(self.W, a, b, box=True)
        return _diag(a, b, self.W)

    @torch.no_grad()
    def project_head_(self) -> "SimplexBilinearCritic":
        self.W.clamp_(-1, 1)
        return self


@dataclass(frozen=True)
class _LocalQuadratic:
    a: Tensor
    b: Tensor
    ga: Tensor
    gb: Tensor
    denominator: int
    off: bool
    weight: float


class QuadraticHead:
    """A snapshot of fixed feature values; HVP storage is O(n(r+s)+r²+s²).

    J(W)=<delta,W>-<W,H0(W)>/2 and F=J-ridge*||W||²/2.
    HVP includes ridge when requested by construction. No (r*s)^2 matrix.
    Never reuse this snapshot after features, Q, or scene weights change.
    """

    def __init__(self, scenes: Sequence[SceneBatch], *, weights: Sequence[float], ridge: float = 0.0) -> None:
        scenes, weights = _inputs(scenes, weights)
        self.ridge = _ridge(ridge)
        self.shape = (scenes[0].a.shape[1], scenes[0].b.shape[1])
        self.delta = scenes[0].a.new_zeros(self.shape)
        self._locals: list[_LocalQuadratic] = []
        trace_bound = self.delta.new_zeros(())
        with torch.no_grad():
            for scene, weight in zip(scenes, weights):
                a, b = scene.a.detach().clone(), scene.b.detach().clone()
                if scene.q_mode == "independent":
                    q_a, q_b = scene.q_pools()
                    q_a, q_b = q_a.detach().clone(), q_b.detach().clone()
                else:
                    q_a, q_b = a, b
                off = scene.q_mode == "u"
                denominator = len(q_a) * len(q_b) - (len(a) if off else 0)
                pos_cross = a.T @ b
                q_cross = torch.outer(q_a.sum(0), q_b.sum(0))
                q_trace = q_a.square().sum() * q_b.square().sum()
                pos_trace = (a.square().sum(1) * b.square().sum(1)).sum()
                if off:
                    q_cross = q_cross - pos_cross
                    q_trace = q_trace - pos_trace
                self.delta = self.delta + weight * (pos_cross / len(a) - q_cross / denominator)
                trace_bound = trace_bound + weight * (pos_trace / len(a) + q_trace / denominator)
                self._locals.append(_LocalQuadratic(a, b, q_a.T @ q_a, q_b.T @ q_b, denominator, off, weight))
        # trace(H0)+ridge bounds lambda_max(H0+ridge*I), not trace(Hreg).
        bound = float(trace_bound) + self.ridge
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("Nonpositive/nonfinite Lipschitz bound; inspect dtype and simplex features")
        self.lipschitz_upper_bound = bound * (1 + 64 * torch.finfo(self.delta.dtype).eps)

    @classmethod
    def from_scenes(cls, scenes: Sequence[SceneBatch], *, weights: Sequence[float], ridge: float = 0.0) -> "QuadraticHead":
        return cls(scenes, weights=weights, ridge=ridge)

    def _hvp(self, direction: Tensor) -> Tensor:
        out = self.ridge * direction
        for s in self._locals:
            diag = _diag(s.a, s.b, direction)
            paired = s.a.T @ (s.b * diag[:, None])
            negative = s.ga @ direction @ s.gb
            if s.off:
                negative = negative - paired
            out = out + s.weight * (paired / len(s.a) + negative / s.denominator)
        return out

    def hvp(self, direction: Tensor) -> Tensor:
        """Apply positive semidefinite Hreg; directions need not be box-feasible."""
        first = self._locals[0]
        _matrix(direction, first.a, first.b, box=False)
        return self._hvp(direction)

    def value_and_gradient(self, w: Tensor) -> tuple[Tensor, Tensor]:
        first = self._locals[0]
        _matrix(w, first.a, first.b, box=False)
        hw = self._hvp(w)
        return (self.delta * w).sum() - 0.5 * (w * hw).sum(), self.delta - hw

    def fw_gap(self, w: Tensor) -> Tensor:
        first = self._locals[0]
        _matrix(w, first.a, first.b, box=True)
        g = self.delta - self._hvp(w)
        # Algebraically ||g||_1-<g,W>, evaluated without subtracting large sums.
        return torch.where(g >= 0, g * (1 - w), -g * (1 + w)).sum()


@dataclass(frozen=True)
class SolverResult:
    w: Tensor
    converged: bool
    iterations: int
    fw_gap: float
    j: float
    ridge_penalty: float
    f: float
    tolerance: float
    lipschitz_upper_bound: float
    status: str
    history: tuple[dict[str, float | int], ...]


@torch.no_grad()
def solve_box_head(head: QuadraticHead, *, initial: Tensor | None = None, tolerance: float = 1e-7, max_iterations: int = 5000, accelerated: bool = True, history_every: int = 25) -> SolverResult:
    """Projected ascent, optionally monotone-restarted FISTA, with an FW gap.

    A finite budget need not reach tolerance. Only `converged=True` reports that
    the computed empirical gap met it; this is not a population certificate.
    float64 is recommended for demanding head tolerances and audit runs.
    """
    if not isinstance(head, QuadraticHead):
        raise ValueError("head must be a QuadraticHead")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if isinstance(history_every, bool) or not isinstance(history_every, int) or history_every < 1:
        raise ValueError("history_every must be a positive integer")
    if initial is None:
        w = torch.zeros_like(head.delta)
    else:
        first = head._locals[0]
        _matrix(initial, first.a, first.b, box=True)
        w = initial.detach().clone()
    y, momentum = w.clone(), 1.0
    step = 1.0 / head.lipschitz_upper_bound
    history: list[dict[str, float | int]] = []
    eps = torch.finfo(w.dtype).eps
    f = (head.delta * w).sum() - 0.5 * (w * head._hvp(w)).sum()
    iterations = 0
    for iteration in range(max_iterations + 1):
        g = head.delta - head._hvp(w)
        gap_t = torch.where(g >= 0, g * (1 - w), -g * (1 + w)).sum()
        gap = float(gap_t)
        f_value = float(f)
        if not math.isfinite(gap) or not math.isfinite(f_value):
            raise FloatingPointError("Head solver produced a nonfinite objective/gap")
        if iteration % history_every == 0 or gap <= tolerance or iteration == max_iterations:
            history.append({"iteration": iteration, "f": f_value, "fw_gap": gap})
        if gap <= tolerance or iteration == max_iterations:
            iterations = iteration
            break
        grad_y = head.delta - head._hvp(y)
        candidate = (y + step * grad_y).clamp(-1, 1)
        f_candidate = (head.delta * candidate).sum() - 0.5 * (candidate * head._hvp(candidate)).sum()
        # Restart momentum when extrapolation loses objective progress. The
        # unaccelerated step uses the PSD trace-based upper bound on curvature.
        rounding_allowance = 64 * eps * max(1.0, abs(f_value))
        if float(f_candidate) < f_value - rounding_allowance:
            momentum = 1.0
            candidate = (w + step * g).clamp(-1, 1)
            f_candidate = (head.delta * candidate).sum() - 0.5 * (candidate * head._hvp(candidate)).sum()
            if float(f_candidate) < f_value - rounding_allowance:
                raise FloatingPointError("Projected step lost objective: inspect numerical precision/HVP")
        next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2 if accelerated else 1.0
        y = candidate + ((momentum - 1) / next_momentum) * (candidate - w)
        w, f, momentum = candidate, f_candidate, next_momentum
    penalty = 0.5 * head.ridge * float(w.square().sum())
    return SolverResult(w.detach().clone(), gap <= tolerance, iterations, gap, f_value + penalty, penalty, f_value, tolerance, head.lipschitz_upper_bound, "gap_tolerance" if gap <= tolerance else "iteration_budget_exhausted", tuple(history))


@dataclass(frozen=True)
class ProfiledResult:
    terms: ObjectiveTerms
    solution: SolverResult


def profiled_objective(scenes: Sequence[SceneBatch], *, weights: Sequence[float], ridge: float = 0.0, initial: Tensor | None = None, tolerance: float = 1e-7, max_iterations: int = 5000, require_converged: bool = True) -> ProfiledResult:
    """M1: solve frozen values, then evaluate F on the SAME original tensors.

    Backpropagate `-result.terms.f`; the solved W is detached. With ridge>0,
    F (not a separately profiled unregularized J) is the outer objective.
    Exact envelope/Danskin conclusions require an exact optimum and their
    regularity conditions. A finite FW tolerance alone is no gradient theorem.
    """
    scenes, weights = _inputs(scenes, weights)
    head = QuadraticHead(scenes, weights=weights, ridge=ridge)
    solution = solve_box_head(head, initial=initial, tolerance=tolerance, max_iterations=max_iterations)
    if require_converged and not solution.converged:
        raise RuntimeError(f"M1 head did not meet its empirical gap tolerance: gap={solution.fw_gap:.6g}, tolerance={tolerance:.6g}; increase an explicitly approved solve budget or mark approximate-head mode")
    terms = objective_terms(scenes, solution.w.detach(), weights=weights, ridge=ridge)
    return ProfiledResult(terms, solution)
