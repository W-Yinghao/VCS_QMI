"""R22-A objectives that replace the hinge of the SGR student (v9.n2.train.WeightedStudent(objective=...)).

Theory: Variational_Cauchy_Schwarz_divergence.pdf (sha256 58ab8a8a...), eqs. (33)-(40).
  sims  (B, B) SGR similarities s_ij in (0, 1) of image i and caption j; batch pair i is the diagonal entry.
  P     the B diagonal pairs (x_i, y_i): the observed pairs, a sample of P_XY (noisy: in U, rho = 0.4 are mismatched).
  Q     the B(B-1) off-diagonal pairs (x_i, y_j), i != j: draws from P_X P_Y when the batch rows are i.i.d. (eq. 39 mismatches with
        one permutation; averaging over all off-diagonal pairs is the U-statistic over all such permutations).
  critic T_ij = tanh(f_ij), f_ij = softplus(a) * s_ij + b (eq. 36; f is increasing in s, so ranking by s = ranking by T).
  VCS   J = E_P T - E_Q T - 1/2 E_P T^2 - 1/2 E_Q T^2 (eq. 40), loss = 1 - J  (= 1/2 E_P (1 - T)^2 + 1/2 E_Q (1 + T)^2).
  VCSW  P replaced by the weighted joint P_w(i) proportional to v_i (v = 1 on A, the locked N2F VCS weight on U):
        E_Pw g = mean_i(v_i g_ii) / mean_v_full with the host's fixed full-pool normaliser mean_v_full = (n_A + sum_U w) / n_pool,
        i.e. the same normaliser as the weighted hinge (unbiased over the pool).  Q stays unweighted (product of the data marginals).
  BCE   matched control, same critic, logit 2f (sigmoid(2f) = (1 + tanh f) / 2, so the BCE optimum 2f* = log p/q gives the same
        T* = tanh(PMI / 2)): loss = E_P softplus(-2f) + E_Q softplus(2f), P and Q each a mean (balanced).
  NCE   InfoNCE control: symmetric cross entropy of s / tau over rows (caption retrieval) and columns (image retrieval), tau = 0.05
        fixed, no critic parameters, no weights.
  loss_scale "sum" multiplies every objective by B: the host's sum-over-anchors convention (official SGRAF/NCR scale), under which
  the locked grad_clip 2.0 acts as in the hinge runs.  Off-diagonal means are (total - trace) / (B(B-1)): no boolean indexing, so the
  deterministic-algorithms setting has nothing to refuse.
Each forward returns (loss, ell): ell (B,) is a per-anchor positive-pair term used only for the trainer's U-risk log.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA0, B0, TAU = 5.0, -2.5, 0.05          # critic init: f = 5 s - 2.5 is 0 at s = 0.5 (T = 0 is the J optimum while all s are equal)


def _offdiag_mean(m: torch.Tensor) -> torch.Tensor:
    n = m.size(0)
    return (m.sum() - m.diagonal().sum()) / (n * (n - 1))


def _scale(loss: torch.Tensor, n: int, loss_scale: str) -> torch.Tensor:
    if loss_scale not in ("mean", "sum"):
        raise ValueError(f"loss_scale {loss_scale!r}")
    return loss * n if loss_scale == "sum" else loss


class Critic(nn.Module):
    """T = tanh(softplus(a) s + b): two learnable scalars, trained jointly with the student by the same optimiser."""

    def __init__(self, alpha0: float = ALPHA0, b0: float = B0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(math.log(math.expm1(alpha0)), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(b0), dtype=torch.float32))
        self.alpha0, self.b0 = alpha0, b0

    def f(self, sims: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a) * sims + self.b

    def values(self) -> dict:
        return {"alpha": float(F.softplus(self.a.detach())), "b": float(self.b.detach())}


def vcs_terms(T: torch.Tensor, wP: torch.Tensor | None = None) -> dict:
    """E_P T, E_P T^2 (weighted when wP is given: mean_i(wP_i g_ii), wP = v / mean_v_full), E_Q T, E_Q T^2 and J (eq. 40)."""
    d = T.diagonal()
    if wP is None:
        ep, ep2 = d.mean(), (d * d).mean()
    else:
        ep, ep2 = (wP * d).mean(), (wP * d * d).mean()
    eq, eq2 = _offdiag_mean(T), _offdiag_mean(T * T)
    return {"EP_T": ep, "EP_T2": ep2, "EQ_T": eq, "EQ_T2": eq2, "J": ep - eq - 0.5 * ep2 - 0.5 * eq2}


class VCSObjective(nn.Module):
    def __init__(self, weighted: bool = False, alpha0: float = ALPHA0, b0: float = B0):
        super().__init__()
        self.critic, self.weighted = Critic(alpha0, b0), bool(weighted)
        self.name = "vcs_weighted_P" if weighted else "vcs"
        self._last = {}

    def forward(self, sims, v, mean_v_full, loss_scale="sum"):
        T = torch.tanh(self.critic.f(sims))
        wP = (v.to(T.dtype) / float(mean_v_full)) if self.weighted else None
        t = vcs_terms(T, wP)
        self._last = {k: x.detach() for k, x in t.items()}
        d = T.diagonal()
        return _scale(1.0 - t["J"], T.size(0), loss_scale), 0.5 * (1.0 - d.detach()) ** 2

    def stats(self) -> dict:
        return {**{k: float(x) for k, x in self._last.items()}, **self.critic.values()}

    def config(self) -> dict:
        return {"name": self.name, "critic": "tanh(softplus(a) s + b)", "alpha0": self.critic.alpha0, "b0": self.critic.b0,
                "P": "diagonal pairs" + (", weighted v/mean_v_full (fixed full-pool normaliser)" if self.weighted else ", unweighted"),
                "Q": "all off-diagonal pairs, U-statistic mean, unweighted", "loss": "1 - J (PDF eq. 40)"}

    def describe(self) -> str:
        return f"R22-A {self.name}: 1 - J, J = E_P T - E_Q T - E_P T^2/2 - E_Q T^2/2, T = tanh(softplus(a) s + b); no hard-negative phase"


class BCEObjective(nn.Module):
    def __init__(self, weighted: bool = False, alpha0: float = ALPHA0, b0: float = B0):
        super().__init__()
        self.critic, self.weighted = Critic(alpha0, b0), bool(weighted)
        self.name = "bce_weighted_P" if weighted else "bce"
        self._last = {}

    def forward(self, sims, v, mean_v_full, loss_scale="sum"):
        logit = 2.0 * self.critic.f(sims)
        pos = F.softplus(-logit.diagonal())
        lp = ((v.to(pos.dtype) / float(mean_v_full)) * pos).mean() if self.weighted else pos.mean()
        lq = _offdiag_mean(F.softplus(logit))
        self._last = {"bce_P": lp.detach(), "bce_Q": lq.detach()}
        return _scale(lp + lq, logit.size(0), loss_scale), pos.detach()

    def stats(self) -> dict:
        return {**{k: float(x) for k, x in self._last.items()}, **self.critic.values()}

    def config(self) -> dict:
        return {"name": self.name, "critic": "tanh(softplus(a) s + b); logit 2f", "alpha0": self.critic.alpha0, "b0": self.critic.b0,
                "P": "diagonal pairs" + (", weighted" if self.weighted else ", unweighted"), "Q": "all off-diagonal pairs, unweighted",
                "loss": "E_P softplus(-2f) + E_Q softplus(2f)"}

    def describe(self) -> str:
        return f"R22-A {self.name}: balanced BCE P vs Q with logit 2 (softplus(a) s + b); no hard-negative phase"


class InfoNCEObjective(nn.Module):
    def __init__(self, tau: float = TAU):
        super().__init__()
        self.tau, self.name, self._last = float(tau), "infonce", {}

    def forward(self, sims, v, mean_v_full, loss_scale="sum"):
        z = sims / self.tau
        row = -torch.log_softmax(z, dim=1).diagonal()             # image i -> its caption among the batch captions
        col = -torch.log_softmax(z, dim=0).diagonal()             # caption i -> its image among the batch images
        ell = 0.5 * (row + col)
        loss = ell.mean()
        self._last = {"nce": loss.detach()}
        return _scale(loss, z.size(0), loss_scale), ell.detach()

    def stats(self) -> dict:
        return {k: float(x) for k, x in self._last.items()}

    def config(self) -> dict:
        return {"name": self.name, "tau": self.tau, "loss": "symmetric InfoNCE over rows and columns of s / tau", "weights": "none"}

    def describe(self) -> str:
        return f"R22-A infonce: symmetric InfoNCE, tau {self.tau} fixed; no hard-negative phase"


# arm -> objective factory (None: the unchanged hinge path, smoke timing reference only)
OBJECTIVES = {
    "VCS": lambda: VCSObjective(weighted=False),
    "VCSW": lambda: VCSObjective(weighted=True),
    "BCE": lambda: BCEObjective(weighted=False),
    "NCE": lambda: InfoNCEObjective(TAU),
    "HINGE": lambda: None,
}
