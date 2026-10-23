"""R22-A2 objectives (R22_ALTERNATIVES_CN.md section A2): the R22-A objectives with a sharp fixed-slope critic.  losses.py is unchanged.

Theory as in losses.py (Variational_Cauchy_Schwarz_divergence.pdf eqs. (33)-(40)); only the critic parameterisation changes:
  f_ij = ALPHA * s_ij + b,  ALPHA = 20 fixed (a Python constant, not a parameter; the InfoNCE scale 1 / tau = 1 / 0.05),
  b learnable, initialised at -10 (f = 0 at s = 0.5), in its own Adam group: base lr 1e-3, x0.1 at epoch lr_update (30) like the student
  (the A critic followed the student's schedule; A2 changes only its base lr).  T = tanh(f) stays in the bounded class of eq. (36).
  Far negatives sit at T ~ -1 where d tanh = 1 - T^2 ~ 0, so the Q gradient concentrates on hard negatives without a mining step.
  VCS_SHARP   loss 1 - J (eq. 40), P = diagonal, Q = all off-diagonal pairs, P and Q means separate, x B (loss_scale "sum") as in A.
  VCSW_SHARP  P weighted by the locked N2F VCS weights exactly as A's VCSW (v / mean_v_full, fixed full-pool normaliser); Q unweighted.
  BCE_SHARP   same f, logit 2f, balanced BCE (P mean + Q mean), no weights.
Each objective keeps a per-epoch record (train.py calls epoch_stats() once per epoch): b, mean T on P and Q, the fraction of Q pairs with
T > -0.9 (hard-negative fraction), J-hat of eq. (40) on the batch (unweighted; VCSW also the weighted training J), and T quantiles on P / Q
from every POOL_EVERY-th update (detached, off-diagonal entries taken without boolean indexing).  Nothing here touches the training graph.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import _offdiag_mean, _scale, vcs_terms

ALPHA, B0, B_LR = 20.0, -10.0, 1e-3
HARD, POOL_EVERY, QUANTILES = -0.9, 20, (0.05, 0.25, 0.5, 0.75, 0.95)


class SharpCritic(nn.Module):
    """T = tanh(ALPHA s + b): ALPHA a fixed float, b the only parameter."""

    def __init__(self, alpha: float = ALPHA, b0: float = B0, b_lr: float = B_LR):
        super().__init__()
        self.alpha, self.b0, self.b_lr = float(alpha), float(b0), float(b_lr)
        self.b = nn.Parameter(torch.tensor(self.b0, dtype=torch.float32))

    def f(self, sims: torch.Tensor) -> torch.Tensor:
        return self.alpha * sims + self.b

    def values(self) -> dict:
        return {"alpha": self.alpha, "b": float(self.b.detach())}


def _offdiag(m: torch.Tensor) -> torch.Tensor:
    n = m.size(0)
    return m.reshape(-1)[1:].reshape(n - 1, n + 1)[:, :-1]          # the n(n-1) off-diagonal entries, row-major


class _Sharp(nn.Module):
    name = "sharp"

    def __init__(self, weighted: bool = False):
        super().__init__()
        self.critic, self.weighted, self._last = SharpCritic(), bool(weighted), {}
        self._reset()

    def param_groups(self) -> list:
        return [{"params": [self.critic.b], "lr": self.critic.b_lr, "base_lr": self.critic.b_lr, "name": "r22a2_critic_b"}]

    def _reset(self):
        self._acc, self._n, self._pool_p, self._pool_q = None, 0, [], []

    @torch.no_grad()
    def _observe(self, T: torch.Tensor, j_train: torch.Tensor | None):
        T = T.detach()
        d, off = T.diagonal(), _offdiag(T)
        j = vcs_terms(T)["J"]
        vals = torch.stack([x.float() for x in (d.mean(), off.mean(), (off > HARD).float().mean(), (d > HARD).float().mean(), j,
                                                j if j_train is None else j_train.detach())])
        self._acc = vals if self._acc is None else self._acc + vals
        if self._n % POOL_EVERY == 0:
            self._pool_p.append(d.float().clone()); self._pool_q.append(off.reshape(-1).float().clone())
        self._n += 1
        self._last["hard_neg_frac_Q"] = vals[2]

    def epoch_stats(self) -> dict:
        if self._n == 0:
            return {}
        a = (self._acc / self._n).tolist()
        q = torch.tensor(QUANTILES, dtype=torch.float32)
        pp, pq = torch.cat(self._pool_p).cpu(), torch.cat(self._pool_q).cpu()
        out = {"updates": self._n, **self.critic.values(), "EP_T_mean": a[0], "EQ_T_mean": a[1], "hard_neg_frac_Q": a[2],
               "frac_P_above_hard": a[3], "J_hat": a[4], "J_train": a[5], "pool_every": POOL_EVERY, "pool_n_P": pp.numel(), "pool_n_Q": pq.numel(),
               "T_P_quantiles": dict(zip(map(str, QUANTILES), torch.quantile(pp, q).tolist())),
               "T_Q_quantiles": dict(zip(map(str, QUANTILES), torch.quantile(pq, q).tolist()))}
        self._reset()
        return out

    def stats(self) -> dict:
        return {**{k: float(x) for k, x in self._last.items()}, **self.critic.values()}

    def _base_config(self) -> dict:
        c = self.critic
        return {"name": self.name, "alpha": c.alpha, "alpha_learnable": False, "b0": c.b0, "b_lr": c.b_lr,
                "b_optimiser": "own Adam group (betas/eps as the student), base lr 1e-3 x 0.1^(epoch // lr_update) like the student; not clipped",
                "hard_negative_threshold_T": HARD}


class VCSSharpObjective(_Sharp):
    def __init__(self, weighted: bool = False):
        super().__init__(weighted)
        self.name = "vcs_sharp_weighted_P" if weighted else "vcs_sharp"

    def forward(self, sims, v, mean_v_full, loss_scale="sum"):
        T = torch.tanh(self.critic.f(sims))
        wP = (v.to(T.dtype) / float(mean_v_full)) if self.weighted else None
        t = vcs_terms(T, wP)
        self._last = {k: x.detach() for k, x in t.items()}
        self._observe(T, t["J"])
        d = T.diagonal()
        return _scale(1.0 - t["J"], T.size(0), loss_scale), 0.5 * (1.0 - d.detach()) ** 2

    def config(self) -> dict:
        return {**self._base_config(), "critic": "tanh(alpha s + b), alpha fixed",
                "P": "diagonal pairs" + (", weighted v/mean_v_full (fixed full-pool normaliser)" if self.weighted else ", unweighted"),
                "Q": "all off-diagonal pairs, U-statistic mean, unweighted", "loss": "1 - J (PDF eq. 40)"}

    def describe(self) -> str:
        return (f"R22-A2 {self.name}: 1 - J, J = E_P T - E_Q T - E_P T^2/2 - E_Q T^2/2, T = tanh({self.critic.alpha:g} s + b), b0 "
                f"{self.critic.b0:g}, b lr {self.critic.b_lr:g} (own group); no hard-negative phase")


class BCESharpObjective(_Sharp):
    def __init__(self):
        super().__init__(False)
        self.name = "bce_sharp"

    def forward(self, sims, v, mean_v_full, loss_scale="sum"):
        f = self.critic.f(sims)
        logit = 2.0 * f
        pos = F.softplus(-logit.diagonal())
        lp, lq = pos.mean(), _offdiag_mean(F.softplus(logit))
        self._last = {"bce_P": lp.detach(), "bce_Q": lq.detach()}
        self._observe(torch.tanh(f), None)
        return _scale(lp + lq, logit.size(0), loss_scale), pos.detach()

    def config(self) -> dict:
        return {**self._base_config(), "critic": "tanh(alpha s + b), alpha fixed; logit 2f", "P": "diagonal pairs, unweighted",
                "Q": "all off-diagonal pairs, unweighted", "loss": "E_P softplus(-2f) + E_Q softplus(2f)"}

    def describe(self) -> str:
        return f"R22-A2 bce_sharp: balanced BCE P vs Q with logit 2 ({self.critic.alpha:g} s + b), b lr {self.critic.b_lr:g} (own group)"


OBJECTIVES_SHARP = {
    "VCS_SHARP": lambda: VCSSharpObjective(weighted=False),
    "VCSW_SHARP": lambda: VCSSharpObjective(weighted=True),
    "BCE_SHARP": lambda: BCESharpObjective(),
}
