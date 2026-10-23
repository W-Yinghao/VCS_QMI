"""R22-A3 objectives (R22_ALTERNATIVES_CN.md section A3): A's symmetric InfoNCE with a per-positive-pair reliability weight.
losses.py and losses_sharp.py are unchanged.

Theory (Variational_Cauchy_Schwarz_divergence.pdf): the weights are the critic posterior w = (1 - rho)(1 + T) / (1 + (1 - 2 rho) T) of the
locked N2F weight files (T = tanh(f), eq. 36; T* = tanh(PMI / 2)); InfoNCE is the host's retrieval loss, a replaceable implementation.
  sims  (B, B) SGR similarities, batch pair i on the diagonal; z = s / tau, tau = 0.05 (A's NCE arm).
  ell_i = 1/2 (row_i + col_i): row_i = -log softmax_j(z_ij) at j = i (caption retrieval), col_i = -log softmax_j(z_ji) at j = i (image
        retrieval) -- exactly A's InfoNCEObjective terms.
  loss  = mean_i(v_i / mean_v_full * ell_i), x B for loss_scale "sum": v_i = 1 on A, the locked weight on U, and mean_v_full = (n_A + sum_U w)
        / n_pool the trainer's fixed full-pool normaliser -- the weighted hinge's normaliser, written in A's VCSW form (wP = v / mean_v_full).
        Only the positive pair's two CE terms carry w_i: every batch member stays in every softmax denominator unweighted (a low-w pair
        remains an ordinary negative).  With v = 1 (mean_v_full = 1) the loss, its gradient and ell are bitwise A's NCE (test_r22_a3.py).
  NCE_VCSW  weights: the N2F VCS_Weight files (the files of A's VCSW and N2F/N2M hinge+VCS).
  NCE_BCEW  weights: the N2F BCE_Weight files (the files of N2F's BCE_Weight_sum hinge arm); matched control.
No critic parameters; ell (unweighted) is returned for the trainer's U-risk log as in A.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from losses import TAU, _scale


class WeightedInfoNCEObjective(nn.Module):
    def __init__(self, weights: str, tau: float = TAU):
        super().__init__()
        self.tau, self.weights, self._last = float(tau), str(weights), {}
        self.name = {"VCS_Weight": "infonce_vcs_weighted", "BCE_Weight": "infonce_bce_weighted"}[self.weights]

    def forward(self, sims, v, mean_v_full, loss_scale="sum"):
        z = sims / self.tau
        row = -torch.log_softmax(z, dim=1).diagonal()             # image i -> its caption among the batch captions
        col = -torch.log_softmax(z, dim=0).diagonal()             # caption i -> its image among the batch images
        ell = 0.5 * (row + col)
        wP = v.to(ell.dtype) / float(mean_v_full)
        loss = (wP * ell).mean()
        self._last = {"nce_w": loss.detach(), "nce_unweighted": ell.detach().mean(), "batch_mass_P": wP.detach().mean()}
        return _scale(loss, z.size(0), loss_scale), ell.detach()

    def stats(self) -> dict:
        return {k: float(x) for k, x in self._last.items()}

    def config(self) -> dict:
        return {"name": self.name, "tau": self.tau, "loss": "symmetric InfoNCE over rows and columns of s / tau",
                "weights": f"N2F {self.weights} on U, 1 on A; row and column CE terms of positive i x v_i / mean_v_full",
                "normaliser": "fixed full pool (n_A + sum_U w) / n_pool, as the weighted hinge", "negatives": "unweighted (all batch members)"}

    def describe(self) -> str:
        return f"R22-A3 {self.name}: symmetric InfoNCE, tau {self.tau} fixed, positive terms weighted by N2F {self.weights}; no hard-negative phase"


OBJECTIVES_A3 = {
    "NCE_VCSW": lambda: WeightedInfoNCEObjective("VCS_Weight"),
    "NCE_BCEW": lambda: WeightedInfoNCEObjective("BCE_Weight"),
}
