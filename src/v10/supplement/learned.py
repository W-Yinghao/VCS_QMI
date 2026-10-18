"""Fixed ordinary-permutation registration for MLP VCS and host InfoNCE."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from v10 import adapter as A


def mlp_energy_rows(critic, reference, moving, permutations):
    """J(tanh f) for [C,N,D] moving descriptors; all R permutations stay fixed.

    Average the negative first and second moments over R, not the logits before
    tanh and not the squared mean of negative scores. No dense N x N allocation.
    """
    if reference.ndim != 2 or moving.ndim != 3:
        raise ValueError("Expected reference[N,D], moving[C,N,D]")
    if moving.shape[1:] != reference.shape or permutations.ndim != 2:
        raise ValueError("Descriptor/permutation shape mismatch")
    if permutations.shape[0] < 1 or permutations.shape[1] != reference.shape[0]:
        raise ValueError("Expected at least one full ordinary permutation")
    u = reference[None].expand(moving.shape[0], -1, -1)
    tp = torch.tanh(critic(torch.cat((u, moving), -1)).squeeze(-1))
    positive = (tp - 0.5 * tp.square()).mean(-1)
    negative = torch.zeros_like(positive)
    for permutation in permutations:
        tq = torch.tanh(critic(torch.cat((u, moving[:, permutation]), -1)).squeeze(-1))
        negative = negative + (tq + 0.5 * tq.square()).mean(-1)
    return positive - negative / permutations.shape[0]


class MLPRigidEnergy(A.Q.RigidEnergy):
    """Unchanged host support/search; extra permutations use an independent RNG."""

    def __init__(self, reference, moving, rng, *, encoder, critic, draws=1,
                 extra_seed=None):
        if draws not in (1, 8):
            raise ValueError("The supplemental protocol defines only perm1 and perm8")
        if draws == 8 and extra_seed is None:
            raise ValueError("perm8 needs an explicit registration-specific extra seed")
        A.assert_frozen(encoder, critic)
        super().__init__(torch, F, "critic", reference, moving, rng,
                         eV=encoder, critic=critic)
        # super consumes the exact host positions and first permutation. Drawing
        # extra permutations must not advance that shared scene/position stream.
        rows = [self.perm]
        extra = np.random.default_rng(extra_seed)
        for _ in range(draws - 1):
            rows.append(torch.as_tensor(extra.permutation(len(self.perm)),
                                        device=self.perm.device))
        self.permutations = torch.stack(rows)
        expected = torch.arange(len(self.perm), device=self.perm.device)
        if not torch.equal(self.permutations.sort(-1).values,
                           expected.expand_as(self.permutations)):
            raise ValueError("Invalid ordinary permutation")

    def j_critic(self, moving):
        return mlp_energy_rows(self.kw["critic"], self.u, moving, self.permutations)
