"""R22 alternative C: the critic's estimated PMI (and E_P[T]) as the rigid registration energy of a fixed V10 critic.

PDF eq. (35): the optimal VCS critic is T* = eta = tanh(PMI / 2), so PMI = 2 atanh(T*).  The host energies
(adapter.MFRigidEnergy, r6_reg_p2.RigidEnergy kind 'critic', supplement.learned.MLPRigidEnergy) all map the moving
descriptors dv [C, N, 16] of C candidate poses to C energies in j_critic(); the unchanged host search (grid, best 3 starts,
100 Adam steps, projection, final exact value) reaches the critic only through batch() / grid_values() / value(), which call
self.j_critic.  A variant replaces j_critic on one energy *instance* (the class methods are not touched, so the J path is the
original code):

  J    the host energy (the critic's J over the N sampled positions with the energy's own product reference)
  PMI  mean_n 2 atanh(clip(T_n, -1 + EPS, 1 - EPS)),  EPS = 1e-4 (fixed before any run)
  EPT  mean_n T_n                                      (E_P[T] alone: no reference term, no atanh)

T_n is the critic on the aligned pair (u_n, dv_n), computed with the same expression as the P term of the J path:
MF  T = a(u)^T W b(dv) (simplex features, W in [-1, 1]);  MLP / InfoNCE host  T = tanh f(u, dv).
Same positions, same permutation draws (unused by PMI / EPT), same moving image and reference, same search.
"""
from __future__ import annotations

import types

import torch

from v10 import adapter as A

EPS = 1e-4
VARIANTS = ("J", "PMI", "EPT")


def positive_scores(energy, dv):
    """T on the aligned pairs: dv [C, N, 16] -> [C, N]."""
    if isinstance(energy, A.MFRigidEnergy):
        B = energy.head.features_b(dv)
        return ((energy.A @ energy.W)[None] * B).sum(-1)              # = mf_energy_rows' t (same ops, same order)
    if isinstance(energy, A.Q.RigidEnergy) and energy.kind == "critic" and energy.kw.get("critic") is not None:
        u = energy.u[None].expand(dv.shape[0], -1, -1)
        return torch.tanh(energy.kw["critic"](torch.cat([u, dv], -1))[..., 0])   # = Q.RigidEnergy.j_critic / mlp_energy_rows tp
    raise TypeError(f"no positive-pair critic readout for {type(energy).__name__}")


def pmi_rows(T):
    return (2.0 * torch.atanh(T.clamp(-1.0 + EPS, 1.0 - EPS))).mean(-1)


def ept_rows(T):
    return T.mean(-1)


_ROWS = {"PMI": pmi_rows, "EPT": ept_rows}


def _variant_j_critic(rows):
    def j_critic(self, dv):
        return rows(positive_scores(self, dv))
    return j_critic


def set_variant(energy, name):
    """Switch one energy instance to J (class method), PMI or EPT.  `energy` is the inner host energy (not FiniteEnergy)."""
    if name == "J":
        energy.__dict__.pop("j_critic", None)
    elif name in _ROWS:
        energy.j_critic = types.MethodType(_variant_j_critic(_ROWS[name]), energy)
    else:
        raise ValueError(name)
    energy.r22_variant = name
    return energy


def inner(energy):
    """The host energy inside a supplement FiniteEnergy wrapper (the wrapper only rejects non-finite values)."""
    return getattr(energy, "energy", energy)


def register_variants(energy, device, truth, variants=VARIANTS):
    """One registration problem with every variant, J first, on the same energy instance (same positions, inputs and search).
    Returns {variant: host per-registration record (A.registration_record) + synchronized wall seconds}.  The instance is left
    on J."""
    import time
    out = {}
    e = inner(energy)
    try:
        for name in variants:
            set_variant(e, name)
            A.synchronize(device)
            t0 = time.perf_counter()
            rec = A.registration_record(energy, device, truth)
            A.synchronize(device)
            rec["wall_seconds"] = time.perf_counter() - t0
            out[name] = rec
    finally:
        set_variant(e, "J")
    return out
