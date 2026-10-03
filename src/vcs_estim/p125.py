"""P125 — v6 V6-STRUCT (VCS_Server_Tasks_v6_CN.md §9): structured pair features for the measurement critic, before the output becomes one number.

Reuses the P108 / P117 conditions, P85 generators, oracles and roles unchanged (imports only; nothing in benchmark.py / p108.py / p117.py is edited).
Candidates (one shared trainer; same roles, indices, RNG roles and finite budget for every candidate and both losses):
  joint      JointMLP reference, concat -> 256 ReLU -> 256 ReLU -> 1 (the P85 / P108 class, unchanged)
  inter      Interaction-MLP: [x, y, x*y, |x - y|] -> h ReLU -> h ReLU -> 1, h chosen so that the parameter count ~= the reference (dims must match)
  quad_full  Quadratic logit f = b + u.x + v.y + x^T A x + y^T B y + x^T C y with full A, B, C (small random init)
  quad_rR    the same with low-rank factors A = Pa Qa^T, B = Pb Qb^T, C = Pc Qc^T of rank R (non-zero random init; R in {4, 16}); large-dim condition only
Losses: VCS (-J, T = tanh f) and matched JS (Deep-InfoMax logit f, posterior T = tanh(f/2)) exactly as P108.  Rotation (P108 E2 fixed orthogonal
maps, Rx != Ry) is applied to joint and inter only: the quadratic class is closed under separate orthogonal rotations, the interaction features are not.
Readouts on the independent EVAL role: posterior MSE vs the oracle eta, J and S_plug bias / RMSE inputs, eval SE, fit / selection seconds, parameter count.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn

from .benchmark import _dev, _native_on, _peak_mem, build_roles, combine, eta_of, truths
from .critics import JointMLP
from .data import setting_from_mi
from .estimators import EMA, estimate
from .p108 import CONDITIONS, LRS, _eval_T, _post_mse, readouts, rotate_roles

PROTOCOL = "P125"
BUDGET = 1000
NS = (4096, 16384)
PILOT_SEEDS = (0, 1)
CONFIRM_SEEDS = (0, 1, 2, 3, 4)
LOW_RANKS = (4, 16)
LARGE_DIM = 50          # conditions with d_total > LARGE_DIM get the low-rank quadratic candidates (C2: 100)


def joint_params(d: int, hidden: int = 256) -> int:
    return (2 * d) * hidden + hidden + hidden * hidden + hidden + hidden + 1


def matched_hidden(d: int, target: int) -> int:
    """Width h of the 2-hidden-layer interaction MLP on 4d inputs whose parameter count is closest to `target`."""
    c = 4 * d + 3; h = (-c + math.sqrt(c * c + 4 * (target - 1))) / 2
    return max(8, int(round(h)))


class InteractionMLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.d = d; self.net = nn.Sequential(nn.Linear(4 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def pairs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != y.shape[1]:
            raise ValueError("Interaction-MLP needs equal x / y dimensions")
        return self.net(torch.cat([x, y, x * y, (x - y).abs()], -1)).squeeze(-1)


class QuadLogit(nn.Module):
    """f = b + u.x + v.y + x^T A x + y^T B y + x^T C y.  rank=None: full matrices; rank=R: A = Pa Qa^T etc. (non-zero random factor init)."""

    def __init__(self, dx: int, dy: int, rank: int | None = None, init_scale: float = 0.1):
        super().__init__()
        self.rank = rank; self.b = nn.Parameter(torch.zeros(1)); self.u = nn.Parameter(torch.randn(dx) * init_scale / math.sqrt(dx))
        self.v = nn.Parameter(torch.randn(dy) * init_scale / math.sqrt(dy))
        if rank is None:
            self.A = nn.Parameter(torch.randn(dx, dx) * init_scale / dx); self.B = nn.Parameter(torch.randn(dy, dy) * init_scale / dy)
            self.C = nn.Parameter(torch.randn(dx, dy) * init_scale / math.sqrt(dx * dy))
        else:
            s = math.sqrt(init_scale)
            mk = lambda d: nn.Parameter(torch.randn(d, rank) * s / math.sqrt(d))
            self.Pa, self.Qa, self.Pb, self.Qb, self.Pc, self.Qc = mk(dx), mk(dx), mk(dy), mk(dy), mk(dx), mk(dy)

    def pairs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lin = self.b + x @ self.u + y @ self.v
        if self.rank is None:
            return lin + ((x @ self.A) * x).sum(-1) + ((y @ self.B) * y).sum(-1) + ((x @ self.C) * y).sum(-1)
        return (lin + ((x @ self.Pa) * (x @ self.Qa)).sum(-1) + ((y @ self.Pb) * (y @ self.Qb)).sum(-1)
                + ((x @ self.Pc) * (y @ self.Qc)).sum(-1))


def factory(cand: str, d: int):
    if cand == "joint":
        return lambda: JointMLP(d, d)
    if cand == "inter":
        h = matched_hidden(d, joint_params(d)); return lambda: InteractionMLP(d, h)
    if cand == "quad_full":
        return lambda: QuadLogit(d, d, None)
    if cand.startswith("quad_r"):
        r = int(cand[len("quad_r"):]); return lambda: QuadLogit(d, d, r)
    raise ValueError(cand)


def candidates_for(d_total: int) -> tuple[str, ...]:
    base = ("joint", "inter", "quad_full")
    return base + tuple(f"quad_r{r}" for r in LOW_RANKS) if d_total > LARGE_DIM else base


def train_critic(make, kind: str, lr: float, roles: dict, B: int, updates: int, seed: int, device, every: int = 100):
    """P85 `train_neural` (product negatives, Adam, SELECT native risk every `every` updates, best-SELECT state kept) with a critic factory.
    RNG roles identical to train_neural: torch.manual_seed(seed*7919+17) before the critic is built; batch stream seed*1000+7."""
    torch.manual_seed(seed * 7919 + 17); critic = make().to(device); opt = torch.optim.Adam(critic.parameters(), lr=lr)
    ema = EMA() if kind == "dv" else None; g = torch.Generator().manual_seed(seed * 1000 + 7); sg = torch.Generator().manual_seed(seed * 1000 + 11)
    F, S = roles["FIT"], roles["SELECT"]; Fx, Fy, Fqx, Fqy = (_dev(t, device) for t in (F.xp, F.yp, F.xq, F.yq)); n = F.n; nonfinite = 0; t0 = time.time()

    def sel_risk():
        critic.eval(); a, b, how, _, _, _ = _native_on(critic, kind, "product", S, device, 512, sg); critic.train()
        v = combine(a, b, how); return -v if math.isfinite(v) else float("inf")

    curve = [(0, sel_risk())]; best = (curve[0][1], 0, {k: v.detach().clone() for k, v in critic.state_dict().items()})
    for step in range(1, updates + 1):
        ip = torch.randint(0, n, (B,), generator=g).to(device); iq = torch.randint(0, n, (B,), generator=g).to(device)
        fp = critic.pairs(Fx[ip], Fy[ip]); fn = critic.pairs(Fqx[iq], Fqy[iq]).unsqueeze(1)
        loss, _ = estimate(kind, fp, fn, ema=ema)
        if not torch.isfinite(loss):
            nonfinite += 1; opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % every == 0:
            v = sel_risk(); curve.append((step, v))
            if v < best[0]:
                best = (v, step, {k: t.detach().clone() for k, t in critic.state_dict().items()})
    critic.load_state_dict(best[2]); critic.eval()
    return critic, {"selected_update": best[1], "select_risk": best[0], "fit_seconds": time.time() - t0, "nonfinite_steps": nonfinite,
                    "n_params": sum(p.numel() for p in critic.parameters())}


def struct_cell(cond: str, N: int, seed: int, rotated: bool, device, budget: int = BUDGET, lrs=LRS, smoke: bool = False, cands=None) -> dict:
    name, ds, dt, I = CONDITIONS[cond]; st = setting_from_mi(name, I, ds)
    if name == "xor_mixture":
        ds = dt = st.d
    t0 = time.time(); roles, sizes = build_roles(st, ds, dt, N, seed, smoke); tr = truths(st, ds, roles["TRUTH"]); S = tr["S"]; eta_roles = None; rot = None
    if rotated:
        roles, rot = rotate_roles(roles, dt); eta_roles = rot.pop("orig_eval")
    E = roles["EVAL"]; Ee = eta_roles or E; ep, eq = eta_of(st, Ee.xp, Ee.yp, ds), eta_of(st, Ee.xq, Ee.yq, ds)
    cands = cands or (("joint", "inter") if rotated else candidates_for(dt)); B = min(256, N); rows = []
    for cand in cands:
        make = factory(cand, dt)
        for kind in ("vcs", "js"):
            half = 1.0 if kind == "vcs" else 0.5; group = []
            for lr in lrs:
                _peak_mem(device); critic, info = train_critic(make, kind, lr, roles, B, budget, seed, device)
                te = time.time(); tp, tq = _eval_T(critic, E, device, half); r = readouts(tp, tq)
                group.append({"candidate": cand, "kind": kind, "lr": lr, "budget_updates": budget, "selected": False, "rotated": rotated, **r,
                              "J_err": r["J"] - S, "S_plug_err": r["S_plug"] - S, "posterior_mse": _post_mse(tp, tq, ep, eq), **info,
                              "eval_seconds": time.time() - te, "memory": _peak_mem(device)})
            best = min(group, key=lambda x: x["select_risk"]); best["selected"] = True
            for x in group:
                x["tuning_seconds_total"] = sum(gg["fit_seconds"] for gg in group)
            rows += group
    return {"protocol": PROTOCOL, "condition": cond, "setting": name, "d_signal": ds, "d_total": dt, "I": I, "N": N, "seed": seed, "rotated": rotated,
            "rotation": rot, "sizes": sizes, "truth": {k: tr[k] for k in ("S", "S_se")}, "rows": rows, "wall_seconds": time.time() - t0, "device": str(device),
            "inter_hidden": matched_hidden(dt, joint_params(dt)), "joint_params": joint_params(dt)}


def struct_cells(seeds=PILOT_SEEDS, Ns=NS, rotations=(False, True)) -> list[tuple]:
    return [(c, N, s, rot) for c in CONDITIONS for N in Ns for s in seeds for rot in rotations]
