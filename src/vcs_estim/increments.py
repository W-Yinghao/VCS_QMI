"""P102 (package v2, Server Spec §11, Plan §8.2): conditional dependence increments between nested information sets.

Binary experiment (conditional): P = P_Y P_{H,N|Y}, Q = P_Y P_{H|Y} P_{N|Y}; C = +1 on P, -1 on Q; M = (P + Q) / 2.
For a critic T with values in [-1, 1]:  J(T) = 1 - E_M (C - T)^2 = E_P T - E_Q T - 1/2 E_P T^2 - 1/2 E_Q T^2 (the VCS objective).
For two nested sets A ⊂ B (A a deterministic function of B):
    Delta_J = J_B - J_A,   D_T = E_M (T_B - T_A)^2,   r_BA = Delta_J - D_T = 2 E_M[(C - T_B)(T_B - T_A)]   (algebraic for any T_A, T_B).
At the oracle T* = E[C | .], r_BA = 0 and Delta = S_B - S_A = E_M (T_B* - T_A*)^2.
For a chain T_0 ⊇ ... ⊇ T_L:  R_orth = sum_l E_M d_l^2 - E_M (T_0 - T_L)^2 = -2 sum_{l<k} E_M d_l d_k,  d_l = T_l - T_{l+1};  zero at the oracle.
The telescoping sum of J differences is an algebraic identity and is NOT used as a check.

Contents: (1) pure functions of per-pair critic outputs (J, increments, R_orth, paired bootstrap by base image); (2) the exact discrete oracle
toy used by the tests and as the runner's self-check; (3) critic classes that receive (features, N, Y) and the VCS / matched-JS fitters with the
T1 (P73) settings: AdamW lr 1e-3, wd 1e-2, 300 full-batch steps, 80 / 20 fit / select split, select every 10 steps.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------------------------------------------------- (1) quantities
def j_from_t(tp: np.ndarray, tq: np.ndarray) -> float:
    """J = E_P T - E_Q T - 1/2 E_P T^2 - 1/2 E_Q T^2 (positive and negative sets averaged separately)."""
    return float(tp.mean() - tq.mean() - 0.5 * (tp ** 2).mean() - 0.5 * (tq ** 2).mean())


def em(xp: np.ndarray, xq: np.ndarray) -> float:
    """E_M of a per-pair quantity given its values on the P pairs and the Q pairs."""
    return float(0.5 * xp.mean() + 0.5 * xq.mean())


def increment(tbp, tbq, tap, taq) -> dict:
    """Delta_J, D_T, r_BA and the residual identity check for one nested pair (B ⊇ A), all on the same P / Q pairs."""
    jb, ja = j_from_t(tbp, tbq), j_from_t(tap, taq)
    d = em((tbp - tap) ** 2, (tbq - taq) ** 2)
    r = (jb - ja) - d
    r_ident = 2.0 * em((1.0 - tbp) * (tbp - tap), (-1.0 - tbq) * (tbq - taq))
    return {"J_B": jb, "J_A": ja, "delta_J": jb - ja, "D_T": d, "r_BA": r, "r_BA_identity_gap": abs(r - r_ident)}


def r_orth(chain_p: list, chain_q: list) -> float:
    """R_orth = sum_l E_M d_l^2 - E_M (T_0 - T_L)^2 for a chain of per-pair outputs [T_0, ..., T_L] (T_0 = the finest set)."""
    s = 0.0
    for l in range(len(chain_p) - 1):
        s += em((chain_p[l] - chain_p[l + 1]) ** 2, (chain_q[l] - chain_q[l + 1]) ** 2)
    return s - em((chain_p[0] - chain_p[-1]) ** 2, (chain_q[0] - chain_q[-1]) ** 2)


def chain_summary(chain_p: list, chain_q: list, names: list) -> dict:
    out = {"J": {nm: j_from_t(p, q) for nm, p, q in zip(names, chain_p, chain_q)}, "steps": {}}
    for l in range(len(names) - 1):
        out["steps"][f"{names[l]}->{names[l + 1]}"] = increment(chain_p[l], chain_q[l], chain_p[l + 1], chain_q[l + 1])
    out["R_orth"] = r_orth(chain_p, chain_q)
    return out


def paired_bootstrap(outputs_p: dict, outputs_q: dict, stat_fn, reps: int = 1000, seed: int = 0) -> np.ndarray:
    """Paired bootstrap by base image: outputs_*[name] are arrays indexed by base image i (the positive pair of image i and its within-class
    negative); every resample draws images with replacement and applies the SAME index to every critic's P and Q outputs.  Returns reps x k."""
    names = list(outputs_p); n = len(outputs_p[names[0]])
    assert all(len(outputs_p[k]) == n and len(outputs_q[k]) == n for k in names), "P and Q outputs must be indexed by the same base images"
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        ix = rng.integers(0, n, n)
        vals.append(stat_fn({k: v[ix] for k, v in outputs_p.items()}, {k: v[ix] for k, v in outputs_q.items()}))
    return np.asarray(vals, dtype=np.float64)


def ci(vals: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> list:
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


# ----------------------------------------------------------------------------------------------------------------------- (2) exact discrete oracle
def toy_joint(n_y: int = 3, n_x: int = 12, seed: int = 0, strength: float = 1.0):
    """A discrete conditional experiment: Y uniform on n_y classes, X | Y on n_x states, N | X, Y binary with P(N = 1 | x, y) = sigmoid(a_y + strength * b_x).
    Returns p[y, x, n] (joint P) and q[y, x, n] = P(y) P(x | y) P(n | y) (the conditional product Q)."""
    rng = np.random.default_rng(seed)
    py = np.full(n_y, 1.0 / n_y)
    px_y = rng.dirichlet(np.ones(n_x), size=n_y)
    a = rng.normal(0, 1, n_y); b = rng.normal(0, 1.5, n_x)
    p1 = 1.0 / (1.0 + np.exp(-(a[:, None] + strength * b[None, :])))            # P(N = 1 | y, x)
    pn_yx = np.stack([1 - p1, p1], -1)                                             # [y, x, n]
    p = py[:, None, None] * px_y[:, :, None] * pn_yx
    pn_y = (px_y[:, :, None] * pn_yx).sum(1)                                       # P(n | y)
    q = py[:, None, None] * px_y[:, :, None] * pn_y[:, None, :]
    return p, q


def coarsen(p: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Push a [y, x, n] table through the deterministic map x -> groups[x]."""
    out = np.zeros((p.shape[0], int(groups.max()) + 1, p.shape[2]))
    for x, g in enumerate(groups):
        out[:, g] += p[:, x]
    return out


def oracle_t(p: np.ndarray, q: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """T*(y, g(x), n) = (p - q) / (p + q) on the coarsened table, broadcast back to the fine [y, x, n] grid."""
    if groups is None:
        groups = np.arange(p.shape[1])
    pc, qc = coarsen(p, groups), coarsen(q, groups)
    tc = np.where(pc + qc > 0, (pc - qc) / np.maximum(pc + qc, 1e-300), 0.0)
    return tc[:, groups, :]


def exact_em(p: np.ndarray, q: np.ndarray, f_p: np.ndarray, f_q: np.ndarray | None = None) -> float:
    """E_M of a function on the [y, x, n] grid (f_q defaults to f_p): 1/2 sum p f_p + 1/2 sum q f_q."""
    f_q = f_p if f_q is None else f_q
    return float(0.5 * (p * f_p).sum() + 0.5 * (q * f_q).sum())


def exact_j(p, q, t) -> float:
    return float((p * t).sum() - (q * t).sum() - 0.5 * (p * t ** 2).sum() - 0.5 * (q * t ** 2).sum())


# ----------------------------------------------------------------------------------------------------------------------- (3) critics with (features, N, Y)
class YPairLinear(nn.Module):
    """f = (<w, [z; onehot(y)]> + b_y) * (2n - 1): the T1 linear class with the class entering explicitly."""
    def __init__(self, d: int, n_classes: int):
        super().__init__(); self.w = nn.Linear(d + n_classes, 1); self.n_classes = n_classes

    def forward(self, z, n, y):
        u = torch.cat((z, F.one_hot(y, self.n_classes).float()), 1)
        return self.w(u).squeeze(1) * (2 * n.float() - 1)


class YPairMLP(nn.Module):
    """The T1 MLP class (hidden 128, two ReLU layers) on [z; onehot(n); onehot(y)]."""
    def __init__(self, d: int, n_classes: int, hidden: int = 128):
        super().__init__(); self.n_classes = n_classes
        self.net = nn.Sequential(nn.Linear(d + 2 + n_classes, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, n, y):
        return self.net(torch.cat((z, F.one_hot(n, 2).float(), F.one_hot(y, self.n_classes).float()), 1)).squeeze(1)


def _j_torch(fp, fq):
    tp, tq = torch.tanh(fp), torch.tanh(fq)
    return (tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean()


def _js_loss(fp, fq):
    """Matched JS (Spec v2 §2.2): posterior q = sigmoid(2 f), balanced 1:1, P and Q averaged separately; same f -> T = tanh(f)."""
    return F.softplus(-2.0 * fp).mean() + F.softplus(2.0 * fq).mean()


def fit_critic(z, n, n_neg, y, *, objective: str, kind: str, n_classes: int, steps: int = 300, seed: int = 0, lr: float = 1e-3,
               wd: float = 1e-2, val_frac: float = 0.2):
    """Fit on 80 % of FIT, select the step on the 20 % by the COMMON squared score J(tanh f) (Spec §11.1: magnitude critics are selected by
    independent regression risk, for VCS and JS alike).  Returns the fixed critic with .val_J and .best_step."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g); nv = max(8, int(val_frac * len(z))); vi, ti = perm[:nv].to(z.device), perm[nv:].to(z.device)
    m = (YPairLinear(z.shape[1], n_classes) if kind == "linear" else YPairMLP(z.shape[1], n_classes)).to(z.device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, best_step, best_state = -float("inf"), 0, {k: v.clone() for k, v in m.state_dict().items()}
    for step in range(1, steps + 1):
        opt.zero_grad()
        fp, fq = m(z[ti], n[ti], y[ti]), m(z[ti], n_neg[ti], y[ti])
        loss = -_j_torch(fp, fq) if objective == "vcs" else _js_loss(fp, fq)
        loss.backward(); opt.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                jv = float(_j_torch(m(z[vi], n[vi], y[vi]), m(z[vi], n_neg[vi], y[vi])))
            if jv > best:
                best, best_step, best_state = jv, step, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval(); m.val_J = best; m.best_step = best_step
    return m


def fit_picked(z, n, n_neg, y, *, objective: str, n_classes: int, steps: int, seed: int):
    """Linear and MLP classes, the one with the higher selection-set J is kept (T1's VAL pick, with the common score)."""
    cands = {k: fit_critic(z, n, n_neg, y, objective=objective, kind=k, n_classes=n_classes, steps=steps, seed=seed) for k in ("linear", "mlp")}
    pick = max(cands, key=lambda k: cands[k].val_J)
    return cands[pick], pick, {k: float(v.val_J) for k, v in cands.items()}
