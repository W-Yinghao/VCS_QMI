"""P156 (SEP-1) — where on the same posterior the quadratic and the logistic scoring rule separate (P134 extended).

Protocol: reports/P156_SEP_SEPARATION_AXIS_PROTOCOL_DRAFT_20261014.md (rev. 1; frozen copy *_FROZEN_* once the gate passes).

Arms = four members of the symmetric Beta family of proper scoring rules (Buja, Stuetzle & Shen 2005, a = b = gamma), all with the
population minimiser T* = eta = (p - q) / (p + q):
  js   gamma 0    logistic, the P85 Deep-InfoMax form on f (T = tanh(f / 2)): softplus(-f) on P, softplus(f) on Q
  b05  gamma 1/2  T = tanh g: -sech g - c gd(g)                  (gd = Gudermannian = 2 atan tanh(g / 2))
  vcs  gamma 1    T = tanh g: the P85 VCS form -J = -(E_P[T - T^2/2] + E_Q[-T - T^2/2])
  b2   gamma 2    T = tanh g: T^2/2 - T^4/4 - c (T - T^3/3)
d loss / d g is proportional to (1 - T^2)^gamma (T - c), c = +1 on P, -1 on Q (tests/test_p156.py).

Critic classes: mlp256 (the P85 JointMLP), mlp8 (the same with 8 hidden units), cos2 (f = a cos(x, y) + b, the application critic form),
quad3 (f = a x.y / d + b (|x|^2 + |y|^2) / d + c, which contains the clean Gaussian log-ratio exactly).
Contamination: clean; 'independent' (P_eps = (1 - eps) P + eps Q; P134 sampler via r1); 'false_negative' (Q_eps = (1 - eps) Q + eps P: eps of the
Q pairs of every role replaced by fresh joint pairs; under cyclic negatives each batch's cyclic negatives get exactly round(eps * 8 B) fresh
conditional draws y ~ P(y | x_anchor)); 'outlier' / 'heavy' (P134 laws, r1, rescaled to d); 'matched' (block B: a clean Gaussian whose rho'
gives S(rho') = S_eps of a block-A cell, from the committed matching table).
P85 / P134 modules are imported, never edited.  Synthetic data only; no image data, no official test set.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .benchmark import Role, _blocks, _dev, build_roles, roles_hash
from .critics import JointMLP
from .data import Gaussian, mi_gaussian, pmi_gaussian, rho_from_mi
from .estimators import scores
from .r1 import HIGH_MI, ContamSetting, Contamination, contaminate_roles
from .run import git_commit
from .synthetic import _gen

PROTOCOL = "P156"
ARMS = {"js": 0.0, "b05": 0.5, "vcs": 1.0, "b2": 2.0}
ARMS_ALL = ("js", "b05", "vcs", "b2")
ARMS_PAIR = ("js", "vcs")
LRS_MLP = (3e-5, 1e-4, 5e-4, 2e-3, 8e-3)
LRS_PAR = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
P85_LRS = (1e-4, 5e-4, 2e-3)
SEEDS = tuple(range(100, 108))
EPS_A = (0.05, 0.1, 0.2, 0.3)
EPS_B = (0.1, 0.3)
DS, IS = (20, 64), (2.0, 4.0, 8.0)
B_DEFAULT = 256
EVAL_BLOCK = 1024
MATCH_N = 200_000
MATCH_TABLE = Path(__file__).resolve().parents[2] / "reports" / "P156_S_MATCHING.json"
CTYPES = ("clean", "independent", "false_negative", "outlier", "heavy", "matched")


# ------------------------------------------------------------------------------------------------------------ scoring rules
def arm_T(arm: str, f: torch.Tensor) -> torch.Tensor:
    """Bounded posterior read-out of a raw critic output f."""
    return torch.tanh(0.5 * f) if arm == "js" else torch.tanh(f)


def _sech(g: torch.Tensor) -> torch.Tensor:
    a = torch.exp(-g.abs()); return 2.0 * a / (1.0 + a * a)                 # overflow-free 1 / cosh g


def _gd(g: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.atan(torch.tanh(0.5 * g))


def pair_loss(arm: str, f: torch.Tensor, c: float) -> torch.Tensor:
    """Per-pair loss of one arm (c = +1 for a P pair, -1 for a Q pair); additive constants are irrelevant."""
    if arm == "js":
        return F.softplus(-c * f)
    if arm == "vcs":
        T = torch.tanh(f); return 0.5 * (T - c) ** 2
    if arm == "b05":
        return -_sech(f) - c * _gd(f)
    if arm == "b2":
        T = torch.tanh(f); return T * T / 2 - T ** 4 / 4 - c * (T - T ** 3 / 3)
    raise ValueError(arm)


def arm_loss(arm: str, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """Balanced loss: separate means over the positive set and the negative set (the P85 convention).  For js and vcs this is exactly the
    P85 `estimate` loss (js: softplus(-fp).mean() + softplus(fn).mean(); vcs: -J)."""
    if arm == "vcs":
        tp, tq = torch.tanh(fp), torch.tanh(fn)
        return -((tp - tp * tp / 2).mean() + (-tq - tq * tq / 2).mean())
    if arm == "js":
        return F.softplus(-fp).mean() + F.softplus(fn).mean()
    return pair_loss(arm, fp, 1.0).mean() + pair_loss(arm, fn, -1.0).mean()


def common_J(tp: torch.Tensor, tq: torch.Tensor) -> float:
    return float((tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean())


# ------------------------------------------------------------------------------------------------------------ critic classes
class _Pairwise(nn.Module):
    def shifted(self, x, y, shifts):
        return torch.stack([self.pairs(x, torch.roll(y, s, 0)) for s in shifts], 1)

    def matrix(self, x, y):
        n = len(x); return self.pairs(x.repeat_interleave(n, 0), y.repeat(n, 1)).view(n, n)


class Cos2(_Pairwise):
    """f = a cos(x, y) + b: the application critic (SSL's fixed scorer, VL's a cos + b); misspecified for the Gaussian."""
    def __init__(self):
        super().__init__(); self.a = nn.Parameter(torch.zeros(())); self.b = nn.Parameter(torch.zeros(()))

    def pairs(self, x, y):
        c = (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-12); return self.a * c + self.b


class Quad3(_Pairwise):
    """f = a x.y / d + b (|x|^2 + |y|^2) / d + c: contains the clean Gaussian log-ratio (and its half) exactly."""
    def __init__(self, d: int):
        super().__init__(); self.d = d
        self.a = nn.Parameter(torch.zeros(())); self.b = nn.Parameter(torch.zeros(())); self.c = nn.Parameter(torch.zeros(()))

    def pairs(self, x, y):
        return self.a * (x * y).sum(-1) / self.d + self.b * ((x * x).sum(-1) + (y * y).sum(-1)) / self.d + self.c


def make_critic(cls: str, d: int, seed: int) -> nn.Module:
    torch.manual_seed(seed * 7919 + 17)                                       # the P85 initialisation stream
    if cls == "mlp256":
        return JointMLP(d, d)
    if cls == "mlp8":
        return JointMLP(d, d, hidden=8)
    if cls == "cos2":
        return Cos2()
    if cls == "quad3":
        return Quad3(d)
    raise ValueError(cls)


def quad3_oracle(rho: float, d: int, arm: str) -> dict:
    """Quad3 parameters that represent the clean optimum: f* = log r / 2 (T = tanh f) or f* = log r (js, T = tanh(f / 2))."""
    s = 1.0 if arm == "js" else 0.5; r2 = rho * rho
    return {"a": s * d * rho / (1 - r2), "b": -s * d * r2 / (2 * (1 - r2)), "c": -s * d / 2 * math.log(1 - r2)}


# ------------------------------------------------------------------------------------------------------------ target algebra
def inv_map(ctype: str, t: torch.Tensor, eps: float) -> torch.Tensor:
    """Known-eps inverse of the contaminated posterior map (oracle diagnostic; docs/03 §7 and its mirror)."""
    if ctype == "independent":
        return t / (1 - eps + eps * t)
    if ctype == "false_negative":
        return t / (1 - eps - eps * t)
    raise ValueError(ctype)


def fwd_map(ctype: str, eta: torch.Tensor, eps: float) -> torch.Tensor:
    if ctype == "independent":
        return (1 - eps) * eta / (1 - eps * eta)
    if ctype == "false_negative":
        return (1 - eps) * eta / (1 + eps * eta)
    raise ValueError(ctype)


def amplification(eps: float) -> float:
    return (1 + eps) ** 2 / (1 - eps)


def log_ratio_fn(pmi: torch.Tensor, eps: float) -> torch.Tensor:
    """log dP / dQ_eps for Q_eps = (1 - eps) Q + eps P:  -log((1 - eps) e^{-pmi} + eps)."""
    if eps == 0.0:
        return pmi
    return -torch.logaddexp(math.log1p(-eps) - pmi, torch.full_like(pmi, math.log(eps)))


def log_ratio_indep(pmi: torch.Tensor, eps: float) -> torch.Tensor:
    """log dP_eps / dQ for P_eps = (1 - eps) P + eps Q (marginals unchanged):  log((1 - eps) e^{pmi} + eps)."""
    if eps == 0.0:
        return pmi
    return torch.logaddexp(math.log1p(-eps) + pmi, torch.full_like(pmi, math.log(eps)))


def truth_from_logr(lp: torch.Tensor, lq: torch.Tensor) -> dict:
    """S, J_oracle and the saturation covariates of a problem from log r on its TRUTH P / Q samples (M = (P + Q) / 2)."""
    lp, lq = lp.double(), lq.double(); ep, eq = torch.tanh(lp / 2), torch.tanh(lq / 2); n, m = len(ep), len(eq)
    sp, sq = ep ** 2, eq ** 2; S = float(0.5 * sp.mean() + 0.5 * sq.mean())
    jp, jq = ep - 0.5 * ep ** 2, -eq - 0.5 * eq ** 2
    mass = lambda fn: float(0.5 * fn(ep).double().mean() + 0.5 * fn(eq).double().mean())
    em_eta = float(0.5 * ep.mean() + 0.5 * eq.mean()); em_eta_se = math.sqrt(0.25 * float(ep.var()) / n + 0.25 * float(eq.var()) / m)
    return {"S": S, "S_se": math.sqrt(0.25 * float(sp.var()) / n + 0.25 * float(sq.var()) / m), "J_oracle": float(jp.mean() + jq.mean()),
            "sat95": mass(lambda e: e.abs() > 0.95), "sat95_P_side": mass(lambda e: e > 0.95), "sat95_Q_side": mass(lambda e: e < -0.95),
            "sat90": mass(lambda e: e.abs() > 0.90), "qc_EM_eta": em_eta, "qc_EM_eta_se": em_eta_se, "qc_EM_eta_ok": abs(em_eta) <= 4 * em_eta_se,
            "n_truth_per_distribution": n}


# ------------------------------------------------------------------------------------------------------------ problems
class Problem:
    """One data law: clean Gaussian (rho), plus a contamination.  `log_ratio(x, y)` = log dP' / dQ' of the actual problem; `clean_pmi` the
    clean Gaussian log-ratio (for the clean-target diagnostic and the region bins)."""
    def __init__(self, ctype: str, eps: float, d: int, rho: float, device=None):
        if ctype not in CTYPES:
            raise ValueError(ctype)
        self.ctype, self.eps, self.d, self.rho, self.device = ctype, float(eps), d, float(rho), device
        self.r1 = None
        if ctype in ("independent", "outlier", "heavy") and eps > 0:
            self.r1 = Contamination(ctype, eps, rho, rho_from_mi(HIGH_MI, d), d)

    def clean_pmi(self, x, y):
        return pmi_gaussian(x.double(), y.double(), self.rho)

    def log_ratio(self, x, y, chunk: int = 65536):
        if self.eps == 0.0 or self.ctype in ("clean", "matched"):
            return self.clean_pmi(x, y)
        if self.ctype == "false_negative":
            return log_ratio_fn(self.clean_pmi(x, y), self.eps)
        return ContamSetting(self.r1, self.device, chunk).pmi(x, y).double()


def contaminate_fn_roles(roles: dict, eps: float, rho: float, seed: int, skip=("GRAD",)) -> tuple[dict, dict]:
    """False-negative type: exactly round(eps n) Q pairs of every role replaced by fresh joint pairs (x ~ N(0, I), y = rho x + sqrt(1 - rho^2) e);
    marginals unchanged; the P side untouched.  Indices recorded with a SHA-256 per role."""
    out, recs = {}, {}
    for name, r in roles.items():
        if name in skip or eps == 0.0:
            out[name] = r; continue
        n = r.n; k = int(round(eps * n)); gi, gn = _gen(("P156-FN-IDX", eps, name, seed)), _gen(("P156-FN-NOISE", eps, name, seed))
        iq = torch.randperm(n, generator=gi)[:k]; d = r.xq.shape[1]
        x = torch.randn(k, d, generator=gn, dtype=torch.float64); e = torch.randn(k, d, generator=gn, dtype=torch.float64)
        xq, yq = r.xq.clone(), r.yq.clone(); xq[iq] = x.to(xq.dtype); yq[iq] = (rho * x + math.sqrt(1 - rho * rho) * e).to(yq.dtype)
        out[name] = Role(r.xp, r.yp, xq, yq, r.base, r.pad_p)
        idx = sorted(iq.tolist())
        recs[name] = {"n": n, "k": k, "idx_sha256": hashlib.sha256(repr(idx).encode()).hexdigest(), "q_idx_head": idx[:16]}
    return out, recs


def inject_cyclic_fn(critic, x: torch.Tensor, fn: torch.Tensor, eps: float, rho: float, gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """Cyclic negatives with false negatives: exactly round(eps * n * K) of the n x K cyclic negatives are replaced by (x_anchor, y'),
    y' ~ P(y | x_anchor) fresh.  Each negative is replaced with the same probability, so the per-pair negative law is Q_eps."""
    n, K = fn.shape; k = int(round(eps * n * K))
    if k == 0:
        return fn, 0
    pos = torch.randperm(n * K, generator=gen)[:k]; rows = (pos // K).to(x.device); cols = (pos % K).to(x.device)
    xs = x[rows]; e = torch.randn(xs.shape, generator=gen, dtype=torch.float64).to(xs.dtype).to(x.device)
    ys = rho * xs + math.sqrt(1 - rho * rho) * e
    fn = fn.clone(); fn[rows, cols] = critic.pairs(xs, ys)
    return fn, k


# ------------------------------------------------------------------------------------------------------------ training
def _forward_role(critic, xs, ys, device) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([critic.pairs(_dev(xs[s:e], device), _dev(ys[s:e], device)).double().cpu() for s, e in _blocks(len(xs), 65536)])


def select_scores(critic, arm: str, variant: str, role: Role, device, fn_cfg, gen) -> tuple[float, float]:
    """(native risk, common quadratic J) of a fixed critic on a role.  Product: the role's P / Q samples; cyclic8: cyclic negatives inside
    EVAL_BLOCK blocks (blocks with <= 8 rows skipped, as P85), with the false-negative injection when fn_cfg is set."""
    with torch.no_grad():
        if variant == "product":
            fp, fn = _forward_role(critic, role.xp, role.yp, device), _forward_role(critic, role.xq, role.yq, device)
        else:
            P, N = [], []
            for s, e in _blocks(role.n, EVAL_BLOCK):
                if e - s <= 8:
                    continue
                x, y = _dev(role.xp[s:e], device), _dev(role.yp[s:e], device)
                a, b = scores(critic, x, y, variant, gen)
                if fn_cfg:
                    b, _ = inject_cyclic_fn(critic, x, b, fn_cfg["eps"], fn_cfg["rho"], gen)
                P.append(a.double().cpu()); N.append(b.double().cpu().reshape(-1))
            fp, fn = torch.cat(P), torch.cat(N)
        risk = float(arm_loss(arm, fp, fn)); J = common_J(arm_T(arm, fp), arm_T(arm, fn))
    return (risk if math.isfinite(risk) else float("inf")), (J if math.isfinite(J) else float("-inf"))


def train_arm(arm, cls, variant, lr, roles, d, B, updates, seed, device, fn_cfg=None, every: int = 100):
    """The P85 trainer (same seeds, minibatch stream, selection cadence) with the arm's loss and the critic class as parameters.  Keeps the
    state with the lowest native SELECT risk (primary) and, separately, the one with the highest common quadratic J on SELECT (sensitivity)."""
    critic = make_critic(cls, d, seed).to(device); opt = torch.optim.Adam(critic.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed * 1000 + 7); sg = torch.Generator().manual_seed(seed * 1000 + 11)
    ig = torch.Generator().manual_seed(seed * 1000 + 13)
    Fr, Sr = roles["FIT"], roles["SELECT"]; Fx, Fy, Fqx, Fqy = (_dev(t, device) for t in (Fr.xp, Fr.yp, Fr.xq, Fr.yq))
    n = Fr.n; nonfinite = 0; injected = 0; t0 = time.time()

    def snap():
        return {k: v.detach().clone() for k, v in critic.state_dict().items()}

    def sel():
        critic.eval(); sgl = torch.Generator().manual_seed(seed * 1000 + 11) if fn_cfg else sg
        r = select_scores(critic, arm, variant, Sr, device, fn_cfg, sgl); critic.train(); return r

    r0, j0 = sel(); curve = [(0, r0, j0)]; best_n = (r0, 0, snap()); best_j = (j0, 0, snap())
    for step in range(1, updates + 1):
        ip = torch.randint(0, n, (B,), generator=g).to(device)
        if variant == "product":
            iq = torch.randint(0, n, (B,), generator=g).to(device)
            fp = critic.pairs(Fx[ip], Fy[ip]); fn = critic.pairs(Fqx[iq], Fqy[iq]).unsqueeze(1)
        else:
            fp, fn = scores(critic, Fx[ip], Fy[ip], variant, g)
            if fn_cfg:
                fn, k = inject_cyclic_fn(critic, Fx[ip], fn, fn_cfg["eps"], fn_cfg["rho"], ig); injected += k
        loss = arm_loss(arm, fp, fn)
        if not torch.isfinite(loss):
            nonfinite += 1; opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % every == 0:
            r, j = sel(); curve.append((step, r, j))
            if r < best_n[0]:
                best_n = (r, step, snap())
            if j > best_j[0]:
                best_j = (j, step, snap())
    critic.load_state_dict(best_n[2]); critic.eval()
    info = {"select_risk": best_n[0], "selected_update": best_n[1], "select_curve": curve, "commonJ_select": best_j[0],
            "commonJ_selected_update": best_j[1], "fit_seconds": time.time() - t0, "nonfinite_steps": nonfinite, "updates": updates, "lr": lr,
            "batch": B, "n_params": sum(p.numel() for p in critic.parameters()), "cyclic_fn_injected": injected,
            "not_converged": best_n[1] == updates, "commonJ_not_converged": best_j[1] == updates}
    return critic, best_j[2], info


# ------------------------------------------------------------------------------------------------------------ read-outs
def readouts(critic, arm: str, E: Role, eta_p, eta_q, cpmi_p, cpmi_q, S: float, device, clean_E: Role | None, prob: Problem) -> dict:
    """Posterior MSE against the problem's posterior on EVAL (primary), J_eval - S, the logistic regret, clean-eta region contributions and
    (types independent / false_negative) the known-eps clean-target diagnostic on the clean EVAL role."""
    tp = arm_T(arm, _forward_role(critic, E.xp, E.yp, device)); tq = arm_T(arm, _forward_role(critic, E.xq, E.yq, device))
    rp, rq = (tp - eta_p) ** 2, (tq - eta_q) ** 2; n, m = len(rp), len(rq)
    e = float(0.5 * rp.mean() + 0.5 * rq.mean()); e_se = math.sqrt(0.25 * float(rp.var()) / n + 0.25 * float(rq.var()) / m)
    J = common_J(tp, tq)

    def kl(eta, t):
        a = ((1 + eta) / 2).clamp(1e-12, 1 - 1e-12); b = ((1 + t) / 2).clamp(1e-12, 1 - 1e-12)
        return a * torch.log(a / b) + (1 - a) * torch.log((1 - a) / (1 - b))
    regret = float(0.5 * kl(eta_p, tp).mean() + 0.5 * kl(eta_q, tq).mean())
    ce_p, ce_q = torch.tanh(cpmi_p / 2), torch.tanh(cpmi_q / 2); regions = {}
    for name, fn in (("clean_eta_le_-0.5", lambda c: c <= -0.5), ("clean_eta_abs_lt_0.5", lambda c: c.abs() < 0.5), ("clean_eta_ge_0.5", lambda c: c >= 0.5)):
        mp, mq = fn(ce_p).double(), fn(ce_q).double()
        regions[name] = {"contribution": float(0.5 * (rp * mp).mean() + 0.5 * (rq * mq).mean()), "mass": float(0.5 * mp.mean() + 0.5 * mq.mean())}
    out = {"posterior_mse": e, "posterior_mse_se": e_se, "J_eval": J, "J_eval_minus_S": J - S, "logistic_regret": regret, "regions": regions,
           "gate_mean": float(0.5 * (1 - tp ** 2).mean() + 0.5 * (1 - tq ** 2).mean()), "abs_T_gt_0.95": float(0.5 * (tp.abs() > 0.95).double().mean() + 0.5 * (tq.abs() > 0.95).double().mean())}
    if clean_E is not None and prob.ctype in ("independent", "false_negative") and prob.eps > 0:
        cp = arm_T(arm, _forward_role(critic, clean_E.xp, clean_E.yp, device)); cq = arm_T(arm, _forward_role(critic, clean_E.xq, clean_E.yq, device))
        hp, hq = inv_map(prob.ctype, cp, prob.eps), inv_map(prob.ctype, cq, prob.eps)
        clipped = float(0.5 * (hp.abs() > 1).double().mean() + 0.5 * (hq.abs() > 1).double().mean())
        hp, hq = hp.clamp(-1, 1), hq.clamp(-1, 1)
        ep_c, eq_c = torch.tanh(prob.clean_pmi(clean_E.xp, clean_E.yp) / 2), torch.tanh(prob.clean_pmi(clean_E.xq, clean_E.yq) / 2)
        out["clean_target"] = {"posterior_mse_clean": float(0.5 * ((hp - ep_c) ** 2).mean() + 0.5 * ((hq - eq_c) ** 2).mean()),
                               "clipped_fraction": clipped, "amplification_max_slope": amplification(prob.eps), "note": "oracle eps; diagnostic only"}
    return out


# ------------------------------------------------------------------------------------------------------------ S matching (block B)
def _crn_clean(d: int, n: int):
    g = _gen(("P156-SMATCH-CRN", d, n)); x = torch.randn(n, d, generator=g, dtype=torch.float64); e = torch.randn(n, d, generator=g, dtype=torch.float64)
    xq = torch.randn(n, d, generator=g, dtype=torch.float64); yq = torch.randn(n, d, generator=g, dtype=torch.float64)
    return x, e, xq, yq


def s_clean(rho: float, d: int, n: int = MATCH_N, crn=None) -> dict:
    x, e, xq, yq = crn or _crn_clean(d, n); y = rho * x + math.sqrt(1 - rho * rho) * e
    return truth_from_logr(pmi_gaussian(x, y, rho), pmi_gaussian(xq, yq, rho))


def s_target(ctype: str, eps: float, d: int, I: float, n: int = MATCH_N) -> dict:
    """Seed-independent Monte Carlo truth of a contaminated block-A problem (dedicated stream)."""
    rho = rho_from_mi(I, d); g = _gen(("P156-STARGET", ctype, eps, d, I, n))
    x = torch.randn(n, d, generator=g, dtype=torch.float64); y = rho * x + math.sqrt(1 - rho * rho) * torch.randn(n, d, generator=g, dtype=torch.float64)
    xq = torch.randn(n, d, generator=g, dtype=torch.float64); yq = torch.randn(n, d, generator=g, dtype=torch.float64)
    k = int(round(eps * n)); idx = torch.randperm(n, generator=g)[:k]
    if ctype == "independent":
        y[idx] = torch.randn(k, d, generator=g, dtype=torch.float64)
        return truth_from_logr(log_ratio_indep(pmi_gaussian(x, y, rho), eps), log_ratio_indep(pmi_gaussian(xq, yq, rho), eps))
    if ctype == "false_negative":
        xn = torch.randn(k, d, generator=g, dtype=torch.float64); xq[idx] = xn
        yq[idx] = rho * xn + math.sqrt(1 - rho * rho) * torch.randn(k, d, generator=g, dtype=torch.float64)
        return truth_from_logr(log_ratio_fn(pmi_gaussian(x, y, rho), eps), log_ratio_fn(pmi_gaussian(xq, yq, rho), eps))
    raise ValueError(ctype)


def match_rho(S_goal: float, d: int, n: int = MATCH_N, tol: float = 1e-6) -> tuple[float, dict]:
    """Bisection for rho' with S_clean(rho'; d) = S_goal on common random numbers (S_clean is then monotone in rho)."""
    crn = _crn_clean(d, n); lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if s_clean(mid, d, n, crn)["S"] < S_goal:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    rho = 0.5 * (lo + hi); return rho, s_clean(rho, d, n, crn)


def match_rho_sat(sat_goal: float, d: int, n: int = MATCH_N, tol: float = 1e-6) -> tuple[float, dict]:
    """Bisection for rho'' with sat95_clean(rho''; d) = sat_goal (the saturated mass P_M(|eta| > 0.95) is non-decreasing in rho on common
    random numbers).  Block B2: the clean control matched on saturation instead of S (S-matching leaves the saturated mass unmatched by a factor
    of about 2 in both directions; see the table)."""
    crn = _crn_clean(d, n); lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if s_clean(mid, d, n, crn)["sat95"] < sat_goal:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    rho = 0.5 * (lo + hi); return rho, s_clean(rho, d, n, crn)


def matching_table(n: int = MATCH_N) -> list[dict]:
    rows = []
    for ctype in ("independent", "false_negative"):
        for eps in EPS_B:
            for d in DS:
                for I in IS:
                    t = s_target(ctype, eps, d, I, n); rho, c = match_rho(t["S"], d, n); rs, cs = match_rho_sat(t["sat95"], d, n)
                    rows.append({"ctype": ctype, "eps": eps, "d": d, "I": I, "S_eps": t["S"], "S_eps_se": t["S_se"], "rho_prime": rho,
                                 "rho_sat": rs, "I_sat": mi_gaussian(rs, d), "S_clean_at_rho_sat": cs["S"], "sat95_at_rho_sat": cs["sat95"],
                                 "I_prime": mi_gaussian(rho, d), "S_clean_at_rho_prime": c["S"], "S_clean_se": c["S_se"],
                                 "sat95_eps": t["sat95"], "sat95_P_side_eps": t["sat95_P_side"], "sat95_Q_side_eps": t["sat95_Q_side"],
                                 "sat95_matched": c["sat95"], "sat95_P_side_matched": c["sat95_P_side"], "sat95_Q_side_matched": c["sat95_Q_side"],
                                 "n_mc": n})
    return rows


def load_matching(path: Path = MATCH_TABLE) -> dict:
    T = json.loads(Path(path).read_text()); return {(r["ctype"], r["eps"], r["d"], r["I"]): r for r in T["rows"]}


# ------------------------------------------------------------------------------------------------------------ cells
def cell(block, cls, variant, ctype, eps, d, I, N, U, arms, lrs, seed, match=None) -> dict:
    c = {"block": block, "cls": cls, "variant": variant, "ctype": ctype, "eps": float(eps), "d": int(d), "I": float(I), "N": int(N),
         "updates": int(U), "B": B_DEFAULT, "arms": list(arms), "lrs": list(lrs), "seed": int(seed)}
    if match:
        c["match"] = {"ctype": match[0], "eps": float(match[1]), "by": match[2] if len(match) > 2 else "S"}
    return c


def cell_name(c: dict) -> str:
    t = f"_{c['tag']}" if c.get("tag") else ""
    m = (f"_m{c['match']['ctype'][:3]}{c['match']['eps']:g}" + ("sat" if c["match"].get("by") == "sat95" else "")) if c.get("match") else ""
    return (f"P156_{c['block']}_{c['cls']}_{c['variant']}_{c['ctype']}{m}_e{c['eps']:g}_d{c['d']}_I{c['I']:g}_N{c['N']}_U{c['updates']}"
            f"_s{c['seed']}{t}")


def cells_for_seed(seed: int) -> list[dict]:
    C = []
    for d in DS:                                                            # block A
        for I in IS:
            C.append(cell("A", "mlp256", "product", "clean", 0.0, d, I, 4096, 2000, ARMS_ALL, LRS_MLP, seed))
            for ct in ("independent", "false_negative"):
                for e in EPS_A:
                    C.append(cell("A", "mlp256", "product", ct, e, d, I, 4096, 2000, ARMS_ALL, LRS_MLP, seed))
    for cls in ("cos2", "quad3", "mlp8"):                                   # block E (capacity)
        for d in DS:
            for I in (4.0, 8.0):
                for e in (0.0, 0.1, 0.3):
                    C.append(cell("E", cls, "product", "clean" if e == 0 else "independent", e, d, I, 4096, 2000, ARMS_ALL,
                                  LRS_MLP if cls == "mlp8" else LRS_PAR, seed))
    for e in (0.0, 0.1, 0.2, 0.3):                                          # block C (budget / convergence)
        for N, U in ((4096, 8000), (16384, 2000), (16384, 8000)):
            C.append(cell("C", "mlp256", "product", "clean" if e == 0 else "independent", e, 20, 4.0, N, U, ARMS_PAIR, LRS_MLP, seed))
    for e in (0.0, 0.3):
        C.append(cell("C", "mlp256", "product", "clean" if e == 0 else "independent", e, 64, 8.0, 4096, 8000, ARMS_PAIR, LRS_MLP, seed))
    for ct in ("outlier", "heavy"):                                         # block D (secondary types)
        for e in (0.1, 0.2, 0.3):
            for d in DS:
                C.append(cell("D", "mlp256", "product", ct, e, d, 4.0, 4096, 2000, ARMS_PAIR, LRS_MLP, seed))
    C.append(cell("Dp", "mlp256", "cyclic8", "clean", 0.0, 20, 4.0, 4096, 2000, ARMS_PAIR, LRS_MLP, seed))   # block D' (cyclic negatives)
    for ct in ("independent", "false_negative"):
        for e in (0.1, 0.2, 0.3):
            C.append(cell("Dp", "mlp256", "cyclic8", ct, e, 20, 4.0, 4096, 2000, ARMS_PAIR, LRS_MLP, seed))
    for ct in ("independent", "false_negative"):                            # block B (S-matched clean controls) last: needs the table
        for e in EPS_B:
            for d in DS:
                for I in IS:
                    C.append(cell("B", "mlp256", "product", "matched", 0.0, d, I, 4096, 2000, ARMS_ALL, LRS_MLP, seed, match=(ct, e)))
    for ct in ("independent", "false_negative"):                            # block B2 (saturation-matched clean controls, eps 0.3; secondary)
        for d in DS:
            for I in IS:
                C.append(cell("B2", "mlp256", "product", "matched", 0.0, d, I, 4096, 2000, ARMS_ALL, LRS_MLP, seed, match=(ct, 0.3, "sat95")))
    return C


LR_EXT_PAR = 0.3                                                            # addendum 1: one point beyond the high edge of LRS_PAR


def addendum1_cells(seed: int) -> list[dict]:
    """Addendum 1 (edge rule, protocol §3): every block-E cos2 / quad3 cell of a seed, all four arms, at lr 0.3 only (tag 'lrext')."""
    out = []
    for c in cells_for_seed(seed):
        if c["block"] == "E" and c["cls"] in ("cos2", "quad3"):
            out.append({**c, "lrs": [LR_EXT_PAR], "tag": "lrext"})
    return out


def merge_extension(orig: dict, ext: dict) -> dict:
    """Original cell + its lr-extension rows -> one cell whose selections are re-made over the extended grid (native risk primary,
    common J sensitivity), with the lr-edge flags recomputed on the extended grid."""
    rows = [dict(r) for r in orig["rows"]] + [dict(r) for r in ext["rows"]]; grid = sorted({r["lr"] for r in rows})
    for arm in {r["arm"] for r in rows}:
        g = [r for r in rows if r["arm"] == arm]
        for r in g:
            r["selected"] = False; r["selected_commonJ"] = False
            r["lr_edge"] = "low" if r["lr"] == grid[0] else "high" if r["lr"] == grid[-1] else None
        fin = lambda v, bad: v if math.isfinite(v) else bad                 # a diverged extension point is never selected (identical to run_cell when finite)
        min(g, key=lambda r: fin(r["select"]["select_risk"], math.inf))["selected"] = True
        max(g, key=lambda r: fin(r["select"]["commonJ_select"], -math.inf))["selected_commonJ"] = True
    cellrec = {**orig["cell"], "lrs_run": grid, "addendum1_extension_file": ext["cell"]["name"]}
    return {"cell": cellrec, "truth": orig["truth"], "rows": rows, "wall_seconds": orig["wall_seconds"] + ext["wall_seconds"]}


def reproduction_cell() -> dict:
    """QC: the P134 clean cell (seed 0, P85 grid, VCS / JS product) must match P134's selected posterior MSE (GPU non-determinism only)."""
    return cell("R", "mlp256", "product", "clean", 0.0, 20, 4.0, 4096, 2000, ARMS_PAIR, P85_LRS, 0)


def run_cell(c: dict, device, smoke: bool = False, matching: dict | None = None) -> dict:
    t_all = time.time(); d = c["d"]; seed = c["seed"]
    if c["ctype"] == "matched":
        m = (matching or load_matching())[(c["match"]["ctype"], c["match"]["eps"], d, c["I"])]
        rho = m["rho_sat"] if c["match"].get("by") == "sat95" else m["rho_prime"]; mi = mi_gaussian(rho, d)
        match_info = m
    else:
        rho = rho_from_mi(c["I"], d); mi = c["I"]; match_info = None
    setting = Gaussian("gaussian", d, -1, rho, mi)
    N, U = (256, 40) if smoke else (c["N"], c["updates"]); lrs = c["lrs"][:1] if smoke else c["lrs"]; every = 20 if smoke else 100
    roles, sizes = build_roles(setting, d, d, N, seed, smoke); clean_hash = roles_hash(roles); clean_roles = roles
    prob = Problem(c["ctype"], c["eps"], d, rho, device); recs = {}
    if c["ctype"] in ("independent", "outlier", "heavy") and c["eps"] > 0:
        roles, recs = contaminate_roles(roles, prob.r1, seed)
    elif c["ctype"] == "false_negative" and c["eps"] > 0 and c["variant"] == "product":
        roles, recs = contaminate_fn_roles(roles, c["eps"], rho, seed)
    elif c["ctype"] == "false_negative" and c["eps"] > 0:                 # cyclic: training / SELECT inject; EVAL / TRUTH need Q_eps samples
        cont, recs = contaminate_fn_roles(roles, c["eps"], rho, seed, skip=("GRAD", "FIT", "TUNE", "SELECT")); roles = cont
    T = roles["TRUTH"]; tr = truth_from_logr(prob.log_ratio(T.xp, T.yp), prob.log_ratio(T.xq, T.yq))
    E = roles["EVAL"]; eta_p, eta_q = torch.tanh(prob.log_ratio(E.xp, E.yp) / 2), torch.tanh(prob.log_ratio(E.xq, E.yq) / 2)
    cpmi_p, cpmi_q = prob.clean_pmi(E.xp, E.yp), prob.clean_pmi(E.xq, E.yq)
    fn_cfg = {"eps": c["eps"], "rho": rho} if (c["ctype"] == "false_negative" and c["eps"] > 0 and c["variant"] != "product") else None
    rows = []
    for arm in c["arms"]:
        group = []
        for lr in lrs:
            critic, stateJ, info = train_arm(arm, c["cls"], c["variant"], lr, roles, d, c["B"], U, seed, device, fn_cfg, every)
            ro = readouts(critic, arm, E, eta_p, eta_q, cpmi_p, cpmi_q, tr["S"], device, clean_roles["EVAL"], prob)
            critic.load_state_dict(stateJ); critic.eval()
            roJ = readouts(critic, arm, E, eta_p, eta_q, cpmi_p, cpmi_q, tr["S"], device, None, prob)
            group.append({"arm": arm, "gamma": ARMS[arm], "lr": lr, "select": {k: v for k, v in info.items() if k != "select_curve"},
                          "select_curve": info["select_curve"], "readout": ro, "readout_commonJ_selected": {k: roJ[k] for k in ("posterior_mse", "J_eval_minus_S")},
                          "selected": False, "selected_commonJ": False})
        bn = min(group, key=lambda r: r["select"]["select_risk"]); bn["selected"] = True
        bj = max(group, key=lambda r: r["select"]["commonJ_select"]); bj["selected_commonJ"] = True
        grid = sorted(lrs)
        for r in group:
            r["lr_edge"] = ("low" if r["lr"] == grid[0] else "high" if r["lr"] == grid[-1] else None) if len(grid) > 1 else None
        rows += group
    return {"cell": {**c, "name": cell_name(c), "protocol_id": PROTOCOL, "rho": rho, "mi": mi, "match_row": match_info, "sizes": sizes,
                     "smoke": smoke, "N_run": N, "updates_run": U, "lrs_run": list(lrs), "clean_roles_hash": clean_hash, "roles_hash": roles_hash(roles),
                     "contamination_records": recs, "code_commit": git_commit(), "device": str(device), "torch": torch.__version__,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "data_authorisation": {"image_data": False, "official_test_accessible": False, "note": "synthetic generators only"}},
            "truth": tr, "rows": rows, "wall_seconds": time.time() - t_all}
