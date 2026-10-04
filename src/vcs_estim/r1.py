"""P134 — R1: estimator-level synthetic contamination on the P85 dependence-staircase cell (spec: NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928 §2 R1;
VCS_QMI_Next_Round_Plan_v2 "R1 估计器层面的污染"; estimator server spec v1 §11.1 for the target definition).

Base cell = P85 `gaussian, d = 20, I = 4 nats, N = 4096, B = 256, 2000 updates`, seeds 0-2, with P85's roles (FIT / TUNE / SELECT / EVAL / GRAD /
TRUTH, built by `benchmark.build_roles` from the same seeded streams), P85's critic class, trainer, lr grid {1e-4, 5e-4, 2e-3}, SELECT rule and
native read-outs (`benchmark.neural_rows`), restricted to the estimators VCS / JS / InfoNCE / NWJ with all their P85 negative constructions.

Contamination: a fraction eps of the joint pairs of every role (FIT, TUNE, SELECT, EVAL, TRUTH; GRAD is unused) is replaced by pairs from a
contamination law C, so the joint becomes P_eps = (1 - eps) P + eps C.  The estimator never knows which pairs are contaminated (selection runs on the
contaminated SELECT role).  Q is always the product of the *actual* marginals of P_eps (server spec v1 §11.1: a contamination that changes a marginal
must change Q), realised exactly from P85's clean Q sample:
  independent    C = P_X (x) P_Y          (marginals unchanged; P_eps = (1 - eps) P + eps Q, the spec's form)  -> Q unchanged
  outlier        x_o = x + 5 (every coordinate, 5 sigma), y_o = rho_h x_o + sqrt(1 - rho_h^2) e: y is drawn from the conditional of the other end of the
                 staircase (I = 10 nats, rho_h) given the shifted x.  C_X = N(5, I), C_Y = N(5 rho_h, I)  -> Q: x-side eps-fraction + 5, y-side
                 (independent eps-fraction) + 5 rho_h
  outlier_indep  the P69-draft reading (data.contaminate 'outlier'): x + 5, y an independent N(0, I) draw (the high step's own x discarded, so y is
                 just a marginal draw).  Descriptive only.  C_X = N(5, I), C_Y = P_Y  -> Q: x-side + 5
  heavy          y_c = y + eta, eta multivariate Student-t_2 (z / sqrt(W), z ~ N(0, I_d), W ~ Exp(1), one scale per pair).  C_Y = N(0, I) (+) t_2
                 -> Q: y-side eps-fraction + an independent t_2 draw
Every replacement uses exactly round(eps * n) rows chosen by a seeded permutation (indices recorded); fresh noise comes from its own seeded stream;
eps = 0 returns the clean P85 tensors unchanged (bit-for-bit).

Truth of the contaminated problem (log r = log p_eps - log q_eps on TRUTH / EVAL samples, closed form for the Gaussian mixtures, a 1-d log-space
trapezoid over the t_2 scale for 'heavy'): S_eps = E_M tanh(log r / 2)^2, JS2_eps = E_P log s(log r) + E_Q log s(-log r) + log 4, MI_eps = E_P log r
(Monte Carlo with standard errors).  eps = 0 uses P85's `truths` (analytic MI).

Not here: any SSL training, image data, the official test set; the P69-draft `data.contaminate` is not used (superseded by this module).
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass

import torch

from .benchmark import (BOOT, EVAL_BLOCK, LRS, Role, build_roles, neural_rows, roles_hash, summarise, truths)
from .data import rho_from_mi, setting_from_mi
from .p85_grid import cell as make_cell, cell_name as p85_cell_name
from .run import git_commit
from .synthetic import _gen

PROTOCOL = "P134"
BASE = dict(setting="gaussian", d_signal=20, d_total=20, I=4.0, N=4096, B=256, updates=2000)
TYPES = ("independent", "outlier", "heavy", "outlier_indep")
PRIMARY_TYPES = ("independent", "outlier", "heavy")
EPS = (0.0, 0.01, 0.05, 0.1, 0.2)
SEEDS = (0, 1, 2)
SHIFT = 5.0
HIGH_MI = 10.0                       # 'the other end of the staircase' = the top P85 level (LEVELS_NATS[-1])
KINDS_R1 = ("vcs", "js", "infonce", "nwj")
LOGW_GRID = (-40.0, 6.0, 3001)       # trapezoid in t = log W for the t_2 scale mixture
LOG2PI = math.log(2 * math.pi)


def cell_name(ctype: str, eps: float, seed: int) -> str:
    return f"P134_{'clean' if eps == 0 else ctype}_eps{eps:g}_s{seed}"


# ------------------------------------------------------------------------------------------------------------ densities
def _lognorm(v: torch.Tensor, var: float | torch.Tensor) -> torch.Tensor:
    """log N(v; 0, var * I_d), summed over the last dimension; var a scalar or broadcastable tensor."""
    d = v.shape[-1]
    return -0.5 * d * (LOG2PI + torch.log(torch.as_tensor(var, dtype=v.dtype, device=v.device))) - (v * v).sum(-1) / (2 * var)


def log_t2_conv(v: torch.Tensor, s0: float, chunk: int = 8192, grid=LOGW_GRID) -> torch.Tensor:
    """log h_{s0}(v) with h_{s0}(v) = E_W[N(v; 0, (s0 + 1/W) I_d)], W ~ Exp(1): density of N(0, s0 I) + multivariate t_2 (z / sqrt(W)).
    Trapezoid in t = log W on [-40, 6] (integrand e^{-W} g(W) W, smooth in t; both ends negligible for |v|^2 < 1e15)."""
    t = torch.linspace(*grid[:2], grid[2], dtype=torch.float64, device=v.device); dt = float(t[1] - t[0])
    w = torch.full_like(t, math.log(dt)); w[0] = w[-1] = math.log(dt / 2)
    s = s0 + torch.exp(-t)                                   # variance at W = e^t
    d = v.shape[-1]; out = []
    for a in range(0, len(v), chunk):
        r2 = (v[a:a + chunk].double() ** 2).sum(-1, keepdim=True)          # [c, 1]
        L = -torch.exp(t) + t - 0.5 * d * (LOG2PI + torch.log(s)) - r2 / (2 * s) + w
        out.append(torch.logsumexp(L, dim=1))
    return torch.cat(out).to(v.dtype)


@dataclass
class Contamination:
    ctype: str
    eps: float
    rho: float                      # clean per-coordinate correlation (I = 4 nats over d = 20)
    rho_h: float                    # correlation at the other end of the staircase (I = 10 nats)
    d: int = 20

    def __post_init__(self):
        if self.ctype not in TYPES:
            raise ValueError(self.ctype)
        if not 0.0 <= self.eps < 1.0:
            raise ValueError(self.eps)

    # clean pieces
    def log_p(self, x, y):
        return _lognorm(x, 1.0) + _lognorm(y - self.rho * x, 1.0 - self.rho ** 2)

    @staticmethod
    def log_px(x):
        return _lognorm(x, 1.0)

    @staticmethod
    def log_py(y):
        return _lognorm(y, 1.0)

    # contamination law C and its marginals
    def log_c(self, x, y):
        if self.ctype == "independent":
            return _lognorm(x, 1.0) + _lognorm(y, 1.0)
        if self.ctype == "outlier":
            return _lognorm(x - SHIFT, 1.0) + _lognorm(y - self.rho_h * x, 1.0 - self.rho_h ** 2)
        if self.ctype == "outlier_indep":
            return _lognorm(x - SHIFT, 1.0) + _lognorm(y, 1.0)
        return _lognorm(x, 1.0) + log_t2_conv(y - self.rho * x, 1.0 - self.rho ** 2)                       # heavy

    def log_cx(self, x):
        return _lognorm(x - SHIFT, 1.0) if self.ctype in ("outlier", "outlier_indep") else _lognorm(x, 1.0)

    def log_cy(self, y):
        if self.ctype == "outlier":
            return _lognorm(y - SHIFT * self.rho_h, 1.0)
        if self.ctype == "heavy":
            return log_t2_conv(y, 1.0)
        return _lognorm(y, 1.0)

    def log_ratio(self, x, y) -> torch.Tensor:
        """log p_eps(x, y) - log p_eps,X(x) - log p_eps,Y(y) (= log dP_eps / dQ_eps); float64."""
        x, y = x.double(), y.double(); e = self.eps
        if e == 0.0:
            return self.log_p(x, y) - self.log_px(x) - self.log_py(y)
        a, b = math.log1p(-e), math.log(e)
        lj = torch.logaddexp(a + self.log_p(x, y), b + self.log_c(x, y))
        lx = torch.logaddexp(a + self.log_px(x), b + self.log_cx(x))
        ly = torch.logaddexp(a + self.log_py(y), b + self.log_cy(y))
        return lj - lx - ly


class ContamSetting:
    """Duck-typed stand-in for a P85 `Setting` inside `benchmark.neural_rows`: only `pmi` is called (for eta on EVAL); returns log r_eps."""
    def __init__(self, C: Contamination, device=None, chunk: int = 65536):
        self.C, self.device, self.chunk, self.name = C, device, chunk, f"gaussian+{C.ctype}"

    def pmi(self, x, y, param=None):
        dev = self.device or x.device
        return torch.cat([self.C.log_ratio(x[a:a + self.chunk].to(dev), y[a:a + self.chunk].to(dev)).cpu() for a in range(0, len(x), self.chunk)])


# ------------------------------------------------------------------------------------------------------------ samplers
def _idx(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    return torch.randperm(n, generator=g)[:k]


def _t2(k: int, d: int, g: torch.Generator, dtype) -> torch.Tensor:
    z = torch.randn(k, d, generator=g, dtype=torch.float64); W = -torch.log(torch.rand(k, 1, generator=g, dtype=torch.float64).clamp_min(1e-300))
    return (z / torch.sqrt(W)).to(dtype)


def contaminate_role(role: Role, C: Contamination, key: tuple) -> tuple[Role, dict]:
    """Contaminate the P side (exactly round(eps n) rows) and realise Q = P_eps,X (x) P_eps,Y from the clean Q (independent index sets per side).
    Returns a new Role (tensors cloned only where changed) and the recorded index sets."""
    n = role.n; k = int(round(C.eps * n)); rec = {"n": n, "k": k}
    if k == 0:
        return role, {**rec, "p_idx": [], "qx_idx": [], "qy_idx": []}
    xp, yp, xq, yq = role.xp.clone(), role.yp.clone(), role.xq, role.yq
    gi, gn = _gen(("P134-IDX", C.ctype, C.eps) + key), _gen(("P134-NOISE", C.ctype, C.eps) + key)
    ip = _idx(n, k, gi); d = xp.shape[1]
    if C.ctype == "independent":
        yp[ip] = torch.randn(k, d, generator=gn, dtype=torch.float64).to(yp.dtype)
    elif C.ctype == "outlier":
        xo = xp[ip] + SHIFT; xp[ip] = xo
        yp[ip] = C.rho_h * xo + math.sqrt(1.0 - C.rho_h ** 2) * torch.randn(k, d, generator=gn, dtype=torch.float64).to(yp.dtype)
    elif C.ctype == "outlier_indep":
        xp[ip] = xp[ip] + SHIFT; yp[ip] = torch.randn(k, d, generator=gn, dtype=torch.float64).to(yp.dtype)
    else:                                                                                                   # heavy
        yp[ip] = yp[ip] + _t2(k, d, gn, yp.dtype)
    iqx = iqy = torch.empty(0, dtype=torch.long)
    if C.ctype in ("outlier", "outlier_indep"):
        iqx = _idx(n, k, gi); xq = xq.clone(); xq[iqx] = xq[iqx] + SHIFT
    if C.ctype == "outlier":
        iqy = _idx(n, k, gi); yq = yq.clone(); yq[iqy] = yq[iqy] + SHIFT * C.rho_h
    if C.ctype == "heavy":
        iqy = _idx(n, k, gi); yq = yq.clone(); yq[iqy] = yq[iqy] + _t2(k, d, gn, yq.dtype)
    rec.update({"p_idx": sorted(ip.tolist()), "qx_idx": sorted(iqx.tolist()), "qy_idx": sorted(iqy.tolist())})
    return Role(xp, yp, xq, yq, role.base, role.pad_p), rec


def contaminate_roles(roles: dict, C: Contamination, seed: int, skip=("GRAD",)) -> tuple[dict, dict]:
    out, recs = {}, {}
    for name, r in roles.items():
        if name in skip or C.eps == 0.0:
            out[name] = r; continue
        out[name], rec = contaminate_role(r, C, (name, seed))
        recs[name] = {"n": rec["n"], "k": rec["k"], "k_qx": len(rec["qx_idx"]), "k_qy": len(rec["qy_idx"]),
                      "idx_sha256": hashlib.sha256(repr((rec["p_idx"], rec["qx_idx"], rec["qy_idx"])).encode()).hexdigest(),
                      "p_idx_head": rec["p_idx"][:16]}
    return out, recs


# ------------------------------------------------------------------------------------------------------------ truth
def contaminated_truths(C: Contamination, T: Role, device=None, clean_mi: float | None = None) -> dict:
    lp = ContamSetting(C, device).pmi(T.xp, T.yp).double(); lq = ContamSetting(C, device).pmi(T.xq, T.yq).double()
    ep, eq = torch.tanh(lp / 2), torch.tanh(lq / 2); sp, sq = ep ** 2, eq ** 2; n = len(lp)
    S = float(0.5 * sp.mean() + 0.5 * sq.mean()); se = math.sqrt(0.25 * float(sp.var()) / n + 0.25 * float(sq.var()) / len(lq))
    jp, jq = ep - 0.5 * ep ** 2, -eq - 0.5 * eq ** 2
    lsp, lsq = torch.nn.functional.logsigmoid(lp), torch.nn.functional.logsigmoid(-lq)
    js2 = float(lsp.mean() + lsq.mean() + math.log(4.0)); js2_se = math.sqrt(float(lsp.var()) / n + float(lsq.var()) / len(lq))
    mi = float(lp.mean()); mi_se = math.sqrt(float(lp.var()) / n)
    # sampler / density agreement: the bounded identity E_M[eta] = (E_P eta + E_Q eta) / 2 = 0 (asserted by the QC, |value| <= 4 se);
    # the importance identities E_Q[r] = 1, E_P[1/r] = 1 are reported only (r is heavy-tailed for the outlier law: rare-event dominated)
    em_eta = float(0.5 * ep.mean() + 0.5 * eq.mean()); em_eta_se = math.sqrt(0.25 * float(ep.var()) / n + 0.25 * float(eq.var()) / len(lq))
    eq_r = float(torch.exp(lq).mean()); ep_inv = float(torch.exp(-lp).mean())
    return {"S": S, "S_se": se, "J_oracle": float(jp.mean() + jq.mean()), "J_oracle_se": math.sqrt(float(jp.var()) / n + float(jq.var()) / len(lq)),
            "JS2": js2, "JS2_se": js2_se, "MI": mi, "MI_se": mi_se, "MI_clean_analytic": clean_mi, "n_truth_per_distribution": n,
            "bayes_denominator_1_minus_S": 1 - S, "bayes_denominator_resolvable": (1 - S) > 3 * se,
            "oracle_gate_mean": float(0.5 * (1 - sp).mean() + 0.5 * (1 - sq).mean()),
            "qc_EM_eta": em_eta, "qc_EM_eta_se": em_eta_se, "qc_EM_eta_ok": abs(em_eta) <= 4 * em_eta_se, "qc_EQ_r": eq_r, "qc_EP_inv_r": ep_inv, "truth_note": "Monte Carlo on the contaminated TRUTH role (200 000 per side); Q = product of the contaminated marginals"}


# ------------------------------------------------------------------------------------------------------------ cell
def run_r1_cell(ctype: str, eps: float, seed: int, device, smoke: bool = False, kinds=KINDS_R1, N: int | None = None, updates: int | None = None) -> dict:
    t_all = time.time()
    base = make_cell(BASE["setting"], BASE["d_signal"], BASE["d_total"], BASE["I"], N or BASE["N"], BASE["B"], updates or BASE["updates"], seed, "neural")
    setting = setting_from_mi(base["setting"], base["I"], base["d_signal"]); d = setting.d
    roles, sizes = build_roles(setting, d, d, base["N"], seed, smoke); clean_hash = roles_hash(roles)
    rho_h = rho_from_mi(HIGH_MI, d)
    if eps == 0.0:                                        # the clean P85 cell (shared by every type; ctype may be 'clean')
        tr = truths(setting, d, roles["TRUTH"]); eta_setting = setting; recs = {}
    else:
        C = Contamination(ctype, eps, setting.param, rho_h, d)
        roles, recs = contaminate_roles(roles, C, seed)
        tr = contaminated_truths(C, roles["TRUTH"], device, clean_mi=setting.mi); eta_setting = ContamSetting(C, device)
    lrs = (5e-4,) if smoke else LRS; block = 128 if smoke else EVAL_BLOCK; reps = 20 if smoke else BOOT
    rows = neural_rows(eta_setting, d, roles, tr, base, device, lrs, smoke_kinds=kinds, block=block, reps=reps)
    for r in rows:
        r.pop("_critic", None); r["protocol_id"] = PROTOCOL
    R = {"cell": {"name": cell_name(ctype, eps, seed), "protocol_id": PROTOCOL, "contamination": {"type": "clean" if eps == 0 else ctype, "eps": eps,
                  "shift_sigma": SHIFT, "high_mi_nats": HIGH_MI, "rho": setting.param, "rho_high": rho_h},
                  "base_p85_cell": p85_cell_name(base), "base": base, "sizes": sizes, "clean_roles_hash": clean_hash, "roles_hash": roles_hash(roles),
                  "contamination_records": recs, "kinds": list(kinds), "lr_grid": list(lrs), "eval_block": block, "bootstrap_reps": reps,
                  "code_commit": git_commit(), "device": str(device), "torch": torch.__version__, "smoke": smoke,
                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "data_authorisation": {"image_data": False, "official_test_accessible": False, "note": "synthetic generators only"}},
         "truth": tr, "rows": rows}
    R["summary"] = summarise(R); R["wall_seconds"] = time.time() - t_all
    return R


# ------------------------------------------------------------------------------------------------------------ reading helpers (aggregate)
def resolution_units(v2: float, v4: float, v6: float) -> dict:
    """The estimator's own adjacent-step gap around I = 4 on the clean P85 staircase: primary = mean of the two adjacent gaps (v6 - v2) / 2;
    the one-sided gaps are kept for the directional variant."""
    return {"u": (v6 - v2) / 2.0, "u_down": v4 - v2, "u_up": v6 - v4}


def normalized_shift(delta: float, u: float, eps: float) -> float:
    """|shift| in resolution units per 0.1 of eps (the spec's bound is <= 1)."""
    return abs(delta) / abs(u) / (eps / 0.1) if u and eps > 0 else float("nan")


def reading_label(m_vcs: dict, m_js: dict, bound: float = 1.0, within: float = 0.20) -> str:
    """m_*: type -> max over eps of the normalized shift (primary types).  Pre-stated reading:
    supported            VCS <= bound in every type and JS not 'alike';
    refuted (JS alike)   VCS <= bound in every type, and JS <= bound and <= (1 + within) x VCS in every type;
    not supported        VCS > bound in some type."""
    types = sorted(m_vcs)
    if any(m_vcs[t] > bound for t in types):
        return "not supported"
    alike = all(m_js[t] <= bound and m_js[t] <= (1.0 + within) * m_vcs[t] for t in types)
    return "refuted (JS alike)" if alike else "supported"
