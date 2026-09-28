"""P85 — direct CS-estimator comparison on shared pair data (package v2 server spec §3, §5, §12; research plan §4–§5).

    python -m vcs_estim.benchmark --setting gaussian --d-signal 2 --d-total 20 --I 1.5 --N 4096 --B 256 --updates 2000 --seed 0 --out <json>

One cell = one synthetic condition with role-separated pools (FIT = N per distribution, TUNE = SELECT = max(64, N // 4), EVAL = 32768,
GRAD = 20000, TRUTH = 200000; every role, side and padding block from its own seeded stream).  Every method sees the same FIT / TUNE / SELECT
data and is evaluated once on the same EVAL pairs; EVAL never feeds a selection.

Methods (rows)
  neural    the six P69 estimators (vcs, js, infonce, nwj, dv, smile) on one JointMLP class, with the negative constructions
            product (independent Q units), cyclic8 (K = 8 cyclic shifts inside the batch) and inbatch (all n(n-1) pairs), each over
            the lr grid {1e-4, 5e-4, 2e-3}; SELECT native risk every 100 updates picks the state and the lr; all lr rows are kept.
  cs_k      CS-K-native: classical kernel CS-QMI plug-in on the FIT joint pairs, bandwidth multiples of the FIT median distance per side;
            own truth (Gaussian only): D_CS^Leb = d log(1 - rho^2/4) - d/2 log(1 - rho^2).  Never written into an S column.
  s_kde     product-kernel Gaussian KDE on the FIT joint pairs, eta_hat on independent EVAL queries; bandwidth by the common quadratic risk
            (max J_kernel on TUNE) and Scott's rule as the method-native sensitivity row; S_plug and J_kernel kept apart.
  s_kernel  RFF features of w = [x; y] (m in {256, 1024, 4096} x bandwidth multiples {0.5, 1, 2}), theta trained on the original J with
            the package trainer (SELECT early stopping, lr grid); Nystrom exact-kernel reference from FIT centres.
  rls       closed-form ridge (RuLSIF alpha = 1/2) on the same RFF features, lambda grid, selected on TUNE by the RuLSIF LS objective;
            raw T is an unbounded diagnostic; rLS-tanh wraps it with two scalars fitted numerically on SELECT.
Readouts per row (spec §5.3): native value + own truth + error, common quadratic evaluation J_eval (bounded posteriors), posterior MSE
E_M (T - eta)^2 where a posterior is defined (vcs, js via T = tanh(f/2), s_kde, s_kernel, rls_tanh; not infonce / nwj / dv / smile —
recorded with the reason), relative Bayes risk with the 'indistinguishable' flag when 1 - S is within 3 se of the oracle S, tail quantiles,
numerical failures, fit / evaluation seconds, peak memory, unit and pair counts, a 200-replicate unit bootstrap of the native value (for the
ordering probability across dependence levels in the aggregate), and the channel derivative d/dtheta of the fixed critic's objective through
the generator (common random numbers) against the oracle's central differences and the envelope form.
The official CIFAR test set and all image data are outside this unit (DATA_AUTHORISATION).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import kernel_cs as KC
from .critics import JointMLP
from .data import Setting, pad_sides, setting_from_mi
from .estimators import EMA, KINDS, TARGET, estimate, scores
from .fitting import native_loss, train
from .objectives import j_hat, risk_hat, score_diagnostics
from .p85_grid import cell as make_cell, cell_name
from .run import SOURCE_REF, git_commit
from .synthetic import _gen

PROTOCOL = "P85"
SOURCE_BASIS = "supplement_af7172c_plus_v2"
DATA_AUTHORISATION = {"image_data": False, "official_test_accessible": False,
                      "note": "synthetic generators only; the official CIFAR-10 test was read once (P67/P68) and the S2 protocol (P75) holds "
                              "its own once-per-run exemption; no P85 unit opens image data"}
LRS = (1e-4, 5e-4, 2e-3)
NEURAL_VARIANTS = {"vcs": ("product", "cyclic8", "inbatch"), "js": ("product", "cyclic8", "inbatch"), "infonce": ("inbatch", "cyclic8"),
                   "nwj": ("product", "inbatch"), "dv": ("product", "inbatch"), "smile": ("product", "inbatch")}
RFF_M = (256, 1024, 4096); RFF_MULT = (0.5, 1.0, 2.0); KDE_MULT = (0.25, 0.5, 1.0, 2.0); CSK_MULT = (0.25, 0.5, 1.0, 2.0); RLS_LAM = (1e-4, 1e-3, 1e-2, 1e-1)
NYSTROM_CENTRES = 512
EVAL_BLOCK = 1024
BOOT = 200
POSTERIOR_ABSENT = "native score has no balanced posterior without a gauge / normalisation step (InfoNCE: finite-candidate softmax; " \
                   "NWJ / DV / SMILE: additive constant free); not converted (spec §2.2)"


@dataclass
class Role:
    xp: torch.Tensor; yp: torch.Tensor; xq: torch.Tensor; yq: torch.Tensor
    base: tuple | None = None          # P-side base variables (signal coordinates) for the channel derivative
    pad_p: tuple | None = None         # padded coordinates of the P side (x_pad, y_pad), theta-free

    @property
    def n(self):
        return len(self.xp)


# ------------------------------------------------------------------------------------------------------------ data
def role_sizes(N: int, smoke: bool = False) -> dict:
    if smoke:
        return {"FIT": N, "TUNE": max(32, N // 4), "SELECT": max(32, N // 4), "EVAL": 512, "GRAD": 512, "TRUTH": 5000}
    return {"FIT": N, "TUNE": max(64, N // 4), "SELECT": max(64, N // 4), "EVAL": 32768, "GRAD": 20000, "TRUTH": 200000}


def build_roles(setting: Setting, d_signal: int, d_total: int, N: int, seed: int, smoke: bool = False) -> tuple[dict, dict]:
    sizes = role_sizes(N, smoke); k = d_total - d_signal; roles = {}
    key = (setting.name, round(setting.mi, 6), d_signal, d_total, N, seed)
    for role, n in sizes.items():
        gp = _gen(("P85-P", role) + key); base = setting.sample_base(n, gp)
        xp, yp = setting.from_base(base, setting.param); xp, yp = xp.double(), yp.double()
        gq = _gen(("P85-Q", role) + key)
        xq = setting.from_base(setting.sample_base(n, gq), setting.param)[0].double(); yq = setting.from_base(setting.sample_base(n, gq), setting.param)[1].double()
        pad_p = None
        if k and role != "TRUTH":
            gpp, gpq = _gen(("P85-PAD-P", role) + key), _gen(("P85-PAD-Q", role) + key)
            xpp, ypp = pad_sides(xp, yp, k, gpp); pad_p = (xpp[:, d_signal:].clone(), ypp[:, d_signal:].clone()); xp, yp = xpp, ypp
            xq, yq = pad_sides(xq, yq, k, gpq)
        roles[role] = Role(xp, yp, xq, yq, tuple(b.double() for b in base), pad_p)
    return roles, sizes


def roles_hash(roles: dict) -> str:
    h = hashlib.sha256()
    for role in ("FIT", "TUNE", "SELECT", "EVAL", "GRAD", "TRUTH"):
        r = roles[role]
        for t in (r.xp, r.yp, r.xq, r.yq):
            h.update(t[:64].numpy().tobytes()); h.update(str(tuple(t.shape)).encode())
    return h.hexdigest()


def pmi_signal(setting: Setting, x, y, d_signal: int, param=None):
    return setting.pmi(x[:, :d_signal], y[:, :d_signal], param)


def eta_of(setting, x, y, d_signal, param=None):
    return torch.tanh(0.5 * pmi_signal(setting, x, y, d_signal, param))


def truths(setting: Setting, d_signal: int, T: Role) -> dict:
    pp, pq = pmi_signal(setting, T.xp, T.yp, d_signal), pmi_signal(setting, T.xq, T.yq, d_signal)
    ep, eq = torch.tanh(pp / 2), torch.tanh(pq / 2); sp, sq = ep ** 2, eq ** 2
    S = float(0.5 * sp.mean() + 0.5 * sq.mean()); se = math.sqrt(0.25 * float(sp.var()) / len(ep) + 0.25 * float(sq.var()) / len(eq))
    jp, jq = ep - 0.5 * ep ** 2, -eq - 0.5 * eq ** 2
    js2 = float(torch.nn.functional.logsigmoid(pp).mean() + torch.nn.functional.logsigmoid(-pq).mean() + math.log(4.0))
    out = {"S": S, "S_se": se, "J_oracle": float(jp.mean() + jq.mean()), "J_oracle_se": math.sqrt(float(jp.var()) / len(ep) + float(jq.var()) / len(eq)),
           "JS2": js2, "MI": float(setting.mi), "n_truth_per_distribution": len(ep), "bayes_denominator_1_minus_S": 1 - S,
           "bayes_denominator_resolvable": (1 - S) > 3 * se, "oracle_gate_mean": float(0.5 * (1 - sp).mean() + 0.5 * (1 - sq).mean())}
    if setting.name == "gaussian":
        out["CS_lebesgue"] = KC.cs_lebesgue_gaussian(setting.param, d_signal); out["CS_lebesgue_note"] = "classical fixed-measure CS (own truth of CS-K-native only)"
    else:
        out["CS_lebesgue"] = None
        out["CS_lebesgue_absent_reason"] = ("cubic: the Lebesgue reference density is not square-integrable" if setting.name == "cubic"
                                           else "xor_mixture: no closed form derived; classical-CS error not evaluated in this setting")
    return out


# ------------------------------------------------------------------------------------------------------------ helpers
def _dev(t, device):
    return t.float().to(device)


def _peak_mem(device) -> dict:
    out = {"cpu_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    if device.type == "cuda":
        # the caching allocator must exist before its statistics can be reset ("Invalid device argument" otherwise: pilot job 1013286);
        # a zero-size allocation initialises it without changing any measurement
        torch.empty(0, device=device)
        try:
            out["cuda_peak_alloc_mb"] = torch.cuda.max_memory_allocated(device) / 2 ** 20; torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError as e:  # never let bookkeeping kill a cell
            out["cuda_peak_alloc_mb"] = None; out["cuda_stat_error"] = str(e)[:120]
    return out


def per_anchor(kind: str, fp: torch.Tensor, fn: torch.Tensor, tau: float = 5.0):
    """Per-anchor decomposition of every native estimate: value = combine(mean a, mean b).  fp [n], fn [n, K]; float64."""
    fp, fn = fp.double(), fn.double(); K = fn.shape[1]
    if kind == "vcs":
        tp, tq = torch.tanh(fp), torch.tanh(fn); return tp - tp * tp / 2, (-tq - tq * tq / 2).mean(1), "sum"
    if kind == "js":
        return -torch.nn.functional.softplus(-fp), -torch.nn.functional.softplus(fn).mean(1) + math.log(4.0), "sum"
    if kind == "infonce":
        return fp - torch.logsumexp(torch.cat([fp[:, None], fn], 1), 1) + math.log(K + 1), torch.zeros_like(fp), "sum"
    if kind == "nwj":
        return fp, -torch.exp(fn - 1.0).mean(1), "sum"
    if kind == "dv":
        return fp, torch.exp(fn).mean(1), "log"
    if kind == "smile":
        return fp, torch.exp(torch.clamp(fn, -tau, tau)).mean(1), "log"
    raise ValueError(kind)


def combine(a, b, how):
    ma, mb = float(a.mean()), float(b.mean())
    return ma + mb if how == "sum" else ma - math.log(mb) if mb > 0 else float("nan")


def boot_units(a, b, how, units, reps, gen):
    """units: None -> a and b are independent unit sets (resampled separately); else an int array mapping rows to blocks (resampled jointly)."""
    vals = []
    if units is None:
        na, nb = len(a), len(b)
        for _ in range(reps):
            ia = torch.randint(0, na, (na,), generator=gen); ib = torch.randint(0, nb, (nb,), generator=gen)
            vals.append(combine(a[ia], b[ib], how))
    else:
        u = torch.as_tensor(units); nb = int(u.max()) + 1; members = [torch.nonzero(u == j).squeeze(1) for j in range(nb)]
        for _ in range(reps):
            pick = torch.randint(0, nb, (nb,), generator=gen); idx = torch.cat([members[int(j)] for j in pick])
            vals.append(combine(a[idx], b[idx], how))
    v = np.array(vals, dtype=float)
    return {"boot_se": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "boot_q05": float(np.quantile(v, 0.05)), "boot_q95": float(np.quantile(v, 0.95)),
            "boot_values": [round(float(x), 6) for x in v]}


def posterior_readouts(tp, tn, eta_p, eta_n, tr: dict, gen, reps) -> dict:
    """Common quadratic evaluation of a bounded posterior T on the EVAL P / Q units + bootstrap of J_eval and the posterior MSE."""
    tp, tn = tp.double().cpu(), tn.double().cpu(); ap, aq = tp - 0.5 * tp ** 2, -tn - 0.5 * tn ** 2
    rp, rq = (tp - eta_p) ** 2, (tn - eta_n) ** 2; res = torch.cat([rp, rq]).sqrt()
    S = tr["S"]; J = float(ap.mean() + aq.mean()); pm = float(0.5 * rp.mean() + 0.5 * rq.mean())
    out = {"defined": True, "J_eval": J, "J_eval_se": math.sqrt(float(ap.var()) / len(ap) + float(aq.var()) / len(aq)), "R_eval": float(risk_hat(tp, tn)),
           "J_eval_minus_S": J - S, "posterior_mse": pm, "posterior_rmse": math.sqrt(pm),
           "excess_to_bayes_risk": pm / (1 - S) if tr["bayes_denominator_resolvable"] else None,
           "excess_to_bayes_risk_flag": "ok" if tr["bayes_denominator_resolvable"] else "indistinguishable (1 - S within 3 se of the oracle)",
           "abs_residual_quantiles": {q: float(torch.quantile(res, q)) for q in (0.5, 0.9, 0.99)}, "abs_residual_max": float(res.max()),
           "gate_mean": float(0.5 * (1 - tp ** 2).mean() + 0.5 * (1 - tn ** 2).mean()), "score_diagnostics": score_diagnostics(tp.numpy(), tn.numpy())}
    bj = boot_units(ap, aq, "sum", None, reps, gen); bm = boot_units(0.5 * rp, 0.5 * rq, "sum", None, reps, gen)
    out["J_eval_boot_se"] = bj["boot_se"]; out["posterior_mse_boot_se"] = bm["boot_se"]; out["posterior_mse_boot_q05_q95"] = [bm["boot_q05"], bm["boot_q95"]]
    return out


def yaml_block(**kw) -> dict:
    base = {"experiment_family": "direct_cs", "protocol_id": PROTOCOL, "source_commit": git_commit(), "source_basis": SOURCE_BASIS,
            "estimator": None, "estimand": None, "evaluation_readout": None, "loss_scale": None, "reference_measure": None, "critic_class": None,
            "gradient_routing": None, "n_independent_units": None, "n_positive_pairs": None, "n_negative_pairs": None, "split_manifest_hash": None,
            "noise_target_kind": "none", "noise_tau": None, "noise_sigma_coordinate": None, "fit_seconds": None, "evaluation_seconds": None, "status": "completed"}
    base.update(kw); return base


# ------------------------------------------------------------------------------------------------------------ neural rows
def _blocks(n: int, block: int):
    return [(s, min(n, s + block)) for s in range(0, n, block)] if n > block else [(0, n)]


def _native_on(critic, kind, variant, role: Role, device, block: int, gen):
    """Native score arrays of a fixed critic on one role: per-anchor (a, b, how) + the unit map for the bootstrap."""
    with torch.no_grad():
        if variant == "product":
            fp = torch.cat([critic.pairs(_dev(role.xp[s:e], device), _dev(role.yp[s:e], device)).double().cpu() for s, e in _blocks(role.n, 65536)])
            fn = torch.cat([critic.pairs(_dev(role.xq[s:e], device), _dev(role.yq[s:e], device)).double().cpu() for s, e in _blocks(role.n, 65536)])
            a, b, how = per_anchor(kind, fp, fn[:, None]); return a, b, how, None, len(fp), len(fn)
        A, Bv, units, n_neg = [], [], [], 0
        for j, (s, e) in enumerate(_blocks(role.n, block)):
            if e - s <= 8:
                continue
            fp, fn = scores(critic, _dev(role.xp[s:e], device), _dev(role.yp[s:e], device), variant, gen)
            a, b, how = per_anchor(kind, fp.cpu(), fn.cpu()); A.append(a); Bv.append(b); units.append(torch.full((len(a),), j)); n_neg += fn.numel()
        a, b = torch.cat(A), torch.cat(Bv); return a, b, how, torch.cat(units).numpy(), len(a), n_neg


def train_neural(kind, variant, lr, roles: dict, d_total: int, B: int, updates: int, seed: int, device, every: int = 100, block: int = EVAL_BLOCK):
    torch.manual_seed(seed * 7919 + 17); critic = JointMLP(d_total, d_total).to(device); opt = torch.optim.Adam(critic.parameters(), lr=lr)
    ema = EMA() if kind == "dv" else None; g = torch.Generator().manual_seed(seed * 1000 + 7); sg = torch.Generator().manual_seed(seed * 1000 + 11)
    F, S = roles["FIT"], roles["SELECT"]; Fx, Fy, Fqx, Fqy = (_dev(t, device) for t in (F.xp, F.yp, F.xq, F.yq))
    n = F.n; nonfinite = 0; t0 = time.time()

    def sel_risk():
        critic.eval(); a, b, how, _, _, _ = _native_on(critic, kind, variant, S, device, block, sg); critic.train()
        v = combine(a, b, how); return -v if math.isfinite(v) else float("inf")

    curve = [(0, sel_risk())]; best = (curve[0][1], 0, {k: v.detach().clone() for k, v in critic.state_dict().items()})
    for step in range(1, updates + 1):
        ip = torch.randint(0, n, (B,), generator=g).to(device)
        if variant == "product":
            iq = torch.randint(0, n, (B,), generator=g).to(device)
            fp = critic.pairs(Fx[ip], Fy[ip]); fn = critic.pairs(Fqx[iq], Fqy[iq]).unsqueeze(1)
        else:
            fp, fn = scores(critic, Fx[ip], Fy[ip], variant, g)
        loss, _ = estimate(kind, fp, fn, ema=ema)
        if not torch.isfinite(loss):
            nonfinite += 1; opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % every == 0:
            v = sel_risk(); curve.append((step, v))
            if v < best[0]:
                best = (v, step, {k: t.detach().clone() for k, t in critic.state_dict().items()})
    critic.load_state_dict(best[2]); critic.eval()
    return critic, {"select_curve": curve, "selected_update": best[1], "select_risk": best[0], "fit_seconds": time.time() - t0, "nonfinite_steps": nonfinite,
                    "updates": updates, "lr": lr, "batch": B, "n_params": sum(p.numel() for p in critic.parameters())}


def neural_rows(setting, d_signal, roles, tr, cell, device, lrs, smoke_kinds=None, block=EVAL_BLOCK, reps=BOOT) -> list[dict]:
    rows = []; E = roles["EVAL"]; eta_p, eta_n = eta_of(setting, E.xp, E.yp, d_signal), eta_of(setting, E.xq, E.yq, d_signal)
    kinds = smoke_kinds or KINDS
    for kind in kinds:
        for variant in NEURAL_VARIANTS[kind]:
            group = []
            for lr in lrs:
                _peak_mem(device)
                critic, info = train_neural(kind, variant, lr, roles, cell["d_total"], cell["B"], cell["updates"], cell["seed"], device, block=block)
                t0 = time.time(); gen = torch.Generator().manual_seed(cell["seed"] * 31 + 5)
                a, b, how, units, n_pos, n_neg = _native_on(critic, kind, variant, E, device, block, gen)
                value = combine(a, b, how); own = tr[TARGET[kind]]
                row = yaml_block(estimator=f"neural_{kind}", estimand={"S": "S", "JS2": "JS", "MI": f"MI (InfoNCE bounded by log(K+1))" if kind == "infonce" else "MI"}[TARGET[kind]],
                                 evaluation_readout="native", loss_scale={"vcs": "-J (P and Q means separately)", "js": "Deep-InfoMax softplus form (f = log p/q at the optimum)"}.get(kind, "native"),
                                 reference_measure="mixture_equal" if kind in ("vcs", "js") else "native_base", critic_class="JointMLP concat -> 256 ReLU -> 256 ReLU -> 1",
                                 gradient_routing="full (data fixed; critic only)", n_independent_units=roles["FIT"].n, n_positive_pairs=n_pos, n_negative_pairs=n_neg,
                                 fit_seconds=info["fit_seconds"])
                row.update({"family": "neural", "kind": kind, "negative_construction": variant, "lr": lr, "selected": False, "method": f"neural:{kind}:{variant}:lr{lr:g}",
                            "native": {"value": value, "own_truth": own, "signed_error": value - own, "abs_error": abs(value - own), **boot_units(a, b, how, units, reps, gen)},
                            "select": {"risk": info["select_risk"], "selected_update": info["selected_update"], "curve": info["select_curve"]},
                            "numerical": {"nonfinite_steps": info["nonfinite_steps"], "value_finite": math.isfinite(value)}, "n_params": info["n_params"]})
                if kind in ("vcs", "js"):
                    with torch.no_grad():
                        fp = torch.cat([critic.pairs(_dev(E.xp[s:e], device), _dev(E.yp[s:e], device)).double().cpu() for s, e in _blocks(E.n, 65536)])
                        fn = torch.cat([critic.pairs(_dev(E.xq[s:e], device), _dev(E.yq[s:e], device)).double().cpu() for s, e in _blocks(E.n, 65536)])
                    half = 1.0 if kind == "vcs" else 0.5
                    row["posterior"] = posterior_readouts(torch.tanh(half * fp), torch.tanh(half * fn), eta_p, eta_n, tr, gen, reps)
                    row["posterior"]["parameterisation"] = "T = tanh f" if kind == "vcs" else "T = tanh(f/2) = 2 sigmoid(f) - 1 (matched posterior of the Deep-InfoMax logit)"
                else:
                    row["posterior"] = {"defined": False, "reason": POSTERIOR_ABSENT}
                row["evaluation_seconds"] = time.time() - t0; row["memory"] = _peak_mem(device); row["_critic"] = critic
                group.append(row)
            best = min(group, key=lambda r: r["select"]["risk"]); best["selected"] = True; rows += group
    return rows


# ------------------------------------------------------------------------------------------------------------ kernel rows
def cs_k_rows(roles, tr, cell, device, mults=CSK_MULT) -> list[dict]:
    F = roles["FIT"]; x, y = F.xp.to(device), F.yp.to(device); rows = []
    mx, my = KC.median_distance(F.xp), KC.median_distance(F.yp)
    for mult in mults:
        _peak_mem(device); t0 = time.time()
        with torch.no_grad():
            r = KC.kernel_cs_native(x, y, mult * mx, mult * my, chunk=2048)
        v = float(r["D_CS"]); own = tr["CS_lebesgue"]
        row = yaml_block(estimator="cs_kernel_native", estimand="native_CS (Lebesgue reference)", evaluation_readout="native", loss_scale="D_CS = log A + log B_q - 2 log C",
                         reference_measure="native_base", critic_class="none (Gaussian kernel plug-in, effective kernel sqrt(2) h)", gradient_routing="native autodiff (not used here)",
                         n_independent_units=F.n, n_positive_pairs=F.n, n_negative_pairs=None, fit_seconds=time.time() - t0, evaluation_seconds=0.0)
        row.update({"family": "kernel", "kind": "cs_k_native", "method": f"cs_k:mult{mult:g}", "bandwidth": {"multiple": mult, "sigma_x": mult * mx, "sigma_y": mult * my, "fit_median_x": mx, "fit_median_y": my},
                    "selected": mult == 1.0, "selection_rule": "pre-declared default multiple 1.0 (median heuristic); all multiples reported",
                    "native": {"value": v, "own_truth": own, "signed_error": (v - own) if own is not None else None, "abs_error": abs(v - own) if own is not None else None,
                               "components": {"A": float(r["A"]), "B_q": float(r["B_q"]), "C": float(r["C"])}, "boot_values": None, "boot_se": None,
                               "boot_note": "U-statistic on the FIT sample; no unit bootstrap (a bootstrap of a plug-in with self-pairs is biased)"},
                    "posterior": {"defined": False, "reason": "classical kernel CS has no posterior; different target from S"},
                    "numerical": {"guard_activations": r["guard_activations"], "value_finite": math.isfinite(v)}, "memory": _peak_mem(device)})
        rows.append(row)
    return rows


def s_kde_rows(setting, d_signal, roles, tr, cell, device, mults=KDE_MULT, reps=BOOT) -> list[dict]:
    F, TU, E = roles["FIT"], roles["TUNE"], roles["EVAL"]; dev = lambda t: t.to(device)
    eta_p, eta_n = eta_of(setting, E.xp, E.yp, d_signal), eta_of(setting, E.xq, E.yq, d_signal); rows = []
    _peak_mem(device); t0 = time.time()
    sel = KC.select_kde_bandwidth(dev(F.xp), dev(F.yp), dev(TU.xp), dev(TU.yp), dev(TU.xq), dev(TU.yq), multipliers=mults)
    tsel = time.time() - t0
    scott = {"h": KC.scott_bandwidth(F.xp), "b": KC.scott_bandwidth(F.yp)}
    for name, hb, rule in (("common_risk", sel, "max J_kernel on TUNE (common quadratic risk)"), ("scott", scott, "Scott's rule per side (method-native, sensitivity)")):
        t1 = time.time()
        with torch.no_grad():
            tp = KC.s_kde_eta(dev(E.xp), dev(E.yp), dev(F.xp), dev(F.yp), hb["h"], hb["b"]).cpu(); tn = KC.s_kde_eta(dev(E.xq), dev(E.yq), dev(F.xp), dev(F.yp), hb["h"], hb["b"]).cpu()
        gen = torch.Generator().manual_seed(cell["seed"] * 31 + 9)
        post = posterior_readouts(tp, tn, eta_p, eta_n, tr, gen, reps); S_plug = float(0.5 * (tp ** 2).mean() + 0.5 * (tn ** 2).mean())
        row = yaml_block(estimator="s_kde", estimand="S", evaluation_readout="S_plugin_and_J_common (both kept)", loss_scale="none (plug-in)", reference_measure="mixture_equal",
                         critic_class="product Gaussian KDE eta_hat = tanh(1/2 (log p_hat - log q_hat))", gradient_routing="none", n_independent_units=F.n,
                         n_positive_pairs=E.n, n_negative_pairs=E.n, fit_seconds=tsel if name == "common_risk" else 0.0, evaluation_seconds=time.time() - t1)
        row.update({"family": "kernel", "kind": "s_kde", "method": f"s_kde:{name}", "selected": name == "common_risk", "selection_rule": rule, "bandwidth": {k: hb[k] for k in ("h", "b")},
                    "bandwidth_grid": sel["grid"] if name == "common_risk" else None,
                    "native": {"value": S_plug, "own_truth": tr["S"], "signed_error": S_plug - tr["S"], "abs_error": abs(S_plug - tr["S"]), "readout": "S_plug = E_M eta_hat^2",
                               **boot_units(0.5 * tp ** 2, 0.5 * tn ** 2, "sum", None, reps, gen)},
                    "posterior": post, "numerical": {"nonfinite_eta": int((~torch.isfinite(tp)).sum() + (~torch.isfinite(tn)).sum()), "value_finite": math.isfinite(S_plug)},
                    "memory": _peak_mem(device)})
        rows.append(row)
    return rows


def _median_w(F: Role) -> float:
    W = torch.cat([torch.cat([F.xp, F.yp], 1), torch.cat([F.xq, F.yq], 1)], 0)
    return KC.median_distance(W)


def s_kernel_and_rls_rows(setting, d_signal, roles, tr, cell, device, lrs, ms=RFF_M, mults=RFF_MULT, lams=RLS_LAM, reps=BOOT, nystrom=NYSTROM_CENTRES) -> list[dict]:
    F, TU, S, E = roles["FIT"], roles["TUNE"], roles["SELECT"], roles["EVAL"]; dim = 2 * cell["d_total"]
    eta_p, eta_n = eta_of(setting, E.xp, E.yp, d_signal), eta_of(setting, E.xq, E.yq, d_signal)
    medw = _median_w(F); rows = []; rff_group, rls_group = [], []

    def feats_chunked(model, x, y, chunk=4096):
        return torch.cat([model.features(_dev(x[s:e], device), _dev(y[s:e], device)).double() for s, e in _blocks(len(x), chunk)])

    def score_rows(model, tag, kind, extra, t_fit, sel_risk, selected_key):
        t1 = time.time(); gen = torch.Generator().manual_seed(cell["seed"] * 31 + 13)
        with torch.no_grad():
            fp = torch.cat([model(_dev(E.xp[s:e], device), _dev(E.yp[s:e], device)).double().cpu() for s, e in _blocks(E.n, 16384)])
            fn = torch.cat([model(_dev(E.xq[s:e], device), _dev(E.yq[s:e], device)).double().cpu() for s, e in _blocks(E.n, 16384)])
        post = posterior_readouts(torch.tanh(fp), torch.tanh(fn), eta_p, eta_n, tr, gen, reps)
        row = yaml_block(estimator=kind, estimand="S", evaluation_readout="J_common", loss_scale="-J (P and Q means separately)", reference_measure="mixture_equal",
                         critic_class=f"{kind}: fixed features + trainable read-out with intercept, tanh", gradient_routing="full (data fixed; read-out only)",
                         n_independent_units=F.n, n_positive_pairs=E.n, n_negative_pairs=E.n, fit_seconds=t_fit, evaluation_seconds=time.time() - t1)
        row.update({"family": "kernel", "kind": kind, "method": tag, "selected": False, **extra, "select": {"risk": sel_risk},
                    "native": {"value": post["J_eval"], "own_truth": tr["S"], "signed_error": post["J_eval"] - tr["S"], "abs_error": abs(post["J_eval"] - tr["S"]), "readout": "J_eval (common quadratic)",
                               **boot_units(torch.tanh(fp) - 0.5 * torch.tanh(fp) ** 2, -torch.tanh(fn) - 0.5 * torch.tanh(fn) ** 2, "sum", None, reps, gen)},
                    "posterior": post, "numerical": {"value_finite": math.isfinite(post["J_eval"])}, "memory": _peak_mem(device), "_model": model})
        return row

    for m in ms:
        for mult in mults:
            sigma = mult * medw
            for lr in lrs:
                _peak_mem(device); torch.manual_seed(cell["seed"] * 7919 + m)
                model = KC.RFFTanh(dim, m, sigma, seed=cell["seed"] * 100 + m)
                model, info = train(model, native_loss("vcs"), F, S, lr=lr, batch=cell["B"], updates=cell["updates"], seed=cell["seed"], device=device)
                rff_group.append(score_rows(model, f"s_kernel:rff:m{m}:mult{mult:g}:lr{lr:g}", "s_kernel_rff",
                                            {"rff": {"m": m, "bandwidth_multiple": mult, "sigma": sigma, "fit_median_w": medw, "feature_seed": cell["seed"] * 100 + m, "n_params": m + 1}, "lr": lr,
                                             "selected_update": info["selected_update"]}, info["fit_seconds"], info["select_risk"], None))
            # rLS on the same features (closed form; lambda grid; selected on TUNE by the RuLSIF LS objective), reusing the last model's Omega / b
            t0 = time.time(); ref = rff_group[-1]["_model"]
            with torch.no_grad():
                Fp, Fq = feats_chunked(ref, F.xp, F.yp), feats_chunked(ref, F.xq, F.yq)
                Tp, Tq = feats_chunked(ref, TU.xp, TU.yp), feats_chunked(ref, TU.xq, TU.yq)
                mom = KC.rls_moments(Fp, Fq)
            for lam in lams:
                fit = KC.rls_from_moments(mom, lam, penalise_intercept=False)
                gp, gq = KC.rls_apply(Tp, fit["theta_g"]), KC.rls_apply(Tq, fit["theta_g"])
                tune_ls = KC.rulsif_ls_objective(gp, gq); tune_J = float(j_hat(gp - 1, gq - 1))
                rls_group.append({"m": m, "mult": mult, "lam": lam, "fit": fit, "model": ref, "tune_ls": tune_ls, "tune_J_raw": tune_J,
                                  "identity_residual": abs(tune_ls - (-0.5 - 0.5 * tune_J)), "fit_seconds": time.time() - t0})
            del Fp, Fq, Tp, Tq
    best = min(rff_group, key=lambda r: r["select"]["risk"]); best["selected"] = True; rows += rff_group
    # Nystrom exact-kernel reference (FIT centres, bandwidth multiple 1) over the lr grid
    nc = min(nystrom, F.n); gc = torch.Generator().manual_seed(cell["seed"] * 31 + 21); idx = torch.randperm(F.n, generator=gc)[:nc]
    Wc = torch.cat([F.xp[idx], F.yp[idx]], 1); ny_group = []
    for lr in lrs:
        _peak_mem(device); model = KC.NystromTanh(Wc, sigma=medw)
        model, info = train(model, native_loss("vcs"), F, S, lr=lr, batch=cell["B"], updates=cell["updates"], seed=cell["seed"], device=device)
        ny_group.append(score_rows(model, f"s_kernel:nystrom:c{nc}:lr{lr:g}", "s_kernel_nystrom", {"nystrom": {"centres": nc, "sigma": medw, "n_params": nc + 1}, "lr": lr,
                                                                                                  "selected_update": info["selected_update"]}, info["fit_seconds"], info["select_risk"], None))
    b2 = min(ny_group, key=lambda r: r["select"]["risk"]); b2["selected"] = True; rows += ny_group
    # rLS selected on TUNE (native LS objective) -> raw diagnostic row + rLS-tanh row (scalars fitted on SELECT)
    sel = min(rls_group, key=lambda r: r["tune_ls"]); ref, fit = sel["model"], sel["fit"]; t1 = time.time()
    with torch.no_grad():
        Sp, Sq = feats_chunked(ref, S.xp, S.yp), feats_chunked(ref, S.xq, S.yq); Ep, Eq = feats_chunked(ref, E.xp, E.yp), feats_chunked(ref, E.xq, E.yq)
        rp, rq = KC.rls_apply(Ep, fit["theta"]).cpu(), KC.rls_apply(Eq, fit["theta"]).cpu(); sp_, sq_ = KC.rls_apply(Sp, fit["theta"]).cpu(), KC.rls_apply(Sq, fit["theta"]).cpu()
    gen = torch.Generator().manual_seed(cell["seed"] * 31 + 17)
    J_raw = float(j_hat(rp, rq)); S_raw = float(0.5 * (rp ** 2).mean() + 0.5 * (rq ** 2).mean())
    raw = yaml_block(estimator="rls_raw", estimand="S (relative density ratio r_1/2 - 1 = T)", evaluation_readout="J_common (unbounded T: diagnostic only)", loss_scale="RuLSIF alpha = 1/2 LS = -1/2 - J/2",
                     reference_measure="mixture_equal", critic_class="linear read-out on RFF features, closed-form ridge (Cholesky)", gradient_routing="none (closed form)",
                     n_independent_units=F.n, n_positive_pairs=E.n, n_negative_pairs=E.n, fit_seconds=sel["fit_seconds"], evaluation_seconds=time.time() - t1)
    raw.update({"family": "kernel", "kind": "rls_raw", "method": f"rls:raw:m{sel['m']}:mult{sel['mult']:g}:lam{sel['lam']:g}", "selected": True, "bounded": False,
                "selection_rule": "min RuLSIF LS objective on TUNE over (m, bandwidth multiple, lambda)", "rls": {"m": sel["m"], "bandwidth_multiple": sel["mult"], "lambda": sel["lam"],
                "penalise_intercept": False, "solver": fit["solver"], "identity_residual_tune": sel["identity_residual"], "grid": [{k: r[k] for k in ("m", "mult", "lam", "tune_ls", "tune_J_raw")} for r in rls_group]},
                "native": {"value": J_raw, "own_truth": tr["S"], "signed_error": J_raw - tr["S"], "abs_error": abs(J_raw - tr["S"]), "readout": "J_hat of the raw linear T (may leave [-3, 1])",
                           "S_plug_raw": S_raw, "T_range": [float(min(rp.min(), rq.min())), float(max(rp.max(), rq.max()))], "boot_values": None, "boot_se": None},
                "posterior": {"defined": False, "reason": "raw ridge T is unbounded; the bounded candidate is rls_tanh"}, "numerical": {"value_finite": math.isfinite(J_raw)}, "memory": _peak_mem(device)})
    rows.append(raw)
    t2 = time.time(); wrap = KC.fit_tanh_wrap(sp_, sq_); tp, tn = KC.tanh_wrap(rp, wrap["c"], wrap["b0"]), KC.tanh_wrap(rq, wrap["c"], wrap["b0"])
    post = posterior_readouts(tp, tn, eta_p, eta_n, tr, gen, reps)
    wr = yaml_block(estimator="rls_tanh", estimand="S", evaluation_readout="J_common", loss_scale="tanh(c T_raw + b0), (c, b0) by Nelder-Mead on J_hat (SELECT)", reference_measure="mixture_equal",
                    critic_class="tanh-wrapped closed-form ridge read-out on RFF features", gradient_routing="none", n_independent_units=F.n, n_positive_pairs=E.n, n_negative_pairs=E.n,
                    fit_seconds=sel["fit_seconds"] + (time.time() - t2), evaluation_seconds=0.0)
    wr.update({"family": "kernel", "kind": "rls_tanh", "method": f"rls:tanh:m{sel['m']}:mult{sel['mult']:g}:lam{sel['lam']:g}", "selected": True, "wrap": wrap,
               "native": {"value": post["J_eval"], "own_truth": tr["S"], "signed_error": post["J_eval"] - tr["S"], "abs_error": abs(post["J_eval"] - tr["S"]), "readout": "J_eval (common quadratic)",
                          **boot_units(tp - 0.5 * tp ** 2, -tn - 0.5 * tn ** 2, "sum", None, reps, gen)},
               "posterior": post, "numerical": {"value_finite": math.isfinite(post["J_eval"])}, "memory": _peak_mem(device),
               "_model": (ref, fit["theta"], wrap)})
    rows.append(wr)
    return rows


# ------------------------------------------------------------------------------------------------------------ channel derivative (spec §5.3)
def channel_derivative(setting: Setting, d_signal: int, G: Role, T_fns: dict, device, deltas=(1e-2, 3e-3, 1e-3), chunk=4096) -> dict:
    """d/dtheta E_{P_theta}[T - T^2/2] with the critic fixed, the gradient flowing through the generator (theta = rho or c); Q is theta-free in
    every setting.  Oracle: dS/dtheta by central differences with common random numbers and the envelope form (eta_theta0 fixed)."""
    theta0 = float(setting.param); base = G.base; xpad, ypad = G.pad_p if G.pad_p is not None else (None, None); n = G.n
    out = {"parameter": setting.param_name, "theta0": theta0, "n_units": n}

    def y_theta(sl, theta):
        b = tuple(t[sl] for t in base); x, y = setting.from_base(b, theta)
        if xpad is not None:
            x, y = torch.cat([x, xpad[sl]], 1), torch.cat([y, ypad[sl]], 1)
        return x, y

    def learned(T_fn):
        """Per-unit derivatives (one theta per row) -> mean and MC standard error."""
        gs = []
        for s, e in _blocks(n, chunk):
            th = torch.full((e - s, 1), theta0, dtype=torch.float64, requires_grad=True); x, y = y_theta(slice(s, e), th)
            t = T_fn(x.to(device), y.to(device)).double().cpu(); h = (t - 0.5 * t * t).sum()
            (g,) = torch.autograd.grad(h, th); gs.append(g.detach().squeeze(1))
        g = torch.cat(gs); return {"value": float(g.mean()), "se": float(g.std() / math.sqrt(len(g)))}

    def S_at(theta):
        vals = []
        for s, e in _blocks(n, chunk):
            x, y = y_theta(slice(s, e), theta); vals.append(eta_of(setting, x, y, d_signal, theta) ** 2)
        ep = torch.cat(vals); eq = eta_of(setting, G.xq, G.yq, d_signal, theta) ** 2
        return 0.5 * ep + 0.5 * eq

    fd = []
    for dl in deltas:
        diff = (S_at(theta0 + dl) - S_at(theta0 - dl)) / (2 * dl); fd.append({"delta": dl, "dS_dtheta": float(diff.mean()), "se": float(diff.std() / math.sqrt(len(diff)))})
    env = learned(lambda x, y: eta_of(setting, x, y, d_signal, theta0)); ref = fd[1]
    out["oracle_fd"] = fd; out["oracle_envelope"] = env
    out["oracle_envelope_minus_fd_in_se"] = (env["value"] - ref["dS_dtheta"]) / math.sqrt(env["se"] ** 2 + ref["se"] ** 2) if (env["se"] or ref["se"]) else None
    out["learned"] = {}
    for name, fn in T_fns.items():
        try:
            v = learned(fn)
            out["learned"][name] = {"dJ_dtheta": v["value"], "se": v["se"], "ratio_to_oracle": v["value"] / ref["dS_dtheta"] if ref["dS_dtheta"] else None,
                                    "minus_oracle_in_se": (v["value"] - ref["dS_dtheta"]) / math.sqrt(v["se"] ** 2 + ref["se"] ** 2) if (v["se"] or ref["se"]) else None}
        except Exception as e:  # noqa: BLE001
            out["learned"][name] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ------------------------------------------------------------------------------------------------------------ cell driver
def run_cell(cell: dict, device, smoke: bool = False, methods: str | None = None) -> dict:
    methods = methods or cell.get("methods", "all"); t_all = time.time()
    setting = setting_from_mi(cell["setting"], cell["I"], cell["d_signal"]); d_signal = setting.d  # xor_mixture fixes its own d
    d_total = max(cell["d_total"], d_signal); cell = {**cell, "d_signal": d_signal, "d_total": d_total}
    roles, sizes = build_roles(setting, d_signal, d_total, cell["N"], cell["seed"], smoke); tr = truths(setting, d_signal, roles["TRUTH"])
    lrs = (5e-4,) if smoke else LRS
    kw = dict(ms=(64,), mults=(1.0,), lams=(1e-2,), reps=20, nystrom=32) if smoke else {}
    R = {"cell": {**cell, "name": cell_name(cell), "protocol_id": PROTOCOL, "source_basis": SOURCE_BASIS, "code_commit": git_commit(), "source_reference_commit": SOURCE_REF,
                  "setting_param": {setting.param_name: setting.param}, "mi_nats": setting.mi, "sizes": sizes, "roles_hash": roles_hash(roles), "device": str(device),
                  "torch": torch.__version__, "smoke": smoke, "methods": methods, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data_authorisation": DATA_AUTHORISATION,
                  "grids": {"lr": lrs, "rff_m": kw.get("ms", RFF_M), "rff_mult": kw.get("mults", RFF_MULT), "kde_mult": (0.5, 1.0) if smoke else KDE_MULT, "cs_k_mult": (0.25, 1.0) if smoke else CSK_MULT,
                            "rls_lambda": kw.get("lams", RLS_LAM), "eval_block": 128 if smoke else EVAL_BLOCK, "bootstrap_reps": kw.get("reps", BOOT)}},
         "truth": tr, "rows": []}
    T_fns = {}
    if methods in ("all", "neural", "vcs_kernel"):
        kinds = ("vcs",) if methods == "vcs_kernel" else None
        rows = neural_rows(setting, d_signal, roles, tr, cell, device, lrs, smoke_kinds=kinds, block=128 if smoke else EVAL_BLOCK, reps=kw.get("reps", BOOT))
        for r in rows:
            if r["selected"] and r["negative_construction"] == "product" and r["kind"] in ("vcs", "js"):
                c = r["_critic"]; half = 1.0 if r["kind"] == "vcs" else 0.5
                T_fns[f"neural_{r['kind']}_product"] = (lambda c, half: (lambda x, y: torch.tanh(half * c.pairs(x.float(), y.float()))))(c, half)
        R["rows"] += rows
    if methods in ("all", "kernel", "vcs_kernel"):
        R["rows"] += cs_k_rows(roles, tr, cell, device, mults=(0.25, 1.0) if smoke else CSK_MULT)
        R["rows"] += s_kde_rows(setting, d_signal, roles, tr, cell, device, mults=(0.5, 1.0) if smoke else KDE_MULT, reps=kw.get("reps", BOOT))
        rows = s_kernel_and_rls_rows(setting, d_signal, roles, tr, cell, device, lrs, **kw); R["rows"] += rows
        for r in rows:
            if r["selected"] and r["kind"] == "s_kernel_rff":
                T_fns["s_kernel_rff"] = (lambda m: (lambda x, y: torch.tanh(m(x.float(), y.float()))))(r["_model"])
            if r["kind"] == "rls_tanh":
                ref, theta, wrap = r["_model"]
                T_fns["rls_tanh"] = (lambda ref, theta, wrap: (lambda x, y: KC.tanh_wrap(KC.rls_apply(ref.features(x.float(), y.float()).double(), theta.to(x.device)), wrap["c"], wrap["b0"])))(ref, theta, wrap)
        Fr = roles["FIT"]; skde = next(r for r in R["rows"] if r.get("method") == "s_kde:common_risk"); hb = skde["bandwidth"]
        T_fns["s_kde"] = (lambda h, b: (lambda x, y: KC.s_kde_eta(x, y, Fr.xp.to(x.device), Fr.yp.to(x.device), h, b)))(hb["h"], hb["b"])
    t0 = time.time(); R["channel_derivative"] = channel_derivative(setting, d_signal, roles["GRAD"], T_fns, device, chunk=256 if smoke else 4096); R["channel_derivative"]["seconds"] = time.time() - t0
    for r in R["rows"]:
        r.pop("_critic", None); r.pop("_model", None); r["split_manifest_hash"] = R["cell"]["roles_hash"]
    R["summary"] = summarise(R); R["wall_seconds"] = time.time() - t_all
    return R


def summarise(R: dict) -> list[dict]:
    """Selected rows only: one line per method with the numbers the aggregate reads."""
    out = []
    for r in R["rows"]:
        if not r.get("selected"):
            continue
        p = r.get("posterior", {})
        out.append({"method": r["method"], "family": r["family"], "kind": r["kind"], "estimand": r["estimand"], "native_value": r["native"]["value"], "own_truth": r["native"]["own_truth"],
                    "native_abs_error": r["native"]["abs_error"], "native_boot_se": r["native"].get("boot_se"), "J_eval": p.get("J_eval"), "posterior_mse": p.get("posterior_mse"),
                    "excess_to_bayes_risk": p.get("excess_to_bayes_risk"), "fit_seconds": r["fit_seconds"], "evaluation_seconds": r["evaluation_seconds"],
                    "peak_mem": r.get("memory"), "n_params": r.get("n_params") or (r.get("rff") or r.get("nystrom") or {}).get("n_params")})
    return out


def truth_table(settings=("gaussian",), d_list=(2, 20), I_list=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0), n=200000) -> list[dict]:
    rows = []
    for st in settings:
        for d in d_list:
            for I in I_list:
                s = setting_from_mi(st, I, d); g = torch.Generator().manual_seed(1)
                x, y = s.sample(n, g); xq = s.sample(n, g)[0]; yq = s.sample(n, g)[1]
                tr = truths(s, s.d, Role(x.double(), y.double(), xq.double(), yq.double()))
                rows.append({"setting": st, "d": s.d, "I": I, s.param_name: s.param, "S": tr["S"], "S_se": tr["S_se"], "JS2": tr["JS2"], "CS_lebesgue": tr.get("CS_lebesgue")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="gaussian"); ap.add_argument("--d-signal", type=int, default=20); ap.add_argument("--d-total", type=int, default=20)
    ap.add_argument("--I", type=float, default=4.0); ap.add_argument("--N", type=int, default=4096); ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--updates", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--methods", default="all", choices=("all", "neural", "kernel", "vcs_kernel"))
    ap.add_argument("--out", default=None); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--truth-table", action="store_true")
    a = ap.parse_args(argv)
    if a.truth_table:
        rows = truth_table(); print(json.dumps(rows, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(rows, indent=1))
        return 0
    device = torch.device("cuda", 0) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    c = make_cell(a.setting, a.d_signal, a.d_total, a.I, a.N, a.B, a.updates, a.seed, a.methods)
    R = run_cell(c, device, smoke=a.smoke, methods=a.methods)
    out = Path(a.out or f"{cell_name(c)}.json"); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(R)); tmp.replace(out)
    for s in R["summary"]:
        print(f"{s['method']:<44s} native {s['native_value']:+.4f} (truth {s['own_truth'] if s['own_truth'] is None else round(s['own_truth'], 4)})  J_eval {s['J_eval']}  pMSE {s['posterior_mse']}  fit {s['fit_seconds']:.1f}s")
    print(f"-> {out} ({R['wall_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
