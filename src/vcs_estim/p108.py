"""P108 — package v4 module E (E1, E2, E3 cost fields) on the P85 synthetic generators, oracles and critic classes.

Nothing in the P85 / P86 modules is modified; this file only imports them (``benchmark.build_roles`` / ``truths`` / ``train_neural`` /
``_native_on`` / ``combine``, ``kernel_cs`` RFF and KDE helpers, ``fitting.train``).

E1  identical fitted T, identical independent EVAL sample, two readouts:
      variational  J_hat(T)      = mean_P (T − T²/2) + mean_Q (−T − T²/2)
      plug-in      S_plug_hat(T) = ½ mean_P T² + ½ mean_Q T²
    population identities for T = eta + e (M = (P + Q)/2, S = E_M eta²):  J(T) − S = −E_M e²,  S_plug(T) − S = 2 E_M[eta e] + E_M e².
    (a) mechanism: T_t = (1 − t) eta + t U for U = 0 and one fixed bounded U independent of the data (exact on a discrete toy; Monte Carlo on
        the P85 TRUTH sample with common random numbers);  (b) real fits: JointMLP critics trained with the VCS loss and with the JS loss
        (T = tanh(f/2) for JS) on independent FIT samples of several sizes and solver budgets; both readouts on the same EVAL units.
E2  fixed orthogonal rotations per side (drawn once from a named stream, before any split), original vs rotated coordinates with identical
    roles, widths, kernel selection program and budgets; error vs independent samples and vs updates / seconds; tuning and chosen-model cost.
E3  cost fields: fit / evaluation seconds, peak memory, error at fixed budget points (stopping points), refit spread over independent seeds.
Wording: an L-BFGS-converged fit is a numerical solution, not a closed form; a tanh-wrapped ridge is not the exact optimum of the wrapped objective.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import kernel_cs as KC
from .benchmark import (Role, _dev, _native_on, _peak_mem, build_roles, combine, eta_of, train_neural, truths)
from .data import setting_from_mi
from .estimators import TARGET
from .fitting import native_loss, train
from .synthetic import _gen

PROTOCOL = "P108"
BUDGETS = (250, 1000, 4000)
LRS = (1e-4, 5e-4, 2e-3)
REF_KINDS = (("infonce", "inbatch"), ("nwj", "product"), ("dv", "product"), ("smile", "product"))
T_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
# E1 conditions, named before any run (v4 §E1): moderate Gaussian, high-dimensional with irrelevant padding, one nonlinear condition
CONDITIONS = {"C1_gauss_mid": ("gaussian", 20, 20, 4.0), "C2_gauss_pad": ("gaussian", 2, 100, 1.5), "C3_xor": ("xor_mixture", 10, 10, 4.0)}
E2_CONDITIONS = ("C1_gauss_mid", "C2_gauss_pad")


# ------------------------------------------------------------------------------------------------------------ readouts
def readouts(tp: torch.Tensor, tq: torch.Tensor) -> dict:
    """Both readouts of one bounded T on one sample of P units and Q units (independent sets), with their sampling SEs."""
    tp, tq = tp.double(), tq.double(); ap, aq = tp - 0.5 * tp ** 2, -tq - 0.5 * tq ** 2; sp, sq = 0.5 * tp ** 2, 0.5 * tq ** 2
    se = lambda a, b: math.sqrt(float(a.var()) / len(a) + float(b.var()) / len(b))
    return {"J": float(ap.mean() + aq.mean()), "J_se": se(ap, aq), "S_plug": float(sp.mean() + sq.mean()), "S_plug_se": se(sp, sq)}


def population_readouts(T: np.ndarray, p: np.ndarray, q: np.ndarray) -> dict:
    """Exact readouts on a discrete space (float64): p, q probability vectors, T values per state."""
    m = 0.5 * (p + q); eta = np.where(m > 0, (p - q) / np.where(m > 0, p + q, 1.0), 0.0); S = float((m * eta ** 2).sum())
    J = float((p * (T - 0.5 * T ** 2)).sum() + (q * (-T - 0.5 * T ** 2)).sum()); Sp = float(0.5 * (p * T ** 2).sum() + 0.5 * (q * T ** 2).sum())
    e = T - eta
    return {"S": S, "J": J, "S_plug": Sp, "J_bias": J - S, "S_plug_bias": Sp - S, "pred_J_bias": float(-(m * e ** 2).sum()),
            "pred_S_plug_bias": float(2 * (m * eta * e).sum() + (m * e ** 2).sum())}


def discrete_toy(k: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed); p = r.dirichlet(np.ones(k)); q = r.dirichlet(np.ones(k)); return p, q


def u_fixed(x: torch.Tensor, y: torch.Tensor, scale: float = 0.8) -> torch.Tensor:
    """One fixed bounded function of (x, y) drawn independently of every data stream (named generator 'P108-U'); |U| < scale."""
    g = _gen(("P108-U", x.shape[1], y.shape[1])); wx = torch.randn(x.shape[1], generator=g, dtype=torch.float64) / math.sqrt(x.shape[1])
    wy = torch.randn(y.shape[1], generator=g, dtype=torch.float64) / math.sqrt(y.shape[1]); c = float(torch.randn(1, generator=g))
    return scale * torch.tanh(x.double() @ wx - y.double() @ wy + 0.5 * (x.double() @ wx) * (y.double() @ wy) + c)


def mechanism_mc(cond: str, seed: int = 0, ts=T_GRID, smoke: bool = False) -> dict:
    """E1(a) on the P85 TRUTH sample of a condition (signal coordinates): T_t = (1 − t) eta + t U, U in {0, u_fixed}.
    S_hat = S_plug(eta) on the same units.  Per t: readout errors J(T_t) − S_hat and S_plug(T_t) − S_hat; identity-predicted biases from the
    sample moments (−E_M e²; 2 E_M[eta e] + E_M e²); the plug-in identity is algebraic and holds exactly on any sample (residual reported);
    the J identity uses E_P T − E_Q T = 2 E_M[eta T], true in expectation only, so its sample discrepancy D_J is reported with its SE (z).
    Orders: log-log slopes of the identity-predicted |bias| against t (all t, and t <= 0.1)."""
    name, ds, dt, I = CONDITIONS[cond]; st = setting_from_mi(name, I, ds); ds = st.d if name == "xor_mixture" else ds
    roles, _ = build_roles(st, ds, ds, 4096, seed, smoke)            # TRUTH is never padded; padding does not change eta
    T = roles["TRUTH"]; ep, eq = eta_of(st, T.xp, T.yp, ds).double(), eta_of(st, T.xq, T.yq, ds).double(); tr = truths(st, ds, T)
    S_hat = float(0.5 * (ep ** 2).mean() + 0.5 * (eq ** 2).mean())
    out = {"condition": cond, "S_hat_truth_sample": S_hat, "S_se": tr["S_se"], "n_per_distribution": T.n, "U": {}}
    for uname, (up, uq) in {"zero": (torch.zeros_like(ep), torch.zeros_like(eq)), "fixed": (u_fixed(T.xp, T.yp), u_fixed(T.xq, T.yq))}.items():
        rows = []
        for t in ts:
            tp, tq = (1 - t) * ep + t * up, (1 - t) * eq + t * uq; r = readouts(tp, tq); e_p, e_q = tp - ep, tq - eq
            m_e2 = float(0.5 * (e_p ** 2).mean() + 0.5 * (e_q ** 2).mean()); m_eta_e = float(0.5 * (ep * e_p).mean() + 0.5 * (eq * e_q).mean())
            dp = tp - 0.5 * tp ** 2 - 0.5 * ep ** 2 + 0.5 * e_p ** 2; dq = -tq - 0.5 * tq ** 2 - 0.5 * eq ** 2 + 0.5 * e_q ** 2
            DJ = float(dp.mean() + dq.mean()); DJ_se = math.sqrt(float(dp.var()) / len(dp) + float(dq.var()) / len(dq))
            pj, pp = -m_e2, 2 * m_eta_e + m_e2
            rows.append({"t": t, **r, "J_err": r["J"] - S_hat, "S_plug_err": r["S_plug"] - S_hat, "pred_J_bias": pj, "pred_S_plug_bias": pp,
                         "D_J": DJ, "D_J_se": DJ_se, "D_J_z": DJ / DJ_se if DJ_se > 0 else 0.0, "S_plug_identity_residual": (r["S_plug"] - S_hat) - pp})
        sl = lambda rs, k: float(np.polyfit(np.log([x["t"] for x in rs]), np.log(np.abs([x[k] for x in rs]) + 1e-300), 1)[0])
        small = [x for x in rows if x["t"] <= 0.1]
        out["U"][uname] = {"rows": rows, "loglog_slope_all_t": {"J": sl(rows, "pred_J_bias"), "S_plug": sl(rows, "pred_S_plug_bias")},
                           "loglog_slope_t_le_0.1": {"J": sl(small, "pred_J_bias"), "S_plug": sl(small, "pred_S_plug_bias")},
                           "max_abs_D_J_z": max(abs(x["D_J_z"]) for x in rows), "max_abs_S_plug_identity_residual": max(abs(x["S_plug_identity_residual"]) for x in rows)}
    return out


# ------------------------------------------------------------------------------------------------------------ rotations (E2)
def orthogonal(d: int, tag: str) -> torch.Tensor:
    """Fixed Haar-random orthogonal matrix from a named stream (drawn once per (dimension, side tag), before any data split)."""
    g = _gen(("P108-ROT", tag, d)); A = torch.randn(d, d, generator=g, dtype=torch.float64); Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R))[None, :]


def rotate_roles(roles: dict, d_total: int) -> tuple[dict, dict]:
    """x -> x Rx^T, y -> y Ry^T on every role except TRUTH (truth uses the original coordinates; eta needs the inverse map, i.e. the originals,
    which are returned for the EVAL role).  Rx != Ry (separate tags)."""
    Rx, Ry = orthogonal(d_total, "x"), orthogonal(d_total, "y"); out = {}; orig_eval = roles["EVAL"]
    for k, r in roles.items():
        if k == "TRUTH":
            out[k] = r; continue
        out[k] = Role(r.xp @ Rx.T, r.yp @ Ry.T, r.xq @ Rx.T, r.yq @ Ry.T, r.base, r.pad_p)
    return out, {"orig_eval": orig_eval, "Rx_sha_prefix": float(Rx[0, 0]), "Ry_sha_prefix": float(Ry[0, 0]),
                 "orthogonality_error": float(max((Rx @ Rx.T - torch.eye(d_total, dtype=Rx.dtype)).abs().max(), (Ry @ Ry.T - torch.eye(d_total, dtype=Ry.dtype)).abs().max()))}


# ------------------------------------------------------------------------------------------------------------ fitted rows
def _eval_T(critic, E: Role, device, half: float):
    with torch.no_grad():
        fp = torch.cat([critic.pairs(_dev(E.xp[s:s + 65536], device), _dev(E.yp[s:s + 65536], device)).double().cpu() for s in range(0, E.n, 65536)])
        fn = torch.cat([critic.pairs(_dev(E.xq[s:s + 65536], device), _dev(E.yq[s:s + 65536], device)).double().cpu() for s in range(0, E.n, 65536)])
    return torch.tanh(half * fp), torch.tanh(half * fn)


def _post_mse(tp, tq, ep, eq) -> float:
    return float(0.5 * ((tp - ep) ** 2).mean() + 0.5 * ((tq - eq) ** 2).mean())


def neural_budget_rows(roles, tr, st, ds, dt, seed, device, kinds=("vcs", "js"), budgets=BUDGETS, lrs=LRS, B=256, eta_roles=None) -> list[dict]:
    """For each loss kind and budget b: P85 trainer (product negatives, SELECT native risk every 100 updates) for every lr; lr chosen by the
    lowest SELECT risk; both readouts of the resulting T on EVAL.  The same seed is used at every budget, so the run at budget b is the
    prefix of the run at a larger budget (early stopping restricted to updates <= b)."""
    E = roles["EVAL"]; Ee = eta_roles or E; ep, eq = eta_of(st, Ee.xp, Ee.yp, ds), eta_of(st, Ee.xq, Ee.yq, ds); rows = []
    for kind in kinds:
        half = 1.0 if kind == "vcs" else 0.5
        for b in budgets:
            group = []
            for lr in lrs:
                _peak_mem(device); critic, info = train_neural(kind, "product", lr, roles, dt, B, b, seed, device)
                t0 = time.time(); tp, tq = _eval_T(critic, E, device, half); r = readouts(tp, tq)
                group.append({"method": f"neural:{kind}", "kind": kind, "budget_updates": b, "lr": lr, "selected": False, "T_map": "tanh f" if kind == "vcs" else "tanh(f/2)",
                              **r, "J_err": r["J"] - tr["S"], "S_plug_err": r["S_plug"] - tr["S"], "posterior_mse": _post_mse(tp, tq, ep, eq),
                              "select_risk": info["select_risk"], "selected_update": info["selected_update"], "fit_seconds": info["fit_seconds"],
                              "eval_seconds": time.time() - t0, "nonfinite_steps": info["nonfinite_steps"], "memory": _peak_mem(device)})
            best = min(group, key=lambda r: r["select_risk"]); best["selected"] = True
            for r in group:
                r["tuning_seconds_total"] = sum(g["fit_seconds"] for g in group)
            rows += group
    return rows


def reference_rows(roles, tr, dt, seed, device, budget=BUDGETS[-1], lrs=LRS, B=256) -> list[dict]:
    """MINE (DV) / NWJ / InfoNCE / SMILE at the largest budget, lr by SELECT; native value against its OWN truth only (no common raw number)."""
    rows = []
    for kind, variant in REF_KINDS:
        group = []
        for lr in lrs:
            critic, info = train_neural(kind, variant, lr, roles, dt, B, budget, seed, device)
            gen = torch.Generator().manual_seed(seed * 31 + 5); a, b, how, _, _, _ = _native_on(critic, kind, variant, roles["EVAL"], device, 1024, gen)
            v = combine(a, b, how); own = tr[TARGET[kind]]
            group.append({"method": f"neural:{kind}:{variant}", "kind": kind, "budget_updates": budget, "lr": lr, "selected": False, "native_value": v, "own_truth": own,
                          "own_target": TARGET[kind], "signed_error": v - own, "rel_error": (v - own) / own if own else None, "select_risk": info["select_risk"],
                          "fit_seconds": info["fit_seconds"], "nonfinite_steps": info["nonfinite_steps"]})
        min(group, key=lambda r: r["select_risk"])["selected"] = True; rows += group
    return rows


def kernel_rows(roles, tr, st, ds, dt, seed, device, budgets=BUDGETS, mults=(0.5, 1.0, 2.0), m=1024, lr=5e-4, B=256, eta_roles=None) -> list[dict]:
    """E2 kernel side, same selection programme in both coordinate systems: S-KDE (bandwidth by max J_kernel on TUNE; S_plug and J both read)
    and the RFF S-kernel read-out (m features, bandwidth multiples of the FIT median, trained on J with SELECT early stopping, per budget)."""
    F, TU, S, E = roles["FIT"], roles["TUNE"], roles["SELECT"], roles["EVAL"]; Ee = eta_roles or E
    ep, eq = eta_of(st, Ee.xp, Ee.yp, ds), eta_of(st, Ee.xq, Ee.yq, ds); rows = []; dev = lambda t: t.to(device)
    _peak_mem(device); t0 = time.time()
    sel = KC.select_kde_bandwidth(dev(F.xp), dev(F.yp), dev(TU.xp), dev(TU.yp), dev(TU.xq), dev(TU.yq))
    t_sel = time.time() - t0; t1 = time.time()
    with torch.no_grad():
        tp = KC.s_kde_eta(dev(E.xp), dev(E.yp), dev(F.xp), dev(F.yp), sel["h"], sel["b"]).cpu().double()
        tq = KC.s_kde_eta(dev(E.xq), dev(E.yq), dev(F.xp), dev(F.yp), sel["h"], sel["b"]).cpu().double()
    r = readouts(tp, tq)
    rows.append({"method": "s_kde:common_risk", "kind": "s_kde", "budget_updates": None, "selected": True, **r, "J_err": r["J"] - tr["S"], "S_plug_err": r["S_plug"] - tr["S"],
                 "posterior_mse": _post_mse(tp, tq, ep, eq), "fit_seconds": t_sel, "tuning_seconds_total": t_sel, "eval_seconds": time.time() - t1,
                 "bandwidth": {"h": sel["h"], "b": sel["b"]}, "memory": _peak_mem(device)})
    W = torch.cat([torch.cat([F.xp, F.yp], 1), torch.cat([F.xq, F.yq], 1)], 0); medw = KC.median_distance(W)
    for b in budgets:
        group = []
        for mult in mults:
            _peak_mem(device); torch.manual_seed(seed * 7919 + m); model = KC.RFFTanh(2 * dt, m, mult * medw, seed=seed * 100 + m)
            model, info = train(model, native_loss("vcs"), F, S, lr=lr, batch=B, updates=b, seed=seed, device=device)
            t2 = time.time()
            with torch.no_grad():
                fp = torch.cat([model(_dev(E.xp[s:s + 16384], device), _dev(E.yp[s:s + 16384], device)).double().cpu() for s in range(0, E.n, 16384)])
                fn = torch.cat([model(_dev(E.xq[s:s + 16384], device), _dev(E.yq[s:s + 16384], device)).double().cpu() for s in range(0, E.n, 16384)])
            tp, tq = torch.tanh(fp), torch.tanh(fn); r = readouts(tp, tq)
            group.append({"method": "s_kernel:rff", "kind": "s_kernel_rff", "budget_updates": b, "rff_m": m, "bandwidth_multiple": mult, "lr": lr, "selected": False, **r,
                          "J_err": r["J"] - tr["S"], "S_plug_err": r["S_plug"] - tr["S"], "posterior_mse": _post_mse(tp, tq, ep, eq), "select_risk": info["select_risk"],
                          "selected_update": info["selected_update"], "fit_seconds": info["fit_seconds"], "eval_seconds": time.time() - t2, "memory": _peak_mem(device)})
        min(group, key=lambda r: r["select_risk"])["selected"] = True
        for r in group:
            r["tuning_seconds_total"] = sum(g["fit_seconds"] for g in group)
        rows += group
    return rows


# ------------------------------------------------------------------------------------------------------------ cells
def e1_cell(cond: str, N: int, seed: int, device, smoke: bool = False, refs: bool = True) -> dict:
    name, ds, dt, I = CONDITIONS[cond]; st = setting_from_mi(name, I, ds)
    if name == "xor_mixture":
        ds = dt = st.d
    t0 = time.time(); roles, sizes = build_roles(st, ds, dt, N, seed, smoke); tr = truths(st, ds, roles["TRUTH"])
    budgets = (50, 100) if smoke else BUDGETS; lrs = (5e-4,) if smoke else LRS
    rows = neural_budget_rows(roles, tr, st, ds, dt, seed, device, budgets=budgets, lrs=lrs, B=min(256, N))
    ref = reference_rows(roles, tr, dt, seed, device, budget=budgets[-1], lrs=lrs, B=min(256, N)) if refs else []
    return {"protocol": PROTOCOL, "unit": "E1", "condition": cond, "setting": name, "d_signal": ds, "d_total": dt, "I": I, "N": N, "seed": seed, "sizes": sizes,
            "truth": {k: tr[k] for k in ("S", "S_se", "JS2", "MI", "J_oracle")}, "rows": rows, "reference_rows": ref, "wall_seconds": time.time() - t0, "device": str(device)}


def e2_cell(cond: str, rotated: bool, N: int, seed: int, device, smoke: bool = False) -> dict:
    name, ds, dt, I = CONDITIONS[cond]; st = setting_from_mi(name, I, ds)
    t0 = time.time(); roles, sizes = build_roles(st, ds, dt, N, seed, smoke); tr = truths(st, ds, roles["TRUTH"]); rot = None; eta_roles = None
    if rotated:
        roles, rot = rotate_roles(roles, dt); eta_roles = rot.pop("orig_eval")
    budgets = (50, 100) if smoke else BUDGETS; lrs = (5e-4,) if smoke else LRS
    rows = neural_budget_rows(roles, tr, st, ds, dt, seed, device, budgets=budgets, lrs=lrs, B=min(256, N), eta_roles=eta_roles)
    rows += kernel_rows(roles, tr, st, ds, dt, seed, device, budgets=budgets, m=256 if smoke else 1024, B=min(256, N), eta_roles=eta_roles)
    return {"protocol": PROTOCOL, "unit": "E2", "condition": cond, "rotated": rotated, "rotation": rot, "setting": name, "d_signal": ds, "d_total": dt, "I": I, "N": N,
            "seed": seed, "sizes": sizes, "truth": {k: tr[k] for k in ("S", "S_se", "JS2", "MI", "J_oracle")}, "rows": rows, "wall_seconds": time.time() - t0, "device": str(device)}


def e1_cells(seeds=(0, 1, 2, 3, 4), Ns=(256, 1024, 4096, 16384)) -> list[tuple]:
    return [("E1", c, N, s) for c in CONDITIONS for N in Ns for s in seeds]


def e2_cells(seeds=(0, 1, 2), Ns=(1024, 4096, 16384)) -> list[tuple]:
    return [("E2", c, rot, N, s) for c in E2_CONDITIONS for rot in (False, True) for N in Ns for s in seeds]
