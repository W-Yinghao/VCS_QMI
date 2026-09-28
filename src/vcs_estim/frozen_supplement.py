"""P83 addendum 1 (package v2 Spec §8.3, §9.1; Plan §2.3, §8.1): the limited supplement to the P84 frozen-checkpoint diagnostics.

    python -m vcs_estim.frozen_supplement --run-dir <outputs/RUN> --ckpt epoch_800 --p84 <P84 json> --out <json> [--smoke] [--cpu]
    python -m vcs_estim.frozen_supplement --stage aggregate --inputs <jsons> --out <md>

Reads the same checkpoints with the same roles / seeds / pair constructions as `vcs_estim.frozen` (roles hash asserted against the P84
JSON) and adds only:
  1. the actual gate 1 - E_M T^2 of the training critic (P and Q sides, quantiles) next to 1 - J, and its Jacobian |dT/dz| on EVAL;
  2. dictionary disagreement D(w) = diag(G)'w - w'Gw (G = E_M T T' on TUNE) for the P84 mix weights (read from the P84 JSON; the member
     outputs are re-fitted with P84's seeds because P84 stored no per-member TUNE outputs — disclosed), the refit's own simplex weights,
     max_w D(w) over the simplex (concave QP) and the global bound U_dict (range and pair forms), with block-bootstrap uncertainty;
  3. the converged cosine C0 (full-batch L-BFGS on FIT) with its fitting trajectory, reported as a different object from the 2000-step C0;
  4. the P48 tanh-wrapped ridge on phi = [z1 * z2; 1] (FIT moments, lambda = 1e-4 x mean diag, c fitted on FIT) as a bounded member; the raw
     ridge listed separately (unbounded, never in the bounded set);
  5. the extended dictionary {C0, C1, C2, C0_conv, tanh_ridge}: TUNE simplex fit, D(w), max D, U_dict, EVAL J vs the best single member;
  6. the converged-C0 + C2 two-member combination as a labelled diagnostic;
  7. scores and Jacobians of every member in the training critic's units (|dT/dz_left|, |dT/dz_right|, gate) on the same EVAL rows.
EVAL is used once for reporting; no selection is made on it (Spec §4, §8.3).  Identity check on EVAL: J(T_w) = sum_j w_j J(T_j) + D(w).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar

from . import candidates as C
from .bounded_core import fit_small_simplex
from .fitting import native_loss, outputs, train
from .frozen import DICT, block_J, block_pairs, critic_diag, encode_role, fit_pairs, load_models
from .kernel_cs import dictionary_disagreement, dictionary_upper_bound
from .pairing import ROLE_SEED, check_disjoint, role_split, split_hash
from .run import SOURCE_REF, git_commit

QS = [0.01, 0.1, 0.5, 0.9, 0.99]


# ------------------------------------------------------------------------------------------------------------ dictionary geometry
def m_weights(n_p: int, n_q: int) -> np.ndarray:
    """Row weights of the empirical M = 1/2 P + 1/2 Q for stacked [P rows; Q rows]."""
    return np.concatenate([np.full(n_p, 0.5 / n_p), np.full(n_q, 0.5 / n_q)])


def gram(Tm: np.ndarray, wM: np.ndarray) -> np.ndarray:
    return (Tm * wM[:, None]).T @ Tm / wM.sum()


def disagreement_from_G(G: np.ndarray, w: np.ndarray) -> float:
    return float(np.diag(G) @ w - w @ G @ w)


def max_disagreement(G: np.ndarray, starts: str = "all") -> dict:
    """max_{w in simplex} diag(G)'w - w'Gw: concave (G is PSD), so SLSQP from the uniform point reaches the global maximum; the vertex
    starts are a safety net."""
    m = len(G); f = lambda w: -(np.diag(G) @ w - w @ G @ w); jac = lambda w: -(np.diag(G) - 2.0 * G @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(m)}]
    w0s = [np.full(m, 1.0 / m)] + ([np.eye(m)[j] * 0.9 + 0.1 / m for j in range(m)] if starts == "all" else [])
    best = None
    for w0 in w0s:
        r = minimize(f, w0, jac=jac, bounds=[(0.0, 1.0)] * m, constraints=cons, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
        w = np.clip(r.x, 0.0, None); w = w / w.sum(); v = disagreement_from_G(G, w)
        if best is None or v > best[1]:
            best = (w, v)
    return {"weights": best[0].tolist(), "max_D": best[1]}


def bounds_from_G(G: np.ndarray, range_term: float) -> dict:
    m = len(G); ub = 0.5 * (1.0 - 1.0 / m) * max(G[j, j] + G[k, k] - 2.0 * G[j, k] for j in range(m) for k in range(m))
    return {"U_dict": min(0.25 * range_term, ub), "range_bound": 0.25 * range_term, "pair_bound": ub}


def block_bootstrap(TP: np.ndarray, TQ: np.ndarray, weight_sets: dict, reps: int = 500, seed: int = 20260932) -> dict:
    """TUNE blocks: row i of TP (positive pair) and row i of TQ (product pair) share the anchor -> one block.  Resamples blocks and
    recomputes D(w) for each weight set, max_w D(w) and U_dict."""
    n, m = TP.shape; rng = np.random.default_rng(seed)
    Gi = 0.5 * (TP[:, :, None] * TP[:, None, :] + TQ[:, :, None] * TQ[:, None, :])
    ri = 0.5 * ((TP.max(1) - TP.min(1)) ** 2 + (TQ.max(1) - TQ.min(1)) ** 2)
    acc = {k: [] for k in ("max_D", "U_dict", "range_bound", "pair_bound", *[f"D_{k}" for k in weight_sets])}
    for _ in range(reps):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64); c /= c.sum()
        G = np.einsum("i,ijk->jk", c, Gi); b = bounds_from_G(G, float(c @ ri))
        acc["max_D"].append(max_disagreement(G, starts="uniform")["max_D"])
        for k in ("U_dict", "range_bound", "pair_bound"):
            acc[k].append(b[k])
        for k, w in weight_sets.items():
            acc[f"D_{k}"].append(disagreement_from_G(G, np.asarray(w, np.float64)))
    return {k: {"sd": float(np.std(v, ddof=1)), "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]} for k, v in acc.items()} | {"reps": reps, "n_blocks": n}


# ------------------------------------------------------------------------------------------------------------ extra members
class RidgeMember(torch.nn.Module):
    """f(x, y) = c * w'[x * y; 1]; T = tanh f.  c = 1 and no tanh gives the raw ridge."""
    def __init__(self, w: torch.Tensor, c: float):
        super().__init__(); self.register_buffer("w", w.double()); self.register_buffer("c", torch.tensor(float(c), dtype=torch.float64))

    def phi(self, x, y):
        return torch.cat([x * y, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], 1)

    def forward(self, x, y):
        return (self.c * (self.phi(x, y) @ self.w.to(x.dtype))).to(x.dtype)


def _j(tp, tn):
    return float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean())


def fit_tanh_ridge(FIT, lam_rel: float = 1e-4) -> tuple[RidgeMember, dict]:
    """P48 closed form on FIT: d = E_P phi - E_Q phi, A = 1/2 (E_P phi phi' + E_Q phi phi'), w* = 1/2 (A + lam I)^-1 d; c on FIT only."""
    t0 = time.time(); one = lambda x: torch.cat([x, torch.ones(len(x), 1, dtype=torch.float64)], 1)
    pp, pq = one((FIT.xp * FIT.yp).double()), one((FIT.xq * FIT.yq).double())
    d = pp.mean(0) - pq.mean(0); A = 0.5 * (pp.T @ pp / len(pp) + pq.T @ pq / len(pq)); scale = float(torch.diag(A).mean())
    w = 0.5 * torch.linalg.solve(A + lam_rel * scale * torch.eye(len(A), dtype=torch.float64), d)
    rp, rq = pp @ w, pq @ w
    Jt = lambda c: _j(torch.tanh(c * rp), torch.tanh(c * rq))
    cs = torch.logspace(-1, 1.5, 40); grid = [Jt(float(c)) for c in cs]; c0 = float(cs[int(np.argmax(grid))])
    r = minimize_scalar(lambda lc: -Jt(math.exp(lc)), bounds=(math.log(c0) - 0.5, math.log(c0) + 0.5), method="bounded", options={"xatol": 1e-6})
    c = math.exp(r.x) if -r.fun >= max(grid) else c0
    info = {"lambda_rel": lam_rel, "lambda_abs": lam_rel * scale, "dim": int(len(w)), "c": c, "J_fit_tanh": Jt(c), "J_fit_raw": _j(rp, rq),
            "frac_abs_T_gt_1_fit_pos": float((rp.abs() > 1).double().mean()), "frac_abs_T_gt_1_fit_neg": float((rq.abs() > 1).double().mean()),
            "J_star_formula_fit": float(0.25 * d @ torch.linalg.solve(A + lam_rel * scale * torch.eye(len(A), dtype=torch.float64), d)), "fit_seconds": time.time() - t0,
            "source": "P48 / plan appendix B closed form; features phi = [z1 * z2; 1] (P48 class P1); c fitted on FIT (grid + bounded refinement)"}
    return RidgeMember(w, c), info


def fit_c0_converged(d: int, c0_std, FIT, device, max_rounds: int = 40, iters_per_round: int = 25) -> tuple[torch.nn.Module, dict]:
    """Full-batch L-BFGS on FIT for the two-parameter cosine class (same FIT standardisation as P84), logging the trajectory per round."""
    t0 = time.time(); model = C.build("C0", d, c0_a0=0.0, c0_std=c0_std).to(device); Fp = [t.float().to(device) for t in (FIT.xp, FIT.yp, FIT.xq, FIT.yq)]
    lossf = native_loss("vcs"); evals = [0]
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=iters_per_round, tolerance_grad=1e-10, tolerance_change=1e-13, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); l = lossf(model, *Fp); l.backward(); evals[0] += 1; return l

    def snap(r):
        with torch.no_grad():
            l = float(lossf(model, *Fp))
        return {"round": r, "closure_evals": evals[0], "loss_minus_J_fit": l, "a": float(model.a), "b": float(model.b)}
    traj = [snap(0)]
    for r in range(1, max_rounds + 1):
        opt.step(closure); traj.append(snap(r))
        if abs(traj[-1]["loss_minus_J_fit"] - traj[-2]["loss_minus_J_fit"]) < 1e-12:
            break
    mu, sd = c0_std; a, b = float(model.a), float(model.b)
    return model.eval(), {"trajectory": traj, "rounds": len(traj) - 1, "closure_evals": evals[0], "converged": abs(traj[-1]["loss_minus_J_fit"] - traj[-2]["loss_minus_J_fit"]) < 1e-12 if len(traj) > 1 else False,
                          "a_std": a, "b_std": b, "a_eff": a / sd, "b_eff": b - a * mu / sd, "J_fit": -traj[-1]["loss_minus_J_fit"], "fit_seconds": time.time() - t0,
                          "note": "a_eff / b_eff are in the training critic's units (f = a_eff <z1, z2> + b_eff)"}


def jacobian_norms(T_fn, P, n: int, device) -> dict:
    """|dT/dz_left|, |dT/dz_right| and the gate on the first n EVAL positive and product rows (same rows for every member)."""
    out = {}
    for side, (x, y) in (("pos", (P.xp, P.yp)), ("neg", (P.xq, P.yq))):
        xx = x[:n].float().to(device).requires_grad_(True); yy = y[:n].float().to(device).requires_grad_(True)
        T = T_fn(xx, yy); gx, gy = torch.autograd.grad(T.sum(), (xx, yy))
        out[side] = {"dT_dz_left_mean": float(gx.norm(dim=1).mean()), "dT_dz_right_mean": float(gy.norm(dim=1).mean()), "gate_mean": float((1 - T.detach() ** 2).mean()),
                     "T_quantiles": np.quantile(T.detach().cpu().numpy(), QS).tolist(), "n": int(len(xx))}
    return out


def scores_of(model, P, device):
    return torch.tanh(outputs(model, P.xp, P.yp, device)), torch.tanh(outputs(model, P.xq, P.yq, device))


def dictionary_block(members: dict, TU, E, device, base_J, w_ref: dict | None, reps: int) -> dict:
    """TUNE geometry (G, D, max D, U_dict, bootstrap) and the EVAL readout of the members, the refit mix and the reference weights."""
    names = list(members); S_T = {k: scores_of(m, TU, device) for k, m in members.items()}; S_E = {k: scores_of(m, E, device) for k, m in members.items()}
    TTp = torch.stack([S_T[k][0] for k in names], 1).numpy(); TTq = torch.stack([S_T[k][1] for k in names], 1).numpy()
    ETp = torch.stack([S_E[k][0] for k in names], 1).numpy(); ETq = torch.stack([S_E[k][1] for k in names], 1).numpy()
    wM = m_weights(len(TTp), len(TTq)); Tm = np.concatenate([TTp, TTq]); G = gram(Tm, wM)
    wE = m_weights(len(ETp), len(ETq)); Em = np.concatenate([ETp, ETq])
    fit = fit_small_simplex(TTp, TTq); w_fit = np.asarray(fit.weights, np.float64)
    ub_lib = dictionary_upper_bound(Tm, wM); ub_G = bounds_from_G(G, float((wM * (Tm.max(1) - Tm.min(1)) ** 2).sum() / wM.sum()))
    memberJ = {k: block_J(torch.tensor(ETp[:, i]), torch.tensor(ETq[:, i])) for i, k in enumerate(names)}
    best = max(memberJ, key=lambda k: memberJ[k]["J"])

    def readout(w, label):
        w = np.asarray(w, np.float64); mix = block_J(torch.tensor(ETp @ w), torch.tensor(ETq @ w))
        D_t, D_e = dictionary_disagreement(Tm, w, wM), dictionary_disagreement(Em, w, wE)
        return {"label": label, "weights": w.tolist(), "D_tune": D_t, "D_tune_from_G": disagreement_from_G(G, w), "J_eval": mix["J"], "J_eval_se_block": mix["J_se_block"],
                "sum_wJ_eval": float(sum(w[i] * memberJ[k]["J"] for i, k in enumerate(names))), "D_eval": D_e,
                "identity_residual_eval": float(mix["J"] - sum(w[i] * memberJ[k]["J"] for i, k in enumerate(names)) - D_e),
                "gain_over_best_single_eval": mix["J"] - memberJ[best]["J"], "gain_bounded_by_D_tune": bool(mix["J"] - memberJ[best]["J"] <= D_t + 1e-9),
                "delta_J_vs_training_critic": (mix["J"] - base_J) if base_J is not None else None}
    out = {"members": names, "G_tune": G.tolist(), "member_J_eval": memberJ, "best_single_eval": best, "simplex_fit_tune": readout(w_fit, "refit simplex (TUNE)"),
           "tune_objective": fit.objective, "kkt_gap": fit.kkt_gap, "max_D": max_disagreement(G), "U_dict": {**ub_lib, "from_G": ub_G},
           "bound_check": {"D_fit_le_maxD": bool(disagreement_from_G(G, w_fit) <= max_disagreement(G)["max_D"] + 1e-9), "maxD_le_U": bool(max_disagreement(G)["max_D"] <= ub_lib["U_dict"] + 1e-9)}}
    ws = {"fit": w_fit}
    if w_ref is not None:
        out["reference_weights"] = readout(w_ref["weights"], w_ref["label"]); ws["ref"] = np.asarray(w_ref["weights"], np.float64)
        out["reference_weights"]["max_abs_weight_diff_vs_refit"] = float(np.abs(ws["ref"] - w_fit).max())
    out["bootstrap"] = block_bootstrap(TTp, TTq, ws, reps=reps)
    return out


# ------------------------------------------------------------------------------------------------------------ main stage
def run_checkpoint(a) -> dict:
    from vcs_ssl.data.cifar import load_cifar10_train
    from vcs_ssl.data.transforms import build_two_view_transform
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"; t_all = time.time()
    run_dir = Path(a.run_dir); cfg, encoder, projector, critic, meta = load_models(run_dir, a.ckpt, dev)
    P84 = json.load(open(a.p84)) if a.p84 and Path(a.p84).exists() else None
    eps = float(cfg["model"]["normalization"]["eps"]); tf = build_two_view_transform(cfg["views"])
    manifest = json.load(open(cfg["data"]["manifest"])); roles = role_split(manifest["fit_uids"], ROLE_SEED); check_disjoint(roles)
    if a.smoke:
        roles = {r: v[: {"FIT": 1024, "TUNE": 256, "SELECT": 256, "EVAL": 256}[r]] for r, v in roles.items()}
    rh = split_hash(roles)
    if P84 is not None and not a.smoke and P84["roles"]["hash"] != rh:
        raise RuntimeError("roles hash differs from the P84 JSON; the supplement must read the same roles")
    images = load_cifar10_train(cfg["data"]["root"]).data
    t0 = time.time(); Z = {}
    for i, (r, ids) in enumerate(roles.items()):
        Z[r] = encode_role(encoder, projector, images, ids, tf, 20260928 + 17 * i, dev, eps, workers=a.workers)
    feat_s = time.time() - t0
    FIT, fit_info = fit_pairs(Z["FIT"][0], Z["FIT"][1], 8, 20260929); RP = {"FIT": FIT}
    for i, r in enumerate(("TUNE", "SELECT", "EVAL")):
        RP[r], _ = block_pairs(Z[r][0], Z[r][1], roles[r], 20260930 + i)
    E, TU = RP["EVAL"], RP["TUNE"]; d = FIT.xp.shape[1]
    R = {"checkpoint": {"run_dir": str(run_dir), "ckpt": a.ckpt, **meta, "method_cfg": cfg["run"]["method"]}, "code_commit": git_commit(), "source_reference_commit": SOURCE_REF,
         "roles": {"seed": ROLE_SEED, "sizes": {r: len(v) for r, v in roles.items()}, "hash": rh, "matches_p84": (P84["roles"]["hash"] == rh) if P84 else None},
         "p84_json": a.p84, "device": dev, "feature_seconds": feat_s, "eval_once": "EVAL read for reporting only; no selection on EVAL",
         "disclosure": "P84 stored no per-member TUNE outputs; the C0 / C1 / C2 members are re-fitted here with P84's seeds and budget (reproduction recorded)"}

    # 1. training critic: actual gate and Jacobian
    base_J = None
    if critic is not None:
        crit = copy.deepcopy(critic).eval()
        for p in crit.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            tp, tn = crit(E.xp.to(dev), E.yp.to(dev)).double().cpu(), crit(E.xq.to(dev), E.yq.to(dev)).double().cpu()
        bj = block_J(tp, tn); base_J = bj["J"]; gp, gn = 1 - tp ** 2, 1 - tn ** 2
        R["training_critic"] = {**bj, "critic_class": type(critic).__name__, "params": ({k: float(v) for k, v in critic.named_parameters()} if sum(p.numel() for p in critic.parameters()) <= 4 else None),
                                "actual_gate": {"E_M_gate": float(0.5 * gp.mean() + 0.5 * gn.mean()), "one_minus_J": 1 - bj["J"], "E_M_T2": float(0.5 * (tp ** 2).mean() + 0.5 * (tn ** 2).mean()),
                                                "gate_pos": {"mean": float(gp.mean()), "quantiles": np.quantile(gp.numpy(), QS).tolist()},
                                                "gate_neg": {"mean": float(gn.mean()), "quantiles": np.quantile(gn.numpy(), QS).tolist()},
                                                "note": "1 - E_M T^2 is the actual mean gate; 1 - J is not a substitute (Plan §6.2)"},
                                "score_diagnostics": critic_diag(tp, tn), "jacobian_eval": jacobian_norms(crit, E, a.jac_n, dev),
                                "fit_gate": (lambda tpf, tnf: {"E_M_gate": float(0.5 * (1 - tpf ** 2).mean() + 0.5 * (1 - tnf ** 2).mean()), "J_fit": _j(tpf, tnf)})(
                                    *[crit(x.to(dev), y.to(dev)).double().cpu() for x, y in ((FIT.xp, FIT.yp), (FIT.xq, FIT.yq))] if not a.smoke else (tp, tn))}
        T_train = crit
    else:
        R["training_critic"] = None; R["training_critic_absent_reason"] = "method has no critic (measurement critics only)"; T_train = None

    # 2. members: P84 refits (same seeds), converged C0, tanh ridge (+ raw ridge separately)
    s_fit = torch.cat([(FIT.xp * FIT.yp).sum(-1), (FIT.xq * FIT.yq).sum(-1)]).double(); c0_std = (float(s_fit.mean()), float(s_fit.std()))
    members, refit_info = {}, {}
    for fam in DICT:
        torch.manual_seed(1000 * a.seed + {"C0": 1, "C1": 2, "C2": 3}[fam])
        m, info = train(C.build(fam, d, c0_a0=a.c0_a0, c0_std=c0_std), native_loss("vcs"), FIT, RP["SELECT"], updates=a.updates, seed=a.seed, device=dev)
        members[fam] = m; refit_info[fam] = {k: info[k] for k in ("selected_update", "select_risk", "fit_seconds")}
        if fam == "C0":
            mu, sd = c0_std; refit_info[fam].update({"a_std": float(m.a), "b_std": float(m.b), "a_eff": float(m.a) / sd, "b_eff": float(m.b) - float(m.a) * mu / sd})
    c0c, c0c_info = fit_c0_converged(d, c0_std, FIT, dev, max_rounds=(8 if a.smoke else 40)); members["C0_conv"] = c0c
    ridge, ridge_info = fit_tanh_ridge(FIT); members["tanh_ridge"] = ridge.to(dev)
    raw = RidgeMember(ridge.w.cpu(), 1.0).to(dev)
    raw_T = {k: (outputs(raw, P.xp, P.yp, dev), outputs(raw, P.xq, P.yq, dev)) for k, P in (("TUNE", TU), ("EVAL", E))}
    R["members"] = {"p84_refits": refit_info, "C0_converged": c0c_info, "tanh_ridge": ridge_info,
                    "raw_ridge_separate": {k: {"J": _j(*v), "frac_abs_T_gt_1_pos": float((v[0].abs() > 1).double().mean()), "frac_abs_T_gt_1_neg": float((v[1].abs() > 1).double().mean())} for k, v in raw_T.items()}
                    | {"note": "unbounded raw ridge: reference only, never a member of a bounded dictionary"}}
    if P84 is not None:
        rows = {(r["estimator"], r["critic_family"]): r for r in P84["rows"]}
        R["members"]["p84_reproduction"] = {fam: {"J_eval_p84": rows[("vcs_single", fam)]["J_eval"], "J_eval_refit": block_J(*scores_of(members[fam], E, dev))["J"],
                                                  "selected_update_p84": rows[("vcs_single", fam)]["selected_checkpoint"], "selected_update_refit": refit_info[fam]["selected_update"]} for fam in DICT}
        conv = [r for r in P84["rows"] if r["estimator"] == "vcs_single_diagnostic"]
        if conv:
            R["members"]["C0_converged"]["p84_converged_C0"] = {"a_std": conv[0]["a"], "b_std": conv[0]["b"], "J_eval": conv[0]["J_eval"]}

    # 3. dictionaries
    w84 = {"weights": P84["combos"]["vcs_mix"]["combination_weights"], "label": "P84 stored mix weights"} if P84 is not None else None
    R["dictionary_p84"] = dictionary_block({k: members[k] for k in DICT}, TU, E, dev, base_J, w84, a.boot_reps)
    if P84 is not None:
        R["dictionary_p84"]["p84_mix_J_eval_stored"] = P84["combos"]["vcs_mix"]["J_eval"]
    R["dictionary_extended"] = dictionary_block(members, TU, E, dev, base_J, None, a.boot_reps)
    R["c0conv_plus_c2"] = dictionary_block({"C0_conv": members["C0_conv"], "C2": members["C2"]}, TU, E, dev, base_J, None, a.boot_reps)
    R["c0conv_plus_c2"]["label"] = "labelled diagnostic: converged two-parameter cosine + C2 (not the spec's per-candidate budget)"

    # 4. Jacobians / scores in training-critic units, same EVAL rows
    jac = {k: jacobian_norms(lambda x, y, m=m: torch.tanh(m(x, y)), E, a.jac_n, dev) for k, m in members.items()}
    if T_train is not None:
        jac["training_critic"] = R["training_critic"]["jacobian_eval"]
        ref = jac["training_critic"]
        for k in members:
            jac[k]["ratio_to_training_critic"] = {s: {"dT_dz_left": jac[k][s]["dT_dz_left_mean"] / max(ref[s]["dT_dz_left_mean"], 1e-12), "gate": jac[k][s]["gate_mean"] / max(ref[s]["gate_mean"], 1e-12)} for s in ("pos", "neg")}
    R["jacobians_eval"] = jac
    R["wall_seconds"] = time.time() - t_all
    return R


def markdown(results: list[dict]) -> str:
    L = ["# P83 addendum 1 — P84 supplement: actual gate, dictionary disagreement bounds, converged cosine, ridge member, Jacobians", "",
         "TUNE geometry (D, max D, U_dict; block-bootstrap 95 % CI), EVAL read once.  D(w) = diag(G)'w − w'Gw is the exact gain of the mix over the weighted member average; "
         "J(T_w) − max_j J(T_j) ≤ D(w).  C0_conv = converged two-parameter cosine (L-BFGS); C0 = the 2000-update member.  Jacobians |∂T/∂z_left| on the same EVAL rows.", "",
         "| run | ckpt | train-critic J | actual gate E_M(1−T²) (1−J) | |∂T/∂z| train | D(w_P84) [CI] | max D P84 dict [CI] | U_dict P84 [CI] | ext dict: max D / U_dict | ext mix J_eval − best single | C0_conv+C2 J_eval − C2 | C0_conv a_eff, b_eff, J_fit (C0: a_eff, b_eff) | tanh-ridge J_eval (raw |T|>1) | |∂T/∂z| C2 / C0_conv / ridge (÷ train) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for R in results:
        ck, tc = R["checkpoint"], R.get("training_critic"); D84, Dx, Dc = R["dictionary_p84"], R["dictionary_extended"], R["c0conv_plus_c2"]; jac = R["jacobians_eval"]
        b84 = D84["bootstrap"]; ref = D84.get("reference_weights")
        ci = lambda v, b: f"{v:.4f} [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]"
        c0 = R["members"]["C0_converged"]; c0r = R["members"]["p84_refits"]["C0"]
        jr = lambda k: (f"{jac[k]['pos']['dT_dz_left_mean']:.3g}" + (f" (×{jac[k]['ratio_to_training_critic']['pos']['dT_dz_left']:.2f})" if "ratio_to_training_critic" in jac[k] else ""))
        L.append(f"| {Path(ck['run_dir']).name} | {ck['ckpt']} | {(f'{tc['J']:.4f}') if tc else '—'} | "
                 + (f"{tc['actual_gate']['E_M_gate']:.4f} ({tc['actual_gate']['one_minus_J']:.4f})" if tc else "—") + " | " + (f"{jac['training_critic']['pos']['dT_dz_left_mean']:.3g}" if tc else "—") + " | "
                 + (ci(ref["D_tune"], b84["D_ref"]) if ref else "—") + f" | {ci(D84['max_D']['max_D'], b84['max_D'])} | {ci(D84['U_dict']['U_dict'], b84['U_dict'])} | "
                 f"{Dx['max_D']['max_D']:.4f} / {Dx['U_dict']['U_dict']:.4f} | {Dx['simplex_fit_tune']['gain_over_best_single_eval']:+.4f} ({Dx['best_single_eval']}) | "
                 f"{Dc['simplex_fit_tune']['J_eval'] - Dc['member_J_eval']['C2']['J']:+.4f} | {c0['a_eff']:.3f}, {c0['b_eff']:.3f}, {c0['J_fit']:.4f} ({c0r['a_eff']:.3f}, {c0r['b_eff']:.3f}) | "
                 f"{Dx['member_J_eval']['tanh_ridge']['J']:.4f} ({R['members']['raw_ridge_separate']['EVAL']['frac_abs_T_gt_1_pos']:.2f}) | {jr('C2')} / {jr('C0_conv')} / {jr('tanh_ridge')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="supplement", choices=["supplement", "aggregate"])
    ap.add_argument("--run-dir"); ap.add_argument("--ckpt"); ap.add_argument("--p84", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="*"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--c0-a0", type=float, default=0.0); ap.add_argument("--boot-reps", type=int, default=500); ap.add_argument("--jac-n", type=int, default=2048)
    a = ap.parse_args()
    if a.stage == "aggregate":
        res = [json.load(open(p)) for p in a.inputs]; Path(a.out).write_text(markdown(res)); print(Path(a.out).read_text()); return
    if a.smoke:
        a.updates = min(a.updates, 100); a.boot_reps = min(a.boot_reps, 50); a.jac_n = min(a.jac_n, 128)
    R = run_checkpoint(a); R["args"] = vars(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(R, open(a.out, "w"), indent=1, default=float)
    tc = R["training_critic"]; D = R["dictionary_p84"]; X = R["dictionary_extended"]
    print(f"{a.run_dir} {a.ckpt}: " + (f"train J {tc['J']:.4f} gate {tc['actual_gate']['E_M_gate']:.4f} (1-J {tc['actual_gate']['one_minus_J']:.4f})" if tc else "no critic")
          + f" | P84 dict maxD {D['max_D']['max_D']:.4f} U {D['U_dict']['U_dict']:.4f}" + (f" D(w84) {D['reference_weights']['D_tune']:.5f}" if "reference_weights" in D else "")
          + f" | ext maxD {X['max_D']['max_D']:.4f} gain {X['simplex_fit_tune']['gain_over_best_single_eval']:+.4f} | C0conv a_eff {R['members']['C0_converged']['a_eff']:.3f} | {R['wall_seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
