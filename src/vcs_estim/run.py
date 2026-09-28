"""Estimator package v1, stages 1–2 on the Gaussian conditions (§6.1–6.4, §7.1–7.3 first batch).

    python -m vcs_estim.run --stage estimator_probe --I 0.5 --d 20 --seed 0 --out <json> [--updates 2000] [--smoke]

Per condition: VCS and matched-JS fits of C0 / C1 / C2 (+ the CQ diagnostic class); VCS C2 restarts (best-of-3 on SELECT, restart control);
the VCS dictionary {C0, C1, C2} simplex fit on TUNE; one VCS residual candidate (C2 structure, lambda0 = 0.5, lambda fitted on TUNE); the JS
dictionary control (native log-loss simplex weights on TUNE); every fixed model evaluated once on EVAL; channel-parameter gradients (§6.4).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import candidates as C
from .convex_mix import fit_js_mixture, fit_small_simplex, js_mixture_logq, residual_step
from .evaluation import drho_learned, drho_oracle_envelope, drho_oracle_fd, full_row, metrics
from .fitting import native_loss, outputs, residual_loss, train
from .objectives import js_native
from .synthetic import eta, make_gaussian, roles_hash, truth

SOURCE_REF = "7b7402c05161c33d77a4301f6efc27bb55420e6a"
DICT = ("C0", "C1", "C2")


def git_commit():
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src"], cwd=Path(__file__).resolve().parents[2]) != 0
        return c + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_probe(a) -> dict:
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    sizes = {"FIT": 512, "TUNE": 256, "SELECT": 256, "EVAL": 2048, "TRUTH": 20000} if a.smoke else None
    rho, D = make_gaussian(a.I, a.d, a.seed, sizes); tr = truth(a.I, a.d, rho, D["TRUTH"]); S = tr["S_truth"]
    E, TU = D["EVAL"], D["TUNE"]
    eta_E = (eta(E.xp, E.yp, rho), eta(E.xq, E.yq, rho))
    common = {"code_commit": git_commit(), "source_reference_commit": SOURCE_REF, "setting": "gaussian", "seed": a.seed, "target_kind": "S",
              "roles_hash": roles_hash(D), "negative_construction": "independent_units", "gradient_routing": "critic only (data fixed)",
              "noise_coordinate_sd": 0.0, "noise_total_rms": 0.0, "n_independent_units": {k: len(v.xp) for k, v in D.items()},
              "S_truth": S, "S_truth_se": tr["S_truth_se"]}
    R = {"condition": {**tr, "sizes": {k: len(v.xp) for k, v in D.items()}, "device": dev, "torch": torch.__version__,
                       "updates": a.updates, "lr": a.lr, "batch": 256}, "rows": [], "combos": {}, "gradients": {}}
    R["rows"].append({**common, "run_id": f"I{a.I}_oracle", "estimator": "oracle_eta", "critic_family": "oracle", "status": "completed",
                      **full_row(eta_E[0], eta_E[1], *eta_E, S)})
    fitted = {}

    def fit_one(kind, fam, seed, tag=None):
        torch.manual_seed(1000 * seed + {"C0": 1, "C1": 2, "C2": 3, "CQ": 4}[fam] + (0 if kind == "vcs" else 500))
        m = C.build(fam, a.d)
        m, info = train(m, native_loss(kind), D["FIT"], D["SELECT"], lr=a.lr, updates=a.updates, seed=seed, device=dev)
        t0 = time.time(); fE = (outputs(m, E.xp, E.yp, dev), outputs(m, E.xq, E.yq, dev)); ev = time.time() - t0
        fF = (outputs(m, D["FIT"].xp, D["FIT"].yp, dev), outputs(m, D["FIT"].xq, D["FIT"].yq, dev))
        eta_F = (eta(D["FIT"].xp, D["FIT"].yp, rho), eta(D["FIT"].xq, D["FIT"].yq, rho))
        row = {**common, "run_id": f"I{a.I}_{kind}_{fam}_s{seed}", "status": "completed", "estimator": f"{kind}_single",
               "critic_family": fam, "loss_scale_convention": "minus_J" if kind == "vcs" else "matched_JS", "trainable_parameters": C.n_params(m),
               "selected_checkpoint": info["selected_update"], "select_risk": info["select_risk"], "fit_seconds": info["fit_seconds"], "eval_seconds": ev,
               "select_curve": info["select_curve"], **full_row(torch.tanh(fE[0]), torch.tanh(fE[1]), *eta_E, S),
               **metrics(torch.tanh(fF[0]), torch.tanh(fF[1]), *eta_F, prefix="FIT_")}
        if kind == "js":
            from .objectives import js_match_loss
            row["JS_native_eval"] = js_native(float(js_match_loss(fE[0], fE[1])))
            row["note"] = "J_eval / posterior_mse of a JS model = the common-posterior VCS regression evaluation, not JS's native estimate"
        R["rows"].append(row); fitted[tag or (kind, fam, seed)] = (m, row)
        return m, row

    for kind in ("vcs", "js"):
        for fam in DICT + ("CQ",):
            fit_one(kind, fam, a.seed)
    for extra in (1, 2):                                    # restart control: same family / budget, independent initialisations
        fit_one("vcs", "C2", a.seed + extra)
    c2 = [fitted[("vcs", "C2", a.seed + s)] for s in (0, 1, 2)]
    best_restart = min(c2, key=lambda t: t[1]["select_risk"])
    R["combos"]["vcs_C2_best_of_3_restarts"] = {"selected_run": best_restart[1]["run_id"], "J_eval": best_restart[1]["J_eval"],
                                                "posterior_mse": best_restart[1]["posterior_mse"]}
    # best single of the dictionary by SELECT native risk
    sel = {fam: fitted[("vcs", fam, a.seed)][1]["select_risk"] for fam in DICT}
    best_fam = min(sel, key=sel.get)
    R["combos"]["vcs_best_single_by_select"] = {"family": best_fam, "select_risk": sel, "J_eval": fitted[("vcs", best_fam, a.seed)][1]["J_eval"],
                                                "posterior_mse": fitted[("vcs", best_fam, a.seed)][1]["posterior_mse"]}

    def dict_scores(kind, role):
        return (torch.stack([outputs(fitted[(kind, f, a.seed)][0], role.xp, role.yp, dev) for f in DICT], 1),
                torch.stack([outputs(fitted[(kind, f, a.seed)][0], role.xq, role.yq, dev) for f in DICT], 1))

    # VCS dictionary: exact simplex QP on TUNE, applied unchanged to EVAL
    fTp, fTn = dict_scores("vcs", TU); fEp, fEn = dict_scores("vcs", E)
    fit = fit_small_simplex(torch.tanh(fTp).numpy(), torch.tanh(fTn).numpy()); w = torch.tensor(fit.weights)
    tEp, tEn = torch.tanh(fEp) @ w, torch.tanh(fEn) @ w
    R["combos"]["vcs_mix"] = {**common, "estimator": "vcs_mix", "critic_family": "+".join(DICT), "combination_fit_role": "TUNE",
                              "combination_weights": fit.weights.tolist(), "tune_objective": fit.objective, "kkt_gap": fit.kkt_gap,
                              "tune_J_vertices": [float(torch.tanh(fTp[:, j]).mean() - torch.tanh(fTn[:, j]).mean()
                                                        - 0.5 * (torch.tanh(fTp[:, j]) ** 2).mean() - 0.5 * (torch.tanh(fTn[:, j]) ** 2).mean()) for j in range(3)],
                              **full_row(tEp, tEn, *eta_E, S)}
    # JS dictionary control: native balanced log-loss weights on TUNE
    gTp, gTn = dict_scores("js", TU); gEp, gEn = dict_scores("js", E)
    jf = fit_js_mixture(gTp.numpy(), gTn.numpy()); wj = np.array(jf["weights"])
    lqp, _ = js_mixture_logq(gEp.numpy(), wj); _, l1qn = js_mixture_logq(gEn.numpy(), wj)
    qEp = torch.tensor(np.exp(lqp)); qEn = torch.tensor(np.exp(js_mixture_logq(gEn.numpy(), wj)[0]))
    R["combos"]["js_mix"] = {**common, "estimator": "js_mix", "critic_family": "+".join(DICT), "combination_fit_role": "TUNE", **jf,
                             "JS_native_eval": math.log(2) - 0.5 * float(-lqp.mean() - l1qn.mean()),
                             "note": "J_eval / posterior_mse use T = 2 q_w - 1: the common-posterior VCS regression evaluation",
                             **full_row(2 * qEp - 1, 2 * qEn - 1, *eta_E, S)}
    # One VCS residual candidate on the SELECT-best base
    base = fitted[("vcs", best_fam, a.seed)][0]
    for p in base.parameters():
        p.requires_grad_(False)
    torch.manual_seed(7919 + a.seed)
    U, info = train(C.build("C2", a.d), residual_loss(base, a.lam0), D["FIT"], D["SELECT"], lr=a.lr, updates=a.updates, seed=a.seed + 100, device=dev)
    b = lambda role: (torch.tanh(outputs(base, role.xp, role.yp, dev)), torch.tanh(outputs(base, role.xq, role.yq, dev)))
    u = lambda role: (torch.tanh(outputs(U, role.xp, role.yp, dev)), torch.tanh(outputs(U, role.xq, role.yq, dev)))
    (b0p, b0n), (u0p, u0n) = b(TU), u(TU)
    st = residual_step(b0p.numpy(), b0n.numpy(), u0p.numpy(), u0n.numpy())
    (bEp, bEn), (uEp, uEn) = b(E), u(E)
    lam = st.coefficient
    R["combos"]["vcs_residual"] = {**common, "estimator": "vcs_residual", "critic_family": f"{best_fam} + C2 residual", "lambda0_train": a.lam0,
                                   "combination_fit_role": "TUNE", "lambda": lam, "A": st.A, "B": st.B, "predicted_tune_gain": st.predicted_gain,
                                   "zero_direction": st.zero_direction, "selected_checkpoint": info["selected_update"], "fit_seconds": info["fit_seconds"],
                                   "trainable_parameters": C.n_params(U), "base_J_eval": metrics(bEp, bEn)["J"],
                                   **full_row(bEp + lam * (uEp - bEp), bEn + lam * (uEn - bEn), *eta_E, S),
                                   "U_alone": metrics(uEp, uEn, *eta_E, S)}
    # §6.4 gradients through the generator on the TRUTH base noise
    Tr = D["TRUTH"]
    g = {"oracle_fd": drho_oracle_fd(Tr.xp, Tr.ep, Tr.xq, Tr.yq, rho), "oracle_envelope": drho_oracle_envelope(Tr.xp, Tr.ep, rho)}
    for kind in ("vcs", "js"):
        for fam in DICT + ("CQ",):
            m = fitted[(kind, fam, a.seed)][0].float()
            g[f"{kind}_{fam}"] = drho_learned(lambda x, y: torch.tanh(m(x, y)), Tr.xp.float(), Tr.ep.float(), rho, dev)
    R["gradients"] = g
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="estimator_probe", choices=["estimator_probe"])
    ap.add_argument("--I", type=float, required=True); ap.add_argument("--d", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=2000); ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--lam0", type=float, default=0.5)
    ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.updates = min(a.updates, 200)
    t0 = time.time(); R = run_probe(a); R["wall_seconds"] = time.time() - t0; R["args"] = vars(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(R, open(a.out, "w"), indent=1, default=float)
    print(f"I={a.I} S={R['condition']['S_truth']:.4f} -> {a.out} ({R['wall_seconds']:.0f}s)")
    for r in R["rows"]:
        print(f"  {r['run_id']:<22} J_eval {r['J_eval']:.4f}  pmse {r.get('posterior_mse', float('nan')):.4f}")
    for k, v in R["combos"].items():
        print(f"  {k:<22} J_eval {v.get('J_eval', float('nan')):.4f}  pmse {v.get('posterior_mse', float('nan')):.4f}")


if __name__ == "__main__":
    main()
