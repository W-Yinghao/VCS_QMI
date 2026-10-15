"""V9-N1-Exact (package §04.1, §04.5): proportion / reliability quantification on finite-support P / Q with exact truth.

    python scripts/v9/n1_exact.py --out <dir> [--designs clear,moderate,weak,shift] [--rhos 0,0.1,0.3,0.5] [--sizes 128,512,2048]
                                  [--repeats 200] [--smoke]

Support: K = 9 points s_k = linspace(−1, 1, 9) carrying the common scalar score; masses p ∝ exp(κ s), q ∝ exp(−κ s) with κ = 2.5 (clear),
1.2 (moderate), 0.3 (weak: a − b small, weak identification).  The `shift` design is the N1-Stress check of §04.1: the reference P / Q are the
moderate masses but the batch's P component is a shifted P' (mass moved one support point up), so the fixed-score mixture assumption fails and
no interval coverage is claimed.  R = iid mixture (1 − rho) P + rho Q with the component label recorded for the offline scorer only.
Per repeat (fresh draws): critic-fit n P + n Q, critic-select n/2 + n/2, reference-moments n + n, batch n; truth η, w*, S exact from the masses.
Fixed-score controls (no fitting): oracle η; affine 2.5η − 0.4 (identical estimate expected); anti η (−η: identical by |u| invariance); constant
(unidentified); noise (T drawn independently of η once per repeat).  Fitted methods: every §04.4 method via `v9.quant` (no Adam sensitivity here).
Outputs <dir>/<design>_rho<rho>_n<n>.json (skip-if-done; .partial.json while running) with per-repeat records and a summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v9 import quant as Q  # noqa: E402
from v9 import reference_core as RC  # noqa: E402

SUPPORT = np.linspace(-1.0, 1.0, 9)
KAPPA = {"clear": 2.5, "moderate": 1.2, "weak": 0.3, "shift": 1.2}
CONTROLS = ("oracle_eta", "affine_eta", "anti_eta", "constant", "noise")


def masses(design: str):
    k = KAPPA[design]; p = np.exp(k * SUPPORT); q = np.exp(-k * SUPPORT); p, q = p / p.sum(), q / q.sum()
    p_batch = p.copy()
    if design == "shift":                                   # batch P component: mass shifted one support point up (stress; assumption fails)
        p_batch = np.zeros_like(p); p_batch[1:] = p[:-1]; p_batch[-1] += p[-1]
    return p, q, p_batch


def draw(mass: np.ndarray, n: int, rng) -> np.ndarray:
    return SUPPORT[rng.choice(len(mass), size=n, p=mass)]


def eta_of(p, q):
    return RC.finite_target(p, q)["eta"]


def lookup(values: np.ndarray, s: np.ndarray) -> np.ndarray:
    return values[np.rint((s + 1) * 4).astype(int)]


def one_repeat(design: str, rho: float, n: int, rep: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, rep, int(rho * 1000), n]); p, q, p_batch = masses(design); eta = eta_of(p, q); S = RC.finite_target(p, q)["S"]
    sp_fit, sq_fit, sp_sel, sq_sel = draw(p, n, rng), draw(q, n, rng), draw(p, n // 2, rng), draw(q, n // 2, rng)
    sp_ref, sq_ref = draw(p, n, rng), draw(q, n, rng)
    is_p = rng.random(n) >= rho; s_batch = np.where(is_p, draw(p_batch, n, rng), draw(q, n, rng)); truth = {"is_p": is_p, "rho": rho}
    rec = {"rep": rep, "rho_realized": float(1 - is_p.mean()), "S": S, "methods": {}, "controls": {}}
    # fixed-score controls on the oracle η (no fitting)
    ctrl = {"oracle_eta": lambda s: lookup(eta, s), "affine_eta": lambda s: 2.5 * lookup(eta, s) - 0.4, "anti_eta": lambda s: -lookup(eta, s),
            "constant": lambda s: np.full(len(s), 0.3), "noise": (lambda tbl: (lambda s: lookup(tbl, s)))(rng.uniform(-1, 1, len(SUPPORT)))}
    for name, fn in ctrl.items():
        m = Q.rho_moment(fn(sp_ref), fn(sq_ref), fn(s_batch)); w = Q.weights(np.clip(fn(s_batch), -1, 1), m["rho_clipped"]) if name == "oracle_eta" else None
        rec["controls"][name] = {k: m[k] for k in ("rho_raw", "rho_clipped", "status", "gap")}
        if w is not None and w["w"] is not None:
            rec["controls"][name]["brier"] = RC.brier(is_p.astype(float), w["w"]); rec["controls"][name]["ess_fraction"] = w["ess_fraction"]
    w_true = RC.clean_posterior(np.clip(lookup(eta, s_batch), -1 + 1e-9, 1 - 1e-9), min(max(rho, 1e-9), 1 - 1e-9))
    rec["oracle_w_brier"] = RC.brier(is_p.astype(float), w_true); rec["oracle_w_logloss"] = RC.logloss(is_p.astype(float), w_true)
    # finite-sample set on the oracle η (independent iid draws: every sample is its own cluster)
    rows = lambda role, s, tag: [{"role": role, "cluster_id": f"{tag}{i}", "units": [f"{tag}{i}"], "score": float(v)} for i, v in enumerate(lookup(eta, s))]
    ci = Q.confidence_set(rows("P", sp_ref, "p"), rows("Q", sq_ref, "q"), rows("R", s_batch, "r"), fixed_score_outside_eval=True, specified_mixture=(design != "shift"))
    rec["controls"]["oracle_eta"]["interval"] = {k: ci[k] for k in ("lower", "upper", "status")}
    # fitted panel (fit / select roles only) and every method on the batch
    t0 = time.time(); panel = Q.fit_panel(sp_fit, sq_fit, sp_sel, sq_sel, seed=rep, adam_sensitivity=False); rec["fit_seconds"] = time.time() - t0
    rec["panel"] = {"vcs": {"a": panel.vcs.a, "b": panel.vcs.b}, "bce": {"a": panel.bce.a, "b": panel.bce.b}, "J_select": panel.info["J_select"],
                    "kde_mult": panel.kde.info["multiplier"], "rulsif": [panel.rulsif.info["sigma_mult"], panel.rulsif.info["lambda"], panel.rulsif.info["at_edge"]],
                    "nystrom": [panel.nystrom.info["sigma_mult"], panel.nystrom.info["lambda"], panel.nystrom.info["at_edge"]]}
    t0 = time.time(); out = Q.quantify_batch(panel, sp_ref, sq_ref, s_batch, truth); rec["batch_seconds"] = time.time() - t0
    for m in Q.METHODS:
        o = out[m]; rec["methods"][m] = {k: o.get(k) for k in ("rho_raw", "rho_clipped", "status")}
        w = o.get("weights") or {}
        rec["methods"][m].update({k: w.get(k) for k in ("brier", "logloss", "auroc", "ess_fraction", "clip_rate_T")})
    rec["J_batch"] = out["_J_batch"]
    # finite-sample set on the fitted VCS critic (fixed before the reference / batch draws of this repeat)
    T = panel.vcs.T
    rowsT = lambda role, s, tag: [{"role": role, "cluster_id": f"{tag}{i}", "units": [f"{tag}{i}"], "score": float(v)} for i, v in enumerate(T(s))]
    ciT = Q.confidence_set(rowsT("P", sp_ref, "p"), rowsT("Q", sq_ref, "q"), rowsT("R", s_batch, "r"), fixed_score_outside_eval=True, specified_mixture=(design != "shift"))
    rec["methods"]["vcs_moment"]["interval"] = {k: ciT[k] for k in ("lower", "upper", "status")}
    ciB = Q.confidence_set(*[[{"role": r, "cluster_id": f"{t}{i}", "units": [f"{t}{i}"], "score": float(v)} for i, v in enumerate(panel.bce.T(s))] for r, s, t in (("P", sp_ref, "p"), ("Q", sq_ref, "q"), ("R", s_batch, "r"))],
                           fixed_score_outside_eval=True, specified_mixture=(design != "shift"))
    rec["methods"]["bce_moment"]["interval"] = {k: ciB[k] for k in ("lower", "upper", "status")}
    return rec


def summarise(records: list, rho: float) -> dict:
    def stats(vals):
        v = np.array([x for x in vals if x is not None], float); return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None, "sd": float(v.std(ddof=1)) if len(v) > 1 else None}
    summ = {"methods": {}, "controls": {}}
    for m in Q.METHODS:
        est = [r["methods"][m]["rho_clipped"] for r in records]; raw = [r["methods"][m]["rho_raw"] for r in records]; ok = [x for x in est if x is not None]
        d = {"n_identified": len(ok), "mae": float(np.mean(np.abs(np.array(ok) - rho))) if ok else None, "rmse": float(np.sqrt(np.mean((np.array(ok) - rho) ** 2))) if ok else None,
             "bias": float(np.mean(np.array(ok) - rho)) if ok else None, "raw_outside": int(sum(1 for x in raw if x is not None and (x < 0 or x > 1))),
             "brier": stats([r["methods"][m].get("brier") for r in records]), "logloss": stats([r["methods"][m].get("logloss") for r in records]),
             "auroc": stats([r["methods"][m].get("auroc") for r in records]), "ess_fraction": stats([r["methods"][m].get("ess_fraction") for r in records])}
        iv = [r["methods"][m].get("interval") for r in records if r["methods"][m].get("interval")]
        if iv:
            d["interval"] = interval_stats(iv, rho)
        summ["methods"][m] = d
    for c in CONTROLS:
        est = [r["controls"][c]["rho_clipped"] for r in records]; ok = [x for x in est if x is not None]
        summ["controls"][c] = {"n_identified": len(ok), "mae": float(np.mean(np.abs(np.array(ok) - rho))) if ok else None, "rmse": float(np.sqrt(np.mean((np.array(ok) - rho) ** 2))) if ok else None}
        iv = [r["controls"][c].get("interval") for r in records if r["controls"][c].get("interval")]
        if iv:
            summ["controls"][c]["interval"] = interval_stats(iv, rho)
    summ["oracle_w_brier"] = stats([r["oracle_w_brier"] for r in records]); summ["rho_realized"] = stats([r["rho_realized"] for r in records])
    summ["seconds"] = stats([r["fit_seconds"] + r["batch_seconds"] for r in records]); return summ


def interval_stats(iv: list, rho: float) -> dict:
    n = len(iv); cov = sum(1 for z in iv if z["status"] in ("bounded_interval", "full_parameter_space") and z["lower"] <= rho <= z["upper"])
    width = [z["upper"] - z["lower"] for z in iv if z["status"] == "bounded_interval"]
    return {"n": n, "coverage": cov / n, "mean_width_bounded": float(np.mean(width)) if width else None, "rate_empty": sum(z["status"] == "empty" for z in iv) / n,
            "rate_full": sum(z["status"] == "full_parameter_space" for z in iv) / n, "rate_bounded": len(width) / n}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--designs", default="clear,moderate,weak,shift")
    ap.add_argument("--rhos", default="0,0.1,0.3,0.5"); ap.add_argument("--sizes", default="128,512,2048"); ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=901); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    designs = a.designs.split(","); rhos = [float(x) for x in a.rhos.split(",")]; sizes = [int(x) for x in a.sizes.split(",")]; reps = a.repeats
    if a.smoke:
        designs, rhos, sizes, reps = designs[:2], rhos[:2], [128], (a.repeats if a.repeats != 200 else 3)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t_all = time.time()
    for design in designs:
        for rho in rhos:
            for n in sizes:
                f = out / f"{design}_rho{rho:g}_n{n}.json"
                if f.exists():
                    print(f"[skip] {f.name}"); continue
                t0 = time.time(); records = [one_repeat(design, rho, n, rep, a.seed) for rep in range(reps)]
                p, q, p_batch = masses(design); res = {"design": design, "rho": rho, "n": n, "repeats": reps, "seed": a.seed, "support": SUPPORT.tolist(),
                       "masses": {"p": p.tolist(), "q": q.tolist(), "p_batch": p_batch.tolist()}, "S": RC.finite_target(p, q)["S"], "eta": eta_of(p, q).tolist(),
                       "a_minus_b_oracle": float(2 * RC.finite_target(p, q)["S"]), "summary": summarise(records, rho), "records": records, "seconds": time.time() - t0, "smoke": a.smoke}
                tmp = f.with_suffix(".partial.json"); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(f)
                s = res["summary"]["methods"]; print(f"[{design} rho={rho:g} n={n}] {reps} reps in {time.time() - t0:.0f}s: RMSE vcs {s['vcs_moment']['rmse']:.3f} bce {s['bce_moment']['rmse']:.3f} "
                      f"em_bcts {s['em_bcts']['rmse']:.3f} kde {s['kde_moment']['rmse']:.3f} rulsif {s['rulsif_moment']['rmse']:.3f} nystrom {s['nystrom_moment']['rmse']:.3f} mpe {s['kernel_mpe']['rmse'] if s['kernel_mpe']['rmse'] is not None else float('nan'):.3f}; "
                      f"oracle interval cov {res['summary']['controls']['oracle_eta'].get('interval', {}).get('coverage')}", flush=True)
    print(f"total {time.time() - t_all:.0f}s"); return 0


if __name__ == "__main__":
    sys.exit(main())
