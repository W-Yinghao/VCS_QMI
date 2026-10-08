"""P152 aggregate (frozen reading, prereg §3): the E-Stability fields per method x dependence level, mean +- sd over the 3 seeds.
Per (kind, I): gradient-norm last-half mean / sd / CV and whole-run CV and q99 / max; loss last-half sd; non-finite steps; eval variance conditional on
the fit (64 independent EVAL blocks of 512) at the SELECT-best state and at updates 250 / 1000 / 2000; |T| > 0.95 / 0.99 fractions on P and Q;
native value, own truth, signed error; posterior MSE for VCS / JS; reproduction |delta| vs P85.  Corrected SMILE shares the JS fit (flagged).
    python scripts/p152_aggregate.py [--out reports/P152_results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

O = Path("/home/infres/yinwang/CS_QMI/outputs/P152_e1")
LEVELS, SEEDS = (2, 6, 10), (0, 1, 2)
KINDS = ("vcs", "js", "infonce", "nwj", "dv", "smile")
CKPTS = ("u250", "u1000", "u2000", "select_best")


HALF = {"vcs": 1.0, "js": 0.5}  # registered T coordinate: VCS T = tanh f, matched JS T = tanh(u / 2) (benchmark.neural_rows)


def sat_from_hist(h: dict, half: float, level: float) -> float:
    """Fraction with |tanh(half * f)| > level from the 201-bin f histogram on [-10, 10] plus overflow counts (bin-centre approximation)."""
    c = np.asarray(h["f_counts"], float); centres = np.linspace(-10, 10, 202)[:-1] + 10.0 / 201; thr = np.arctanh(level) / half
    n = c.sum() + h["f_below"] + h["f_above"]
    return float((c[np.abs(centres) > thr].sum() + h["f_below"] + h["f_above"]) / n) if thr < 10 else float((h["f_below"] + h["f_above"]) / n)


def ms(x) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return {"mean": float(x.mean()) if len(x) else None, "sd": float(x.std(ddof=1)) if len(x) > 1 else None, "n": int(len(x))}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/P152_results.json"); a = ap.parse_args()
    units, missing = {}, []
    for I in LEVELS:
        for s in SEEDS:
            f = O / f"I{I}_s{s}.json"
            if not f.exists():
                missing.append(f.name); continue
            d = json.load(open(f))
            if len(d["kinds"]) < len(KINDS):
                missing.append(f"{f.name} ({len(d['kinds'])} kinds)"); continue
            units[(I, s)] = d
    table = {}
    for kind in KINDS:
        for I in LEVELS:
            rs = [units[(I, s)]["kinds"][kind] for s in SEEDS if (I, s) in units]
            if not rs:
                continue
            g = [r["gradient_norm"] for r in rs]
            row = {"lr": sorted({r["lr_from_P85_selection"] for r in rs}),
                   "gnorm_last_half_mean": ms([x["last_half_mean"] for x in g]), "gnorm_last_half_cv": ms([x["last_half_sd"] / x["last_half_mean"] if x["last_half_mean"] > 0 else None for x in g]),
                   "gnorm_cv_whole": ms([x["cv"] for x in g]), "gnorm_q99": ms([x["quantiles"]["q99"] for x in g]), "gnorm_max": ms([x["quantiles"]["max"] for x in g]),
                   "loss_last_half_sd": ms([r["loss"]["last_half_sd"] for r in rs]), "nonfinite_steps": [r["fit"]["nonfinite_steps"] for r in rs],
                   "selected_update": [r["fit"]["selected_update"] for r in rs], "fit_seconds": ms([r["fit"]["fit_seconds"] for r in rs]),
                   "reproduction_abs_diff_max": max((r["reproduction_vs_P85"]["abs_diff"] for r in rs if "reproduction_vs_P85" in r), default=None), "checkpoints": {}}
            for c in CKPTS:
                e = [r["checkpoints"][c] for r in rs]
                row["checkpoints"][c] = {"native_value": ms([x["native_value"] for x in e]), "own_truth": e[0]["own_truth"], "signed_error": ms([x["signed_error"] for x in e]),
                                         "eval_block_sd": ms([np.sqrt(x["eval_blocks"]["var_conditional_fit"]) if x["eval_blocks"]["var_conditional_fit"] is not None else None for x in e]),
                                         "eval_block_rel_sd": ms([np.sqrt(x["eval_blocks"]["var_conditional_fit"]) / abs(x["native_value"]) if x["eval_blocks"]["var_conditional_fit"] is not None and x["native_value"] else None for x in e]),
                                         "eval_block_mean": ms([x["eval_blocks"]["mean"] for x in e]), "eval_block_n": e[0]["eval_blocks"].get("block_n"),
                                         "eval_block_nonfinite": [x["eval_blocks"]["n_nonfinite"] for x in e],
                                         "f_overflow_P": [x["hist_P"]["f_below"] + x["hist_P"]["f_above"] for x in e], "f_overflow_Q": [x["hist_Q"]["f_below"] + x["hist_Q"]["f_above"] for x in e],
                                         "f_q99_P": ms([x["hist_P"]["f_quantiles"]["q99"] for x in e]), "f_q01_Q": ms([x["hist_Q"]["f_quantiles"]["q01"] for x in e])}
                if kind in HALF:  # saturation on the registered T coordinate (the recorder's sat_T fields use tanh f for every kind)
                    for side in ("P", "Q"):
                        row["checkpoints"][c][f"sat95_{side}"] = ms([sat_from_hist(x[f"hist_{side}"], HALF[kind], 0.95) for x in e])
                        row["checkpoints"][c][f"sat99_{side}"] = ms([sat_from_hist(x[f"hist_{side}"], HALF[kind], 0.99) for x in e])
                if kind in ("vcs", "js"):
                    row["checkpoints"][c]["posterior_mse"] = ms([x["posterior"].get("posterior_mse") for x in e])
            if kind == "smile":
                same = [units[(I, s)]["kinds"]["smile"]["lr_from_P85_selection"] == units[(I, s)]["kinds"]["js"]["lr_from_P85_selection"] for s in SEEDS if (I, s) in units]
                row["same_lr_as_js"] = same
                row["note"] = ("corrected SMILE trains on the JS gradient: where P85 selected the same lr as for JS the fit, gradient series and histograms equal "
                               "JS's (only the value read-out and the SELECT state differ); elsewhere it is a JS-gradient fit at a different lr")
            table[f"{kind}/I{I}"] = row
    json.dump({"table": table, "missing": missing}, open(a.out, "w"), indent=1)
    f = lambda m, k=3: "—" if m["mean"] is None else (f"{m['mean']:.{k}f}" + (f"±{m['sd']:.{k}f}" if m["sd"] is not None else ""))
    print("kind/I        gnorm(last half)   CV(last half)   CV(whole)   nonfinite   value (truth)           eval-block sd (rel)          sat95 P/Q (own T)  post-MSE   repro")
    for k, r in table.items():
        sb = r["checkpoints"]["select_best"]; rep = r["reproduction_abs_diff_max"]; rep = "—" if rep is None else f"{rep:.1e}"
        sat = f"{f(sb['sat95_P'], 2)}/{f(sb['sat95_Q'], 2)}" if "sat95_P" in sb else f"|f|>10: {sum(sb['f_overflow_P'])}/{sum(sb['f_overflow_Q'])}"
        post = f(sb["posterior_mse"], 3) if "posterior_mse" in sb else ""
        print(f"{k:12s} {f(r['gnorm_last_half_mean']):>16s} {f(r['gnorm_last_half_cv']):>15s} {f(r['gnorm_cv_whole']):>11s} {str(r['nonfinite_steps']):>10s}  "
              f"{f(sb['native_value']):>14s} ({sb['own_truth']:.3f})  {f(sb['eval_block_sd'], 4):>15s} ({f(sb['eval_block_rel_sd'], 3)})  {sat:>18s}  {post:>9s}  {rep}")
    if missing:
        print("missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
