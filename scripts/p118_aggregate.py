"""P118 aggregate (v5 NEXT-I-NEST): truth-compression table + the frozen "error reduced" rule, visual nested increments per encoder seed,
the seeds-1/2 I1 audits, and the existing P109 prediction effects for the same encoders / versions.

    python scripts/p118_aggregate.py --out reports/P118_nest_results
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_ssl.utils import atomic_write_json  # noqa: E402

R = Path(__file__).resolve().parents[1] / "reports"
VARIANTS = ("indep_sampled", "indep_exact", "nested_sampled", "nested_exact", "zero")
BASE = ("indep_sampled", "readout_sampled")                 # P102 baseline (as P109 I2)


def rule(sm_by_cond: dict, obj: str, v: str, rd: str) -> dict:
    """Frozen rule, per objective: in BOTH non-null conditions vs the baseline —
    (1) RMSE of the M→A increment error ≤ 0.75 × baseline; (2) |mean B→M error| ≤ baseline's; (3) posterior MSE at B and at A ≤ 1.10 × baseline;
    (4) |mean M→A increment| ≥ 0.5 × the oracle increment (no shrinkage toward zero).  All four → 'error reduced'.  (4) failing while R_orth falls
    → 'shrinkage'.  The constant-zero control is evaluated by the same rule (it must fail (1)/(4))."""
    out = {}
    for cond in ("weak", "moderate"):
        sm, b = sm_by_cond[cond]["summary"][f"{obj}/{v}/{rd}"], sm_by_cond[cond]["summary"][f"{obj}/{BASE[0]}/{BASE[1]}"]
        true_inc = sm_by_cond[cond]["oracle"]["M"]["S"] - sm_by_cond[cond]["oracle"]["A"]["S"]
        c1 = sm["inc_rmse_M->A"] <= 0.75 * b["inc_rmse_M->A"]; c2 = abs(sm["inc_err_B->M"]["mean"]) <= abs(b["inc_err_B->M"]["mean"])
        c3 = sm["post_B"]["mean"] <= 1.10 * b["post_B"]["mean"] and sm["post_A"]["mean"] <= 1.10 * b["post_A"]["mean"]
        c4 = abs(sm["inc_M->A"]["mean"]) >= 0.5 * true_inc
        out[cond] = {"c1_rmse": c1, "c2_null_step": c2, "c3_posterior": c3, "c4_no_shrinkage": c4,
                     "R_orth_lower": abs(sm["R_orth"]["mean"]) < abs(b["R_orth"]["mean"])}
    allc = all(all(x[k] for k in ("c1_rmse", "c2_null_step", "c3_posterior", "c4_no_shrinkage")) for x in out.values())
    shrink = any((not x["c4_no_shrinkage"]) and x["R_orth_lower"] for x in out.values())
    out["verdict"] = "error reduced" if allc else ("shrinkage" if shrink else "no clear reduction")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--reports-dir", default=None); a = ap.parse_args()
    global R
    R = Path(a.reports_dir) if a.reports_dir else R
    J = {"truth": None, "visual": {}, "audit": {}, "effects": {}}; L = ["# P118 — nested critics (v5 NEXT-I-NEST): results", ""]
    tf = R / "P118_truth.json"
    if tf.is_file():
        T = json.load(open(tf)); J["truth"] = {"cells": {k: {kk: vv for kk, vv in c.items() if kk != "repeats"} for k, c in T["cells"].items()}, "rule": {}}
        L += ["## 6.1 Truth compression (mean over repeats; increment errors vs oracle; posterior MSE vs exact eta)", "",
              "| cond | true M→A | obj | variant | readout | err B→M | err M→A (RMSE) | err B→A | post B | post M | post A | R_orth | violations |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cond, c in T["cells"].items():
            ti = c["oracle"]["M"]["S"] - c["oracle"]["A"]["S"]
            for obj in ("vcs", "js"):
                for v in VARIANTS:
                    for rd in ("readout_sampled", "readout_exact"):
                        s = c["summary"][f"{obj}/{v}/{rd}"]
                        L.append(f"| {cond} | {ti:.4f} | {obj} | {v} | {rd[8:]} | {s['inc_err_B->M']['mean']:+.4f} | {s['inc_err_M->A']['mean']:+.4f} ({s['inc_rmse_M->A']:.4f}) | "
                                 f"{s['inc_err_B->A']['mean']:+.4f} | {s['post_B']['mean']:.4f} | {s['post_M']['mean']:.4f} | {s['post_A']['mean']:.4f} | {s['R_orth']['mean']:+.4f} | "
                                 f"{s['ordering_violations']['mean']:.1f} |")
        if all(k in T["cells"] for k in ("weak", "moderate")):
            L += ["", "### Frozen rule vs indep_sampled / sampled readout (both non-null conditions)", "", "| obj | variant | readout | verdict | weak (c1 c2 c3 c4) | moderate |", "|---|---|---|---|---|---|"]
            for obj in ("vcs", "js"):
                for v in VARIANTS:
                    for rd in ("readout_sampled", "readout_exact"):
                        if (v, rd) == BASE:
                            continue
                        r = rule(T["cells"], obj, v, rd); J["truth"]["rule"][f"{obj}/{v}/{rd}"] = r
                        f = lambda x: " ".join("✓" if x[k] else "✗" for k in ("c1_rmse", "c2_null_step", "c3_posterior", "c4_no_shrinkage"))
                        L.append(f"| {obj} | {v} | {rd[8:]} | {r['verdict']} | {f(r['weak'])} | {f(r['moderate'])} |")
        if "null" in T["cells"]:
            L += ["", "Null floor (null condition): |mean increment error| B→M / M→A per variant (exact readout, VCS): " + "; ".join(
                f"{v} {abs(T['cells']['null']['summary'][f'vcs/{v}/readout_exact']['inc_err_B->M']['mean']):.4f} / "
                f"{abs(T['cells']['null']['summary'][f'vcs/{v}/readout_exact']['inc_err_M->A']['mean']):.4f}" for v in VARIANTS)]
    vf = sorted(f for f in glob.glob(str(R / "P118_visual_*.json")) if not f.endswith("partial.json"))
    if vf:
        L += ["", "## 6.2 Visual nested increments (per encoder seed; main pair h vs logit_h; mean ± refit sd over repeats)", "",
              "| run | cell | obj | variant | readout | ΔJ main | R_orth | violations | step h→pca64 | step pca64→pca16 | step pca16→logit16 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for f in vf:
            V = json.load(open(f)); J["visual"][V["run"]] = {k: c["summary"] for k, c in V["cells"].items()}
            for cell, c in V["cells"].items():
                for obj in ("vcs", "js"):
                    for v in VARIANTS:
                        for rd in ("readout_sampled", "readout_exact"):
                            s = c["summary"][f"{obj}/{v}/{rd}"]
                            L.append(f"| {V['run']} | {cell} | {obj} | {v} | {rd[8:]} | {s['delta_J']['mean']:+.4f} ± {s['delta_J']['sd_refit'] or 0:.4f} | {s['R_orth']['mean']:+.4f} | "
                                     f"{s['nesting_violations']['mean']:.1f} | {s['step_h->pca64']['mean']:+.4f} | {s['step_pca64->pca16']['mean']:+.4f} | {s['step_pca16->logit16']['mean']:+.4f} |")
    af = sorted(f for f in glob.glob(str(R / "P118_i1_*.json")) if not f.endswith("partial.json"))
    if af:
        L += ["", "## 6.2 I1 audits, seeds 1–2 (rejection rate; level flag > 0.09 at R = 200)", "", "| run | family | cell | R | h vcs / js | logits vcs / js | any-layer maxT vcs / js | flag |", "|---|---|---|---|---|---|---|---|"]
        for f in af:
            A = json.load(open(f))
            for key, c in A["cells"].items():
                s = c["summary"]; lvl = c["mode"] != "planted"
                flag = any(s[k]["power"] > 0.09 for k in s if not k.startswith("any_")) if lvl else False
                J["audit"][f"{A['run']}/{A['family']}/{key}"] = {"summary": s, "flag": flag}
                L.append(f"| {A['run']} | {A['family']} | {key} | {c['repeats']} | {s['h/vcs_closed']['power']:.2f} / {s['h/js_exact']['power']:.2f} | "
                         f"{s['logits/vcs_closed']['power']:.2f} / {s['logits/js_exact']['power']:.2f} | {s['any_layer_maxT/vcs_closed']['power']:.2f} / "
                         f"{s['any_layer_maxT/js_exact']['power']:.2f} | {'FLAG' if flag else '—'} |")
    L += ["", "## Paired prediction effects (existing P109 effect files, seeds 1–2; no new computation)", "", "| run | version | Δ p(true) [CI] | Δ acc | flip rate |", "|---|---|---|---|---|"]
    for r in ("P107_AP3_views4_800ep_seed1", "P107_AP3_views4_800ep_seed2", "P41_simclr_views4_800ep_seed1", "P41_simclr_views4_800ep_seed2"):
        for fam, ver in (("colour", "colour_s0.1"), ("blur", "blur_s0.25")):
            p = R / f"P109_i1_{r}_{fam}_effects.json"
            if p.is_file():
                e = json.load(open(p))["prediction_effects"][ver]["overall"]; J["effects"][f"{r}/{ver}"] = e
                L.append(f"| {r} | {ver} | {e['d_prob_true']:+.4f} [{e['d_prob_true_ci'][0]:+.4f}, {e['d_prob_true_ci'][1]:+.4f}] | {e['d_acc']:+.4f} | {e['prediction_flip_rate']:.3f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); atomic_write_json(Path(a.out + ".json"), J); print(f"-> {a.out}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
