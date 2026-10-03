"""P117 aggregate (v5 NEXT-E-CAL): per condition × N × fit loss over the 5 fit seeds; applies the pre-stated readings of the P117 prereg.

    python scripts/p117_aggregate.py --in-dir outputs/P117_cal --out reports/P117_cal_results
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim.p117 import CANDS, M_BINS  # noqa: E402
from vcs_ssl.utils import atomic_write_json  # noqa: E402

ROWS = CANDS + ("selected",)


def ms(x):
    x = np.asarray(x, float); return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else None)


def rmse(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, float)))))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in-dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cells = [json.load(open(f)) for f in sorted(glob.glob(f"{a.in_dir}/P117_*.json")) if "smoke" not in f]
    G = defaultdict(list)
    for c in cells:
        for kind, v in c["per_kind"].items():
            G[(c["condition"], c["N"], kind)].append((c, v))
    res, L = [], ["# P117 — fixed-critic calibration (v5 NEXT-E-CAL): results", "",
                  "Mean over fit seeds (sd).  Mechanism = same frozen T0 (FIT), calibrators on CAL, readouts on EVAL; decomposition on DIAG (finest m-bins).", "",
                  "| condition | N | loss | seeds | candidate | posterior MSE | J bias | J RMSE | S_plug bias | S_plug RMSE | eval SE (J) | selected count |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    dec_lines = ["", "## Decomposition S − J(g) = A + B (DIAG, finest m-bins) and the pre-stated readings", "",
                 "| condition | N | loss | A (score loss) | B identity | B selected | A share of identity gap | fixable share (B_id − B_sel) / (A + B_id) | max abs residual | reading |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
    e2e_lines = ["", "## End-to-end at equal total sample budget (identity on FIT ∪ CAL vs T0 on FIT + selected calibrator on CAL)", "",
                 "| condition | N | loss | identity(FIT∪CAL) J RMSE / post MSE | calibrated J RMSE / post MSE | paired Δ post MSE (cal − id), mean ± se |",
                 "|---|---|---|---|---|---|"]
    for (cond, N, kind), lst in sorted(G.items()):
        rec = {"condition": cond, "N": N, "kind": kind, "seeds": len(lst), "candidates": {}}
        for r in ROWS:
            m = [v["mechanism_same_T0"][r] for _, v in lst]
            rec["candidates"][r] = {"posterior_mse": ms([x["posterior_mse"] for x in m]), "J_bias": ms([x["J_err"] for x in m]), "J_rmse": rmse([x["J_err"] for x in m]),
                                    "S_plug_bias": ms([x["S_plug_err"] for x in m]), "S_plug_rmse": rmse([x["S_plug_err"] for x in m]), "J_se": ms([x["J_se"] for x in m])}
            sel_count = sum(v["selected"] == r for _, v in lst) if r != "selected" else ""
            cr = rec["candidates"][r]
            L.append(f"| {cond} | {N} | {kind} | {len(lst)} | {r} | {cr['posterior_mse'][0]:.4f} ({cr['posterior_mse'][1] or 0:.4f}) | {cr['J_bias'][0]:+.4f} | {cr['J_rmse']:.4f} | "
                     f"{cr['S_plug_bias'][0]:+.4f} | {cr['S_plug_rmse']:.4f} | {cr['J_se'][0]:.4f} | {sel_count} |")
        fin = str(M_BINS[-1]); d = [v["decomposition_DIAG"][fin] for _, v in lst]
        A = np.array([x["A_score_loss"] for x in d]); Bid = np.array([x["per_calibrator"]["identity"]["B_calibration"] for x in d])
        Bsel = np.array([x["per_calibrator"][v["selected"]]["B_calibration"] for x, (_, v) in zip(d, lst)])
        resid = max(abs(x["per_calibrator"][c]["residual"]) for x in d for c in CANDS); gap = A + Bid
        a_share = float(np.mean(A / np.where(gap > 0, gap, 1))); fix = float(np.mean((Bid - Bsel) / np.where(gap > 0, gap, 1)))
        post_id, post_sel = rec["candidates"]["identity"]["posterior_mse"][0], rec["candidates"]["selected"]["posterior_mse"][0]
        reading = ("calibration repairs most of the error" if post_sel <= 0.5 * post_id else
                   "score has lost most of the dependence (A dominates)" if a_share >= 0.5 else "mixed: partial repair, A < half")
        rec["decomposition"] = {"A": ms(A), "B_identity": ms(Bid), "B_selected": ms(Bsel), "A_share": a_share, "fixable_share": fix, "max_abs_residual": resid, "reading": reading}
        dec_lines.append(f"| {cond} | {N} | {kind} | {A.mean():.4f} | {Bid.mean():.4f} | {Bsel.mean():.4f} | {a_share:.2f} | {fix:.2f} | {resid:.1e} | {reading} |")
        ei = [v["end_to_end_equal_budget"]["identity_on_FIT_plus_CAL"] for _, v in lst]; ec = [v["end_to_end_equal_budget"]["T0_on_FIT_plus_selected_calibrator_on_CAL"] for _, v in lst]
        dpost = np.array([c_["posterior_mse"] - i_["posterior_mse"] for c_, i_ in zip(ec, ei)])
        rec["end_to_end"] = {"identity_FITCAL": {"J_rmse": rmse([x["J_err"] for x in ei]), "post": ms([x["posterior_mse"] for x in ei])},
                             "calibrated": {"J_rmse": rmse([x["J_err"] for x in ec]), "post": ms([x["posterior_mse"] for x in ec])},
                             "paired_d_post": (float(dpost.mean()), float(dpost.std(ddof=1) / np.sqrt(len(dpost))) if len(dpost) > 1 else None)}
        e = rec["end_to_end"]
        e2e_lines.append(f"| {cond} | {N} | {kind} | {e['identity_FITCAL']['J_rmse']:.4f} / {e['identity_FITCAL']['post'][0]:.4f} | {e['calibrated']['J_rmse']:.4f} / "
                         f"{e['calibrated']['post'][0]:.4f} | {e['paired_d_post'][0]:+.4f} ± {e['paired_d_post'][1] or 0:.4f} |")
        res.append(rec)
    Path(a.out + ".md").write_text("\n".join(L + dec_lines + e2e_lines) + "\n"); atomic_write_json(Path(a.out + ".json"), {"cells": res, "n_files": len(cells)})
    print(f"-> {a.out}.md ({len(cells)} cell files, {len(res)} groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
