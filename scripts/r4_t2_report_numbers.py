"""Descriptive numbers for the R4-T2 report (not a frozen reading): variance inflation of sparse shifts Var Ĵ_K / Var Ĵ, width ratios of the
bounded and block intervals to the jackknife (cross-pair) or image-t (image-local) interval, and the stage-3 crossover n at which the training
variance equals the EVAL variance (EVAL variance scaled as 1/n from n = 512).  Reads outputs/R4_T2/*.json; writes reports/R4/R4_T2_report_numbers.json.
    python scripts/r4_t2_report_numbers.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

IN = Path("/home/infres/yinwang/CS_QMI/outputs/R4_T2"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R4" / "R4_T2_report_numbers.json"


def main() -> int:
    res = {"sparse_inflation": [], "width_ratios": [], "stage3_crossover": {}}
    for f in sorted(IN.glob("1a_*.json")) + sorted(IN.glob("2_*_n*.json")):
        r = json.load(open(f))
        if r.get("constant_kernel"):
            continue
        J = np.array(r["draws"]["J"]); vJ = J.var(ddof=1); n = r["n"]; jk_w = 2 * 1.959964 * float(np.mean(r["draws"]["se_jk"]))
        m = np.array(r["draws"]["m_blk"]); blk_w = float(np.mean(2 * 2.0 * np.array(r["draws"]["sd_blk"]) / np.sqrt(m)))      # t ≈ 2 for the ratio only
        row = {"stage": r["stage"], "setting": r.get("setting", "finite"), "level": r["level"], "critic": r["critic"], "n": n}
        res["sparse_inflation"].append({**row, **{f"K{K}": float(np.var(r["draws"][f"K{K}.J"], ddof=1) / vJ) for K in (1, 4, 16) if f"K{K}.J" in r["draws"]},
                                        **{f"evals_K{K}_frac": (n + K * n) / n ** 2 for K in (1, 4, 16)}})
        res["width_ratios"].append({**row, "jk_width": jk_w, "bd_over_jk": 2 * math.sqrt(32 * math.log(40) / n) / jk_w, "blk_t_over_jk": blk_w / jk_w,
                                    "blk_hoeff_over_jk": 2 * math.sqrt(8 * math.log(40) / (n // 2)) / jk_w})
    for f in sorted(IN.glob("1b_*.json")):
        r = json.load(open(f))
        if r.get("constant_kernel"):
            continue
        N = r["n"]; t_w = float(np.mean(2 * 2.0 * np.array(r["draws"]["sd_img"]) / math.sqrt(N)))
        res["width_ratios"].append({"stage": "1b", "setting": "finite", "level": r["level"], "critic": r["critic"], "n": N, "img_t_width": t_w,
                                    "img_hoeff_over_img_t": 2 * math.sqrt(8 * math.log(40) / N) / t_w})
    s3 = {}
    for f in sorted(IN.glob("3_*_n512.json")):
        r = json.load(open(f)); s3.setdefault(r["kind"], []).append((r["J"], float(np.var(r["draws"]["J"], ddof=1))))
    for kind, v in s3.items():
        Js = np.array([x[0] for x in v]); ve = float(np.mean([x[1] for x in v])); vt = float(Js.var(ddof=1))
        res["stage3_crossover"][kind] = {"var_train": vt, "var_eval_n512": ve, "share_train_at_512": vt / (vt + ve), "n_crossover": 512 * ve / vt,
                                         "share_train_at_32768": vt / (vt + ve * 512 / 32768)}
    OUT.write_text(json.dumps(res, indent=1))
    def rng(xs):
        return f"{min(xs):.2f}–{max(xs):.2f}" if xs else "—"
    for K in (1, 4, 16):
        xs = [d[f"K{K}"] for d in res["sparse_inflation"] if d["n"] == 512 and f"K{K}" in d]; print(f"n=512 K={K}: Var Ĵ_K / Var Ĵ {rng(xs)} (evals {(512 + K * 512) / 512 ** 2:.4f} of full)")
    for key in ("bd_over_jk", "blk_hoeff_over_jk", "blk_t_over_jk", "img_hoeff_over_img_t"):
        for n in (32, 512):
            xs = [d[key] for d in res["width_ratios"] if key in d and d["n"] == n]; print(f"{key} n={n}: {rng(xs)}")
    print(json.dumps(res["stage3_crossover"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
