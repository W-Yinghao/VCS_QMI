"""P130 addendum 1 — ResNet-50 CIFAR-100, seeds 0-2, A-P3 / JS-AP3 / SimCLR (frozen reading, P130_ADDENDUM1_SEEDS_FROZEN_20261007.md):
A-P3 − SimCLR and A-P3 − JS-AP3 paired by seed, 95 % t interval, P114 labels (close |mean| < 0.3; clear |mean| ≥ 0.3 and the interval excludes 0;
else inconclusive); kNN alongside; per-method ResNet-50 − ResNet-18 descriptive (ResNet-18 seeds 0-2: P107 / P120 / P91).
Writes reports/P130A1_results.json; exits 1 if a run is missing (no partial reading)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

O = Path("/home/infres/yinwang/CS_QMI/outputs"); REP = Path(__file__).resolve().parents[1] / "reports"
R50 = {"A-P3": "P130_AP3_c100_r50_views4_800ep_seed{s}", "JS-AP3": "P130_JS_AP3_c100_r50_views4_800ep_seed{s}", "SimCLR": "P130_simclr_c100_r50_views4_800ep_seed{s}"}
R18 = {"A-P3": "P107_AP3_c100_views4_800ep_seed{s}", "JS-AP3": "P120_JS_AP3_c100_views4_800ep_seed{s}", "SimCLR": "P91_c100_simclr_views4_800ep_seed{s}"}
SEEDS = (0, 1, 2)


def acc(run: str):
    f = O / run / "evaluations" / "evaluation_epoch_800.json"
    if not f.exists():
        return None
    d = json.load(open(f)); return float(d["linear_val_top1_pct"]), float(d["knn_val_top1_pct"])


def label(x) -> dict:
    x = np.asarray(x, float); m = float(x.mean()); sd = float(x.std(ddof=1)); h = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
    lab = "close" if abs(m) < 0.3 else ("clear" if (m - h > 0 or m + h < 0) else "inconclusive")
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "label": lab, "values": x.tolist()}


def main() -> int:
    r50 = {m: [acc(R50[m].format(s=s)) for s in SEEDS] for m in R50}; r18 = {m: [acc(R18[m].format(s=s)) for s in SEEDS] for m in R18}
    missing = [f"{m}/r50/s{s}" for m in r50 for s, v in zip(SEEDS, r50[m]) if v is None] + [f"{m}/r18/s{s}" for m in r18 for s, v in zip(SEEDS, r18[m]) if v is None]
    if missing:
        print("missing:", missing); return 1
    out = {"per_method": {}, "contrasts": {}}
    for m in R50:
        out["per_method"][m] = {"r50_lin": [v[0] for v in r50[m]], "r50_knn": [v[1] for v in r50[m]], "r18_lin": [v[0] for v in r18[m]],
                                "r50_lin_mean": float(np.mean([v[0] for v in r50[m]])), "r50_lin_sd": float(np.std([v[0] for v in r50[m]], ddof=1)),
                                "r50_minus_r18_lin": label([a[0] - b[0] for a, b in zip(r50[m], r18[m])])}
    for other in ("SimCLR", "JS-AP3"):
        out["contrasts"][f"A-P3_minus_{other}"] = {"lin": label([a[0] - b[0] for a, b in zip(r50["A-P3"], r50[other])]),
                                                   "knn": label([a[1] - b[1] for a, b in zip(r50["A-P3"], r50[other])])}
    json.dump(out, open(REP / "P130A1_results.json", "w"), indent=1)
    for m, v in out["per_method"].items():
        d = v["r50_minus_r18_lin"]
        print(f"{m:7s} R50 lin {[round(x, 2) for x in v['r50_lin']]} mean {v['r50_lin_mean']:.2f} ± {v['r50_lin_sd']:.2f} | kNN {[round(x, 2) for x in v['r50_knn']]} | R50 − R18 {d['mean']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]")
    for k, v in out["contrasts"].items():
        print(f"{k}: lin {v['lin']['mean']:+.2f} [{v['lin']['ci95'][0]:+.2f},{v['lin']['ci95'][1]:+.2f}] {v['lin']['label']} | knn {v['knn']['mean']:+.2f} [{v['knn']['ci95'][0]:+.2f},{v['knn']['ci95'][1]:+.2f}] {v['knn']['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
