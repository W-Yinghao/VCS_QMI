"""P145 addendum 1 — CIFAR-100 STRESS (eps 0.10) seeds 0-2 for VCS / JS / SimCLR, paired with their clean runs by seed (frozen reading).
Writes reports/P145A1_results.json; prints the table.  Missing runs are listed and the script exits 1 (no partial reading)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

O = Path("/home/infres/yinwang/CS_QMI/outputs")
CLEAN = {"vcs": "P107_AP3_c100_views4_800ep_seed{s}", "js": "P120_JS_AP3_c100_views4_800ep_seed{s}", "simclr": "P91_c100_simclr_views4_800ep_seed{s}"}
STRESS = "P145_STRESS10_{m}_c100_seed{s}"
SEEDS = (0, 1, 2)


def acc(run: str):
    f = O / run / "evaluations" / "evaluation_epoch_800.json"
    if not f.exists():
        return None
    d = json.load(open(f)); return float(d["linear_val_top1_pct"]), float(d["knn_val_top1_pct"])


def label(x: np.ndarray) -> dict:
    m = float(x.mean()); sd = float(x.std(ddof=1)); h = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
    lab = "close" if abs(m) < 0.3 else ("clear" if (m - h > 0 or m + h < 0) else "inconclusive")
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "label": lab, "values": x.tolist()}


def main() -> int:
    rows, missing = {}, []
    for m in CLEAN:
        for s in SEEDS:
            c, t = acc(CLEAN[m].format(s=s)), acc(STRESS.format(m=m, s=s))
            if c is None or t is None:
                missing.append((m, s, c is None, t is None)); continue
            rows[(m, s)] = {"clean_lin": c[0], "clean_knn": c[1], "stress_lin": t[0], "stress_knn": t[1], "d_lin": t[0] - c[0], "d_knn": t[1] - c[1]}
    if missing:
        print("missing (method, seed, clean_missing, stress_missing):", missing); return 1
    out = {"per_seed": {f"{m}/s{s}": r for (m, s), r in rows.items()}, "per_method": {}, "interaction": {}}
    for m in CLEAN:
        d = np.array([rows[(m, s)]["d_lin"] for s in SEEDS]); dk = np.array([rows[(m, s)]["d_knn"] for s in SEEDS])
        out["per_method"][m] = {"delta_lin": label(d), "delta_knn": label(dk),
                                "clean_lin_mean": float(np.mean([rows[(m, s)]["clean_lin"] for s in SEEDS])),
                                "stress_lin_mean": float(np.mean([rows[(m, s)]["stress_lin"] for s in SEEDS]))}
    for m in ("js", "simclr"):
        x = np.array([rows[("vcs", s)]["d_lin"] - rows[(m, s)]["d_lin"] for s in SEEDS])
        xk = np.array([rows[("vcs", s)]["d_knn"] - rows[(m, s)]["d_knn"] for s in SEEDS])
        out["interaction"][f"vcs_minus_{m}"] = {"lin": label(x), "knn": label(xk)}
    json.dump(out, open("reports/P145A1_results.json", "w"), indent=1)
    for m, v in out["per_method"].items():
        dl = v["delta_lin"]; print(f"{m:7s} clean {v['clean_lin_mean']:.2f} stress {v['stress_lin_mean']:.2f}  Δ {dl['mean']:+.2f} [{dl['ci95'][0]:+.2f},{dl['ci95'][1]:+.2f}] {dl['label']}  per seed {[round(x, 2) for x in dl['values']]}")
    for k, v in out["interaction"].items():
        print(f"{k}: lin {v['lin']['mean']:+.2f} [{v['lin']['ci95'][0]:+.2f},{v['lin']['ci95'][1]:+.2f}] {v['lin']['label']} | knn {v['knn']['mean']:+.2f} {v['knn']['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
