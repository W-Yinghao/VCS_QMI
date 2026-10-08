"""P138 addendum 1 — K = 16 sampled pairs vs all pairs, seeds 0-2, four cells (frozen reading): per cell, K16 - parent paired by seed with a 95 % t
interval; non-inferior if the lower bound > -0.30 (CIFAR-10) / -0.50 (CIFAR-100), else "non-inferiority not shown".  kNN alongside.
Writes reports/P138A1_results.json; exits 1 (no reading) while any run is missing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

O = Path("/home/infres/yinwang/CS_QMI/outputs")
CELLS = {"vcs_c10": ("P138_PAIR_K16_vcs_c10_seed{s}", "P107_AP3_views4_800ep_seed{s}", 0.30),
         "js_c10": ("P138_PAIR_K16_js_c10_seed{s}", "P114_JSAP3_views4_800ep_seed{s}", 0.30),
         "vcs_c100": ("P138_PAIR_K16_vcs_c100_seed{s}", "P107_AP3_c100_views4_800ep_seed{s}", 0.50),
         "js_c100": ("P138_PAIR_K16_js_c100_seed{s}", "P120_JS_AP3_c100_views4_800ep_seed{s}", 0.50)}
SEEDS = (0, 1, 2)


def acc(run: str):
    f = O / run / "evaluations" / "evaluation_epoch_800.json"
    if not f.exists():
        return None
    d = json.load(open(f)); return float(d["linear_val_top1_pct"]), float(d["knn_val_top1_pct"])


def tint(x: np.ndarray) -> tuple[float, float, float]:
    m = float(x.mean()); h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))); return m, m - h, m + h


def main() -> int:
    out, missing = {}, []
    for cell, (k16, parent, margin) in CELLS.items():
        rows = []
        for s in SEEDS:
            a, b = acc(k16.format(s=s)), acc(parent.format(s=s))
            if a is None or b is None:
                missing.append((cell, s, a is None, b is None)); continue
            rows.append({"seed": s, "k16_lin": a[0], "k16_knn": a[1], "parent_lin": b[0], "parent_knn": b[1], "d_lin": a[0] - b[0], "d_knn": a[1] - b[1]})
        out[cell] = {"rows": rows, "margin": margin}
        if len(rows) == len(SEEDS):
            m, lo, hi = tint(np.array([r["d_lin"] for r in rows])); mk, lok, hik = tint(np.array([r["d_knn"] for r in rows]))
            out[cell].update({"d_lin_mean": m, "d_lin_ci95": [lo, hi], "d_knn_mean": mk, "d_knn_ci95": [lok, hik],
                              "reading": "non-inferior" if lo > -margin else "non-inferiority not shown"})
    if missing:
        print("missing (cell, seed, k16_missing, parent_missing):", missing); return 1
    json.dump(out, open("reports/P138A1_results.json", "w"), indent=1)
    for cell, v in out.items():
        print(f"{cell:9s} Δlin {v['d_lin_mean']:+.2f} [{v['d_lin_ci95'][0]:+.2f},{v['d_lin_ci95'][1]:+.2f}] (margin −{v['margin']:.2f}) -> {v['reading']} | "
              f"Δknn {v['d_knn_mean']:+.2f} [{v['d_knn_ci95'][0]:+.2f},{v['d_knn_ci95'][1]:+.2f}] | per seed {[round(r['d_lin'], 2) for r in v['rows']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
