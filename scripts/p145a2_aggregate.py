"""P145 addendum 2 — STRESS eps 0.20, seed 0 (frozen reading, P145_ADDENDUM2_EPS020_PREREG_FROZEN_20261008.md): Delta_m = A_m(0.20) − A_m(0)
against the same clean seed-0 parents as P145 (read from reports/P145_results.json), trigger |Delta_VCS − Delta_m| >= 0.50 (CIFAR-10) / 1.00
(CIFAR-100) for m in {JS, SimCLR}; descriptive dose curve eps 0 -> 0.10 -> 0.20 (seed 0).  Writes reports/P145A2_results.json; exits 1 if a
run is missing (no partial reading)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

O = Path("/home/infres/yinwang/CS_QMI/outputs"); REP = Path(__file__).resolve().parents[1] / "reports"
THR = {"c10": 0.50, "c100": 1.00}


def acc(run: str):
    f = O / run / "evaluations" / "evaluation_epoch_800.json"
    if not f.exists():
        return None
    d = json.load(open(f)); return float(d["linear_val_top1_pct"]), float(d["knn_val_top1_pct"])


def main() -> int:
    p145 = json.load(open(REP / "P145_results.json")); out = {"per_cell": {}, "trigger": {}, "missing": []}
    for ds in ("c10", "c100"):
        for m in ("vcs", "js", "simclr"):
            clean = p145[f"{m}/{ds}"]["clean"]; c, e1, e2 = acc(clean), acc(f"P145_STRESS10_{m}_{ds}_seed0"), acc(f"P145_STRESS20_{m}_{ds}_seed0")
            if None in (c, e1, e2):
                out["missing"].append(f"{m}/{ds}"); continue
            out["per_cell"][f"{m}/{ds}"] = {"clean_run": clean, "lin": {"eps0": c[0], "eps0.10": e1[0], "eps0.20": e2[0]},
                                            "knn": {"eps0": c[1], "eps0.10": e1[1], "eps0.20": e2[1]}, "d_lin": e2[0] - c[0], "d_knn": e2[1] - c[1]}
    if out["missing"]:
        print("missing:", out["missing"]); return 1
    for ds in ("c10", "c100"):
        dv = out["per_cell"][f"vcs/{ds}"]["d_lin"]
        for m in ("js", "simclr"):
            x = dv - out["per_cell"][f"{m}/{ds}"]["d_lin"]
            out["trigger"][f"{ds}/vcs_minus_{m}"] = {"value": x, "threshold": THR[ds], "fires": abs(x) >= THR[ds]}
    out["seeds_triggered_for"] = sorted({k.split("/")[0] for k, v in out["trigger"].items() if v["fires"]})
    json.dump(out, open(REP / "P145A2_results.json", "w"), indent=1)
    for k, v in out["per_cell"].items():
        l = v["lin"]; print(f"{k:11s} lin eps 0 / 0.10 / 0.20: {l['eps0']:.2f} / {l['eps0.10']:.2f} / {l['eps0.20']:.2f}  Δ(0.20) {v['d_lin']:+.2f}  kNN Δ {v['d_knn']:+.2f}")
    for k, v in out["trigger"].items():
        print(f"{k}: {v['value']:+.2f} (threshold {v['threshold']:.2f}) {'FIRES' if v['fires'] else 'no'}")
    print("seeds 1–2 triggered for:", out["seeds_triggered_for"] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
