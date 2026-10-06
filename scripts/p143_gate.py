"""P143 stage 0 — reproduction gate for the regenerated P109 I1 cache (the original cache was deleted by cleanup rule R6).

Pass (frozen in the P143 prereg): every layer x statistic power of the regenerated A-P3 seed-1 colour cells (planted 0.1 and both nulls) within
+-0.05 of reports/P118_i1_P107_AP3_views4_800ep_seed1_colour_{power,level}.json, and the colour-0.1 paired effects (d_acc, d_prob_true) inside the
recorded 95 % intervals of reports/P109_i1_P107_AP3_views4_800ep_seed1_colour_effects.json.  Exit 0 = pass, 3 = fail (stops stages 1-2).
    python scripts/p143_gate.py --out reports/P143/stage0_gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = "P107_AP3_views4_800ep_seed1"
TOL = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows, ok = [], True
    for kind in ("power", "level"):
        old = json.load(open(REPO / f"reports/P118_i1_{RUN}_colour_{kind}.json"))["cells"]
        new = json.load(open(REPO / f"reports/P143/stage0_{RUN}_colour_{kind}.json"))["cells"]
        for ck, c in old.items():
            for key, v in c["summary"].items():
                d = new[ck]["summary"][key]["power"] - v["power"]
                rows.append({"cell": ck, "stat": key, "old": v["power"], "new": new[ck]["summary"][key]["power"], "diff": d, "ok": abs(d) <= TOL})
                ok &= abs(d) <= TOL
    eo = json.load(open(REPO / f"reports/P109_i1_{RUN}_colour_effects.json"))["prediction_effects"]["colour_s0.1"]["overall"]
    en = json.load(open(REPO / f"reports/P143/stage0_{RUN}_colour_effects.json"))["prediction_effects"]["colour_s0.1"]["overall"]
    eff = {}
    for k, ci in (("d_acc", "d_acc_ci"), ("d_prob_true", "d_prob_true_ci")):
        inside = eo[ci][0] <= en[k] <= eo[ci][1]; ok &= inside
        eff[k] = {"old": eo[k], "old_ci": eo[ci], "new": en[k], "inside": inside}
    eff["flip_old_new"] = [eo["prediction_flip_rate"], en["prediction_flip_rate"]]
    res = {"run": RUN, "tolerance_power": TOL, "max_abs_power_diff": max(abs(r["diff"]) for r in rows), "effects": eff, "pass": bool(ok), "rows": rows}
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    print(f"P143 stage-0 gate: pass={ok} max |dpower| {res['max_abs_power_diff']:.3f}; effects {eff}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
