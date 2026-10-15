"""V8-03 lr rule: per arm, the tuning lr (seed 17, {5e-6, 1e-5, 2e-5}) with the lowest CAL_SELECT mean-rating MSE at its selected epoch;
exact ties -> the lower lr.  Refuses unless all 24 tuning units of configs/TUNING_UNITS.csv are done (formal units start only after all of them).
Reads done.json / log.json of the tuning units only (no label beyond the CAL_SELECT metric each unit already recorded)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

TUNING_CSV = Path("/home/infres/yinwang/CS_QMI/outputs/V8/package_ref_v1/VCS_V8_Utility_Aligned_CS_Server_v1_20261014/configs/TUNING_UNITS.csv")


def choose(units_dir: Path, tuning_csv: Path = TUNING_CSV) -> dict:
    rows = list(csv.DictReader(tuning_csv.read_text(encoding="utf-8-sig").splitlines()))
    missing = [r["unit_id"] for r in rows if not (units_dir / r["unit_id"] / "done.json").exists()]
    if len(rows) != 24 or missing:
        raise SystemExit(f"[v8_select_lr] {len(rows)} tuning units listed, {len(missing)} not done: {missing[:4]}")
    by_arm: dict[str, list] = {}
    for r in rows:
        d = json.loads((units_dir / r["unit_id"] / "done.json").read_text())
        by_arm.setdefault(r["arm"], []).append({"unit": r["unit_id"], "lr": r["lr"], "lr_value": float(r["lr"]), "selected_epoch": d["selected_epoch"],
                                                "cal_select_mse": d["cal_select_mse"], "gpu": d.get("gpu")})
    out = {}
    for arm, g in by_arm.items():
        g.sort(key=lambda x: x["lr_value"]); best = min(g, key=lambda x: (x["cal_select_mse"], x["lr_value"]))
        grid = [x["lr_value"] for x in g]
        out[arm] = {"lr": best["lr"], "cal_select_mse": best["cal_select_mse"], "selected_epoch": best["selected_epoch"],
                    "lr_edge": "low" if best["lr_value"] == grid[0] else "high" if best["lr_value"] == grid[-1] else None, "tuning": g}
    return {"rule": "lowest CAL_SELECT mean-rating MSE at the selected epoch; exact ties -> lower lr (V8-03)", "arms": out}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--units", default="/projects/EEG-foundation-model/yinghao/outputs_v8/units")
    ap.add_argument("--out", required=True); ap.add_argument("--arm", default=None, help="print only this arm's lr (for the group wrapper)")
    a = ap.parse_args(); res = choose(Path(a.units)); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)
    if a.arm:
        print(res["arms"][a.arm]["lr"])
    else:
        for arm, r in res["arms"].items():
            print(f"{arm}: lr {r['lr']} (CAL_SELECT MSE {r['cal_select_mse']:.5f}, epoch {r['selected_epoch']}, edge {r['lr_edge']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
