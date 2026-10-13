"""V7 tuning -> formal learning rate per M arm (V7-03): the lr whose selected epoch has the lowest CAL_SELECT natural element Brier among the three
tuning units (seed 17, full FIT); ties -> the lower lr.  Reads only the units' own logs (CAL_SELECT, already used for epoch selection).
    python -I scripts/v7/v7_tune_select.py --units-dir D --out lr_choice.json"""
import argparse
import json
from pathlib import Path

ARMS = ("M0_semantic_l1", "M1_semantic_logistic", "M2_semantic_vcs"); LRS = ("5e-06", "1e-05", "2e-05")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--units-dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); D = Path(a.units_dir); res = {}
    for arm in ARMS:
        rows = []
        for lr in LRS:
            u = D / f"V7-TUNE-{arm}-lr{lr}"
            if not (u / "done.json").exists():
                raise SystemExit(f"tuning unit missing: {u.name}")
            L = json.loads((u / "log.json").read_text()); rows.append((L["selected_cal_select_brier"], float(lr), L["selected_epoch"], u.name))
        best = min(rows, key=lambda r: (r[0], r[1]))
        res[arm] = {"chosen_lr": best[1], "chosen_unit": best[3], "candidates": [{"lr": r[1], "cal_select_brier": r[0], "selected_epoch": r[2], "unit": r[3]} for r in rows]}
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps({k: v["chosen_lr"] for k, v in res.items()}))


if __name__ == "__main__":
    main()
