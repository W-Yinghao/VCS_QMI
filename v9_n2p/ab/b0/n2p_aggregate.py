"""SYSTEM_FULLDEV_RESULTS.csv from the B1 units' RESULT.json (one row per family x seed; missing units are rows with status NOT_COMPLETE).

    python -I n2p_aggregate.py --root <N2P_OUT> --out <SYSTEM_FULLDEV_RESULTS.csv>
"""
import argparse
import csv
import json
from pathlib import Path

RECALLS = ("i2t_R1", "i2t_R5", "i2t_R10", "t2i_R1", "t2i_R5", "t2i_R10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for method in ("NCR_A", "L2RM_A"):
        for seed in (901, 902, 903):
            unit = Path(a.root) / f"V9N2P_f30k_r04_s{seed}_{method}"
            row = {"unit_id": unit.name, "method": method, "seed": seed, "rho": 0.4, "selection": "fullDEV1014 best rSum (earliest exact tie)"}
            res = unit / "RESULT.json"
            if not res.exists():
                rows.append({**row, "status": "NOT_COMPLETE"})
                continue
            r = json.loads(res.read_text())
            b = r["primary_fullDEV_best"]
            attempts = sorted(p.name for p in (unit / "attempts").glob("attempt_*")) if (unit / "attempts").exists() else []
            row.update(status="COMPLETE" if not r["missing_candidates"] else "INCOMPLETE_CANDIDATES", row_label=r["row_label"],
                       best_candidate=b["candidate_id"], **{k: b["fullDEV"][k] for k in RECALLS}, rsum=b["fullDEV"]["rsum"], mR=b["fullDEV"]["mR"],
                       last_rsum=r["last"]["fullDEV"]["rsum"] if r["last"] else None, mean_last5_rsum=r["mean_last5_rsum"],
                       candidates_expected=r["candidates_expected"], candidates_scored=r["candidates_scored"],
                       missing_candidates=";".join(r["missing_candidates"]),
                       native_selector_candidate=(r["native_sensitivity"]["official_native_selector"] or {}).get("candidate_id"),
                       native_selector_native200_rsum=(r["native_sensitivity"]["official_native_selector"] or {}).get("trainer_native_rsum"),
                       native200_best_candidate=r["native_sensitivity"]["native200_recomputed_best"]["candidate_id"],
                       train_wall_s=r["costs"].get("train", {}).get("wall_seconds"), forward_wall_s=r["costs"].get("forward", {}).get("wall_seconds"),
                       score_wall_s=r["costs"].get("score", {}).get("wall_seconds"), gpu=r["costs"].get("train", {}).get("gpu"),
                       earlier_failed_attempts=";".join(attempts), checkpoint=b["checkpoint"], checkpoint_sha256=b["checkpoint_sha256"])
            rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{a.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
