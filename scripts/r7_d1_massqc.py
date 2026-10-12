"""r7 R7-30 D1-10 QC (FIT shards only — never CAL / DEV before the pre-registered readout): distribution of the first-token yes / no
probability mass of the teacher passes (full, text-only, masked), the fraction of low-mass items, and the decoded-answer-free view of
which questions are affected (structural / semantic type).
    python scripts/r7_d1_massqc.py --teacher-dir T --shards FIT-000 [FIT-001 ...] --out O
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_vl_common as V  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher-dir", required=True); ap.add_argument("--shards", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if any(not s.startswith("FIT-") for s in a.shards):
        raise SystemExit("QC reads FIT shards only before the CAL readout")
    rows = [r for s in a.shards for r in V.read_jsonl(Path(a.teacher_dir) / f"{s}.jsonl")]
    full = np.array([sum(r["full"]) for r in rows]); text = np.array([sum(r["text"]) for r in rows])
    masked = np.array([sum(r["masked"][m][k]) for r in rows for m in r["masked"] for k in r["masked"][m]])
    def dist(x):
        return {"n": int(len(x)), "min": float(x.min()), "p01": float(np.percentile(x, 1)), "p05": float(np.percentile(x, 5)), "median": float(np.median(x)),
                "frac_below_0.9": float(np.mean(x < 0.9)), "frac_below_0.5": float(np.mean(x < 0.5))}
    low = [r for r in rows if sum(r["full"]) < 0.9]; by = {}
    for r in low:
        by[(r["structural"], r["semantic"])] = by.get((r["structural"], r["semantic"]), 0) + 1
    allby = {}
    for r in rows:
        allby[(r["structural"], r["semantic"])] = allby.get((r["structural"], r["semantic"]), 0) + 1
    res = {"shards": a.shards, "full": dist(full), "text_only": dist(text), "masked": dist(masked),
           "low_full_mass_by_type": {f"{k[0]}/{k[1]}": {"low": v, "of": allby[k]} for k, v in sorted(by.items(), key=lambda kv: -kv[1])}}
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "mass_qc_FIT.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
