"""V8.1 D02 supplement — first/second score gap per candidate pool (FULL_PLAN_CN 04 §D02 row 'first/second score gap').

Scores only, no labels: DEV (combined/dev_all.jsonl) and GenAI-Rank (labelfree/genai_rank_v8fit.jsonl, the label-free export allowed by
docs/07 §2).  Per pool with >= 2 candidates: gap = top score - second score in the method's native units (V8 arms, V7 units, kernels and
FGA are on the 1-5 rating scale; the other external systems are not, so gaps are compared only within a scale), and gap / within-pool IQR
(numpy linear percentiles; NA when IQR = 0).  Added 2026-10-16 after the V8.1 report check found the row missing; descriptive only.
Writes results/d02_score_gap.csv (one row per dataset x method x seed) and results/d02_score_gap.json (seed means per method)."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v81_common as C  # noqa: E402

FILES = {"DEV": C.NORM / "combined" / "dev_all.jsonl", "GenAI-Rank": C.NORM / "labelfree" / "genai_rank_v8fit.jsonl"}


def pools(path: Path) -> dict:
    out = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["method"], r["seed"], r["score_kind"])][r["pool_id"]].append(float(r["score"]))
    return out


def main():
    csv_out, js_out = C.RESULTS / "d02_score_gap.csv", C.RESULTS / "d02_score_gap.json"
    for p in (csv_out, js_out):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    rows = []
    for ds, path in FILES.items():
        for (meth, seed, kind), P in sorted(pools(path).items(), key=lambda t: (str(t[0][0]), str(t[0][1]))):
            g, gi, ties, iqr0 = [], [], 0, 0
            for s in P.values():
                if len(s) < 2:
                    continue
                a = np.sort(np.asarray(s))[::-1]; gap = float(a[0] - a[1]); g.append(gap); ties += gap == 0.0
                iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
                if iqr == 0.0:
                    iqr0 += 1
                else:
                    gi.append(gap / iqr)
            rows.append({"dataset": ds, "method": meth, "seed": seed, "score_kind": kind, "n_pools": len(g), "n_top_tie": ties,
                         "median_gap": float(np.median(g)), "mean_gap": float(np.mean(g)), "p90_gap": float(np.percentile(g, 90)),
                         "median_gap_over_iqr": float(np.median(gi)) if gi else None, "n_iqr_zero": iqr0})
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["method"], r["score_kind"])].append(r)
    summary = {"script": "scripts/v81/d02_score_gap.py", "status": "development_descriptive_label_free", "inputs": {k: str(v) for k, v in FILES.items()},
               "seed_means": [{"dataset": ds, "method": m, "score_kind": k, "n_seeds": len(rs), "n_pools": rs[0]["n_pools"],
                               **{x: float(np.mean([r[x] for r in rs])) for x in ("median_gap", "mean_gap", "p90_gap", "n_top_tie", "n_iqr_zero")},
                               "median_gap_over_iqr": (float(np.mean([r["median_gap_over_iqr"] for r in rs])) if all(r["median_gap_over_iqr"] is not None for r in rs) else None)}
                              for (ds, m, k), rs in sorted(agg.items())]}
    js_out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for s in summary["seed_means"]:
        if s["method"] in C.ARMS or s["method"].startswith(("B0", "B1", "K-NYSTROM-pca0", "K-RULSIF-pca0")):
            print(f"{s['dataset']:10s} {s['method']:26s} pools {s['n_pools']:5d} median gap {s['median_gap']:.4f} mean {s['mean_gap']:.4f} gap/IQR {s['median_gap_over_iqr']} ties {s['n_top_tie']:.1f}")


if __name__ == "__main__":
    main()
