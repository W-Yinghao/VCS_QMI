"""Print the AGIQA-3K alignment pilot TEST summary from the package's metric and compare JSONs (read-only; --write saves
TEST_SUMMARY.json). PILOT_DIAGNOSTIC_ONLY: TEST seen once, no selection after TEST."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ORDER = ["task_only", "reference_nce_text", "bce_text", "vcs_exact_text", "vcs_sampled8_text", "corrected_cmim_both", "native_imagereward"]
KEYS = ["srcc", "plcc", "within_prompt_pairwise_macro", "normalized_regret"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-dir", default="/home/infres/yinwang/CS_QMI/outputs_apps/aigiqa_20261020/results/test")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    T = Path(a.test_dir)
    fam = defaultdict(list)
    for p in sorted((T / "metrics").glob("*.json")):
        fam[p.stem.split("__seed")[0]].append(json.loads(p.read_text()))
    rows = {}
    for f in ORDER:
        ms = fam.get(f, [])
        if not ms:
            continue
        stat = {}
        for k in KEYS:
            v = [m[k] for m in ms if m.get(k) is not None]
            stat[k] = {"mean": statistics.mean(v) if v else None, "sd": statistics.stdev(v) if len(v) > 1 else None, "n_defined": len(v)}
        rows[f] = {"n": len(ms), **stat}
    comps = {p.stem: json.loads(p.read_text())["differences"] for p in sorted((T / "compare").glob("*.json"))}
    print("AGIQA-3K alignment TEST, PILOT_DIAGNOSTIC_ONLY (one look, no selection after TEST); mean (sd) over seeds")
    print(f"{'family':22s} n  " + "  ".join(f"{k[:14]:>16s}" for k in KEYS))
    for f, r in rows.items():
        cells = [("n/a" if r[k]["mean"] is None else f"{r[k]['mean']:.4f}") + (f" ({r[k]['sd']:.4f})" if r[k]["sd"] is not None else "         ")
                 for k in KEYS]
        print(f"{f:22s} {r['n']}  " + "  ".join(f"{c:>16s}" for c in cells))
    print("VCS-exact minus BCE, same seed (95% prompt-bootstrap CI; regret: lower is better)")
    for name, d in comps.items():
        fmt = lambda x: "n/a" if x is None else f"{x:+.4f}"
        print(f"  {name}: " + "; ".join(f"{k} {fmt(d[k]['a_minus_b'])} [{', '.join(map(fmt, d[k]['ci95'] or [None, None]))}]" for k in KEYS))
    if a.write:
        (T / "TEST_SUMMARY.json").write_text(json.dumps({"claim_status": "PILOT_DIAGNOSTIC_ONLY", "families": rows, "vcs_exact_minus_bce": comps},
                                                       indent=1) + "\n")


if __name__ == "__main__":
    main()
