"""P116 aggregation — applies the pre-stated reading of P116_V5_SOLVE_PREREG (§6): per cell, paired over repeats vs A1, differences in the EVAL
common score J(T), own-statistic and common-statistic rejection (mean and 95 % normal interval), cost (seconds, closures), residuals, A2 chose-A1
fraction; level flags (> 0.09 at the null cells).  "Fits better" and "permutation power higher" are reported separately.

    python scripts/p116_aggregate.py --in reports/P116 --prefix full_ --out reports/P116_results
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_ssl.utils import atomic_write_json  # noqa: E402

ORDER = ("A1", "A2@50", "A2@200", "A2@800", "A2L@50", "A2L@200", "A2L@800", "A3@50", "A3@200", "A3@800", "JS_P105")


def paired(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    d = np.asarray(x, float) - np.asarray(y, float); m = float(d.mean())
    h = 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return m, m - h, m + h


def cell_rows(path: str):
    d = json.load(open(path))
    for case, c in d["cases"].items():
        for n, b in c["by_n"].items():
            yield Path(path).stem, case, c["mode"], int(n), b


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="reports/P116"); ap.add_argument("--prefix", default="")
    ap.add_argument("--out", required=True); a = ap.parse_args()
    files = sorted(f for f in glob.glob(f"{a.inp}/{a.prefix}*.json") if not f.endswith(".partial.json"))
    res = {"files": files, "cells": []}
    L = [f"# P116 results ({a.prefix or 'all'}) — paired vs A1 over repeats; 95 % normal intervals", ""]
    for src, case, mode, n, b in (r for f in files for r in cell_rows(f)):
        inst = b["instances"]; R = len(inst)
        names = [k for k in ORDER if k in inst[0]["critics"]]
        cell = {"source": src, "case": case, "mode": mode, "n": n, "repeats": R, "algos": {}}
        L += [f"## {src} — {case}, n {n}, R {R} ({mode})", "",
              "| algo | power own | power common | J(T) EVAL | ΔJ(T) EVAL vs A1 [95 %] | Δpower own vs A1 [95 %] | Δpower common vs A1 [95 %] | own risk VAL | sec fit | closures | perm sec | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        base = {k: np.array([i["critics"]["A1"][k] if k == "Jcommon_eval" else i["critics"]["A1"]["test"][k.split(":")[1]]["reject"] for i in inst], float)
                for k in ("Jcommon_eval", "test:own", "test:common")}
        for nm in names:
            rows = [i["critics"][nm] for i in inst]
            po = np.array([r["test"]["own"]["reject"] for r in rows], float); pc = np.array([r["test"]["common"]["reject"] for r in rows], float)
            je = np.array([r["Jcommon_eval"] for r in rows], float)
            dj, djl, dju = paired(je, base["Jcommon_eval"]); dpo = paired(po, base["test:own"]); dpc = paired(pc, base["test:common"])
            sec = np.nanmean([r["meta"].get("seconds_total", r["meta"].get("seconds", np.nan)) for r in rows])
            clo = np.mean([r["meta"].get("closures_used", 0) for r in rows]); ps = np.mean([r["test"]["seconds"] for r in rows])
            risk_val = np.mean([r["own_risk_val"] for r in rows])
            note = []
            if nm.startswith("A2@"):
                note.append(f"chose A1 {np.mean([r['meta']['chose_A1'] for r in rows]):.2f}")
            if nm.startswith("A3@") or nm == "JS_P105":
                terms = [t["termination"] for r in rows for t in (r["meta"].get("per_lam") or r["meta"].get("solver", {}).get("per_lam", []))]
                if terms:
                    note.append("term " + ",".join(f"{t}:{terms.count(t)}" for t in sorted(set(terms))))
            flag = ""
            if mode != "planted" and (po.mean() > 0.09 or pc.mean() > 0.09):
                flag = "LEVEL>0.09"
            fits_better = nm != "A1" and djl > 0
            power_higher_own = nm != "A1" and dpo[1] > 0
            power_higher_common = nm != "A1" and dpc[1] > 0
            cell["algos"][nm] = {"power_own": po.mean(), "power_common": pc.mean(), "J_eval": je.mean(), "dJ_eval": [dj, djl, dju], "dpower_own": list(dpo),
                                 "dpower_common": list(dpc), "own_risk_val": risk_val, "seconds": sec, "closures": clo, "perm_seconds": ps,
                                 "fits_better_than_A1": bool(fits_better), "power_higher_own": bool(power_higher_own), "power_higher_common": bool(power_higher_common),
                                 "level_flag": flag}
            L.append(f"| {nm} | {po.mean():.2f} | {pc.mean():.2f} | {je.mean():+.4f} | {dj:+.4f} [{djl:+.4f}, {dju:+.4f}] | {dpo[0]:+.2f} [{dpo[1]:+.2f}, {dpo[2]:+.2f}] | "
                     f"{dpc[0]:+.2f} [{dpc[1]:+.2f}, {dpc[2]:+.2f}] | {risk_val:+.4f} | {sec:.2f} | {clo:.0f} | {ps:.3f} | {'; '.join(note)} {flag} |")
        res["cells"].append(cell); L.append("")
    atomic_write_json(Path(a.out + ".json"), res); Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print(f"-> {a.out}.md ({len(res['cells'])} cells from {len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
