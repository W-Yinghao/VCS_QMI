"""P141 aggregator: per method x site x task tables (mean ± sample sd over seeds) and seed-paired contrasts with the P114 labels.

    python scripts/p141_aggregate.py --in reports/P141 --out reports/P141_v7_layer_readout_results
Primary readout = recipe_raw (the P124 primary rule) at every site; raw_std and kNN reported alongside.  Contrasts (paired by seed, 95 % t interval):
A-P3 − JS-AP3 (seeds 0–4) and A-P3 − SimCLR (seeds 0–2) at each site and task; labels: close |Δ| < 0.3; clear |Δ| ≥ 0.3 with the interval excluding
0; otherwise inconclusive; n = 1 → "single seed".  Descriptive: A-P3 − tuned JS (3, 0.5) (seeds 0–2), ResNet-50 − ResNet-18 at seed 0, and the
site profile layer3 → h → r → z.  No site replaces the h (backbone) metric.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from p141_layers import FAMILIES  # noqa: E402

SITES = ("layer3", "h", "r", "z")
TASKS = ("coarse", "fine", "conditional")
RUN2METHOD = {r: m for m, rs in FAMILIES.items() for r in rs}


def seed_of(run: str) -> int:
    return int(re.search(r"seed(\d+)$", run).group(1))


def value(res: dict, site: str, task: str, readout: str) -> float | None:
    t = res["sites"].get(site, {}).get("tasks", {}).get(task, {}).get(readout)
    if t is None:
        return None
    return t["macro_pct"] if task == "conditional" else t["acc_pct"]


def label(d: np.ndarray) -> tuple[float, float, float, str]:
    n = len(d); m = float(d.mean())
    if n < 2:
        return m, float("nan"), float("nan"), "single seed"
    from scipy import stats  # noqa: PLC0415
    h = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)); lo, hi = m - h, m + h
    lab = "close" if abs(m) < 0.3 else ("clear" if (lo > 0 or hi < 0) else "inconclusive")
    return m, lo, hi, lab


def paired(R: dict, ma: str, mb: str, site: str, task: str, readout: str = "recipe_raw"):
    def vals(method):
        out = {}
        for r, res in R.items():
            if RUN2METHOD.get(r) == method:
                x = value(res, site, task, readout)
                if x is not None:
                    out[seed_of(r)] = x
        return out
    a, b = vals(ma), vals(mb)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    d = np.array([a[s] - b[s] for s in seeds]); m, lo, hi, lab = label(d)
    return {"a": ma, "b": mb, "site": site, "task": task, "readout": readout, "seeds": seeds, "delta": d.tolist(), "mean": m, "ci95": [lo, hi], "label": lab}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="reports/P141"); ap.add_argument("--out", required=True)
    ap.add_argument("--include-smoke", action="store_true", help="gate only: also read smoke result files")
    a = ap.parse_args()
    R = {}
    for f in sorted(Path(a.inp).glob("layers_*.json")):
        d = json.loads(f.read_text())
        if d.get("smoke") and not a.include_smoke:
            continue
        R[d["run"]] = d
    readouts = sorted({k for d in R.values() for k in d.get("readouts", [])}, key=lambda k: ["recipe_raw", "raw_unstd", "raw_std", "l2_std", "knn"].index(k))
    lines = ["# P141 — CIFAR-100 four-site (layer3 / h / r / z) readouts: results", "",
             f"{len(R)} encoders.  Primary readout recipe_raw (P124 rule); conditional = macro 5-way given the coarse label.  h is the backbone metric.", ""]
    table = []
    for method in FAMILIES:
        runs = [r for r in R if RUN2METHOD.get(r) == method]
        if not runs:
            continue
        for site in SITES:
            row = {"method": method, "site": site, "seeds": sorted(seed_of(r) for r in runs), "dim": R[runs[0]]["sites"].get(site, {}).get("dim")}
            for task in TASKS:
                for ro in readouts:
                    v = np.array([x for r in runs if (x := value(R[r], site, task, ro)) is not None])
                    row[f"{task}:{ro}"] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else None
            table.append(row)
    hdr = ["method", "site", "dim", "n"] + [f"{t} ({ro})" for t in TASKS for ro in readouts]
    lines += ["## 1. Per method and site (mean ± sd over seeds)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for row in table:
        cells = [row["method"], row["site"], str(row["dim"]), str(len(row["seeds"]))]
        for t in TASKS:
            for ro in readouts:
                v = row[f"{t}:{ro}"]; cells.append("—" if v is None else f"{v[0]:.2f} ± {v[1]:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    contrasts = []
    for mb in ("JS-AP3", "SimCLR", "JS (3,0.5) tuned"):
        for site in SITES:
            for task in TASKS:
                c = paired(R, "A-P3", mb, site, task)
                if c:
                    contrasts.append(c)
    lines += ["", "## 2. Seed-paired contrasts, recipe_raw (A-P3 − JS-AP3 and A-P3 − SimCLR frozen labels; vs tuned JS descriptive)", "",
              "| contrast | site | task | seeds | mean Δ | 95 % CI | label |", "|---|---|---|---|---|---|---|"]
    for c in contrasts:
        lines.append(f"| A-P3 − {c['b']} | {c['site']} | {c['task']} | {len(c['seeds'])} | {c['mean']:+.2f} | [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] | {c['label']} |")
    r50 = []
    for m18, m50 in (("A-P3", "R50 A-P3"), ("JS-AP3", "R50 JS-AP3"), ("SimCLR", "R50 SimCLR")):
        for site in SITES:
            for task in TASKS:
                c = paired(R, m50, m18, site, task)
                if c:
                    r50.append(c)
    if r50:
        lines += ["", "## 3. ResNet-50 − ResNet-18 at seed 0 (descriptive)", "", "| method | site | task | Δ |", "|---|---|---|---|"]
        for c in r50:
            lines.append(f"| {c['b']} | {c['site']} | {c['task']} | {c['mean']:+.2f} |")
    qc = {r: d.get("qc") for r, d in R.items()}
    lines += ["", "## QC", "", "max |h − P124-cached h| per run (expected ~0): " +
              ", ".join(f"{r}: {max((v for k, v in q.items() if k.endswith('maxabs') and v is not None), default=float('nan')):.2e}" for r, q in qc.items() if q)]
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")
    Path(a.out + ".json").write_text(json.dumps({"table": table, "contrasts": contrasts, "r50_minus_r18": r50, "qc": qc, "n_runs": len(R)}, indent=1, default=float))
    print(f"-> {a.out}.md / .json ({len(R)} encoders)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
