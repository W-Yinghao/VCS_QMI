"""Appendix ablation table of the SSL exploration, from reports/results_v7.json (raw per-run readouts; development split).  Each finished run
of an exploration line is listed with what it changed relative to the A-P3 seed-0 parent of its dataset and its final linear / kNN and the delta.
Analysis only; writes reports/ABLATION_TABLE.md.
    python scripts/ablation_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = json.load(open(REPO / "reports" / "results_v7.json"))["table_A"]
PARENT = {"cifar10": "P107_AP3_views4_800ep_seed0", "cifar100": "P107_AP3_c100_views4_800ep_seed0"}
LINES = {"P112": "scorer scale sensitivity", "P115": "augmentation factors", "P126": "curved scorer", "P129": "tuned JS scorer", "P133": "momentum keys",
         "P135": "learning rate", "P136": "schedule / views / batch", "P137": "projector width", "P138": "K = 16 sampled pairs", "P145": "pair contamination (STRESS)",
         "P147": "batch 512 + lr scaling", "P148": "weight decay", "P153": "weak crop"}
FIELDS = ["lr", "views", "crop_min", "epochs", "batch_images", "projector", "backbone"]


def diff(run: dict, par: dict) -> str:
    out = []
    for f in FIELDS:
        if run.get(f) != par.get(f):
            out.append(f"{f} {par.get(f)}→{run.get(f)}")
    for grp in ("scorer", "pairing"):
        for k, v in (run.get(grp) or {}).items():
            if (par.get(grp) or {}).get(k) != v:
                out.append(f"{grp}.{k} {(par.get(grp) or {}).get(k)}→{v}")
    return "; ".join(out) or "(seed / stage only)"


def main() -> int:
    by_id = {r["run_id"]: r for r in R}
    rows = ["| line | run | dataset | family | change vs A-P3 seed 0 | linear | kNN | Δ linear | Δ kNN |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(R, key=lambda r: (r["line"], r["dataset"], r["run_id"])):
        if r["line"] not in LINES or r["dataset"] not in PARENT or not r.get("final"):
            continue
        par = by_id[PARENT[r["dataset"]]]; fl, fk = r["final"]["linear"], r["final"].get("knn"); pl, pk = par["final"]["linear"], par["final"].get("knn")
        rows.append(f"| {r['line']} {LINES[r['line']]} | {r['run_id']} | {r['dataset']} | {r['family']} | {diff(r, par)} | {fl:.2f} | {fk:.2f} | {fl - pl:+.2f} | {fk - pk:+.2f} |")
    text = ("# SSL exploration — appendix ablation table (development split, from results_v7.json)\n\nParents: A-P3 seed 0 = "
            f"{PARENT['cifar10']} ({by_id[PARENT['cifar10']]['final']['linear']:.2f}) and {PARENT['cifar100']} ({by_id[PARENT['cifar100']]['final']['linear']:.2f}).  "
            "Single-seed cells are descriptive; multi-seed lines have their own reports.\n\n" + "\n".join(rows) + "\n")
    (REPO / "reports" / "ABLATION_TABLE.md").write_text(text); print(f"{len(rows) - 2} rows -> reports/ABLATION_TABLE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
