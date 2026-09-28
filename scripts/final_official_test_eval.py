"""P67 — one-shot official CIFAR-10 test evaluation driver (owner's rule: the test set is used once, at the very end).

    VCS_FINAL_ROUND=1 python scripts/final_official_test_eval.py --units slurm/final_official_test_units.txt --out reports/P68_FINAL_OFFICIAL_TEST
    python scripts/final_official_test_eval.py --units ... --out reports/P68_FINAL_OFFICIAL_TEST_standin --standin --limit 1   # smoke: selection split

Units file: one ``run_id checkpoint cell_label`` per line (cell_label without spaces; '#' comments allowed).  Every unit is evaluated with
``vcs_ssl.evaluate`` protocol ``final_official_test`` (pilot head/bank/transform, scored on the 10,000 official test images); a unit whose
result file already exists is read, not re-run (the protocol refuses to overwrite).  Output: per-run rows and per-cell mean ± sd of the
official-test linear / kNN accuracy next to the selection-split numbers the decisions were made on (from ``evaluations/evaluation_<tag>.json``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_ssl.evaluate import evaluate_run  # noqa: E402
from vcs_ssl.utils import atomic_write_json, read_json  # noqa: E402

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/home/infres/yinwang/CS_QMI/outputs"))


def parse_units(path: Path) -> list[tuple[str, str, str]]:
    units = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"units line needs 'run_id checkpoint cell_label': {line!r}")
        units.append((parts[0], parts[1], parts[2]))
    return units


def selection_numbers(run_dir: Path, tag: str) -> dict:
    p = run_dir / "evaluations" / f"evaluation_{tag}.json"
    if not p.is_file():
        return {"linear_val_top1_pct": None, "knn_val_top1_pct": None, "source": None}
    e = read_json(p)
    return {"linear_val_top1_pct": e.get("linear_val_top1_pct"), "knn_val_top1_pct": e.get("knn_val_top1_pct"), "source": p.name}


def msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0), len(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--standin", action="store_true", help="smoke only: selection split instead of the official test partition")
    ap.add_argument("--limit", type=int, default=None, help="smoke only: first N units")
    ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    units = parse_units(Path(a.units))
    if a.limit:
        units = units[: a.limit]
    if not a.standin and os.environ.get("VCS_FINAL_ROUND", "") != "1":
        print("refusing: the official run needs VCS_FINAL_ROUND=1 (set by slurm/final_official_test.sbatch); use --standin for smoke", file=sys.stderr)
        return 3
    rows = []; t0 = time.time()
    for run_id, ckpt, label in units:
        run_dir = OUTPUT_ROOT / run_id; tag = Path(ckpt).stem
        res_path = run_dir / "evaluations" / (f"final_standin_{tag}.json" if a.standin else f"final_official_test_{tag}.json")
        if not (run_dir / "checkpoints" / ckpt).is_file():
            rows.append({"run_id": run_id, "checkpoint": ckpt, "cell": label, "status": "MISSING_CHECKPOINT"}); print(f"[{run_id}] {ckpt}: MISSING", flush=True); continue
        if res_path.is_file() and not a.force:
            res = read_json(res_path); status = "read"
        else:
            res = evaluate_run(run_dir, ckpt, protocol="final_official_test", standin_selection=a.standin, force=a.force, num_workers=a.num_workers); status = "evaluated"
        key_l = "linear_val_top1_pct" if a.standin else "linear_test_top1_pct"; key_k = "knn_val_top1_pct" if a.standin else "knn_test_top1_pct"
        sel = selection_numbers(run_dir, tag)
        rows.append({"run_id": run_id, "checkpoint": ckpt, "cell": label, "status": status, "method": res.get("method"), "seed": res.get("seed"),
                     "test_linear": res[key_l], "test_knn": res[key_k], "sel_linear": sel["linear_val_top1_pct"], "sel_knn": sel["knn_val_top1_pct"],
                     "checkpoint_sha256": res.get("checkpoint_sha256"), "eval_split": res.get("eval_split"), "utc": res.get("utc")})
        print(f"[{run_id}] {ckpt} ({label}): {'STANDIN' if a.standin else 'TEST'} linear {res[key_l]:.2f} knn {res[key_k]:.2f} | selection linear {sel['linear_val_top1_pct']} knn {sel['knn_val_top1_pct']} [{status}] ({time.time() - t0:.0f}s)", flush=True)
    cells = {}
    for r in rows:
        if r["status"] == "MISSING_CHECKPOINT":
            continue
        c = cells.setdefault(r["cell"], {"test_linear": [], "test_knn": [], "sel_linear": [], "sel_knn": [], "runs": []})
        for k in ("test_linear", "test_knn", "sel_linear", "sel_knn"):
            c[k].append(r[k])
        c["runs"].append(r["run_id"])
    summary = {}
    for label, c in cells.items():
        summary[label] = {"n": len(c["runs"]), "runs": c["runs"]}
        for k in ("test_linear", "test_knn", "sel_linear", "sel_knn"):
            m, s, n = msd(c[k]); summary[label][k] = {"mean": m, "sd": s, "n": n}
    out = {"protocol": "final_official_test", "standin": a.standin, "units_file": a.units, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "VCS_FINAL_ROUND": os.environ.get("VCS_FINAL_ROUND"), "rows": rows, "cells": summary}
    atomic_write_json(Path(a.out + ".json"), out)
    what = "selection split (STAND-IN smoke — not the official test)" if a.standin else "OFFICIAL CIFAR-10 TEST partition (10 000 images), evaluated once"
    L = [f"# P68 — final evaluation on the {what} — {out['utc']}", "",
         "Head and kNN bank: the 45 000-image fit split, frozen pilot probe hyper-parameters, clean transform; the checkpoint of each run is the one every",
         "decision was made on.  Selection-split numbers (the development metric) are shown next to the test numbers; single-seed cells are marked in the label.", "",
         "## Per cell (mean ± sd over seeds; n)", "", "| cell | n | test linear % | test kNN % | selection linear % | selection kNN % | runs |", "|---|---|---|---|---|---|---|"]
    def f(d):
        return "—" if d["mean"] is None else (f"{d['mean']:.2f} ± {d['sd']:.2f}" if d["n"] > 1 else f"{d['mean']:.2f}")
    for label, c in summary.items():
        L.append(f"| {label} | {c['n']} | {f(c['test_linear'])} | {f(c['test_knn'])} | {f(c['sel_linear'])} | {f(c['sel_knn'])} | {', '.join(c['runs'])} |")
    L += ["", "## Per run", "", "| run | checkpoint | cell | test linear % | test kNN % | selection linear % | selection kNN % | status |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] == "MISSING_CHECKPOINT":
            L.append(f"| {r['run_id']} | {r['checkpoint']} | {r['cell']} | — | — | — | — | MISSING_CHECKPOINT |"); continue
        L.append(f"| {r['run_id']} | {r['checkpoint']} | {r['cell']} | {r['test_linear']:.2f} | {r['test_knn']:.2f} | {r['sel_linear']} | {r['sel_knn']} | {r['status']} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
