"""V9-G2 CLI (src/v9/reg/g2.py): the V9-G1 extension on EPFL RGB–NIR.  Outputs under $V9_REG_EPFL_ROOT (default
/projects/EEG-foundation-model/yinghao/outputs_v9/reg_epfl); the r6 REG-P3 root is read-only.

    python scripts/v9/reg_g2.py units [--csv configs/V9_REG_EPFL_UNITS.csv]      write the unit table (status, reuse decision, realised FIT units)
    python scripts/v9/reg_g2.py run --unit <id> [--smoke]                         train (or CoMIR / kernel fit) + REG-P3 registration for one unit; skips finished steps
    python scripts/v9/reg_g2.py reuse-check                                       MI / NMI reuse check (hashes of REG-P3's perturbations and manifest)
    python scripts/v9/reg_g2.py aggregate [--out reports/V9/V9_G2_results]        readings (src/v9/reg/g2_aggregate.py)
Smoke units live under <root>/smoke/units (a REG-P3-shaped smoke root is built once at <root>/smoke/p3root from the existing cache: 4 FIT windows,
2 CAL, 2 further CAL windows relabelled DEV — no DEV window is read); a formal run refuses a run dir made with other steps / epochs or holding
checkpoints without train.json (no silent mixing of a smoke or an interrupted run with a formal one).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v9.reg import g2 as G  # noqa: E402


def parse_unit(uid):
    for u, m, f, e, s in G.all_units() + G.kernel_units():
        if u == uid:
            return m, f, e, s
    raise SystemExit(f"unknown unit {uid}")


def cmd_run(a):
    method, frac, eps, seed = parse_unit(a.unit); p3_root = G.P3_ROOT
    if a.smoke:                                                                   # smoke units and the smoke REG-P3 root live under <root>/smoke
        G.V9_ROOT = G.V9_ROOT / "smoke"; p3_root = G.make_smoke_root(G.V9_ROOT / "p3root")
    u = G.ensure_layout(a.unit, method, seed, p3_root, a.smoke)
    res = G.result_path(a.unit, method, seed, a.smoke)
    if res.exists() and not a.force:
        print(f"[skip] {a.unit}: {res} exists"); return 0
    run = u / "run"
    if method == "comir":
        meta = run / "comir.json"; want = {"epochs": (2 if a.smoke else (a.comir_epochs or 23))}
        if meta.exists() and not a.force:
            have = {"epochs": json.load(open(meta)).get("epochs")}
            if have != want:
                raise SystemExit(f"{a.unit}: existing run has {have}, this call wants {want} — refusing to mix (use --force after moving the old run away)")
        else:
            G.cmd_comir(a.unit, frac, eps, seed, smoke=a.smoke, p3_root=p3_root, comir_epochs=a.comir_epochs)
    elif method != "kernel":
        meta = run / "train.json"; want = {"steps": 200 if a.smoke else a.steps}
        if meta.exists() and not a.force:
            have = {"steps": json.load(open(meta)).get("steps")}
            if have != want:
                raise SystemExit(f"{a.unit}: existing run has {have}, this call wants {want} — refusing to mix (use --force after moving the old run away)")
        else:
            if any(run.glob("ckpt_*.pt")):
                raise SystemExit(f"{a.unit}: checkpoints without train.json in {run} (interrupted run) — move them away before training")
            G.cmd_train(a.unit, method, frac, eps, seed, steps=a.steps, smoke=a.smoke, p3_root=p3_root)
    rc = G.cmd_register(a.unit, method, seed, smoke=a.smoke, n_pert=a.n_pert, frac=frac, eps=eps, p3_root=p3_root)
    print(f"[done] {a.unit} -> {res}"); return rc


def cmd_units(a):
    rows = G.unit_table(); a.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_new = sum(1 for r in rows if not r["reuse"].startswith("r6:")); print(f"{len(rows)} units ({n_new} new, {len(rows) - n_new} reusable) -> {a.csv}"); return 0


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--unit", required=True); r.add_argument("--smoke", action="store_true"); r.add_argument("--force", action="store_true")
    r.add_argument("--steps", type=int, default=3000); r.add_argument("--n-pert", type=int, default=G.P.N_PERT); r.add_argument("--comir-epochs", type=int, default=0)
    u = sub.add_parser("units"); u.add_argument("--csv", type=Path, default=REPO / "configs" / "V9_REG_EPFL_UNITS.csv")
    sub.add_parser("reuse-check")
    g = sub.add_parser("aggregate"); g.add_argument("--out", type=Path, default=REPO / "reports" / "V9" / "V9_G2_results"); g.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "units":
        return cmd_units(a)
    if a.cmd == "reuse-check":
        print(json.dumps(G.mi_nmi_reuse_check(), indent=1)); return 0
    from v9.reg import g2_aggregate as A  # noqa: E402
    if a.smoke:
        G.V9_ROOT = G.V9_ROOT / "smoke"
    return A.main(a.out, a.smoke)


if __name__ == "__main__":
    sys.exit(main())
