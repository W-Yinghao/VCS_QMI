"""V9-G1 CLI (src/v9/reg/g1.py).  Outputs under $V9_REG_ROOT (default /projects/EEG-foundation-model/yinghao/outputs_v9/reg); r6 roots read-only.

    python scripts/v9/reg_g1.py units [--csv configs/V9_REG_UNITS.csv]           write the unit table (status, reuse decision, realised FIT counts)
    python scripts/v9/reg_g1.py run --unit <id> [--smoke]                          train (or CoMIR / kernel fit) + REG-P2 registration for one unit; skips finished steps
    python scripts/v9/reg_g1.py reuse-check                                        MI / NMI reuse check (hashes of the perturbations and cache manifest)
    python scripts/v9/reg_g1.py aggregate [--out reports/V9/V9_G1_results]         readings (src/v9/reg/g1_aggregate.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v9.reg import g1 as G  # noqa: E402


def parse_unit(uid):
    for u, m, f, e, s in G.all_units() + G.kernel_units():
        if u == uid:
            return m, f, e, s
    raise SystemExit(f"unknown unit {uid}")


def cmd_run(a):
    method, frac, eps, seed = parse_unit(a.unit); u = G.ensure_layout(a.unit, method, seed)
    if a.smoke:                                                                   # smoke roots: REG-P1's smoke cache / manifest, REG-P2 perturbations generated into the unit's smoke root
        p1s, p2s = u / "p1" / "smoke", u / "p2" / "smoke"; (p1s / "runs").mkdir(parents=True, exist_ok=True); p2s.mkdir(parents=True, exist_ok=True)
        for link, target in ((p1s / "cache", G.P1_ROOT / "smoke" / "cache"), (p1s / "manifest_smoke.json", G.P1_ROOT / "smoke" / "manifest_smoke.json"), (p1s / "runs" / f"{method}_s{seed}", u / "run")):
            if not link.is_symlink() and not link.exists():
                link.symlink_to(target)
        if not (p2s / "perturbations_smoke.json").exists():
            G.Q.cmd_perturb(G._Args(root=p2s, p1_root=p1s, smoke=True))
    res = G.result_path(a.unit, method, seed, a.smoke)
    if res.exists() and not a.force:
        print(f"[skip] {a.unit}: {res} exists"); return 0
    p1 = (u / "p1" / "smoke") if a.smoke else (u / "p1"); run = u / "run"
    if method == "comir":
        if not (run / "comir.json").exists() or a.force:
            G.cmd_comir(a.unit, frac, eps, seed, smoke=a.smoke, p1_root=p1, comir_epochs=a.comir_epochs)
    elif method != "kernel":
        if not (run / "train.json").exists() or a.force:
            G.cmd_train(a.unit, method, frac, eps, seed, steps=a.steps, smoke=a.smoke, p1_root=p1)
    rc = G.cmd_register(a.unit, method, seed, smoke=a.smoke, n_pert=a.n_pert, frac=frac, eps=eps)
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
    u = sub.add_parser("units"); u.add_argument("--csv", type=Path, default=REPO / "configs" / "V9_REG_UNITS.csv")
    sub.add_parser("reuse-check")
    g = sub.add_parser("aggregate"); g.add_argument("--out", type=Path, default=REPO / "reports" / "V9" / "V9_G1_results"); g.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "units":
        return cmd_units(a)
    if a.cmd == "reuse-check":
        print(json.dumps(G.mi_nmi_reuse_check(), indent=1)); return 0
    from v9.reg import g1_aggregate as A  # noqa: E402
    return A.main(a.out, a.smoke)


if __name__ == "__main__":
    sys.exit(main())
