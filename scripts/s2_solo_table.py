"""P76 harvest for S2 / P75 (solo-learn CIFAR-10 protocol, VCS vs SimCLR, 1000 epochs, 3 seeds).

Reads `outputs/P75_S2_solo/<method>_seed<s>/final_metrics.json` (written once at the end of epoch 1000) and the Lightning CSV logs,
applies the pre-committed reading of `reports/P75_S2_SOLO_LEARN_PREREG_FROZEN_20260928.md` verbatim (gap = mean SimCLR − mean VCS of the
online Acc@1 at epoch 1000: holds < 1.5, refuted ≥ 1.5, "at the threshold" if |gap − 1.5| < 2 × pooled SE; QC: SimCLR mean within 1.0 of
90.74), and records the QC sentinels (1000 epochs completed, `source` of final_metrics, VCS critic scalars a / b, J, saturation trajectory).

    python scripts/s2_solo_table.py --out reports/P76_s2_solo        # writes P76_s2_solo.{md,json}
Results-only: no interpretation beyond the pre-committed rule.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import time
from pathlib import Path

OUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/P75_S2_solo")
METHODS = ("vcs", "simclr")
SEEDS = (0, 1, 2)
THRESHOLD = 1.5
PROTOCOL_REF = 90.74  # solo-learn README, SimCLR ResNet-18 CIFAR-10 1000 ep, at 9187ea3
EPOCH_MARKS = (0, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)


def csv_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for fn in sorted(glob.glob(str(run_dir / "lightning_logs" / "version_*" / "metrics.csv"))):
        with open(fn) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def per_epoch(rows: list[dict], key: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in rows:
        v = r.get(key, "")
        if v not in ("", None):
            out[int(r["epoch"])] = float(v)
    return out


def summarise_run(method: str, seed: int) -> dict:
    d = OUT_ROOT / f"{method}_seed{seed}"
    rec: dict = {"method": method, "seed": seed, "dir": str(d), "final": None, "status": "MISSING"}
    fm = d / "final_metrics.json"
    if fm.is_file():
        rec["final"] = json.load(open(fm))
        rec["status"] = "FINAL"
    rows = csv_rows(d)
    if rows:
        acc = per_epoch(rows, "train_acc1_epoch")
        rec["last_epoch_logged"] = max(acc) if acc else None
        rec["online_train_acc1_marks"] = {str(e): round(acc[e], 2) for e in EPOCH_MARKS if e in acc}
        if rec["status"] != "FINAL":
            rec["status"] = "RUNNING_OR_STOPPED"
        if method == "vcs":
            traj = {}
            for key, name in (("critic_a_epoch", "a"), ("critic_b_epoch", "b"), ("train_vcs_J_epoch", "J"), ("train_vcs_t_pos_epoch", "t_pos"),
                              ("train_vcs_t_neg_epoch", "t_neg"), ("train_vcs_sat_pos_epoch", "sat_pos"), ("train_vcs_sat_neg_epoch", "sat_neg")):
                pe = per_epoch(rows, key)
                traj[name] = {str(e): round(pe[e], 4) for e in EPOCH_MARKS if e in pe}
            rec["vcs_trajectory"] = traj
    return rec


def final_acc(rec: dict) -> float | None:
    f = rec.get("final")
    if not f:
        return None
    m = f.get("metrics", f)  # main_pretrain._s2_final_metrics nests the Lightning metrics under "metrics"
    for k in ("val_acc1", "test_acc1", "acc1"):
        if k in m:
            return float(m[k])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/P76_s2_solo")
    a = ap.parse_args()
    runs = [summarise_run(m, s) for m in METHODS for s in SEEDS]
    by = {m: [final_acc(r) for r in runs if r["method"] == m] for m in METHODS}
    complete = {m: [x for x in by[m] if x is not None] for m in METHODS}
    res: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prereg": "reports/P75_S2_SOLO_LEARN_PREREG_FROZEN_20260928.md",
                 "threshold_gap": THRESHOLD, "protocol_reference_simclr": PROTOCOL_REF, "runs": runs, "n_complete": {m: len(complete[m]) for m in METHODS}}
    verdict = "PENDING (not all six units have final_metrics.json)"
    if all(len(complete[m]) == 3 for m in METHODS):
        mean = {m: statistics.mean(complete[m]) for m in METHODS}
        sd = {m: statistics.stdev(complete[m]) for m in METHODS}
        gap = mean["simclr"] - mean["vcs"]
        pooled_se = math.sqrt(sd["simclr"] ** 2 / 3 + sd["vcs"] ** 2 / 3)
        if abs(gap - THRESHOLD) < 2 * pooled_se:
            verdict = f"AT THE THRESHOLD (gap {gap:.2f}, |gap − 1.5| < 2 × pooled SE {pooled_se:.2f})"
        elif gap < THRESHOLD:
            verdict = f"HOLDS (gap {gap:.2f} < 1.5)"
        else:
            verdict = f"REFUTED (gap {gap:.2f} ≥ 1.5)"
        qc = abs(mean["simclr"] - PROTOCOL_REF) <= 1.0
        res.update({"mean": mean, "sd": sd, "gap": gap, "pooled_se": pooled_se, "qc_protocol_reproduced": qc,
                    "qc_note": None if qc else f"SimCLR mean {mean['simclr']:.2f} is more than 1.0 from the README value {PROTOCOL_REF}: protocol not reproduced; gap reported with this caveat"})
    res["verdict"] = verdict
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    L = [f"# P76 — S2 solo-learn protocol results (results only; {res['utc']})", "",
         "Pre-committed reading (P75, frozen): gap = mean SimCLR − mean VCS of the online linear Acc@1 on the official CIFAR-10 test split at the end of epoch 1000; "
         "holds < 1.5, refuted ≥ 1.5, 'at the threshold' if |gap − 1.5| < 2 × pooled SE. QC: SimCLR 3-seed mean within 1.0 of 90.74.", "",
         "| unit | status | epochs logged | final Acc@1 | final Acc@5 | final source |", "|---|---|---|---|---|---|"]
    for r in runs:
        f = r.get("final") or {}
        fm = f.get("metrics", f)
        L.append(f"| {r['method']}_seed{r['seed']} | {r['status']} | {r.get('last_epoch_logged', '—')} | {fm.get('val_acc1', '—')} | {fm.get('val_acc5', '—')} | {f.get('source', '—')} |")
    L += ["", f"**Verdict:** {verdict}"]
    if "mean" in res:
        L += ["", f"SimCLR {res['mean']['simclr']:.2f} ± {res['sd']['simclr']:.2f}, VCS {res['mean']['vcs']:.2f} ± {res['sd']['vcs']:.2f}; gap {res['gap']:.2f}, pooled SE {res['pooled_se']:.2f}; "
              f"QC protocol reproduced: {res['qc_protocol_reproduced']}" + (f" ({res['qc_note']})" if res.get("qc_note") else "")]
    L += ["", "## Online train Acc@1 at epoch marks (Lightning CSV; training-set online classifier, not the test metric)", ""]
    marks = [str(e) for e in EPOCH_MARKS]
    L += ["| unit | " + " | ".join(marks) + " |", "|---|" + "---|" * len(marks)]
    for r in runs:
        m = r.get("online_train_acc1_marks", {})
        L.append(f"| {r['method']}_seed{r['seed']} | " + " | ".join(str(m.get(e, "")) for e in marks) + " |")
    L += ["", "## VCS critic scalars and objective (QC sentinel of P75: a, b logged per step; LARS-excluded plain SGD-momentum at the scheduled lr)", ""]
    for r in runs:
        if r["method"] != "vcs" or "vcs_trajectory" not in r:
            continue
        t = r["vcs_trajectory"]
        L.append(f"- {r['method']}_seed{r['seed']}: " + "; ".join(
            f"ep{e}: a {t['a'].get(e, '—')}, b {t['b'].get(e, '—')}, J {t['J'].get(e, '—')}, t± {t['t_pos'].get(e, '—')}/{t['t_neg'].get(e, '—')}, sat± {t['sat_pos'].get(e, '—')}/{t['sat_neg'].get(e, '—')}"
            for e in marks if e in t["a"]))
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print("...")
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
