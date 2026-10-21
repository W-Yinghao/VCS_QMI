#!/usr/bin/env python3
"""Phase A: build the explicit `--extra-records` lists for the package exporter (v9/evidence/export_evidence.py).

Two channels, written as two JSON files (never mixed):
  * evidence  -> N2F production weights + locks (BCE / RuLSIF / RankMass / VCS_Weight), observed tables, noise manifests, roles,
                 teacher logs, and the baseline health / timing / smoke logs that BASELINE_HEALTH_STATUS.json and
                 HISTORICAL_TEST_ACCESS_AUDIT.json cite.  No mechanism labels.
  * eval_only -> the three EVAL_ONLY label files only (kind eval_only_labels), for the separate evaluator-only archive.

Every path is an existing server file named here explicitly (no globbing of unknown locations).  Each N2F production input must
match the hash frozen in the N2M package provenance (N2F_FROZEN.json, frozen 2026-10-17 before N2M); a mismatch aborts.
Reads bytes only (sha256 + size); never parses label values.

    python -I make_extra_records.py --n2f-frozen <N2F_FROZEN.json> --out-evidence <json> --out-eval-only <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

N2F = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4")
N2 = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/n2")
TIMING = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/timing")
LOGS = Path("/home/infres/yinwang/CS_QMI/slurm_logs")
SEEDS = (901, 902, 903)


def ident(path: Path, kind: str, note: str) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "kind": kind, "note": note}


def evidence_records(frozen: dict) -> list[dict]:
    rows = []
    for s in SEEDS:
        d = N2F / f"s{s}"
        for m in ("VCS_Weight", "BCE_Weight", "RuLSIF_Weight", "RankMass"):
            for suffix in (".csv", ".csv.lock.json"):
                p = d / "weights" / f"{m}{suffix}"
                r = ident(p, "weights", f"N2F production {m} seed {s} (frozen in N2F_FROZEN.json)")
                if frozen.get(str(p)) != r["sha256"]:
                    raise SystemExit(f"N2F frozen hash mismatch or missing: {p}")
                rows.append(r)
        for name, kind in (("pairs.observed_pairs.json", "observed"), ("pairs.noise_manifest.json", "observed"), ("roles.json", "observed")):
            p = d / name
            r = ident(p, kind, f"N2F shared stage seed {s}")
            if str(p) in frozen and frozen[str(p)] != r["sha256"]:
                raise SystemExit(f"N2F frozen hash mismatch: {p}")
            rows.append(r)
        rows.append(ident(d / "teacher" / "teacher_log.json", "analysis", f"frozen SGR teacher training log seed {s} (checkpoint not exported)"))
    health = [
        (N2 / "ncr_health_1043807" / "ncr_health.json", "NCR_A health 1043807 verdict"),
        (N2 / "ncr_health_1043807" / "launch_manifest.json", "NCR_A health 1043807 launch manifest"),
        (N2 / "ncr_health_1043807" / "official_train.log", "NCR_A health 1043807 trainer log"),
        (N2 / "ncr_health_1043807" / "v9_export_manifest.json", "NCR_A health 1043807 export manifest"),
        (N2 / "ncr_health_1043807" / "official_out" / "config.json", "NCR_A health 1043807 trainer config"),
        (N2 / "ncr_health_1043807" / "code" / "ncr" / "V9_PATCH.json", "NCR_A health 1043807 patch record"),
        (LOGS / "v9n2_ncrhealth_1043807.out", "NCR_A health 1043807 Slurm stdout"),
        (N2 / "official_smoke_1043800" / "smoke_report.json", "smoke rerun 1043800 report"),
        (N2 / "official_smoke_1043800" / "smoke_ncr.log", "smoke rerun 1043800 NCR driver log"),
        (N2 / "official_smoke_1043800" / "smoke_l2rm.log", "smoke rerun 1043800 L2RM driver log"),
        (N2 / "official_smoke_1043800" / "unit_ncr" / "launch_manifest.json", "smoke 1043800 NCR launch manifest"),
        (N2 / "official_smoke_1043800" / "unit_ncr" / "official_train.log", "smoke 1043800 NCR trainer log"),
        (N2 / "official_smoke_1043800" / "unit_l2rm" / "launch_manifest.json", "smoke 1043800 L2RM launch manifest"),
        (N2 / "official_smoke_1043800" / "unit_l2rm" / "official_train.log", "smoke 1043800 L2RM trainer log"),
        (N2 / "official_smoke_1043800" / "unit_l2rm" / "official_eval.log", "smoke 1043800 L2RM collect (TEST subset) log"),
        (N2 / "official_smoke_1043800" / "unit_l2rm" / "result.json", "smoke 1043800 L2RM collect result (contains TEST-subset recalls)"),
        (N2 / "official_smoke_1043800" / "unit_l2rm" / "done.json", "smoke 1043800 L2RM done record"),
        (LOGS / "v9_n2_offsmoke_1043800.out", "smoke rerun 1043800 Slurm stdout"),
        (N2 / "official_smoke_1042220" / "smoke_report.json", "smoke 1042220 report"),
        (N2 / "official_smoke_1042220" / "smoke_ncr.log", "smoke 1042220 NCR driver log"),
        (N2 / "official_smoke_1042220" / "smoke_l2rm.log", "smoke 1042220 L2RM driver log"),
        (N2 / "official_smoke_1042220" / "unit_ncr" / "official_train.log", "smoke 1042220 NCR trainer log"),
        (N2 / "official_smoke_1042220" / "unit_l2rm" / "official_train.log", "smoke 1042220 L2RM trainer log"),
        (LOGS / "v9_n2_offsmoke_1042220.out", "smoke 1042220 Slurm stdout"),
        (TIMING / "group_1042033" / "l2rm_train.log", "timing 1042033 L2RM log (native end-of-training TEST attempt)"),
        (TIMING / "group_1042033" / "ncr_train.log", "timing 1042033 NCR log"),
        (TIMING / "group_1042033" / "sgr_test_eval.log", "timing 1042033 SGR TEST evalrank attempt"),
        (LOGS / "v9_timing_group_1042033.out", "timing 1042033 Slurm stdout"),
        (TIMING / "ncr_cotrain_1042195" / "ncr_cotrain.log", "timing 1042195 NCR log (native end-of-training TEST read)"),
        (TIMING / "ncr_cotrain_1042195" / "config.json", "timing 1042195 NCR config"),
        (LOGS / "v9_timing_ncr_cotrain_1042195.out", "timing 1042195 Slurm stdout"),
    ]
    rows += [ident(p, "baseline_health", n) for p, n in health]
    return rows


def eval_only_records(frozen: dict) -> list[dict]:
    rows = []
    for s in SEEDS:
        p = N2F / f"s{s}" / "eval_only" / "pairs.eval_only.json"
        r = ident(p, "eval_only_labels", f"EVAL_ONLY mechanism labels seed {s}; evaluator channel only")
        if frozen.get(str(p)) != r["sha256"]:
            raise SystemExit(f"N2F frozen hash mismatch or missing: {p}")
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n2f-frozen", required=True)
    ap.add_argument("--out-evidence", required=True)
    ap.add_argument("--out-eval-only", required=True)
    a = ap.parse_args()
    frozen = json.loads(Path(a.n2f_frozen).read_text())["input_sha256"]
    for path, rows in ((a.out_evidence, evidence_records(frozen)), (a.out_eval_only, eval_only_records(frozen))):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as fh:            # never overwrite
            json.dump(rows, fh, indent=1)
        print(json.dumps({"out": str(out), "records": len(rows), "bytes": sum(r["bytes"] for r in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
