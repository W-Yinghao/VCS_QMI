#!/usr/bin/env python3
"""EPFL FIT-only functional gate of RuLSIF_exact_v2 under the amended level-A rule (amendment A1), on the EPFL FIT windows.  Slurm only.

Rule (stated in v10/epfl/EPFL_RULSIF_V2_ADDENDUM.json, carried over by revisions 2 and 3, before this gate runs):
  sample      (FIT window, perturbation i), i = 0..9, for all 334 FIT windows of the frozen group record v2 (3340 samples per candidate):
              positions Q.RigidEnergy draw with seed 'R6-REG-P2|pos|<window>|<i>'; reference = RGB luminance, moving = REG-P3 Inputs3
              (linear NIR, robust-normalised) at the zero perturbation; imposed perturbation Q.perturbations_p2(window)[i] (FIT rule read)
  model       the deployed fit of each candidate: seed 1017 (the CAL seed), all 334 FIT windows, EPFL stream tag, revision-3 centres
              (T = 1024 primary or T = 2048 pre-registered capacity sensitivity, --centres), i.e. exactly the CAL fit of that T
  statistics  F_true (identity pose), F_imposed (imposed perturbation); F_local (six near-truth poses) and the truth split: diagnostic
  verdict     level A of A1, unchanged: mean F_true > 0, F_true > 0 for >= 75% of samples, F_true > F_imposed for >= 75% of samples
  candidates  the 8 A1-eligible candidates (orders 0-7) in frozen order; a candidate is EPFL-eligible iff it passes; ineligible ones are
              excluded (not re-tuned, not replaced); the EPFL family gate passes iff at least one candidate is eligible
  level B     (4-fold cross-fit) is a diagnostic under A1 and is not run on EPFL
Reads FIT windows only (DEV refused by the GATE_CHECKS record; CAL never touched).  Per-candidate results are written as they complete and
reused on a re-run of the same commit and pins.  --smoke --device cpu: smoke roles (4 FIT windows), 2 candidates; never a gate result.

  python -I scripts/v10_epfl/rulsif_v2_fit_gate.py --centres 1024|2048 --device cuda|cpu
Each T has its own job, out-root (<V10_EPFL_OUT>/rulsif_v2_fit_gate/T<T>/<job>) and eligible set; per-candidate centre-provenance sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.epfl import runtime as RT  # noqa: E402

FAMILY, SEED, LEVEL = "RuLSIF_exact_v2", 1017, "deployed_position_heldout"
ADDENDUM = REPO / "v10" / "epfl" / "EPFL_RULSIF_V2_ADDENDUM_REV3.json"


def fingerprint(model):
    w = model.coefficients.detach().double().cpu()
    return {"bandwidth_left": model.bandwidth_left, "bandwidth_right": model.bandwidth_right, "coefficients": len(w),
            "coefficient_sum": float(w.sum()), "coefficient_l2": float(w.norm()), "constant": float(w[0]),
            "centres_sha256": hashlib.sha256(model.centres_left.detach().cpu().numpy().tobytes()
                                             + model.centres_right.detach().cpu().numpy().tobytes()).hexdigest()}


def family_status(entries):
    """EPFL rule: eligible = candidates passing level A, in frozen order; the family passes iff at least one is eligible."""
    eligible = [e["candidate"]["order"] for e in entries if e["status"] == "PASS"]
    return {"status": "PASS" if eligible else "FAIL", "eligible_orders": eligible,
            "excluded_orders": [e["candidate"]["order"] for e in entries if e["status"] != "PASS"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--centres", type=int, required=True, choices=(1024, 2048))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); dev = a.device
    pins = RT.job_pins()
    RT.require(pins.get("group_record_sha256"), "the v2 jobs need the group-record pin V10_EPFL_GROUP_SHA256")
    started = time.time()
    import torch
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    from v10 import adapter as A
    from v10.epfl import host as H, production as PR
    from v10.supplement import rulsif_gate as G
    V2, C = H.rulsif_v2_module()
    addendum = json.loads(ADDENDUM.read_text())
    RT.require(addendum.get("status") == "FROZEN_BEFORE_EPFL_V2_GATES" and addendum.get("revision") == 3, "the EPFL v2 addendum (revision 3) must be frozen before the gate")
    T = a.centres; RT.require(T in H.RULSIF_V2_CENTRES, "total centres must be 1024 (primary) or 2048 (capacity sensitivity)")
    amendment = C.rulsif_v2_amendment()
    RT.require(amendment is not None, "amendment A1 missing")
    thresholds = amendment["amended_rule"]["thresholds"]
    RT.require(thresholds == addendum["epfl_fit_gate"]["thresholds"], "addendum thresholds differ from A1")
    base = RT.guard_from_pins(pins)
    RT.require(base.stage == "GATE_CHECKS", "the v2 FIT gate runs at stage GATE_CHECKS (DEV refused)")
    if a.smoke:
        spec = importlib.util.spec_from_file_location("epfl_run", REPO / "scripts" / "v10_epfl" / "run.py")
        run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run); guard = run.SmokeGuard(base)
    else:
        guard = base
    fit = H.fit_windows(guard)
    cands = C.candidates(FAMILY)[:2] if a.smoke else C.candidates(FAMILY)
    RT.require(a.smoke or [c["order"] for c in cands] == amendment["amended_rule"]["eligible_orders"], "candidates differ from A1")
    root = RT.OUT_ROOT / ("smoke_cpu/rulsif_v2_fit_gate" if a.smoke else "rulsif_v2_fit_gate") / f"T{T}" / pins["job"]; root.mkdir(parents=True, exist_ok=True)
    inputs = H.EPFLGuardedInputs(guard, {"FIT"}, "V10 EPFL RuLSIF_exact_v2 functional gate (FIT only)", {w: [[0.0, 0.0, 0.0]] for w in fit}, dev)
    key = {"commit": pins["commit"], "record_sha256": pins["record_sha256"], "group_record_sha256": pins["group_record_sha256"],
           "device": dev, "smoke": a.smoke, "seed": SEED, "fit_windows": fit, "centres_total": T}
    entries = []
    for cand in cands:
        part = root / "candidates" / f"order_{cand['order']}.json"
        if part.exists():
            prev = json.loads(part.read_text())
            if prev.get("key") == key and prev.get("candidate") == cand:
                entries.append(prev); print(f"[r2fit] order {cand['order']} reused {prev['status']}", flush=True); continue
        t0 = time.time()
        trainer = H.new_trainer(FAMILY, cand, SEED, guard, dev, centres_total=T)
        RT.require(trainer.fit_metadata["stream_tag"] == H.rulsif_v2_stream_tag(SEED), "not the EPFL CAL-seed stream")
        H.verify_centre_provenance(trainer.provenance, trainer.fit_metadata, T)
        side = root / "provenance" / f"order_{cand['order']}.json"; RT.write(side, trainer.provenance)
        info = trainer.step(); fit_seconds = time.time() - t0
        rows = G.evaluate(trainer.ratio, V2.RuLSIFV2RigidEnergy, inputs, guard, fit, shg_input=cand["shg_input"])
        rows_path = root / "rows" / f"order_{cand['order']}.jsonl"; rows_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = rows_path.with_name("." + rows_path.name + ".tmp")
        tmp.write_text("".join(json.dumps({"family": FAMILY, "order": cand["order"], "level": LEVEL, **r}) + "\n" for r in rows)); os.replace(tmp, rows_path)
        summary = G.summarize(rows); verdict = G.verdict(summary, thresholds)
        entry = {"key": key, "candidate": cand, "status": verdict["status"], LEVEL: {"summary": summary, "verdict": verdict},
                 "deployed_fit": fingerprint(trainer.ratio), "fit_info": {k: info[k] for k in ("scene_count", "coefficient_count", "linear_solve_residual",
                                                                                            "mixture_mean_of_g", "negative_coefficients", "J_fit_moments")},
                 "rows": str(rows_path), "rows_sha256": PR.sha(rows_path), "fit_seconds": fit_seconds, "seconds": time.time() - t0,
                 "provenance_sidecar": str(side), "provenance_sidecar_sha256": PR.sha(side),
                 "provenance_summary": {k: trainer.provenance[k] for k in ("cx_sha256", "cy_sha256", "indices_sha256", "fit_patches_sha256",
                                                                           "extra_windows", "medians", "median_pairs")}}
        RT.write(part, entry); entries.append(entry)
        s = summary
        print(f"[r2fit] order {cand['order']} {entry['status']} meanJ {s['mean_F_true']:.4f} J>0 {s['frac_F_true_positive']:.3f} "
              f"true>imposed {s['frac_true_beats_imposed']:.3f} ({entry['seconds']:.0f}s)", flush=True)
        del trainer
        if dev == "cuda":
            torch.cuda.empty_cache()
    fam = family_status(entries)
    roles = sorted({r["role"] for r in base.reads})
    inv = PR.source_inventory()
    report = {"schema": "v10.epfl.rulsif_v2_fit_gate.v1", "stage": "EPFL_FIT_FUNCTIONAL_GATE", "family": FAMILY, "centres_total": T,
              "kernel_role": H.RULSIF_V2_CENTRES[T],
              "status": (("SMOKE_" + fam["status"] + "_NOT_A_GATE_RESULT") if a.smoke else fam["status"]) if roles == ["FIT"] else "FAIL",
              "pins": pins, "device": dev, "utc": PR.utc(), "seed": SEED, "level": LEVEL, "thresholds": thresholds,
              "rule": addendum["epfl_fit_gate"]["rule"], "candidates_evaluated": [e["candidate"] for e in entries],
              "family_gate": fam["status"], "eligible_orders": fam["eligible_orders"], "excluded_orders": fam["excluded_orders"],
              "candidates": [{k: v for k, v in e.items() if k != "key"} for e in entries],
              "levels_not_run": {"crossfit4_core_heldout": "diagnostic only under A1; not run on EPFL"},
              "addendum_sha256": PR.sha(ADDENDUM), "amendment_sha256": PR.sha(C.RULSIF_V2_AMENDMENT), "preregistration_sha256": PR.sha(C.RULSIF_V2_RECORD),
              "source_files": inv, "code_sha256": PR.digest(inv), "roles_read": roles, "dev_read": False, "cal_read": "CAL" in roles,
              "held_test_read": False, "fit_reads": sum(r.get("kind") != "rule" for r in base.reads), "n_fit_windows": len(fit),
              "samples_per_candidate": len(fit) * 10, "seconds": time.time() - started}
    RT.write(root / "FIT_GATE.json", report)
    print(f"[r2fit] {report['status']} eligible {fam['eligible_orders']} excluded {fam['excluded_orders']} -> {root / 'FIT_GATE.json'}", flush=True)
    return 0 if report["status"] in ("PASS", "SMOKE_PASS_NOT_A_GATE_RESULT") else 3


if __name__ == "__main__":
    sys.exit(main())
