#!/usr/bin/env python3
"""Read-only V10 handoff audit; stdlib only, no imports of training code.

Checks exported artifacts, their cross-references, CAL selections and descriptive
aggregate arithmetic. Does not train, deserialize checkpoints, read natural TEST
images, or certify the mathematical/experimental validity of recorded results.
Truth-dependent derived tables stay in handoff/evaluator_only/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tracked_hash(repo, paths):
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=repo)
    names = sorted(set(p for p in raw.split(b"\0") if p))
    h = hashlib.sha256()
    for name in names:
        h.update(sha(repo / name.decode()).encode() + b"  " + name + b"\n")
    return h.hexdigest()[:16]


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-14)


def basic_stats(values):
    return {"R": len(values), "mean": statistics.mean(values),
            "rmse": math.sqrt(statistics.mean(x * x for x in values)),
            "sd": statistics.stdev(values), "var": statistics.variance(values)}


def best_checkpoint(checkpoints):
    return min(checkpoints, key=lambda c: (-c["CAL_UJ"], c["step"]))


def best_candidate(ids, records):
    candidates = []
    for ordinal, uid in enumerate(ids):
        r = records[uid]
        for c in r["checkpoints"]:
            candidates.append(((-c["CAL_UJ"], c["step"], ordinal), uid, r, c))
    _, uid, r, c = min(candidates, key=lambda x: x[0])
    return {"unit_id": uid, "lr": r["config"]["lr"], "cap": r["config"]["cap"],
            "step": c["step"], "CAL_UJ": c["CAL_UJ"]}


class Audit:
    def __init__(self):
        self.counts = collections.Counter()
        self.errors = []

    def check(self, condition, category, context):
        self.counts[category] += 1
        if not condition:
            self.errors.append({"category": category, "context": context})

    def file(self, path, expected, category):
        exists = path.is_file()
        self.check(exists, category, f"missing: {path}")
        if exists:
            self.check(path.stat().st_size == expected["bytes"], category, f"size: {path}")
            self.check(sha(path) == expected["sha256"], category, f"sha256: {path}")


def run(root, dest, external=False):
    audit = Audit()
    repo = root / "ssl_pilot_v10_t03"
    bf = root / "ssl_pilot_v10"
    t03 = repo / "v10/T03"
    ledger_path = repo / "v10/protocol/T03_UNITS.csv"
    with ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    ledger = {r["unit_id"]: r for r in rows}
    audit.check(len(rows) == len(ledger) == 1219, "ledger", "unique 1219 unit IDs")
    code_paths = ["src/v10", "scripts/v10_t03"]
    pin_paths = code_paths + ["slurm/v10_t03", "v10/protocol/T03_UNITS.csv",
                            "v10/package_v10_server_v1_20261016/synthetic/src/v10_synthetic.py",
                            "v10/package_v10_server_v1_20261016/synthetic/tests/check_synthetic.py"]
    identities = {"t03_code_hash": tracked_hash(repo, code_paths), "t03_job_pin": tracked_hash(repo, pin_paths),
                  "bf_code_hash": tracked_hash(bf, ["src/v10", "scripts/v10"]),
                  "ledger_sha256": sha(ledger_path), "backend_sha256": sha(bf / "src/v10/mf_vcs.py")}
    audit.check(identities["t03_code_hash"] == "54606921cdd2bf15", "identity", "accepted T03 code")
    audit.check(identities["t03_job_pin"] == "64478d332c0a115a", "identity", "accepted formal job pin")
    audit.check(identities["bf_code_hash"] == "e9b4239450c4e2eb", "identity", "accepted BF core")
    records, evals, record_paths = {}, {}, {}
    manifest_summaries = []
    external_n = external_bytes = 0
    for manifest_path in sorted(t03.glob("*/*RESULTS_MANIFEST.json")):
        m = read(manifest_path)
        base = manifest_path.parent
        for ent in m["files"]:
            path = base / ent["path"]
            audit.check(path.resolve().is_relative_to(base.resolve()), "manifest_path", str(path))
            audit.file(path, ent, "exported_file")
        for ent in m["kept_outside_repository"]:
            if not isinstance(ent, dict) or "sha256" not in ent or "bytes" not in ent:
                continue
            external_n += 1
            external_bytes += ent["bytes"]
            if external:
                audit.file(Path(ent["path"]), ent, "external_artifact")
        manifest_summaries.append({"path": str(manifest_path.relative_to(root)), "sha256": sha(manifest_path),
                                   "job_id": m["job_id"], "exported_files": len(m["files"]),
                                   "exported_bytes": m["total_bytes"]})
        for path in sorted(base.glob("runs/*/result.json")):
            r = read(path)
            uid = r["unit_id"]
            audit.check(uid not in records, "unit", f"duplicate {uid}")
            records[uid], record_paths[uid] = r, path
            epath = path.parent / "evaluator_only/evaluation.json"
            evals[uid] = read(epath)
            audit.check(sha(epath) == r["evaluation"]["sha256"], "evaluation_link", uid)
    audit.check(set(records) == set(ledger), "coverage", "records equal frozen unit ledger")
    shared_data, shared_init, shared_perm, saturated = {}, {}, {}, {}
    for uid, r in records.items():
        e, row, c = evals[uid], ledger[uid], r["config"]
        ident = r["provenance"]["identity"]
        audit.check(r["status"] == "COMPLETE" and e["status"] == "PASS", "unit_status", uid)
        audit.check(r["unit_row"] == row, "unit_config", uid + " frozen row")
        for k in ("unit_id", "block", "method", "family", "selection_rule"):
            audit.check(c[k] == row[k], "unit_config", uid + " " + k)
        for k in ("m", "rho", "K", "data_root_seed", "algo_seed", "steps", "ckpt_every"):
            matches = c[k] is None if row[k] == "NA" else float(c[k]) == float(row[k])
            audit.check(matches, "unit_config", uid + " " + k)
        audit.check(c["dtype"] == "float32" and c["device"] == "cpu", "unit_config", uid + " precision/device")
        audit.check(ident["t03_code_hash"] == identities["t03_code_hash"], "unit_identity", uid)
        expected_pin = "4fef9e2bc122c35d" if c["block"] == "smoke" else identities["t03_job_pin"]
        audit.check(ident["t03_job_pin"] == expected_pin, "unit_identity", uid + " job pin")
        audit.check(ident["units_csv_sha256"] == identities["ledger_sha256"] and not ident["git_dirty"], "unit_identity", uid + " ledger/clean")
        audit.check(not r["truth_barrier"]["denials_during_unit"], "barrier_record", uid)
        for k in ("unit_id", "block", "method", "data_root_seed", "algo_seed"):
            audit.check(e[k] == c[k], "evaluation_link", uid + " " + k)
        audit.check(r["steps_completed"] == c["steps"], "training_record", uid)
        expected_steps = list(range(c["ckpt_every"], c["steps"] + 1, c["ckpt_every"]))
        audit.check([x["step"] for x in r["checkpoints"]] == expected_steps, "training_record", uid + " checkpoints")
        best = best_checkpoint(r["checkpoints"])
        audit.check(best["step"] == r["selection"]["own_cal"]["step"] == e["selected"]["step"], "CAL_selection", uid)
        audit.check(best["CAL_UJ"] == r["selection"]["own_cal"]["CAL_UJ"], "CAL_selection", uid + " score")
        sm = e["selected_metrics"]
        audit.check(abs(sm["S"] - sm["J_pop"] - sm["pop_mse"]) < 2e-13, "recorded_identity", uid)
        audit.check(close(sm["EVAL_UJ"] - sm["S"], sm["EVAL_UJ_minus_S"]), "recorded_identity", uid + " UJ-S")
        audit.check(sm["identity_check"] == "PASS" and all(x == "PASS" for x in e["checks"].values()), "evaluation_checks", uid)
        samples = {**r["data"]["sample_sha256"], "eval": r["data"]["eval_sample_sha256_record"]}
        audit.check(samples == e["sample_sha256"], "data_pairing", uid + " evaluator samples")
        key = (c["data_root_seed"], r["case_id"])
        audit.check(shared_data.setdefault(key, samples) == samples, "data_pairing", uid)
        sat = e["saturated_fit_table"]
        stable_sat = {mode: {k: sat[mode][k] for k in ("pop_mse", "EVAL_UJ", "FIT_UJ", "CAL_UJ", "J_pop")} for mode in ("U", "V")}
        audit.check(saturated.setdefault(key, stable_sat) == stable_sat, "saturated_pairing", uid)
        ik = (c["algo_seed"], r["case_id"], r["critic"]["kind"], c["K"])
        audit.check(shared_init.setdefault(ik, r["critic"]["init_state_sha256"]) == r["critic"]["init_state_sha256"], "initialization_pairing", uid)
        if r["streams"].get("perm"):
            pk = (c["algo_seed"], r["case_id"])
            audit.check(shared_perm.setdefault(pk, r["streams"]["perm"]) == r["streams"]["perm"], "permutation_stream_pairing", uid)
        for dep, expected_sha in r.get("hyperparameter_resolution", {}).get("dependency_result_sha256", {}).items():
            audit.check(dep in record_paths and sha(record_paths[dep]) == expected_sha, "CAL_dependency", uid + " " + dep)
    selection = read(t03 / "CAL/selection/T03_CAL_SELECTION.json")
    group_selected = {}
    for group in selection["groups"]:
        chosen = best_candidate(group["candidates"], records)
        audit.check(group["status"] == "SELECTED", "CAL_group", group["case_id"] + group["method"])
        audit.check(all(chosen[k] == group["selected"][k] for k in chosen), "CAL_group", chosen["unit_id"])
        group_selected[group["case_id"], group["method"]] = chosen
    audit.check(selection["n_groups"] == selection["n_selected"] == len(group_selected) == 60, "CAL_group", "60 groups")
    for uid, r in records.items():
        c = r["config"]
        if c["block"] in ("main", "algo_extra", "capacity"):
            selected = group_selected[r["case_id"], c["method"]]
            audit.check((c["lr"], c["cap"]) == (selected["lr"], selected["cap"]), "CAL_resolution", uid)
    agg_path = t03 / "formal/aggregate/evaluator_only/T03_AGGREGATE.json"
    agg = read(agg_path)
    audit.check(agg["aggregator_sha256"] == sha(repo / "v10/tools/t03_aggregate.py"), "aggregate_identity", "aggregator code")
    numeric_fields = {"EVAL_UJ_minus_S": lambda s: s["EVAL_UJ_minus_S"], "pop_mse": lambda s: s["pop_mse"],
                      "EVAL_UJ_minus_J_pop": lambda s: s["EVAL_UJ_minus_J_pop"], "EVAL_U_minus_V": lambda s: s["EVAL_U_minus_V"],
                      "perm1_var_fit": lambda s: s["perm_conditional_variance"]["fit"]["perm1_var"]}
    for table, n_cells, reps in (("main", 60, 10), ("algo", 60, 3), ("capacity", 36, 10)):
        audit.check(len(agg[table]) == n_cells, "aggregate_coverage", table)
        for cell in agg[table]:
            ids = cell["unit_ids"]
            audit.check(cell["expected"] == cell["n_complete"] == len(set(ids)) == reps, "aggregate_coverage", table + cell["case_id"])
            for uid in ids:
                r = records[uid]
                matches = r["case_id"] == cell["case_id"] and r["config"]["method"] == cell["method"]
                if table == "main":
                    matches = matches and r["block"] == "main"
                elif table == "algo":
                    matches = matches and r["block"] in ("main", "algo_extra") and r["config"]["data_root_seed"] == 2026101700
                else:
                    matches = matches and r["block"] in ("main", "capacity") and r["config"]["K"] == cell["K"]
                audit.check(matches, "aggregate_membership", table + " " + uid)
            for field, func in numeric_fields.items():
                actual = basic_stats([func(evals[u]["selected_metrics"]) for u in ids])
                audit.check(all(close(value, cell[field][k]) for k, value in actual.items()), "aggregate_arithmetic", table + cell["case_id"] + cell["method"] + field)
    for pair in agg["paired_main"]:
        sides = {}
        for method in (pair["a"], pair["b"]):
            sides[method] = {r["config"]["data_root_seed"]: evals[u]["selected_metrics"] for u, r in records.items()
                             if r["block"] == "main" and r["case_id"] == pair["case_id"] and r["config"]["method"] == method}
        audit.check(set(sides[pair["a"]]) == set(sides[pair["b"]]) and pair["n_pairs"] == 10, "paired_coverage", pair["case_id"])
        for target, source, power in (("diff_pop_mse", "pop_mse", 1), ("diff_sq_err_EVAL_UJ", "EVAL_UJ_minus_S", 2)):
            values = [sides[pair["a"]][s][source] ** power - sides[pair["b"]][s][source] ** power for s in sides[pair["a"]]]
            audit.check(all(close(v, pair[target][k]) for k, v in basic_stats(values).items()), "paired_arithmetic", pair["case_id"] + target)
    t01 = []
    for name in ("TORCH_VALIDATION", "GEOMETRY_GRADIENT_CHECK"):
        path = bf / f"v10/T01/{name}.json"
        d = read(path)
        checks = [c for part in d["parts"].values() for c in part["checks"]]
        required = [c for c in checks if c["required"]]
        audit.check(d["overall_status"] == "PASS" and d["host_adapter_verified"], "T01_record", name)
        audit.check(all(c["status"] == "PASS" for c in required), "T01_record", name + " required")
        t01.append({"path": str(path.relative_to(root)), "sha256": sha(path), "checks": len(checks), "required": len(required),
                    "required_pass": sum(c["status"] == "PASS" for c in required),
                    "nonrequired_fail": sum(not c["required"] and c["status"] == "FAIL" for c in checks)})
    budget_path = bf / "v10/T02/BUDGET.json"
    budget = read(budget_path)
    audit.check(budget["status"] == "MEASURED_PROJECTION" and budget["grid_within_family_caps"], "T02_record", "current budget status")
    projection = budget["projection"]
    audit.check(close(sum(c["tuning_gpu_h"] for c in projection["candidates"]), projection["tuning_total_gpu_h"]), "T02_record", "tuning sum")
    summary = {"schema": "cs_qmi.handoff.audit.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "status": "PASS" if not audit.errors else "FAIL", "audit_tool_sha256": sha(__file__), "identities": identities,
               "scope": "Artifact integrity, recorded provenance, CAL-only choices, pairing and descriptive arithmetic; no training rerun or new natural TEST read.",
               "checks": dict(audit.counts), "n_checks": sum(audit.counts.values()), "errors": audit.errors,
               "manifests": manifest_summaries, "T03_units_by_block": dict(collections.Counter(r["block"] for r in records.values())),
               "T03_CAL_groups": len(group_selected), "T01": t01,
               "T02_budget": {"path": str(budget_path.relative_to(root)), "sha256": sha(budget_path), "status": budget["status"],
                              "tuning_gpu_h": projection["tuning_total_gpu_h"],
                              "total_gpu_h_by_selected_M1_cap_max_over_lr": projection["total_gpu_h_by_selected_M1_cap_max_over_lr"]},
               "external_artifacts": {"checked": external, "count": external_n, "bytes": external_bytes},
               "limitations": ["SHA checks establish consistency with recorded manifests, not independent authenticity.",
                               "Stored identity/CAL metrics are checked; no checkpoint deserialization or numerical forward/gradient rerun.",
                               "CI endpoints and scientific assumptions are not validated by arithmetic checks.",
                               "Slurm controller reachability is separate from archived completion records."]}
    write(dest / "evidence/V10_AUDIT.json", summary)
    if audit.errors:
        print(json.dumps({"status": "FAIL", "errors": audit.errors[:12], "n_errors": len(audit.errors)}))
        return 1
    truth_dest = dest / "evaluator_only"
    truth_dest.mkdir(parents=True, exist_ok=True)
    for table in ("main", "algo", "capacity"):
        fieldnames = ["case_id", "method", "K", "R", "S", "bias_UJ", "sd_UJ", "rmse_UJ", "pop_mse_mean", "head_gap_mean"]
        with (truth_dest / f"T03_{table}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in agg[table]:
                writer.writerow({"case_id": c["case_id"], "method": c["method"], "K": c.get("K", 16 if c["method"].startswith("MF") else "NA"),
                                 "R": c["n_complete"], "S": c["S"], "bias_UJ": c["EVAL_UJ_minus_S"]["mean"],
                                 "sd_UJ": c["EVAL_UJ_minus_S"]["sd"], "rmse_UJ": c["EVAL_UJ_minus_S"]["rmse"],
                                 "pop_mse_mean": c["pop_mse"]["mean"], "head_gap_mean": c.get("head_fw_gap_fit", {}).get("mean", "NA")})
    comparisons = []
    for a, b in sorted(set((p["a"], p["b"]) for p in agg["paired_main"])):
        ps = [p for p in agg["paired_main"] if (p["a"], p["b"]) == (a, b)]
        comparisons.append({"a": a, "b": b, "conditions": len(ps),
                            "a_lower_pop_mse_conditions": sum(p["diff_pop_mse"]["mean"] < 0 for p in ps),
                            "a_lower_eval_squared_error_conditions": sum(p["diff_sq_err_EVAL_UJ"]["mean"] < 0 for p in ps)})
    m1 = [r for r in records.values() if r["block"] == "main" and r["config"]["method"] == "MF_VCS_M1"]
    descriptive = {"aggregate_sha256": sha(agg_path), "paired_direction_counts_descriptive_only": comparisons,
                   "m1_main": {"units": len(m1), "converged_step_fraction_min": min(r["m1"]["converged_fraction"] for r in m1),
                               "converged_step_fraction_median": statistics.median(r["m1"]["converged_fraction"] for r in m1),
                               "converged_step_fraction_max": max(r["m1"]["converged_fraction"] for r in m1),
                               "caps": dict(collections.Counter(r["config"]["cap"] for r in m1))},
                   "main_cost_seconds_mean": {c["method"]: c["wall_seconds_mean"] for c in agg["cost"] if c["block"] == "main"},
                   "max_abs_recorded_identity_residual": max(abs(e["selected_metrics"]["identity_residual"]) for e in evals.values())}
    write(truth_dest / "T03_DESCRIPTIVE.json", descriptive)
    print(json.dumps({"status": "PASS", "n_checks": summary["n_checks"], "units": len(records), "external": summary["external_artifacts"]}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-external-artifacts", action="store_true")
    args = parser.parse_args()
    destination = args.output_dir or args.root / "handoff"
    # A crash must not leave an older PASS looking like the outcome of this run.
    audit_path = destination / "evidence/V10_AUDIT.json"
    write(audit_path, {"status": "RUNNING", "generated_utc": datetime.now(timezone.utc).isoformat()})
    try:
        rc = run(args.root, destination, args.verify_external_artifacts)
    except Exception as exc:
        write(audit_path, {"status": "FAIL", "generated_utc": datetime.now(timezone.utc).isoformat(),
                           "reason": "audit aborted; no new derived tables are certified", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    raise SystemExit(rc)
