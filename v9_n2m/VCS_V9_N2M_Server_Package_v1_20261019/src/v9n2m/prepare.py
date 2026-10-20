"""Freeze one bounded 6–9-run experiment after the inherited N2F block closes."""
from __future__ import annotations

from pathlib import Path

from .common import PACKAGE_ROOT, ARMS, SEEDS, assert_same_file, canonical_sha256, read_json, sha256_file, utc_now, write_csv, write_json
from .identity import closeout_census, pinned_freeze, resolve_inputs, validate_vcs_reference, verify_training_inputs
from .runtime import environment_differences, package_code_identity, verify_runtime_files


def _historical_environment(seed, result, supplied):
    historical = dict(result["hardware"])
    entry = supplied.get(str(seed))
    if entry is None:
        return historical, {"source": "original_full_result.hardware", "supplement": None}
    evidence_path = Path(entry["evidence_path"]).resolve()
    assert_same_file(evidence_path, entry["evidence_sha256"])
    if not entry.get("captured_during_original_job"):
        raise ValueError("A current environment probe cannot be asserted to describe an old allocation")
    if str(entry.get("original_job_id")) != str(historical.get("job_id")):
        raise ValueError(f"Historical environment evidence has the wrong original job ID for seed {seed}")
    extra = entry["environment"]
    for key in ("gpu", "torch", "cuda", "python", "deterministic_algorithms", "tf32"):
        if key in historical and key in extra and historical[key] != extra[key]:
            raise ValueError(f"Historical environment evidence contradicts saved {key} for seed {seed}")
    historical.update(extra)
    return historical, {"source": "original_full_result.hardware+archived_environment_evidence", "supplement": entry}


def prepare_experiment(base, out, *, mode="auto", old_output=None, data=None, vocab=None,
                       verification_path=None, historical_environments=None):
    from .controls import generate_controls
    base, out = Path(base).resolve(), Path(out).resolve()
    if mode not in ("auto", "replay9", "reuse6"):
        raise ValueError("mode must be auto, replay9 or reuse6")
    inputs = resolve_inputs(base, old_output, data, vocab)
    for original in (base, Path(inputs["old_output"]), Path(inputs["data"])):
        if out == original or original in out.parents:
            raise ValueError("New output must be separate from all inherited source/input/output directories")
    experiment_path = out / "experiment.json"
    if experiment_path.exists():
        existing = verify_experiment(experiment_path)
        if existing["base_handoff"] != str(base) or existing["requested_mode"] != mode or existing["inputs"] != inputs:
            raise ValueError("Existing experiment is already frozen with different preparation arguments")
        return existing
    protocol_path = PACKAGE_ROOT / "config/protocol.json"
    protocol = read_json(protocol_path)
    frozen = pinned_freeze(base)
    runtime = verify_runtime_files(out / "runtime")
    verification_path = Path(verification_path or out / "SERVER_VERIFICATION.json").resolve()
    verification = read_json(verification_path)
    code_identity = package_code_identity()
    if verification.get("status") != "PASS" or verification.get("device") != "cuda":
        raise ValueError("A real same-runtime CUDA comparison/resume verification is still needed")
    if verification.get("runtime_receipt_sha256") != sha256_file(out / "runtime/RUNTIME.json"):
        raise ValueError("Server verification belongs to another effective runtime")
    if verification.get("package_code_digest") != code_identity["digest"]:
        raise ValueError("Package code changed after server verification")
    assert_same_file(verification["verification_report"], verification["verification_report_sha256"])
    target = verification["environment"]
    for key in ("torch", "cuda", "python"):
        if target.get(key) != protocol["software"][key]:
            raise ValueError(f"Use the inherited N2F server software, not an upgraded environment: {key}={target.get(key)!r}, expected {protocol['software'][key]!r}")
    if environment_differences(target, target):
        raise ValueError("The target environment record is incomplete; retain driver/library evidence from the allocated GPU")
    source_check = verify_training_inputs(base, inputs)
    census = closeout_census(base, check_binary=True, out_path=out / "N2F_CLOSEOUT.json")
    if not census["ready_for_new_training"]:
        raise ValueError(f"N2F closeout is {census['complete']}/33; preserve the existing executor and finish its remaining units first")
    supplied = read_json(historical_environments) if historical_environments else {}
    references = {(r["seed"], r["method"], r["scale"]): r for r in census["units"]}
    unit_rows, decisions, control_manifests = [], {}, {}
    for seed in SEEDS:
        seed_base = Path(inputs["old_output"]) / "rho0.4" / f"s{seed}"
        control_dir = out / "controls" / f"s{seed}"
        controls = generate_controls(seed_base / "pairs.observed_pairs.json",
                                     seed_base / "weights/VCS_Weight.csv",
                                     seed_base / "weights/VCS_Weight.csv.lock.json", seed, control_dir)
        control_manifests[str(seed)] = {"path": str(control_dir / "manifest.json"),
                                        "sha256": sha256_file(control_dir / "manifest.json")}
        ref = references[(seed, "VCS_Weight", "sum")]
        reference_recipe = validate_vcs_reference(ref["path"], base, seed)
        old_result = read_json(Path(ref["path"]) / "full_result.json")
        old_env, evidence = _historical_environment(seed, old_result, supplied)
        differences = environment_differences(old_env, target)
        action = "REUSE" if mode != "replay9" and not differences else "RUN"
        if mode == "reuse6" and differences:
            raise ValueError(f"Historical seed {seed} is not proved environment-compatible; use auto/replay9. Differences: {differences}")
        decisions[str(seed)] = {"VCS_action": action, "environment_differences": differences,
                                "historical_recipe_verification": reference_recipe,
                                "historical_environment_evidence": evidence,
                                "reference_result_sha256": ref["result_sha256"],
                                "reason": "Complete verified historical reference with matching environment" if action == "REUSE"
                                          else "Explicit replay9" if mode == "replay9"
                                          else "Historical environment is different or insufficiently recorded; use a new matched reference"}
        order = protocol["reference"]["replay_order"][str(seed)]
        for arm in ARMS:
            unit_id = f"s{seed}__{arm}"
            item = controls["arms"][arm]
            row = {"unit_id": unit_id, "seed": seed, "arm": arm,
                   "action": action if arm == "VCS" else "RUN",
                   "order": order.index(arm), "output_dir": str(out / "runs" / unit_id),
                   "observed_path": str(seed_base / "pairs.observed_pairs.json"),
                   "observed_sha256": sha256_file(seed_base / "pairs.observed_pairs.json"),
                   "noise_manifest_path": str(seed_base / "pairs.noise_manifest.json"),
                   "noise_manifest_sha256": sha256_file(seed_base / "pairs.noise_manifest.json"),
                   "weights_path": item["csv"], "weight_lock_path": item["lock"],
                   "weights_sha256": item["sha256"], "weight_lock_sha256": sha256_file(item["lock"]),
                   "controls_dir": str(control_dir)}
            if arm == "VCS":
                row["historical_reference_dir"] = ref["path"]
                if action == "REUSE":
                    row["reference_source_dir"] = ref["path"]
                    row["reference_result_sha256"] = ref["result_sha256"]
            unit_rows.append(row)
    run_count = sum(u["action"] == "RUN" for u in unit_rows)
    if not 6 <= run_count <= 9 or len(unit_rows) != 9:
        raise ValueError("New-run budget or experiment census is invalid")
    experiment = {"schema_version": 1, "protocol_id": protocol["protocol_id"],
                  "protocol_sha256": sha256_file(protocol_path), "created_utc": utc_now(),
                  "base_handoff": str(base), "base_freeze_sha256": sha256_file(base / "FROZEN.json"),
                  "requested_mode": mode, "new_full_runs": run_count,
                  "runtime_dir": str(out / "runtime"), "runtime_receipt_sha256": sha256_file(out / "runtime/RUNTIME.json"),
                  "source_identity": runtime, "package_code": code_identity,
                  "target_environment": target, "server_verification_path": str(verification_path),
                  "server_verification_sha256": sha256_file(verification_path),
                  "N2F_closeout_path": str(out / "N2F_CLOSEOUT.json"),
                  "N2F_closeout_sha256": sha256_file(out / "N2F_CLOSEOUT.json"),
                  "inputs": inputs, "input_identity": source_check,
                  "control_manifests": control_manifests, "reference_decisions": decisions,
                  "units": unit_rows, "scientific_data_role": "TRAIN+reused DEV; developmental mechanism experiment",
                  "test_read": False, "analysis_parameters": protocol["analysis"]}
    write_json(experiment_path, experiment)
    (out / "EXPERIMENT.sha256").write_text(sha256_file(experiment_path) + "  experiment.json\n", encoding="ascii")
    write_csv(out / "UNIT_PLAN.csv", [{k: u.get(k) for k in ("unit_id", "seed", "arm", "action", "order", "output_dir", "reference_source_dir", "weights_sha256")} for u in sorted(unit_rows, key=lambda x: (x["seed"], x["order"]))])
    return experiment


def verify_experiment(experiment_path):
    p = Path(experiment_path).resolve()
    receipt = (p.parent / "EXPERIMENT.sha256").read_text().split()[0]
    assert_same_file(p, receipt)
    exp = read_json(p)
    assert_same_file(PACKAGE_ROOT / "config/protocol.json", exp["protocol_sha256"])
    if package_code_identity() != exp["package_code"]:
        raise ValueError("The execution package differs from its frozen identity")
    assert_same_file(Path(exp["runtime_dir"]) / "RUNTIME.json", exp["runtime_receipt_sha256"])
    runtime = verify_runtime_files(exp["runtime_dir"])
    if runtime != exp["source_identity"]:
        raise ValueError("Effective runtime differs from the experiment source identity")
    assert_same_file(exp["server_verification_path"], exp["server_verification_sha256"])
    assert_same_file(exp["N2F_closeout_path"], exp["N2F_closeout_sha256"])
    for rec in exp["control_manifests"].values():
        assert_same_file(rec["path"], rec["sha256"])
    units = exp["units"]
    if {(u["seed"], u["arm"]) for u in units} != {(s, a) for s in SEEDS for a in ARMS} or len(units) != 9:
        raise ValueError("Unit matrix is not exactly the prescribed three arms and three seeds")
    for u in units:
        for path_key, hash_key in (("observed_path", "observed_sha256"), ("noise_manifest_path", "noise_manifest_sha256"),
                                   ("weights_path", "weights_sha256"), ("weight_lock_path", "weight_lock_sha256")):
            assert_same_file(u[path_key], u[hash_key])
        if u["action"] == "REUSE":
            if u["arm"] != "VCS":
                raise ValueError("Only the VCS reference can be reused")
            assert_same_file(Path(u["reference_source_dir"]) / "full_result.json", u["reference_result_sha256"])
        elif u["action"] != "RUN":
            raise ValueError("Unknown unit action")
    return exp
