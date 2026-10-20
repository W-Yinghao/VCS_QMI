"""Full-trajectory, paired-seed N2M readout without significance claims."""
from __future__ import annotations

import math
from pathlib import Path

from .common import ARMS, PACKAGE_ROOT, SEEDS, read_json, sha256_file, utc_now, write_csv, write_json

RECALLS = ("i2t_R1", "i2t_R5", "i2t_R10", "t2i_R1", "t2i_R5", "t2i_R10")
EPOCHS, STEPS_PER_EPOCH, TOTAL_UPDATES = 40, 1133, 45320


def _finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def _rsum(rec):
    dev = rec.get("dev") if isinstance(rec, dict) else None
    x = dev.get("rsum") if isinstance(dev, dict) else None
    return float(x) if _finite(x) else None


def collapse_diagnostic(train_log):
    """Locate the actual final 200 updates, including the legacy resume bug.

Legacy step counters restart at len(sampled_steps) at each recorded resume.
They can be reconstructed from resumed_at and the completed epoch lengths.
Unknown locations/coverage are retained as unknown, never converted to false.
"""
    epoch_list = train_log.get("epochs", [])
    if not isinstance(epoch_list, list):
        return {"status": "UNKNOWN", "severe_collapse": None, "reason": "Malformed epoch log", "epochs": [], "pairs": []}
    epochs = {r.get("epoch"): r for r in epoch_list if isinstance(r, dict) and isinstance(r.get("epoch"), int)}
    baseline = _rsum(epochs.get(4, {}))
    if baseline is None:
        return {"status": "UNKNOWN", "severe_collapse": None, "reason": "Complete epoch 4 DEV baseline is unavailable", "epochs": [], "pairs": []}
    steps = train_log.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    steps = [r for r in steps if isinstance(r, dict)]
    resumes = train_log.get("resumed_at", [])
    if not isinstance(resumes, list) or any(not isinstance(r, int) or r < 0 for r in resumes):
        resumes = []
        resume_metadata_valid = False
    else:
        resume_metadata_valid = True
    rows = []
    for epoch in sorted(e for e in epochs if e >= 5):
        rec = epochs[epoch]
        nsteps = rec.get("steps_in_epoch")
        dev = _rsum(rec)
        complete_epoch = nsteps == STEPS_PER_EPOCH
        records = [r for r in steps if r.get("epoch") == epoch]
        segment_start = max([0] + [r for r in resumes if r <= epoch])
        previous_epochs = [epochs.get(e, {}).get("steps_in_epoch") for e in range(segment_start, epoch)]
        counter_known = resume_metadata_valid and all(isinstance(v, int) and v > 0 for v in previous_epochs)
        if segment_start:
            # Frozen original train.py: step = len(log['steps']) on resume.
            counter_start = sum(1 for r in steps if isinstance(r.get("epoch"), int) and r["epoch"] < segment_start)
        else:
            counter_start = 0
        counter_before_epoch = counter_start + sum(previous_epochs) if counter_known else None
        located, reasons, location_methods = {}, [], set()
        for sample in records:
            if "step_in_epoch" in sample:
                pos = sample["step_in_epoch"]
                location_methods.add("explicit_step_in_epoch")
            elif counter_before_epoch is not None and isinstance(sample.get("step"), int):
                pos = sample["step"] - counter_before_epoch
                location_methods.add("reconstructed_legacy_counter_and_resume")
            else:
                reasons.append("Sample update cannot be located")
                continue
            if not isinstance(pos, int) or not isinstance(nsteps, int) or not 1 <= pos <= nsteps:
                reasons.append("Sample update location lies outside its epoch")
                continue
            if pos in located:
                reasons.append("Duplicate sampled update within epoch")
                continue
            located[pos] = sample
        final = [(p, s) for p, s in sorted(located.items()) if isinstance(nsteps, int) and p > nsteps - 200]
        risks = [s.get("u_risk_unnormalised", s.get("u_risk")) for p, s in final]
        # Frozen log_every=20 yields exactly ten samples in a 200-update window.
        # We require their actual locations, all 20 apart; not simply last ten rows.
        positions = [p for p, s in final]
        coverage = len(positions) == 10 and all(b-a == 20 for a, b in zip(positions, positions[1:]))
        valid = complete_epoch and not reasons and coverage and all(_finite(v) for v in risks)
        frac = sum(v >= .39 for v in risks) / len(risks) if valid else None
        low_dev = (dev < .5 * baseline) if dev is not None and complete_epoch else None
        if low_dev is False:
            criterion = False
        elif low_dev is True and valid:
            criterion = frac >= .9
        else:
            criterion = None
        rows.append({"epoch": epoch, "complete_epoch": complete_epoch, "dev_rsum": dev,
                     "dev_below_half_epoch4": low_dev, "final_window_updates": 200,
                     "sample_positions_in_epoch": positions, "n_samples": len(final),
                     "sample_coverage_valid": valid, "fraction_U_risk_ge_0p39": frac,
                     "criterion_this_epoch": criterion, "location_methods": sorted(location_methods),
                     "unavailable_reasons": reasons + ([] if coverage else ["Expected ten complete log_every=20 samples in the actual final 200 updates"])})
    by_epoch = {r["epoch"]: r for r in rows}
    pairs = []
    for e in sorted(by_epoch):
        if e+1 not in by_epoch:
            continue
        a, b = by_epoch[e]["criterion_this_epoch"], by_epoch[e+1]["criterion_this_epoch"]
        state = True if a is True and b is True else False if a is False or b is False else None
        pairs.append({"epochs": [e, e+1], "severe": state})
    severe = True if any(p["severe"] is True for p in pairs) else None if any(p["severe"] is None for p in pairs) else False
    full = sorted(epochs) == list(range(EPOCHS)) and all(r.get("steps_in_epoch") == STEPS_PER_EPOCH for r in epochs.values())
    if not full and severe is False:
        severe = None
    return {"status": "YES" if severe is True else "NO" if severe is False else "UNKNOWN", "severe_collapse": severe,
            "epoch4_rsum": baseline, "full_trajectory": full, "epochs": rows, "pairs": pairs,
            "definition": "Two consecutive complete hard epochs: DEV < half own epoch4 and each final-200-update sampled U-risk >= .39 fraction >= .9",
            "not_an_early_stop_rule": True}


def _cost_summary(root, log, result):
    attempts = []
    errors = []
    for path in sorted((root / "attempts").glob("*.json")):
        try:
            obj = read_json(path)
            if not isinstance(obj, dict):
                raise ValueError("Attempt record must be an object")
            elapsed = obj.get("elapsed_s")
            valid = _finite(elapsed) and elapsed >= 0 and obj.get("end_utc") is not None
            attempts.append({"file": str(path), "status": obj.get("status"), "start_utc": obj.get("start_utc"),
                             "end_utc": obj.get("end_utc"), "elapsed_s": float(elapsed) if valid else None,
                             "job_id": obj.get("job_id"), "duration_known": valid})
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{path.name}: {exc}")
    if attempts:
        known = sum(a["elapsed_s"] for a in attempts if a["duration_known"])
        complete = all(a["duration_known"] for a in attempts) and not errors
        return {"source": "per-attempt ledger", "attempts": attempts, "known_attempt_seconds": known,
                "known_attempt_process_hours": known / 3600, "duration_complete": complete,
                "unknown_or_unclosed_attempts": sum(not a["duration_known"] for a in attempts), "errors": errors,
                "interpretation": "Recorded attempt process durations; unrecorded allocation startup or lost work is not assumed zero"}
    epochs = log.get("epochs", [])
    # Each legacy seconds value is cumulative within an attempt. Summing every
    # epoch's seconds would grossly overcount. Sum only segment-ending values.
    resumes = set(e for e in log.get("resumed_at", []) if isinstance(e, int))
    segment_last, known, previous = None, 0.0, None
    segments = []
    for rec in sorted(epochs, key=lambda r: r.get("epoch", -1)):
        value = rec.get("seconds")
        if not _finite(value) or value < 0:
            continue
        starts_new = previous is not None and (rec.get("epoch") in resumes or value < previous)
        if starts_new and segment_last is not None:
            known += segment_last
            segments.append(segment_last)
        segment_last = float(value)
        previous = float(value)
    if segment_last is not None:
        known += segment_last
        segments.append(segment_last)
    return {"source": "legacy epoch cumulative timer segments", "known_completed_epoch_seconds_lower_bound": known if segments else None,
            "segment_end_seconds": segments, "final_attempt_done_seconds": result.get("done", {}).get("seconds"),
            "duration_complete": False, "unknown_or_unclosed_attempts": None, "errors": errors,
            "interpretation": "Lower bound through saved completed epochs. Failed unsaved work, startup and some allocation time are unknown; final done.seconds is not total cost."}


def _gradient_summary(root):
    path = root / "grad_norms.jsonl"
    if not path.exists():
        return {"available": False, "reason": "Gradient window telemetry not present"}
    import json
    records, invalid, duplicate = {}, 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            key = (rec["epoch"], rec["step_in_epoch"])
            if key in records:
                duplicate += 1
            records[key] = rec
        except (ValueError, KeyError, TypeError):
            invalid += 1
    valid = [r for r in records.values() if _finite(r.get("grad_norm_pre_clip")) and type(r.get("clipped")) is bool]
    return {"available": bool(valid), "n_unique_samples": len(valid), "duplicate_records_last_used": duplicate,
            "invalid_lines": invalid, "clip_fraction": sum(r["clipped"] for r in valid) / len(valid) if valid else None,
            "mean_grad_norm_pre_clip": sum(r["grad_norm_pre_clip"] for r in valid) / len(valid) if valid else None,
            "max_grad_norm_pre_clip": max((r["grad_norm_pre_clip"] for r in valid), default=None),
            "coverage": "Epoch4 final200 and epoch5 first200 when enabled; not a whole-training gradient estimate"}


def _unit_readout(unit, experiment):
    root = Path(unit.get("reference_source_dir") or unit["output_dir"]) if unit.get("action") == "REUSE" else Path(unit["output_dir"])
    row = {"unit_id": unit["unit_id"], "seed": unit["seed"], "arm": unit["arm"], "action": unit["action"],
           "provenance": "historical_frozen_VCS" if unit["action"] == "REUSE" else "new_N2M_run", "source_dir": str(root),
           "status": "NOT_STARTED", "complete_for_primary": False, "primary_rsum": None,
           "primary_mR": None, "best_epoch": None, **{k: None for k in RECALLS},
           "errors": [], "missing": [], "artifact_verification": {}}
    result, log, manifest, identity = {}, {}, {}, {}
    payloads = {"full_result.json": result, "train_log.json": log, "run_manifest.json": manifest, "identity.json": identity}
    for filename in list(payloads):
        path = root / filename
        if not path.exists():
            row["missing"].append(filename)
            continue
        try:
            obj = read_json(path)
            if not isinstance(obj, dict):
                raise ValueError("Expected a JSON object")
            payloads[filename] = obj
        except (OSError, ValueError, TypeError) as exc:
            row["errors"].append(f"{filename}: {exc}")
    result, log, manifest, identity = (payloads[k] for k in ("full_result.json", "train_log.json", "run_manifest.json", "identity.json"))
    # A malformed unit must not prevent the other eight units from being
    # summarized. Preserve the error, then provide safe diagnostic containers.
    for key in ("epochs", "steps", "resumed_at"):
        values = log.get(key, [])
        expected_item = int if key == "resumed_at" else dict
        if not isinstance(values, list) or any(not isinstance(v, expected_item) for v in values):
            row["errors"].append(f"train_log.{key} has an invalid schema")
            log[key] = [v for v in values if isinstance(v, expected_item)] if isinstance(values, list) else []
    for key in ("done", "artifact_sha256"):
        if key in result and not isinstance(result[key], dict):
            row["errors"].append(f"full_result.{key} must be an object")
            result[key] = {}
    if "cfg" in manifest and not isinstance(manifest["cfg"], dict):
        row["errors"].append("run_manifest.cfg must be an object")
        manifest["cfg"] = {}
    if unit["action"] == "REUSE" and unit.get("reference_result_sha256") and (root / "full_result.json").exists():
        if sha256_file(root / "full_result.json") != unit["reference_result_sha256"]:
            row["errors"].append("Historical full_result differs from frozen reference_result_sha256")
    if (root / "done.json").exists():
        try:
            if read_json(root / "done.json") != result.get("done"):
                row["errors"].append("done.json differs from full_result.done")
        except (OSError, ValueError, TypeError) as exc:
            row["errors"].append(f"done.json: {exc}")
    if result.get("done", {}).get("manifest_sha256") and (root / "run_manifest.json").exists():
        if result["done"]["manifest_sha256"] != sha256_file(root / "run_manifest.json"):
            row["errors"].append("done.manifest_sha256 differs from run_manifest.json")
    row["hardware"] = result.get("hardware")
    row["cost"] = _cost_summary(root, log, result)
    row["gradient_window"] = _gradient_summary(root)
    row["collapse"] = collapse_diagnostic(log)
    epoch_rows = log.get("epochs", [])
    if not isinstance(epoch_rows, list):
        row["errors"].append("train_log.epochs must be a list")
        epoch_rows = []
    row["epochs_observed"] = len(epoch_rows)
    finite_epochs = [r for r in epoch_rows if isinstance(r, dict) and _rsum(r) is not None and isinstance(r.get("epoch"), int)]
    current = max(finite_epochs, key=lambda r: (_rsum(r), -r["epoch"]), default=None)
    row["current_best_diagnostic"] = {"epoch": current["epoch"], "rsum": _rsum(current)} if current else None
    row["resumed_at"] = log.get("resumed_at", [])
    row["actual_updates_from_epochs"] = sum(r.get("steps_in_epoch", 0) for r in epoch_rows if isinstance(r, dict) and isinstance(r.get("steps_in_epoch"), int))
    row["trajectory"] = [{"epoch": r["epoch"], "rsum": _rsum(r), "steps_in_epoch": r.get("steps_in_epoch"),
                          "mean_loss": r.get("mean_loss"), "mean_u_risk_unnormalised": r.get("mean_u_risk_unnormalised"),
                          "lr": r.get("lr"), "phase": r.get("phase"), **{k: r.get("dev", {}).get(k) for k in (*RECALLS, "mR")}} for r in finite_epochs]
    by_epoch = {r["epoch"]: r for r in finite_epochs}
    row["epoch4_rsum"] = _rsum(by_epoch.get(4, {}))
    row["epoch5_rsum"] = _rsum(by_epoch.get(5, {}))
    hard = [_rsum(r) for r in finite_epochs if r["epoch"] >= 5]
    row["minimum_hard_rsum"] = min(hard) if hard else None
    row["last_rsum"] = _rsum(by_epoch.get(39, {}))
    last5 = [_rsum(by_epoch.get(e, {})) for e in range(35, 40)]
    row["last5_mean_rsum"] = sum(last5) / 5 if all(v is not None for v in last5) else None
    if result:
        if result.get("test_read") is not False:
            row["errors"].append("full_result must explicitly state test_read=false")
        if result.get("full_epochs") != EPOCHS or result.get("done", {}).get("epochs_run") != EPOCHS:
            row["errors"].append("full_result does not certify the prescribed 40 epochs")
        if result.get("n_pool") != 145000 or result.get("n_U") != 130500:
            row["errors"].append("full_result pool/U sizes differ from frozen full data")
        if identity != result.get("identity"):
            row["errors"].append("identity.json and full_result.identity differ")
        if identity.get("seed") != unit["seed"] or identity.get("scale") != "sum":
            row["errors"].append("Result seed/scale differs from unit")
        identity_arm = identity.get("arm", {"VCS_Weight": "VCS"}.get(identity.get("method"), identity.get("method")))
        if identity_arm != unit["arm"]:
            row["errors"].append("Result method/arm differs from unit")
        if unit["action"] == "RUN":
            if identity.get("protocol_sha256") != experiment.get("protocol_sha256") or identity.get("unit_id") != unit["unit_id"]:
                row["errors"].append("New run protocol/unit identity differs from frozen experiment")
            if identity.get("experiment_sha256") != experiment.get("_file_sha256"):
                row["errors"].append("New run experiment SHA256 differs")
            for key in ("observed_sha256", "weights_sha256", "weight_lock_sha256"):
                if key in unit and identity.get(key) != unit[key]:
                    row["errors"].append(f"New run {key} differs from frozen unit")
            expected_runtime = experiment.get("source_identity", {}).get("effective_digest")
            if expected_runtime is not None and identity.get("effective_runtime_digest") != expected_runtime:
                row["errors"].append("New run runtime digest differs from frozen experiment")
            if identity.get("base_freeze_sha256", experiment.get("base_freeze_sha256")) != experiment.get("base_freeze_sha256"):
                row["errors"].append("New run baseline freeze digest differs")
            if result.get("done", {}).get("global_step") != TOTAL_UPDATES:
                row["errors"].append("New run terminal global_step is not 45320")
        elif identity.get("freeze_sha256") != experiment.get("base_freeze_sha256"):
            row["errors"].append("Historical reference is not the frozen N2F source")
        hashes = result.get("artifact_sha256", {})
        for filename in ("train_log.json", "run_manifest.json", "identity.json", "best.pt", "last.pt"):
            path = root / filename
            expected = hashes.get(filename)
            if not isinstance(expected, str) or len(expected) != 64:
                row["errors"].append(f"Result lacks a valid artifact SHA256 for {filename}")
                continue
            if not path.exists():
                explicit_missing = bool(unit.get("checkpoints_not_materialized") or result.get("checkpoints_not_materialized")) and filename.endswith(".pt")
                row["artifact_verification"][filename] = "NOT_MATERIALIZED_EXPLICIT" if explicit_missing else "MISSING"
                row["missing"].append(filename)
            elif sha256_file(path) != expected:
                row["artifact_verification"][filename] = "HASH_MISMATCH"
                row["errors"].append(f"Artifact hash mismatch: {filename}")
            else:
                row["artifact_verification"][filename] = "VERIFIED"
    epoch_numbers = [r.get("epoch") for r in epoch_rows if isinstance(r, dict)]
    full_epochs = epoch_numbers == list(range(EPOCHS))
    if result and not full_epochs:
        row["errors"].append("Completed result requires ordered, unique epochs 0..39")
    if result and (row["actual_updates_from_epochs"] != TOTAL_UPDATES or any(not isinstance(r, dict) or r.get("steps_in_epoch") != STEPS_PER_EPOCH for r in epoch_rows)):
        row["errors"].append("Completed result does not contain 40 x 1133 = 45320 actual updates")
    for rec in epoch_rows:
        if not isinstance(rec, dict):
            row["errors"].append("Non-object epoch record")
            continue
        dev = rec.get("dev", {})
        if not isinstance(dev, dict):
            row["errors"].append(f"Epoch {rec.get('epoch')} dev must be an object")
            continue
        values = [dev.get(k) for k in RECALLS]
        if not all(_finite(v) and 0 <= v <= 100 for v in values):
            row["errors"].append(f"Epoch {rec.get('epoch')} missing/invalid six Recalls")
            continue
        if _rsum(rec) is None or not math.isclose(sum(values), _rsum(rec), rel_tol=0, abs_tol=1e-8):
            row["errors"].append(f"Epoch {rec.get('epoch')} rSum differs from six Recalls")
        if not _finite(dev.get("mR")) or not math.isclose(sum(values) / 6, dev["mR"], rel_tol=0, abs_tol=1e-8):
            row["errors"].append(f"Epoch {rec.get('epoch')} mR differs from six Recalls")
    if manifest:
        if manifest.get("n_pool") != 145000 or manifest.get("n_U") != 130500 or manifest.get("n_A") != 14500:
            row["errors"].append("run_manifest pool roles differ from frozen data")
        if manifest.get("seed") != unit["seed"] or manifest.get("lr_mult") != 1.0:
            row["errors"].append("run_manifest seed/LR multiplier differs")
        if manifest.get("weight_lock_sha256") != unit.get("weights_sha256"):
            row["errors"].append("run_manifest weight identity differs from frozen unit")
        cfg = manifest.get("cfg", {})
        expected_cfg = {"batch_size": 128, "num_epochs": 40, "warmup_epochs": 5, "learning_rate": .0002, "margin": .2, "grad_clip": 2.0, "lr_update": 30}
        if any(cfg.get(k) != v for k, v in expected_cfg.items()):
            row["errors"].append("run_manifest training recipe differs from frozen recipe")
    if current and result:
        selected = result.get("done", {}).get("best", {})
        if selected.get("epoch") != current["epoch"] or not _finite(selected.get("rsum")) or not math.isclose(selected["rsum"], _rsum(current), rel_tol=0, abs_tol=1e-8):
            row["errors"].append("Selected endpoint differs from exact best DEV rSum / earliest tie")
    if row["errors"]:
        row["status"] = "INVALID"
    elif row["missing"]:
        only_pt = set(row["missing"]).issubset({"best.pt", "last.pt"})
        explicitly_missing = all(row["artifact_verification"].get(p) == "NOT_MATERIALIZED_EXPLICIT" for p in row["missing"])
        row["status"] = "COMPLETE_METADATA_ONLY" if result and full_epochs and only_pt and explicitly_missing else "INCOMPLETE" if root.exists() else "NOT_STARTED"
    elif result and full_epochs:
        row["status"] = "COMPLETE_VERIFIED"
        row["complete_for_primary"] = True
    if row["complete_for_primary"]:
        row.update({"primary_rsum": _rsum(current), "primary_mR": current["dev"]["mR"],
                    "best_epoch": current["epoch"], **{k: current["dev"][k] for k in RECALLS}})
    if row["status"] in ("INCOMPLETE", "NOT_STARTED") and row["cost"].get("attempts"):
        latest = sorted(row["cost"]["attempts"], key=lambda a: a.get("start_utc") or "")[-1]
        if latest.get("status") in ("FAILED", "INTERRUPTED", "RUNNING"):
            row["status"] = latest["status"]
    return row


def verify_analysis_metadata(experiment_path):
    """Bind a readout to its frozen experiment without requiring training data.

    This deliberately does not call the training-side verify_experiment: the
    original features, runtime copy and input weight files need not be mounted
    on a machine that only reads exported result artifacts.
    """
    path = Path(experiment_path).resolve()
    receipt_path = path.parent / "EXPERIMENT.sha256"
    tokens = receipt_path.read_text(encoding="ascii").split()
    if not tokens or len(tokens[0]) != 64 or any(c not in "0123456789abcdef" for c in tokens[0].lower()):
        raise ValueError("EXPERIMENT.sha256 is malformed")
    actual = sha256_file(path)
    if actual != tokens[0].lower():
        raise ValueError("Experiment file differs from EXPERIMENT.sha256")
    experiment = read_json(path)
    protocol_path = PACKAGE_ROOT / "config/protocol.json"
    protocol_digest = sha256_file(protocol_path)
    if experiment.get("protocol_sha256") != protocol_digest:
        raise ValueError("Analysis package protocol differs from frozen experiment protocol_sha256")
    protocol = read_json(protocol_path)
    if experiment.get("analysis_parameters") != protocol.get("analysis"):
        raise ValueError("Experiment analysis_parameters differ from its frozen protocol")
    package_code_verified = None
    if "package_code" in experiment:
        from .runtime import package_code_identity
        if experiment["package_code"] != package_code_identity():
            raise ValueError("Analysis package code differs from the experiment's frozen package identity")
        package_code_verified = True
    rule = experiment["analysis_parameters"].get("investment_rule", {})
    if (not _finite(rule.get("mean_rsum_difference_each_at_least"))
            or not isinstance(rule.get("positive_seeds_each_at_least"), int)
            or not 0 <= rule["positive_seeds_each_at_least"] <= len(SEEDS)
            or rule.get("complete_pairs_required") != len(SEEDS)
            or type(rule.get("no_more_severe_collapse")) is not bool):
        raise ValueError("Frozen investment rule is incompatible with the prescribed three paired seeds")
    experiment["_file_sha256"] = actual
    experiment["_analysis_identity"] = {
        "experiment_receipt_sha256": sha256_file(receipt_path), "experiment_verified": True,
        "protocol_verified": True, "analysis_parameters_verified": True,
        "package_code_verified": package_code_verified,
        "original_training_inputs_read": False, "original_runtime_read": False,
        "scope": "Metadata identity only; per-unit result and checkpoint integrity are verified separately below",
    }
    return experiment


def summarize_experiment(experiment_path, out_dir):
    """Only complete verified full40 units enter the two prespecified contrasts."""
    experiment = verify_analysis_metadata(experiment_path)
    analysis_rule = experiment["analysis_parameters"]["investment_rule"]
    required_pairs = analysis_rule["complete_pairs_required"]
    required_mean = analysis_rule["mean_rsum_difference_each_at_least"]
    required_positive = analysis_rule["positive_seeds_each_at_least"]
    require_no_extra_collapse = analysis_rule["no_more_severe_collapse"]
    units = experiment.get("units", [])
    expected = {(s, a) for s in SEEDS for a in ARMS}
    keys = [(u.get("seed"), u.get("arm")) for u in units]
    if len(keys) != 9 or set(keys) != expected or len(set(u.get("unit_id") for u in units)) != 9:
        raise ValueError("Frozen experiment requires exactly the three arms x seeds901,902,903; no missing or duplicate units")
    if any(u.get("action") not in ("RUN", "REUSE") or (u.get("action") == "REUSE" and u.get("arm") != "VCS") for u in units):
        raise ValueError("Only VCS reference units may be REUSE")
    rows = [_unit_readout(u, experiment) for u in sorted(units, key=lambda u: (u["seed"], ARMS.index(u["arm"])))]
    by = {(r["seed"], r["arm"]): r for r in rows}
    contrasts = []
    for control in ("ConstantU", "ShuffleVCS"):
        pairs = []
        for seed in SEEDS:
            a, b = by[seed, "VCS"], by[seed, control]
            valid = a["complete_for_primary"] and b["complete_for_primary"]
            pairs.append({"seed": seed, "eligible": valid, "VCS_rsum": a["primary_rsum"], "control_rsum": b["primary_rsum"],
                          "difference": a["primary_rsum"] - b["primary_rsum"] if valid else None,
                          "VCS_provenance": a["provenance"], "control_provenance": b["provenance"],
                          "VCS_status": a["status"], "control_status": b["status"]})
        differences = [p["difference"] for p in pairs if p["eligible"]]
        complete = len(differences) == required_pairs
        # Partial individual differences remain visible for bookkeeping, but do
        # not turn into a selectively completed two-seed experiment mean.
        mean = sum(differences) / required_pairs if complete else None
        vcs_c = [by[s, "VCS"]["collapse"]["severe_collapse"] for s in SEEDS]
        ctl_c = [by[s, control]["collapse"]["severe_collapse"] for s in SEEDS]
        collapse_known = all(v is not None for v in vcs_c + ctl_c)
        no_extra = sum(vcs_c) <= sum(ctl_c) if collapse_known and complete else None
        collapse_condition = no_extra if require_no_extra_collapse else True
        rule = (mean >= required_mean and sum(d > 0 for d in differences) >= required_positive and collapse_condition) if complete and collapse_condition is not None else None
        contrasts.append({"contrast": f"VCS-{control}", "endpoint": "best full DEV rSum; earliest tie", "n_expected": required_pairs,
                          "n_eligible_pairs": len(differences), "complete_three_seed_contrast": complete, "pairs": pairs,
                          "mean_difference": mean, "minimum_difference": min(differences) if complete else None,
                          "maximum_difference": max(differences) if complete else None,
                          "n_positive": sum(d > 0 for d in differences) if complete else None,
                          "n_zero": sum(d == 0 for d in differences) if complete else None,
                          "no_more_severe_collapse": no_extra, "investment_rule_pass": rule,
                          "p_value": None, "confidence_interval": None,
                          "interpretation": "Paired-seed descriptive contrast; n=3. No significance, equivalence, or confirmation claim."})
    rule_values = [c["investment_rule_pass"] for c in contrasts]
    combined = all(rule_values) if all(v is not None for v in rule_values) else None
    counts = {state: sum(r["status"] == state for r in rows) for state in sorted(set(r["status"] for r in rows))}
    report = {"schema_version": 1, "created_utc": utc_now(), "experiment_sha256": experiment["_file_sha256"],
              "protocol_sha256": experiment.get("protocol_sha256"), "test_read": False,
              "analysis_identity": experiment["_analysis_identity"],
              "analysis_parameters": experiment["analysis_parameters"],
              "n_expected_units": 9, "n_complete_verified_units": sum(r["complete_for_primary"] for r in rows),
              "counts": counts, "units": rows, "contrasts": contrasts,
              "investment_rule": {"description": f"Both paired means >= {required_mean:+g} rSum, at least {required_positive}/{required_pairs} positive seeds for each contrast; require no additional severe collapse: {require_no_extra_collapse}",
                                  "frozen_parameters": analysis_rule,
                                  "all_conditions_known": combined is not None, "pass": combined,
                                  "scope": "Prespecified engineering investment rule, not a significance or equivalence test; no early stopping or seed exclusion."},
              "warnings": ["This is a development mechanism experiment on reused DEV, not an independent confirmation set.",
                           "Unfinished/invalid units do not contribute their instantaneous best scores to primary endpoints.",
                           "Historical reused and new VCS references retain row-level provenance. Inspect matched environment approval recorded by preparation.",
                           "Metadata-only copied checkpoints do not qualify as verified complete primary units.",
                           "Cost is reported by attempts where available; missing or lost segments remain unknown."]}
    out = Path(out_dir)
    write_json(out / "summary.json", report)
    table = []
    epoch_table = []
    for row in rows:
        table.append({k: row[k] for k in ("unit_id", "seed", "arm", "action", "provenance", "status", "complete_for_primary", "primary_rsum", "primary_mR", "best_epoch", *RECALLS, "epochs_observed", "actual_updates_from_epochs", "epoch4_rsum", "epoch5_rsum", "minimum_hard_rsum", "last_rsum", "last5_mean_rsum")})
        table[-1]["severe_collapse"] = row["collapse"]["severe_collapse"]
        table[-1]["cost_duration_complete"] = row["cost"]["duration_complete"]
        table[-1]["errors"] = " | ".join(row["errors"])
        table[-1]["missing"] = " | ".join(row["missing"])
        epoch_table.extend({"unit_id": row["unit_id"], "seed": row["seed"], "arm": row["arm"], "status": row["status"], **r} for r in row["trajectory"])
    write_csv(out / "unit_results.csv", table)
    write_csv(out / "epoch_trajectories.csv", epoch_table, ["unit_id", "seed", "arm", "status", "epoch", "rsum", "steps_in_epoch", "mean_loss", "mean_u_risk_unnormalised", "lr", "phase", *RECALLS, "mR"])
    write_csv(out / "paired_contrasts.csv", [{"contrast": c["contrast"], **p} for c in contrasts for p in c["pairs"]])
    lines = ["# V9 N2M 完整训练结果", "", f"已核验完整单元：{report['n_complete_verified_units']}/9。计数：{counts}。", "",
             "主终点固定为同 checkpoint 六项 Recall 相加的 best full DEV rSum，并列选更早 epoch。未完成和缺失 checkpoint 的分数只在机器记录中标为诊断，不进入主表。", "",
             "| seed | VCS | Constant-U | Shuffle-VCS | VCS−Constant | VCS−Shuffle |", "|---|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "NA" if x is None else f"{x:.6f}"
    for seed in SEEDS:
        v, c, s = [by[seed, arm]["primary_rsum"] for arm in ARMS]
        lines.append("| " + " | ".join([str(seed), fmt(v), fmt(c), fmt(s), fmt(v-c if v is not None and c is not None else None), fmt(v-s if v is not None and s is not None else None)]) + " |")
    lines += ["", "| contrast | 完整配对数 | 三 seed 均差 | seed 差范围 | 正方向 seed 数 | 工程投入规则 |", "|---|---:|---:|---|---:|---|"]
    for c in contrasts:
        interval = f"{fmt(c['minimum_difference'])} — {fmt(c['maximum_difference'])}" if c["complete_three_seed_contrast"] else "NA"
        lines.append(f"| {c['contrast']} | {c['n_eligible_pairs']}/3 | {fmt(c['mean_difference'])} | {interval} | {c['n_positive']} | {c['investment_rule_pass']} |")
    lines += ["", f"工程规则（来自冻结协议）：两项差均值均至少 {required_mean:+g} rSum、各至少 {required_positive}/{required_pairs} seed 为正；要求严重塌陷不更多：{require_no_extra_collapse}。它只决定下一步投入，不表示显著性；未通过也不是等价检验。样本数不足预定配对数时不把已完成 seed 的均值替代预定比较。", "", "所有状态、完整轨迹、六项 Recall、best/末值诊断、严重塌陷末 200 实际更新的定位、恢复记录、逐 attempt 成本与未知项均在 summary.json、unit_results.csv 和 epoch_trajectories.csv。", ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")
    return report
