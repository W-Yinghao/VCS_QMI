"""Preregister an entire CAL grid before training; release TEST only as a unit.

This is a reproducible workflow gate, not filesystem access control or proof that
a person never saw labels. Use ``python -m vcs_common.experiment --help``.
Only the standard library is imported until a native train/freeze/evaluate call.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path


LOCK_SCHEMA = "vcs_global_experiment_lock_v1"
RELEASE_SCHEMA = "vcs_global_test_release_v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _write_state(path, value):
    """Atomic replacement is used only for mutable execution state, never locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _path(value, cwd):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(cwd) / path).resolve()


@contextlib.contextmanager
def _cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _default_code_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "kd").is_dir() or (parent / "aigiqa").is_dir():
            return parent
    raise ValueError("Cannot locate application root; supply --code-root")


def source_record(root, app):
    """Match KD's own source digest exactly; record AIG with the same rule."""
    root = Path(root).resolve()
    files = []
    digest = hashlib.sha256()
    for folder in (root / app, root / "vcs_common", root / "shared" / "vcs_common"):
        if folder.is_dir():
            for path in sorted(folder.rglob("*.py")):
                if "references" in path.parts or "tests" in path.parts:
                    continue
                relative = str(path.relative_to(root))
                content = path.read_bytes()
                digest.update(relative.encode()); digest.update(content)
                files.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
    if not files or not (root / app).is_dir():
        raise ValueError("Application code is missing")
    return {"root": str(root), "sha256": digest.hexdigest(), "files": files}


def _tree_hash(root):
    entries = [{"path": str(p.relative_to(root)), "sha256": file_hash(p)}
               for p in sorted(Path(root).rglob("*.py")) if ".git" not in p.parts]
    if not entries:
        raise ValueError(f"No Python source found in external source_root: {root}")
    return canonical_hash(entries)


def _effective_config(raw, app, cwd):
    cfg = copy.deepcopy(raw)
    if app == "kd":
        if cfg.get("schema") != "kd_config_v1":
            raise ValueError("Expected kd_config_v1")
        for field in ("root", "manifest"):
            cfg["data"][field] = str(_path(cfg["data"][field], cwd))
        for which in ("teacher", "student"):
            for field in ("source_root", "checkpoint", "provenance"):
                if cfg[which].get(field):
                    cfg[which][field] = str(_path(cfg[which][field], cwd))
        if cfg["loss"].get("crd"):
            cfg["loss"]["crd"]["source_root"] = str(_path(cfg["loss"]["crd"]["source_root"], cwd))
    return cfg


def _family(cfg, app):
    if app == "kd":
        return cfg["name"]
    loss = cfg["loss"]
    return cfg.get("method_label", loss["method"] + "_" + loss.get("forward", "none") + "_" + "-".join(loss.get("branches", ["text"])))


def _candidate(cfg, app):
    cfg = copy.deepcopy(cfg)
    cfg.pop("seed", None)
    if app == "aigiqa":
        for field in ("output_dir", "device", "num_workers"):
            cfg.pop(field, None)
        # Exact native AIG tie-break signature, so wrapper and native select alike.
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return canonical_hash(cfg)


def _benchmark(cfg, app, cwd):
    if app == "kd":
        return {k: cfg[k] for k in ("student", "teacher", "data", "training", "device", "track")}
    return {"data_dir": str(_path(cfg["data_dir"], cwd)), "target": cfg["target"],
            "model": cfg["model"], "epochs": cfg.get("epochs", 20), "batch_size": cfg.get("batch_size", 5),
            "image_size": cfg.get("image_size", 224), "evidence_role": cfg.get("evidence_role", "DEV")}


def _assets(cfg, app, cwd, cache=None):
    cache = {} if cache is None else cache
    assets = []
    sections = [cfg["teacher"], cfg["student"], cfg["loss"].get("crd", {})] if app == "kd" else [cfg["model"]]
    for section in sections:
        for key in ("checkpoint", "provenance", "med_config"):
            if section.get(key):
                path = _path(section[key], cwd)
                cache_key = (str(path), "file")
                if cache_key not in cache:
                    cache[cache_key] = file_hash(path)
                actual = cache[cache_key]
                declared = section.get(key + "_sha256")
                if declared is not None and actual != declared:
                    raise ValueError(f"Declared asset hash differs: {path}")
                assets.append({"path": str(path), "kind": "file", "sha256": actual})
        if section.get("source_root"):
            path = _path(section["source_root"], cwd)
            key = (str(path), "python_tree")
            if key not in cache:
                cache[key] = _tree_hash(path)
            assets.append({"path": str(path), "kind": "python_tree", "sha256": cache[key]})
    return assets


def _input_record(path):
    path = Path(path).resolve()
    allowed = {"sample_id", "dataset", "image", "image_sha256", "prompt", "prompt_group", "generator", "split"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if fields != allowed:
            raise ValueError(f"AIG evaluation inputs must use the label-free native input schema: {path}")
        rows = list(reader)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Evaluation inputs must be nonempty with unique sample IDs")
    if len({r["dataset"] for r in rows}) != 1 or not rows[0]["dataset"]:
        raise ValueError("An evaluation input table must identify one nonempty dataset")
    if any(len(r["image_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in r["image_sha256"].lower()) for r in rows):
        raise ValueError("All evaluation input image bytes must have verified SHA256 hashes before registration")
    return {"path": str(path), "sha256": file_hash(path), "n_inputs": len(rows)}


def _host_recipe(job):
    cfg = job["effective_config"]
    if cfg.get("schema") == "kd_config_v1":
        dep = cfg["loss"].get("dependence", {})
        return {"optimizer": cfg["optimizer"], "rank": dep.get("rank", 32),
                "q_mode": dep.get("q_mode", "u"), "temperature": dep.get("temperature", 1.)}
    dep = cfg["loss"]
    return {"learning_rate": cfg.get("learning_rate", 1e-5), "regularizer_learning_rate": cfg.get("regularizer_learning_rate", .001),
            "weight_decay": cfg.get("weight_decay", .001), "lr_step_epochs": cfg.get("lr_step_epochs", 4),
            "rank": dep.get("rank", 32), "q_mode": dep.get("q_mode", "u"), "temperature": dep.get("temperature", 1.),
            "branches": dep.get("branches", ["text"]), "forward": dep.get("forward", "none"),
            "gaussian_warmup_epochs": cfg.get("gaussian_warmup_epochs", 0)}


def _validate_design(jobs, matched_groups):
    if not jobs:
        raise ValueError("The expanded grid is empty")
    if len({j["job_id"] for j in jobs}) != len(jobs) or len({j["run_dir"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate candidate/seed or output directory in grid")
    groups = defaultdict(lambda: defaultdict(list))
    for job in jobs:
        groups[job["family"]][job["candidate"]].append(job["seed"])
    reference = sorted(next(iter(next(iter(groups.values())).values())))
    budget = {}
    for family, candidates in groups.items():
        for seeds in candidates.values():
            if len(set(seeds)) != len(seeds) or sorted(seeds) != reference:
                raise ValueError("Unequal training seed sets across planned candidates/families")
        budget[family] = {"candidate_count": len(candidates), "seed_count": len(reference),
                          "run_count": sum(map(len, candidates.values()))}
    for families in matched_groups:
        if len(families) < 2 or len(set(families)) != len(families):
            raise ValueError("A matched budget group needs at least two distinct families")
        if any(f not in budget for f in families):
            raise ValueError("A declared matched family is absent from the grid")
        if len({budget[f]["candidate_count"] for f in families}) != 1:
            raise ValueError("Unequal CAL candidate budgets in a declared matched group")
        recipes = {f: {canonical_hash(_host_recipe(j)) for j in jobs if j["family"] == f} for f in families}
        if any(recipes[f] != recipes[families[0]] for f in families[1:]):
            raise ValueError("Matched families differ in host optimizer, schedule, rank, Q, temperature or branch recipe")
    return reference, budget


def _state_dir(lock_path):
    path = Path(lock_path).resolve()
    return path.parent / (path.stem + ".state")


def _receipt_path(lock_path, job):
    return _state_dir(lock_path) / "jobs" / (job["job_id"] + ".json")


def _load_lock(path):
    lock = _read(path)
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("Wrong experiment lock schema")
    body = dict(lock); stored = body.pop("lock_sha256", None)
    if canonical_hash(body) != stored:
        raise ValueError("Experiment lock hash changed")
    return lock


def _known_test_or_release(lock_path, jobs):
    state = _state_dir(lock_path)
    if (state / "release.json").exists() or (state / "seal_in_progress.json").exists():
        return True
    return any((Path(j["run_dir"]) / filename).exists() for j in jobs
               for filename in ("test_metrics.json", "test_predictions.csv", "TEST_STARTED.json"))


def register(app, grid, output, *, run_root=None, scope="pilot", matched_groups=None,
             required_native=None, required_external=None, inputs=None, primary_metric=None,
             code_root=None, launch_cwd=None, amend_from=None, amend_reason=None, prior_cal_selection=None):
    """Bind every expanded candidate before any student FIT/CAL training starts."""
    if app not in ("kd", "aigiqa") or scope not in ("pilot", "full_comparison"):
        raise ValueError("Unknown application or scope")
    output, grid = Path(output).resolve(), Path(grid).resolve()
    if output.exists() or _state_dir(output).exists():
        raise FileExistsError("A lock and its execution state cannot be overwritten")
    cwd = _path(launch_cwd or Path.cwd(), Path.cwd())
    source = source_record(code_root or _default_code_root(), app)
    matched_groups = matched_groups or []
    required_external = sorted(set(required_external or []))
    if scope == "full_comparison" and not matched_groups:
        raise ValueError("Full comparison requires explicit matched-family budgets")
    if app == "kd":
        content = _read(grid)
        rows = content.get("rows", content.get("entries"))
        if rows is None:
            raise ValueError("KD grid manifest needs rows or final-grid entries")
        if run_root is None and any(not r.get("run_dir") for r in rows):
            raise ValueError("KD grid needs --run-root, or run_dir in every row")
        primary_metric = primary_metric or "top1"
        if primary_metric != "top1":
            raise ValueError("Native KD selects checkpoints by CAL top1; a different primary metric needs a new protocol")
    else:
        with grid.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    jobs, manifests, benchmarks, assets, asset_cache = [], {}, set(), {}, {}
    for row in rows:
        raw_path = Path(row["config"])
        options = [raw_path] if raw_path.is_absolute() else ([grid.parent / raw_path, cwd / raw_path] if app == "kd" else [cwd / raw_path, grid.parent / raw_path])
        config_path = next((p.resolve() for p in options if p.is_file()), None)
        if config_path is None:
            raise FileNotFoundError(f"Grid config not found: {row['config']}")
        raw = _read(config_path)
        cfg = _effective_config(raw, app, cwd)
        family, seed = _family(cfg, app), cfg["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Training seed must be an integer")
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in family):
            raise ValueError("Method labels may contain only letters, digits, underscore, dot, hyphen")
        if row.get("method", family) != family or int(row.get("seed", seed)) != seed:
            raise ValueError("Grid method/seed does not match its config")
        candidate = _candidate(cfg, app)
        run_dir = _path(row.get("run_dir") or (Path(run_root) / config_path.stem), cwd) if app == "kd" else _path(cfg["output_dir"], cwd)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("Register before candidate training; amended grids also require fresh run directories")
        manifest_path = Path(cfg["data"]["manifest"]) if app == "kd" else _path(cfg["data_dir"], cwd) / "manifest.json"
        manifest = _read(manifest_path)
        manifests[str(manifest_path)] = {"path": str(manifest_path), "sha256": file_hash(manifest_path)}
        if app == "aigiqa":
            metric = cfg.get("selection_metric", "srcc")
            primary_metric = primary_metric or metric
            if metric != primary_metric or metric not in ("srcc", "within_prompt_pairwise_macro"):
                raise ValueError("Every AIG candidate must use the same preregistered CAL metric")
            for filename in ("fit.csv", "cal.csv"):
                path = manifest_path.parent / filename
                actual = file_hash(path)
                if actual != manifest["input_and_label_hashes"][filename]:
                    raise ValueError("Prepared FIT/CAL table hash mismatch")
                assets[str(path)] = {"path": str(path), "kind": "file", "sha256": actual}
        benchmarks.add(canonical_hash(_benchmark(cfg, app, cwd)))
        for asset in _assets(cfg, app, cwd, asset_cache):
            assets[asset["path"]] = asset
        jobs.append({"job_id": f"{family}__{candidate[:16]}__seed{seed}", "family": family,
                     "candidate": candidate, "seed": seed, "config_path": str(config_path),
                     "config_file_sha256": file_hash(config_path), "effective_config_sha256": canonical_hash(cfg),
                     "effective_config": cfg, "run_dir": str(run_dir),
                     "expected_split_sha256": manifest.get("split_sha256")})
    if len(manifests) != 1 or len(benchmarks) != 1:
        raise ValueError("One lock requires one source manifest, target, model pair, and training budget")
    seeds, budget = _validate_design(jobs, matched_groups)
    required_native = sorted(set(required_native or budget))
    if not set(required_native) <= set(budget):
        raise ValueError("A required native baseline is absent from the expanded grid")
    if set(required_native) & set(required_external):
        raise ValueError("Native and external baseline IDs must be distinct")
    evaluation_inputs = [_input_record(_path(p, cwd)) for p in (inputs or [])]
    if app == "aigiqa" and not evaluation_inputs:
        raise ValueError("Preregister label-free AIG TEST/cross-dataset input files before CAL")
    prior = None
    if prior_cal_selection:
        selection_path = _path(prior_cal_selection, cwd)
        selection = _read(selection_path)
        selection_body = dict(selection); selection_sha = selection_body.pop("selection_sha256", None)
        if selection.get("schema") != "vcs_global_cal_selection_v1" or canonical_hash(selection_body) != selection_sha:
            raise ValueError("Prior CAL selection hash/schema is invalid")
        parent_lock = check(selection["lock_path"])
        if _known_test_or_release(selection["lock_path"], parent_lock["jobs"]):
            raise ValueError("Prior selection must be a CAL-only study with no TEST release")
        parent_records = [_completed_record(selection["lock_path"], parent_lock, j) for j in parent_lock["jobs"]]
        if parent_records != selection["all_candidate_records"] or parent_lock["lock_sha256"] != selection["lock_sha256"]:
            raise ValueError("Prior CAL selection evidence changed")
        if parent_lock["app"] != app or parent_lock["source"]["sha256"] != source["sha256"] or parent_lock["source_manifest"] != next(iter(manifests.values())):
            raise ValueError("Final seed study must use the same source, application and data manifest as its CAL pilot")
        chosen = {r["family"]: r["candidate"] for r in parent_records if r["job_id"] in selection["selected_job_ids"]}
        if any(chosen.get(j["family"]) != j["candidate"] for j in jobs):
            raise ValueError("Final configs differ from the prior CAL winner in that family")
        if set(seeds) & set(parent_lock["training_seeds"]):
            raise ValueError("Final seed study must use new student training seeds")
        prior = {"path": str(selection_path), "sha256": file_hash(selection_path), "parent_lock_sha256": parent_lock["lock_sha256"],
                 "parent_training_seeds": parent_lock["training_seeds"], "selected_candidates": chosen}
    amendment = None
    if amend_from:
        if not amend_reason or not amend_reason.strip():
            raise ValueError("An amendment needs an explicit scientific/operational reason")
        parent_path = Path(amend_from).resolve(); parent = _load_lock(parent_path)
        if parent["app"] != app or _known_test_or_release(parent_path, parent["jobs"]):
            raise ValueError("Cannot amend after a TEST release or across applications")
        if (_state_dir(parent_path) / "superseded.json").exists():
            raise ValueError("Parent lock was already superseded")
        parent_statuses = []
        for job in parent["jobs"]:
            receipt = _receipt_path(parent_path, job)
            status = _read(receipt).get("status") if receipt.exists() else "NOT_STARTED"
            if status == "RUNNING":
                raise ValueError("Stop and record any running job before amending")
            parent_statuses.append({"job_id": job["job_id"], "status": status,
                                    "receipt_sha256": file_hash(receipt) if receipt.exists() else None})
        amendment = {"parent_lock_path": str(parent_path), "parent_lock_sha256": parent["lock_sha256"],
                     "reason": amend_reason, "parent_candidate_statuses": parent_statuses,
                     "policy": "Fresh output directories; all candidates retrain under this new lock. No automatic failed-run deletion."}
    elif amend_reason:
        raise ValueError("--amend-reason requires --amend-from")
    body = {"schema": LOCK_SCHEMA, "app": app, "scope": scope, "created_unix_time": time.time(),
            "launch_cwd": str(cwd), "grid": {"path": str(grid), "sha256": file_hash(grid)},
            "source": source, "source_manifest": next(iter(manifests.values())), "assets": sorted(assets.values(), key=lambda x: x["path"]),
            "primary_metric": {"name": primary_metric, "direction": "maximize",
                               "selection": "mean_CAL_across_all_preregistered_seeds",
                               "tie_break": "mean_CAL_nll_then_candidate_hash" if app == "kd" else "candidate_hash"},
            "training_seeds": seeds, "matched_family_groups": matched_groups, "family_budgets": budget,
            "required_native": required_native, "required_external": required_external,
            "external_status_at_registration": {name: "PENDING_READINESS" for name in required_external},
            "jobs": sorted(jobs, key=lambda j: j["job_id"]), "evaluation_inputs": evaluation_inputs,
            "amendment": amendment, "prior_cal_selection": prior,
            "claim_scope": "DIAGNOSTIC_ONLY" if scope == "pilot" else "DECLARED_COMPARISON_ONLY_AFTER_ALL_EVALUATIONS_AND_ANALYSIS",
            "limitations": ["No filesystem access control or proof of unseen labels.",
                            "Equal candidate counts do not by themselves establish equal gradient scale or GPU cost.",
                            "Changing this lock after TEST creates a new exploratory study, not a fresh confirmation set."]}
    body["lock_sha256"] = canonical_hash(body)
    _write_new(output, body)
    if amendment:
        _write_new(_state_dir(amend_from) / "superseded.json", {"new_lock": str(output), "new_lock_sha256": body["lock_sha256"], "reason": amend_reason})
    return body


def check(lock_path, *, allow_released=False):
    lock = _load_lock(lock_path)
    state = _state_dir(lock_path)
    if (state / "superseded.json").exists():
        raise ValueError("Experiment lock was superseded by an explicit amendment")
    if not allow_released and (state / "release.json").exists():
        raise ValueError("TEST was already released; training and resealing are forbidden")
    if file_hash(lock["grid"]["path"]) != lock["grid"]["sha256"]:
        raise ValueError("Expanded grid changed after registration")
    if source_record(lock["source"]["root"], lock["app"]) != lock["source"]:
        raise ValueError("Application/shared source hash changed after registration")
    for record in [lock["source_manifest"], *lock["evaluation_inputs"]]:
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Data manifest or evaluation input hash changed")
    prior = lock.get("prior_cal_selection")
    if prior and file_hash(prior["path"]) != prior["sha256"]:
        raise ValueError("Prior CAL selection evidence hash changed")
    for asset in lock["assets"]:
        actual = _tree_hash(Path(asset["path"])) if asset["kind"] == "python_tree" else file_hash(asset["path"])
        if actual != asset["sha256"]:
            raise ValueError("Checkpoint/provenance/source/FIT-CAL asset hash changed")
    for job in lock["jobs"]:
        if file_hash(job["config_path"]) != job["config_file_sha256"]:
            raise ValueError("Planned config changed after registration")
        actual = _effective_config(_read(job["config_path"]), lock["app"], lock["launch_cwd"])
        if canonical_hash(actual) != job["effective_config_sha256"]:
            raise ValueError("Planned effective config changed")
    seeds, budgets = _validate_design(lock["jobs"], lock["matched_family_groups"])
    if seeds != lock["training_seeds"] or budgets != lock["family_budgets"]:
        raise ValueError("Registered seed or budget metadata is inconsistent")
    return lock


def _job(lock, job_id):
    matches = [j for j in lock["jobs"] if j["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError("Job ID is not uniquely registered")
    return matches[0]


def _native_train(app, job):
    if app == "kd":
        from kd.runner import train
        return train(job["config_path"], job["run_dir"])
    from aigiqa.engine import train
    return train(_read(job["config_path"]))


def _artifact_names(app):
    return ["run.json", "best.pt", "best_cal_metrics.json", "best_cal_predictions.csv", "history.json", "resolved_config.json"] if app == "kd" else ["result.json", "best.pt", "config.json", "cal_predictions.csv", "history.json"]


def train_job(lock_path, job_id):
    lock = check(lock_path)
    job = _job(lock, job_id)
    receipt_path = _receipt_path(lock_path, job)
    if receipt_path.exists():
        raise ValueError("Job already attempted; a failed/interrupted run needs an explicit pre-TEST amendment and fresh grid")
    run_dir = Path(job["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Candidate output is not fresh")
    receipt = {"schema": "vcs_experiment_job_receipt_v1", "status": "RUNNING", "job_id": job_id,
               "lock_sha256": lock["lock_sha256"], "source_sha256": lock["source"]["sha256"],
               "config_file_sha256": job["config_file_sha256"], "started_unix_time": time.time()}
    _write_new(receipt_path, receipt)
    try:
        with _cwd(lock["launch_cwd"]):
            _native_train(lock["app"], job)
        check(lock_path)
        result_path = run_dir / ("run.json" if lock["app"] == "kd" else "result.json")
        receipt.update(status="COMPLETED", result_sha256=file_hash(result_path), finished_unix_time=time.time(),
                       artifact_sha256={name: file_hash(run_dir / name) for name in _artifact_names(lock["app"])})
        _write_state(receipt_path, receipt)
        _completed_record(lock_path, lock, job)
    except BaseException as exc:
        receipt.update(status="FAILED", error_type=type(exc).__name__, error=str(exc), finished_unix_time=time.time())
        _write_state(receipt_path, receipt)
        raise
    return receipt


def record_interruption(lock_path, job_id, reason):
    """After the server operator stops a crashed/killed process, record its state."""
    lock = _load_lock(lock_path)
    if _known_test_or_release(lock_path, lock["jobs"]) or not reason.strip():
        raise ValueError("Interruption requires a reason and must precede TEST release")
    receipt_path = _receipt_path(lock_path, _job(lock, job_id))
    receipt = _read(receipt_path)
    if receipt["status"] != "RUNNING":
        raise ValueError("Only a RUNNING receipt can be marked interrupted")
    receipt.update(status="INTERRUPTED", operator_reason=reason, finished_unix_time=time.time(),
                   caveat="Operator records that the process has stopped; this command does not terminate it.")
    _write_state(receipt_path, receipt)
    return receipt


def _completed_record(lock_path, lock, job):
    run_dir = Path(job["run_dir"])
    receipt_path = _receipt_path(lock_path, job)
    if not receipt_path.is_file():
        raise ValueError(f"Planned candidate has no global training receipt: {job['job_id']}")
    receipt = _read(receipt_path)
    for key, expected in (("status", "COMPLETED"), ("job_id", job["job_id"]), ("lock_sha256", lock["lock_sha256"]),
                          ("source_sha256", lock["source"]["sha256"]), ("config_file_sha256", job["config_file_sha256"])):
        if receipt.get(key) != expected:
            raise ValueError(f"Planned candidate incomplete or receipt mismatch: {job['job_id']} ({key})")
    app = lock["app"]
    expected_artifacts = receipt.get("artifact_sha256", {})
    if set(expected_artifacts) != set(_artifact_names(app)):
        raise ValueError("Training receipt must bind all CAL metric/config/checkpoint/history artifacts")
    if any(file_hash(run_dir / name) != sha for name, sha in expected_artifacts.items()):
        raise ValueError("Completed candidate artifact hash changed")
    result_path = run_dir / ("run.json" if app == "kd" else "result.json")
    if file_hash(result_path) != receipt["result_sha256"]:
        raise ValueError("Completed candidate result hash changed")
    result = _read(result_path)
    if app == "kd":
        cfg = result["config"]
        if result.get("status") != "complete" or result.get("teacher_training") is not False:
            raise ValueError("A KD student candidate is not complete")
        if result["source_sha256"] != lock["source"]["sha256"] or result["split_sha256"] != job["expected_split_sha256"]:
            raise ValueError("KD source/split does not match the planned experiment")
        if result.get("selection_data_role") != "CAL" or result.get("test_loaded_during_train") is not False:
            raise ValueError("Candidate used the wrong data role")
        if result["config_sha256"] != canonical_hash(cfg) or result["epochs_completed"] != cfg["training"]["epochs"]:
            raise ValueError("KD config/epoch budget is inconsistent")
        checkpoint_sha = result["best_checkpoint_sha256"]
        predictions = run_dir / "best_cal_predictions.csv"
        if file_hash(predictions) != result["best_cal_predictions_sha256"]:
            raise ValueError("KD CAL predictions changed")
        metric_path = run_dir / "best_cal_metrics.json"
        metrics = _read(metric_path)
        nll = metrics["nll"]
    else:
        cfg_path = run_dir / "config.json"; cfg = _read(cfg_path)
        if result.get("status") != "COMPLETED_FIT_CAL_ONLY" or (run_dir / "FAILED.json").exists():
            raise ValueError("An AIG candidate is not complete")
        if result.get("test_labels_read") is not False or result.get("selection_metric") != lock["primary_metric"]["name"]:
            raise ValueError("AIG data role/selection metric differs from the plan")
        if result.get("seed") != job["seed"] or result.get("target") != cfg["target"]:
            raise ValueError("AIG result seed/target differs from config")
        if result["prepared_manifest_sha256"] != lock["source_manifest"]["sha256"] or file_hash(cfg_path) != result["config_sha256"]:
            raise ValueError("AIG config/prepared manifest changed")
        history = _read(run_dir / "history.json")
        if len(history) != cfg.get("epochs", 20):
            raise ValueError("AIG candidate did not complete its planned epoch budget")
        checkpoint_sha = result["checkpoint_sha256"]
        predictions = run_dir / "cal_predictions.csv"
        metrics, metric_path, nll = result["cal_metrics"], result_path, None
    if canonical_hash(cfg) != job["effective_config_sha256"] or cfg["seed"] != job["seed"] or _family(cfg, app) != job["family"]:
        raise ValueError("Actual run config/seed/family differs from the preregistered candidate")
    if file_hash(run_dir / "best.pt") != checkpoint_sha:
        raise ValueError("Completed checkpoint hash changed")
    metric = metrics[lock["primary_metric"]["name"]]
    if not isinstance(metric, (float, int)) or not math.isfinite(metric) or (nll is not None and not math.isfinite(nll)):
        raise ValueError("CAL selection metric must be finite; a failed candidate cannot silently disappear")
    return {"job_id": job["job_id"], "family": job["family"], "candidate": job["candidate"], "seed": job["seed"],
            "run_dir": job["run_dir"], "metric": metric, "nll": nll,
            "result_sha256": file_hash(result_path), "metrics_sha256": file_hash(metric_path),
            "cal_predictions_sha256": file_hash(predictions), "checkpoint_sha256": checkpoint_sha,
            "receipt_sha256": file_hash(receipt_path)}


def _select(records):
    families = defaultdict(lambda: defaultdict(list))
    for record in records:
        families[record["family"]][record["candidate"]].append(record)
    table, chosen = [], []
    for family, candidates in sorted(families.items()):
        ranking = []
        for signature, rr in sorted(candidates.items()):
            mean = statistics.mean(r["metric"] for r in rr)
            nll = statistics.mean(r["nll"] for r in rr) if rr[0]["nll"] is not None else 0.
            ranking.append((-mean, nll, signature, rr))
            table.append({"family": family, "candidate": signature, "mean_cal_metric": mean,
                          "mean_cal_nll": nll if rr[0]["nll"] is not None else None,
                          "seeds": sorted(r["seed"] for r in rr)})
        chosen.extend(sorted(ranking, key=lambda r: r[:3])[0][3])
    return table, chosen


def select_cal(lock_path, output):
    """Complete-grid CAL selection only; does not create a native TEST freeze."""
    lock = check(lock_path)
    if _known_test_or_release(lock_path, lock["jobs"]):
        raise ValueError("CAL-only selection must precede TEST release")
    records = [_completed_record(lock_path, lock, j) for j in lock["jobs"]]
    table, chosen = _select(records)
    body = {"schema": "vcs_global_cal_selection_v1", "lock_path": str(Path(lock_path).resolve()),
            "lock_sha256": lock["lock_sha256"], "app": lock["app"], "primary_metric": lock["primary_metric"],
            "all_candidate_records": records, "candidate_table": table,
            "selected_job_ids": [r["job_id"] for r in chosen], "test_released": False,
            "created_unix_time": time.time()}
    body["selection_sha256"] = canonical_hash(body)
    _write_new(output, body)
    return body


def _external_readiness(lock, status_path):
    statuses = _read(status_path) if status_path else {}
    if set(statuses) - set(lock["required_external"]):
        raise ValueError("Unregistered external baseline supplied; amend the plan before TEST")
    records = {}
    for method in lock["required_external"]:
        row = statuses.get(method, {"status": "PENDING"})
        if row.get("status") == "READY_FOR_FROZEN_EVALUATION":
            required = {"source_url", "source_revision", "checkpoint_path", "checkpoint_sha256", "fit_cal_provenance_path", "target_used_for_selection", "extra_inputs", "score_direction"}
            if not required <= set(row) or row["target_used_for_selection"] is not False:
                raise ValueError(f"External readiness needs version, checkpoint and source-only CAL provenance: {method}")
            row = dict(row)
            for field in ("checkpoint_path", "fit_cal_provenance_path"):
                row[field] = str(_path(row[field], lock["launch_cwd"]))
            if file_hash(row["checkpoint_path"]) != row["checkpoint_sha256"]:
                raise ValueError("External checkpoint hash mismatch")
            row["fit_cal_provenance_sha256"] = file_hash(row["fit_cal_provenance_path"])
        elif lock["scope"] == "full_comparison":
            raise ValueError(f"Required external baseline is not ready; full comparison TEST release blocked: {method}")
        records[method] = row
    return records


def _native_freeze(lock, records, chosen, output):
    with _cwd(lock["launch_cwd"]):
        if lock["app"] == "kd":
            from kd.runner import freeze
            return freeze([r["run_dir"] for r in chosen], output)
        from aigiqa.engine import freeze
        return freeze([r["run_dir"] for r in records], [r["path"] for r in lock["evaluation_inputs"]], output)


def seal(lock_path, output, *, external_statuses=None):
    """Require every planned candidate, select on CAL, then create native freeze."""
    output = Path(output).resolve()
    child = output.with_name(output.stem + ".native_freeze.json")
    if output.exists() or child.exists():
        raise FileExistsError("A release and native freeze cannot be overwritten")
    lock = check(lock_path)
    if _known_test_or_release(lock_path, lock["jobs"]):
        raise ValueError("Known TEST output/release predates this seal")
    records = [_completed_record(lock_path, lock, j) for j in lock["jobs"]]
    table, chosen = _select(records)
    externals = _external_readiness(lock, external_statuses)
    marker = _state_dir(lock_path) / "seal_in_progress.json"
    _write_new(marker, {"output": str(output), "lock_sha256": lock["lock_sha256"]})
    try:
        native = _native_freeze(lock, records, chosen, child)
        native_paths = {r["run_path"] for r in native["entries"]} if lock["app"] == "kd" else {str(Path(r["checkpoint"]).parent.resolve()) for r in native["selected_runs"]}
        if native_paths != {r["run_dir"] for r in chosen}:
            raise ValueError("Native CAL winner differs from global selection; no global TEST release was issued")
        body = {"schema": RELEASE_SCHEMA, "lock_path": str(Path(lock_path).resolve()), "lock_sha256": lock["lock_sha256"],
                "app": lock["app"], "scope": lock["scope"], "primary_metric": lock["primary_metric"],
                "all_candidate_records": records, "candidate_table": table, "selected_job_ids": [r["job_id"] for r in chosen],
                "native_freeze": {"path": str(child), "sha256": file_hash(child)},
                "required_native_status": {f: "ALL_PLANNED_CAL_CANDIDATES_COMPLETED" for f in lock["required_native"]},
                "external_readiness": externals, "evaluation_inputs": lock["evaluation_inputs"],
                "created_unix_time": time.time(), "test_labels_read_for_selection": False,
                "claim_status": "PILOT_DIAGNOSTIC_ONLY" if lock["scope"] == "pilot" else "NO_CLAIM_YET__REQUIRED_TEST_RESULTS_AND_ANALYSIS_PENDING"}
        body["release_sha256"] = canonical_hash(body)
        _write_new(output, body)
        _write_new(_state_dir(lock_path) / "release.json", {"path": str(output), "release_sha256": body["release_sha256"]})
        return body
    finally:
        marker.unlink(missing_ok=True)


def check_release(release_path):
    release = _read(release_path)
    body = dict(release); stored = body.pop("release_sha256", None)
    if release.get("schema") != RELEASE_SCHEMA or canonical_hash(body) != stored:
        raise ValueError("TEST release hash changed")
    lock = check(release["lock_path"], allow_released=True)
    state = _read(_state_dir(release["lock_path"]) / "release.json")
    if lock["lock_sha256"] != release["lock_sha256"] or state["release_sha256"] != stored:
        raise ValueError("Release does not match its parent experiment lock/state")
    if file_hash(release["native_freeze"]["path"]) != release["native_freeze"]["sha256"]:
        raise ValueError("Native freeze hash changed")
    records = [_completed_record(release["lock_path"], lock, j) for j in lock["jobs"]]
    if records != release["all_candidate_records"]:
        raise ValueError("CAL candidate evidence changed after release")
    for record in release["external_readiness"].values():
        if record.get("status") == "READY_FOR_FROZEN_EVALUATION":
            for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"), ("fit_cal_provenance_path", "fit_cal_provenance_sha256")):
                if file_hash(record[path_key]) != record[hash_key]:
                    raise ValueError("External frozen checkpoint/provenance changed")
    return release


def evaluate_kd(release_path, job_id):
    release = check_release(release_path)
    if release["app"] != "kd" or job_id not in release["selected_job_ids"]:
        raise ValueError("Only a globally selected KD job can open TEST")
    lock = _load_lock(release["lock_path"])
    from kd.runner import evaluate
    with _cwd(lock["launch_cwd"]):
        return evaluate(_job(lock, job_id)["run_dir"], release["native_freeze"]["path"])


def predict_aigiqa(release_path, run_key, inputs, image_root, output, device=None):
    release = check_release(release_path)
    if release["app"] != "aigiqa":
        raise ValueError("Wrong release application")
    from aigiqa.engine import predict
    lock = _load_lock(release["lock_path"])
    with _cwd(lock["launch_cwd"]):
        return predict(release["native_freeze"]["path"], run_key, inputs, image_root, output, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("register")
    p.add_argument("--app", choices=("kd", "aigiqa"), required=True)
    p.add_argument("--grid", required=True); p.add_argument("--output", required=True)
    p.add_argument("--run-root"); p.add_argument("--scope", choices=("pilot", "full_comparison"), default="pilot")
    p.add_argument("--match", action="append", default=[], help="Comma-separated matched family group; may repeat")
    p.add_argument("--required-native", nargs="+", default=[]); p.add_argument("--required-external", nargs="+", default=[])
    p.add_argument("--inputs", nargs="+", default=[]); p.add_argument("--primary-metric")
    p.add_argument("--code-root"); p.add_argument("--launch-cwd")
    p.add_argument("--amend-from"); p.add_argument("--amend-reason")
    p.add_argument("--prior-cal-selection", help="Bind final fresh-seed configs to a complete CAL-only pilot selection")
    for name in ("check", "list-jobs", "train", "record-interruption"):
        p = commands.add_parser(name); p.add_argument("--lock", required=True)
        if name in ("train", "record-interruption"):
            p.add_argument("--job-id", required=True)
        if name == "record-interruption":
            p.add_argument("--reason", required=True)
    p = commands.add_parser("seal"); p.add_argument("--lock", required=True); p.add_argument("--output", required=True); p.add_argument("--external-statuses")
    p = commands.add_parser("select-cal"); p.add_argument("--lock", required=True); p.add_argument("--output", required=True)
    for name in ("check-release", "evaluate-kd", "predict-aigiqa"):
        p = commands.add_parser(name); p.add_argument("--release", required=True)
        if name == "evaluate-kd":
            p.add_argument("--job-id", required=True)
        elif name == "predict-aigiqa":
            for field in ("run-key", "inputs", "image-root", "output"):
                p.add_argument("--" + field, required=True)
            p.add_argument("--device")
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "register":
        args["matched_groups"] = [[f.strip() for f in s.split(",") if f.strip()] for s in args.pop("match")]
        result = register(**args)
        result = {"lock": str(Path(args["output"]).resolve()), "lock_sha256": result["lock_sha256"], "jobs": len(result["jobs"]), "scope": result["scope"]}
    elif command in ("check", "list-jobs"):
        lock = check(args["lock"])
        result = [{"job_id": j["job_id"], "config": j["config_path"], "run_dir": j["run_dir"]} for j in lock["jobs"]] if command == "list-jobs" else {"status": "VALID", "lock_sha256": lock["lock_sha256"], "jobs": len(lock["jobs"])}
    elif command == "train":
        result = train_job(args["lock"], args["job_id"])
    elif command == "record-interruption":
        result = record_interruption(args["lock"], args["job_id"], args["reason"])
    elif command == "seal":
        value = seal(args["lock"], args["output"], external_statuses=args["external_statuses"])
        result = {"release": args["output"], "release_sha256": value["release_sha256"], "native_freeze": value["native_freeze"], "selected_jobs": value["selected_job_ids"], "claim_status": value["claim_status"]}
    elif command == "select-cal":
        value = select_cal(args["lock"], args["output"])
        result = {"selection": args["output"], "selection_sha256": value["selection_sha256"], "selected_jobs": value["selected_job_ids"], "test_released": False}
    elif command == "check-release":
        value = check_release(args["release"])
        result = {"status": "VALID", "release_sha256": value["release_sha256"], "claim_status": value["claim_status"]}
    elif command == "evaluate-kd":
        result = evaluate_kd(args["release"], args["job_id"])
    else:
        result = predict_aigiqa(args["release"], args["run_key"], args["inputs"], args["image_root"], args["output"], args["device"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
