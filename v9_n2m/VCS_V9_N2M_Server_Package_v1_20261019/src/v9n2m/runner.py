"""Full N2M SGR training against an immutable experiment, one arm at a time."""
from __future__ import annotations

import fcntl
import gc
import math
import os
import signal
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from pathlib import Path

from .common import PACKAGE_ROOT, assert_same_file, read_json, sha256_file, utc_now, write_json
from .identity import validate_complete_unit, verify_training_inputs
from .prepare import verify_experiment
from .runtime import collect_environment, configure_precision, environment_differences

SCIENTIFIC_CONFIG = {
    "img_dim": 2048, "word_dim": 300, "embed_size": 1024, "sim_dim": 256,
    "num_layers": 1, "bi_gru": True, "num_region": 36, "sgr_step": 3,
    "margin": 0.2, "learning_rate": 2e-4, "grad_clip": 2.0,
    "batch_size": 128, "lr_update": 30, "num_epochs": 40, "warmup_epochs": 5,
}


def check_config(cfg):
    mapping = asdict(cfg) if not isinstance(cfg, dict) else cfg
    for key, expected in SCIENTIFIC_CONFIG.items():
        if mapping.get(key) != expected:
            raise ValueError(f"Frozen host configuration differs at {key}: {mapping.get(key)} != {expected}")
    if mapping.get("to_verify"):
        raise ValueError(f"Host has unresolved recipe fields: {mapping['to_verify']}")
    return mapping


def manifests_compatible(previous, current):
    """Input hashes/config are exact; derived float64 diagnostics permit roundoff."""
    if set(previous) != set(current):
        return False
    derived = {"sum_w_U", "mean_v_full_pool", "ess_U"}
    for key in previous:
        if key in derived:
            a, b = previous[key], current[key]
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-12):
                return False
        elif previous[key] != current[key]:
            return False
    return True


@contextmanager
def unit_lock(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "UNIT_LOCK").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process already owns unit {out}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def attempts_summary(out):
    entries = []
    for p in sorted((Path(out) / "attempts").glob("*.json")):
        rec = read_json(p)
        entries.append({"file": str(p), "sha256": sha256_file(p), **rec})
    known = sum(float(x["elapsed_s"]) for x in entries if x.get("elapsed_s") is not None)
    unknown = [x["attempt_id"] for x in entries if x.get("elapsed_s") is None]
    return {"attempts": len(entries), "attempt_wall_s_known": known,
            "missing_duration_attempts": unknown, "total_duration_complete": not unknown,
            "scope": "Process walltime incl verification/load/train/DEV/checkpoint work; Slurm allocation totals are collected separately without counting a shared job twice.",
            "entries": entries}


def load_training_data(exp):
    runtime = Path(exp["runtime_dir"])
    runtime_src = str(runtime / "src")
    if runtime_src not in sys.path:
        sys.path.insert(0, runtime_src)
    from v9.n2.data import eval_arrays, load_precomp, load_vocab
    from v9.n2.fixtures import collate_pairs
    from v9.n2.teacher import load_host_lock
    import v9.n2.train as train_module
    if Path(train_module.__file__).resolve() != runtime / "src/v9/n2/train.py":
        raise ValueError("A different v9 module was already imported into this process")
    vocab = load_vocab(exp["inputs"]["vocab"])
    cfg = load_host_lock(runtime / "HOST_LOCK.yaml", "f30k_precomp", len(vocab))
    check_config(cfg)
    train = load_precomp(exp["inputs"]["data"], "train", vocab, cache_dir=exp["inputs"]["cache"])
    dev = load_precomp(exp["inputs"]["data"], "dev", vocab, cache_dir=exp["inputs"]["cache"])
    if train["n_images"] != 29000 or len(train["captions"]) != 145000 or dev["n_images"] != 1014 or len(dev["captions"]) != 5070:
        raise ValueError("Data loader did not produce the full prescribed train+DEV splits")
    dev_images, dev_captions = eval_arrays(dev)
    return cfg, train, dev_images, dev_captions, partial(collate_pairs, train["images"], train["captions"])


def _unit_identity(exp, unit, experiment_path):
    return {"unit_id": unit["unit_id"], "seed": unit["seed"], "arm": unit["arm"],
            "method": unit["arm"], "scale": "sum", "epochs": 40,
            "protocol_sha256": exp["protocol_sha256"], "experiment_sha256": sha256_file(experiment_path),
            "base_freeze_sha256": exp["base_freeze_sha256"],
            "effective_runtime_digest": exp["source_identity"]["effective_digest"],
            "observed_sha256": unit["observed_sha256"], "weights_sha256": unit["weights_sha256"],
            "weight_lock_sha256": unit["weight_lock_sha256"],
            "selection": "best full DEV rSum; earliest tie", "dataset": "f30k full train",
            "design_rho": 0.4, "learning_rate_multiplier": 1.0, "oracle": False,
            "test_read": False, "reuse": False, "order": unit["order"]}


def _check_predecessors(exp, unit, experiment_path):
    for previous in exp["units"]:
        if previous["seed"] == unit["seed"] and previous["order"] < unit["order"] and previous["action"] == "RUN":
            p = Path(previous["output_dir"])
            try:
                validate_complete_unit(p, expected_identity=_unit_identity(exp, previous, experiment_path), check_binary=True)
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(f"Prespecified same-seed order requires {previous['unit_id']} to finish first") from exc


def run_unit(experiment_path, unit_id):
    from .controls import load_observed, load_weights
    experiment_path = Path(experiment_path).resolve()
    exp = verify_experiment(experiment_path)
    unit = next((u for u in exp["units"] if u["unit_id"] == unit_id), None)
    if unit is None:
        raise ValueError(f"Unknown frozen unit {unit_id}")
    if unit["action"] == "REUSE":
        verified = validate_complete_unit(unit["reference_source_dir"], expected_identity={
            "seed": unit["seed"], "method": "VCS_Weight", "scale": "sum",
            "freeze_sha256": exp["base_freeze_sha256"]}, check_binary=True)
        return {"status": "REUSED", "unit_id": unit_id, "verification": verified}
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full SGR training must run in the server's allocated Slurm GPU job")
    out = Path(unit["output_dir"])
    identity = _unit_identity(exp, unit, experiment_path)
    with unit_lock(out):
        if not (out / "identity.json").exists() and any((out / name).exists() for name in
                ("last.pt", "best.pt", "done.json", "run_manifest.json", "train_log.json", "full_result.json")):
            raise ValueError("Unidentified scientific artifacts cannot be adopted by a new control; each arm must start from its own seeded initialization")
        if (out / "identity.json").exists() and read_json(out / "identity.json") != identity:
            raise ValueError("Existing run directory belongs to a different unit/experiment")
        if (out / "full_result.json").exists():
            verified = validate_complete_unit(out, expected_identity=identity, check_binary=True)
            return {"status": "ALREADY_COMPLETE", "unit_id": unit_id, "verification": verified}
        _check_predecessors(exp, unit, experiment_path)
        write_json(out / "identity.json", identity)
        attempt_id = str(uuid.uuid4())
        attempt_path = out / "attempts" / f"{attempt_id}.json"
        marker = out / "STOP_REQUESTED"
        if marker.exists():
            archived = out / "attempts" / f"{attempt_id}.prior_stop_request.txt"
            archived.parent.mkdir(parents=True, exist_ok=True)
            marker.replace(archived)
        started = time.monotonic()
        attempt = {"attempt_id": attempt_id, "unit_id": unit_id, "job_id": os.environ.get("SLURM_JOB_ID"),
                   "start_utc": utc_now(), "status": "RUNNING", "elapsed_s": None,
                   "experiment_sha256": sha256_file(experiment_path), "checkpoint_at_start_sha256": None}
        if (out / "last.pt").is_file():
            attempt["checkpoint_at_start_sha256"] = sha256_file(out / "last.pt")
        write_json(attempt_path, attempt)
        signals = {"stop_requested": False}
        def request_stop(signum, frame):
            signals["stop_requested"] = True
            marker.write_text(f"signal={signum}; utc={utc_now()}; attempt={attempt_id}\n", encoding="utf-8")
        def terminate(signum, frame):
            request_stop(signum, frame)
            raise InterruptedError("Allocation terminating; the last complete epoch checkpoint is retained")
        previous_handlers = {signal.SIGUSR1: signal.getsignal(signal.SIGUSR1),
                             signal.SIGTERM: signal.getsignal(signal.SIGTERM)}
        signal.signal(signal.SIGUSR1, request_stop)
        signal.signal(signal.SIGTERM, terminate)
        published = False
        try:
            configure_precision()
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("An allocated CUDA device is required for a full unit")
            hardware = collect_environment()
            differences = environment_differences(exp["target_environment"], hardware)
            if differences:
                raise ValueError(f"Allocated environment differs from frozen paired environment: {differences}")
            attempt["hardware"] = hardware
            write_json(attempt_path, attempt)
            verify_training_inputs(exp["base_handoff"], exp["inputs"], seeds=(unit["seed"],))
            rows = load_observed(unit["observed_path"])
            u_ids = [r["pair_id"] for r in rows if r["role"] == "U"]
            if len(rows) != 145000 or len(u_ids) != 130500:
                raise ValueError("Wrong inherited pool or anchor budget")
            weights = load_weights(unit["weights_path"], u_ids, unit["weight_lock_path"])
            lock = read_json(unit["weight_lock_path"])
            cfg, train, dev_images, dev_captions, collate = load_training_data(exp)
            from v9.n2.train import WeightedStudent
            previous_manifest_bytes = (out / "run_manifest.json").read_bytes() if (out / "run_manifest.json").exists() else None
            previous_manifest = read_json(out / "run_manifest.json") if previous_manifest_bytes is not None else None
            student = WeightedStudent(cfg, rows, weights, collate, dev_images, dev_captions, out, unit["seed"], 1.0,
                                      device="cuda", weight_lock=lock, formal=False, loss_scale="sum", grad_log_window=(200, 200))
            if previous_manifest is not None:
                # The inherited constructor writes its manifest. Preserve the previous exact
                # bytes so 1-ULP ESS diagnostics cannot invalidate a correct checkpoint receipt.
                tmp = out / "run_manifest.restore.tmp"
                tmp.write_bytes(previous_manifest_bytes)
                tmp.replace(out / "run_manifest.json")
                if not manifests_compatible(previous_manifest, student.manifest):
                    raise ValueError("Resume training manifest differs beyond diagnostic roundoff")
            done = student.fit(40, log_every=20)
            if done.get("status") == "INTERRUPTED":
                attempt.update(status="INTERRUPTED", end_utc=utc_now(), elapsed_s=time.monotonic()-started,
                               completed_epochs=done["epochs_run"], completed_updates=done["global_step"])
                write_json(attempt_path, attempt)
                return {"status": "INTERRUPTED", "unit_id": unit_id, "done": done}
            log = read_json(out / "train_log.json")
            if [r["epoch"] for r in log["epochs"]] != list(range(40)) or any(r.get("steps_in_epoch") != 1133 for r in log["epochs"]):
                raise ValueError("A full unit must contain every complete epoch and exactly 45,320 updates")
            if done.get("global_step") != 45320:
                raise ValueError("Trainer's global update counter differs from complete epoch accounting")
            del student, train, dev_images, dev_captions, collate
            gc.collect()
            torch.cuda.empty_cache()
            artifacts = {n: sha256_file(out / n) for n in ("best.pt", "last.pt", "train_log.json", "run_manifest.json", "identity.json")}
            attempt.update(status="SUCCEEDED", end_utc=utc_now(), elapsed_s=time.monotonic()-started,
                           completed_epochs=40, completed_updates=45320)
            write_json(attempt_path, attempt)
            result = {"identity": identity, "done": done, "hardware": hardware, "full_epochs": 40,
                      "n_pool": len(rows), "n_U": len(weights), "test_read": False,
                      "elapsed_this_attempt_s": attempt["elapsed_s"], "artifact_sha256": artifacts,
                      "attempts_summary": attempts_summary(out), "status": "COMPLETE"}
            # Verify all final artifacts and selection before publishing a COMPLETE receipt.
            validate_complete_unit(out, expected_identity=identity, check_binary=True, result_override=result)
            write_json(out / "full_result.json", result)
            published = True
            return {"status": "COMPLETE_STOP_REQUESTED" if signals["stop_requested"] else "COMPLETE",
                    "unit_id": unit_id, "best": done["best"], "new_full_runs": exp["new_full_runs"]}
        except BaseException as exc:
            if published:
                # A late scheduler signal must not retroactively mark verified training FAILED.
                raise
            attempt.update(status="INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "FAILED",
                           end_utc=utc_now(), elapsed_s=time.monotonic()-started,
                           error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            write_json(attempt_path, attempt)
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def run_seed(experiment_path, seed):
    exp = verify_experiment(experiment_path)
    units = sorted([u for u in exp["units"] if u["seed"] == int(seed)], key=lambda u: u["order"])
    if len(units) != 3:
        raise ValueError("Seed is not in the frozen three-arm matrix")
    results = []
    for unit in units:
        result = run_unit(experiment_path, unit["unit_id"])
        results.append(result)
        print(result, flush=True)
        if result["status"] in ("INTERRUPTED", "COMPLETE_STOP_REQUESTED"):
            return {"status": "INTERRUPTED", "seed": seed, "units": results}
    return {"status": "COMPLETE", "seed": seed, "units": results}
