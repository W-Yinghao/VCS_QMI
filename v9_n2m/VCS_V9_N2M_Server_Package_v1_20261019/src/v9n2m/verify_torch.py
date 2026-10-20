"""Server-side equivalence check using the actual frozen SGR implementation.

Only synthetic region features and token IDs are created. No original train,
DEV, TEST, caption text, teacher, or mechanism label files are opened. Torch is
imported lazily; an unavailable requested device is NOT_RUN, never PASS.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback

import numpy as np

from .common import sha256_file, utc_now, write_json
from .trainer_patch import BASELINE_TRAIN_SHA256, patch_train_source, telemetry_patch_identity


_DEPENDENCIES = (
    "src/v9/n2/teacher.py",
    "src/v9/n2/sgr_host.py",
    "src/v9/n2/eval.py",
    "src/v9/n2/fixtures.py",
    "src/v9/n2/weights.py",
    "src/v9/n2/noise.py",
    "v9/package_v9_20261016/reference/core.py",
    "v9/package_v9_20261016/reference/torch_losses.py",
)


def _check_source(runtime_root: Path, baseline_root: Path) -> dict:
    baseline = baseline_root / "src/v9/n2/train.py"
    patched = runtime_root / "src/v9/n2/train.py"
    source = baseline.read_text(encoding="utf-8")
    expected = patch_train_source(source)
    if patched.read_bytes() != expected.encode("utf-8"):
        raise ValueError("runtime train.py is not the exact pinned transformation")
    checked = {}
    for relative in _DEPENDENCIES:
        old_hash = sha256_file(baseline_root / relative)
        new_hash = sha256_file(runtime_root / relative)
        if old_hash != new_hash:
            raise ValueError(f"verification dependency differs from baseline: {relative}")
        checked[relative] = old_hash
    return {
        "status": "PASS",
        "baseline_train_sha256": sha256_file(baseline),
        "patched_train_sha256": sha256_file(patched),
        "identical_dependency_sha256": checked,
    }


def _module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _compare_tree(left, right, torch, path="root") -> dict:
    """Exact comparisons on the same deterministic device; no tolerance escape."""
    differences = []
    tensor_count = 0
    max_abs_difference = 0.0

    def walk(a, b, location):
        nonlocal tensor_count, max_abs_difference
        if torch.is_tensor(a) and torch.is_tensor(b):
            tensor_count += 1
            aa, bb = a.detach().cpu(), b.detach().cpu()
            if aa.shape != bb.shape or aa.dtype != bb.dtype:
                differences.append(f"{location}: tensor shape/dtype")
            elif not torch.equal(aa, bb):
                differences.append(f"{location}: tensor values")
                if aa.numel():
                    delta = float((aa.to(torch.float64) - bb.to(torch.float64)).abs().max())
                    if np.isfinite(delta):
                        max_abs_difference = max(max_abs_difference, delta)
            return
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                differences.append(f"{location}: NumPy array")
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                differences.append(f"{location}: mapping keys")
            for key in a.keys() & b.keys():
                walk(a[key], b[key], f"{location}.{key}")
            return
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                differences.append(f"{location}: sequence shape/type")
            for i, (aa, bb) in enumerate(zip(a, b)):
                walk(aa, bb, f"{location}[{i}]")
            return
        if type(a) is not type(b) or a != b:
            differences.append(f"{location}: scalar/type")

    walk(left, right, path)
    return {
        "status": "PASS" if not differences else "FAIL",
        "comparison": "exact on one deterministic runtime/device",
        "tensors_compared": tensor_count,
        "max_abs_tensor_difference": max_abs_difference,
        "difference_count": len(differences),
        "first_differences": differences[:20],
    }


def _scientific_history(log):
    epoch_keys = ("epoch", "lr", "phase", "steps_in_epoch", "mean_loss", "mean_u_risk_unnormalised", "dev")
    step_keys = ("step", "epoch", "loss", "u_risk_unnormalised", "batch_mass", "batch_ess", "lr")
    return {
        "epochs": [{key: rec[key] for key in epoch_keys} for rec in log["epochs"]],
        "steps": [{key: rec[key] for key in step_keys} for rec in log["steps"]],
    }


def verify_runtime(runtime_root, baseline_root, out_path, device="cuda") -> dict:
    """Write and return a PASS / FAIL / NOT_RUN diagnostic report.

    FAIL is returned, not raised, so the caller can retain the concrete report.
    The CLI and the server wrapper must use a nonzero exit status unless PASS.
    All tensor comparisons use exact equality under one deterministic device;
    this does not assert equivalence between different GPUs or software builds.
    """
    runtime_root = Path(runtime_root).resolve()
    baseline_root = Path(baseline_root).resolve()
    out_path = Path(out_path).resolve()
    report = {
        "schema_version": 1,
        "status": "NOT_RUN",
        "started_utc": utc_now(),
        "scope": "synthetic actual-SGR code/resume equivalence; no scientific performance evidence",
        "runtime_root": str(runtime_root),
        "baseline_root": str(baseline_root),
        "baseline_train_sha256": BASELINE_TRAIN_SHA256,
        "patched_train_sha256": None,
        "patch": telemetry_patch_identity(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "requested_device": str(device)},
        "checks": {},
        "original_data_read": False,
        "test_read": False,
    }
    added_sys_path = None
    module_keys_before = set(sys.modules)
    previous_settings = None
    torch = None
    try:
        source_check = _check_source(runtime_root, baseline_root)
        report["checks"]["pinned_runtime_source"] = source_check
        report["patched_train_sha256"] = source_check["patched_train_sha256"]
        # This must be set before CUDA initialisation for deterministic GEMM.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            report["reason"] = f"Torch unavailable: {exc}"
            return _finish(report, out_path)
        report["environment"].update({"torch": torch.__version__, "cuda": torch.version.cuda})
        requested = torch.device(device)
        if requested.type not in ("cpu", "cuda"):
            raise ValueError("verification device must be cpu or cuda; protocol training is CUDA")
        if requested.type == "cuda" and not torch.cuda.is_available():
            report["reason"] = "Requested CUDA device is unavailable; actual SGR verification was not run"
            return _finish(report, out_path)
        previous_settings = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "num_threads": torch.get_num_threads(),
        }
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Small CPU fixtures are much faster with one thread. A CUDA check
        # retains the allocated caller thread count so the checked execution
        # settings match the production runtime envelope.
        if requested.type == "cpu":
            torch.set_num_threads(1)
        report["environment"].update({
            "gpu": torch.cuda.get_device_name(requested) if requested.type == "cuda" else None,
            "device": str(requested),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "fixture_torch_num_threads": torch.get_num_threads(),
        })
        src = (runtime_root / "src").resolve()
        for name, module in list(sys.modules.items()):
            if name == "v9" or name.startswith("v9.") or name.startswith("v9_pkg_ref_"):
                filename = getattr(module, "__file__", None)
                if filename and runtime_root not in Path(filename).resolve().parents:
                    raise ValueError(f"pre-imported {name} belongs to another runtime; invoke verification in a fresh process")
        added_sys_path = str(src)
        sys.path.insert(0, added_sys_path)
        importlib.import_module("v9.n2")
        base_module = _module_from_path("v9.n2._n2m_verify_baseline_train", baseline_root / "src/v9/n2/train.py")
        patched_module = _module_from_path("v9.n2._n2m_verify_patched_train", runtime_root / "src/v9/n2/train.py")
        teacher = importlib.import_module("v9.n2.teacher")
        fixtures = importlib.import_module("v9.n2.fixtures")
        cfg = teacher.HostConfig(
            vocab_size=40, img_dim=8, word_dim=8, embed_size=16, sim_dim=8,
            num_region=3, sgr_step=1, batch_size=8, num_epochs=4,
            warmup_epochs=2, lr_update=3, learning_rate=2e-4, grad_clip=2.0,
            margin=0.2, bi_gru=True,
        )
        seed = 901
        fixture = fixtures.make_fixture(n_images=6, n_caps=5, n_region=3, img_dim=8, vocab=40, seed=20261019)
        rows = [
            {"pair_id": f"synthetic_p{cid}", "caption_id": str(cid),
             "observed_image_id": str(cid // 5), "role": "teacher_warmup" if cid < 5 else "U"}
            for cid in range(30)
        ]
        u_ids = [r["pair_id"] for r in rows if r["role"] == "U"]
        vector = np.linspace(0.0, 0.95, len(u_ids), dtype=np.float64)
        weights = dict(zip(u_ids, vector.tolist()))
        collate = partial(fixtures.collate_pairs, fixture["images"], fixture["captions"])
        dev_captions = [fixture["captions"][cid] for cid in range(30)]
        batch_indices = list(teacher.iterate_batches(len(rows), cfg.batch_size, np.random.default_rng([seed, 9020, 0])))
        report["fixture"] = {
            "synthetic_only": True, "seed": seed, "data_seed": 20261019,
            "cfg": asdict(cfg), "n_pool": len(rows), "n_A": 5, "n_U": len(u_ids),
            "batch_sizes": [len(idx) for idx in batch_indices],
            "expected_updates": len(batch_indices) * cfg.num_epochs,
            "log_every": 3, "stop_after_epoch": 1,
            "note": "Reduced dimensions/schedule exercise warmup, hard negatives, LR change, final nonfull batch, sparse logs, and resume.",
        }
        with tempfile.TemporaryDirectory(prefix="v9n2m_synthetic_verify_", dir=out_path.parent if out_path.parent.exists() else None) as temporary:
            temporary = Path(temporary)

            def student(cls, name):
                student_out = temporary / name
                if name in ("patched", "resumed", "wrong_identity", "production_legacy"):
                    write_json(student_out / "identity.json", {
                        "synthetic_diagnostic_only": True,
                        "unit_id": f"synthetic_{name}", "seed": seed,
                        "arm": "fixture_nonuniform_weights",
                    })
                return cls(
                    cfg, rows, weights, collate, fixture["images"], dev_captions,
                    student_out, seed, 1.0, device=str(requested),
                    weight_lock={"sha256": "SYNTHETIC_DIAGNOSTIC_ONLY"},
                    formal=False, loss_scale="sum", grad_log_window=(2, 2),
                )

            teacher.seed_everything(seed)
            initial_host = teacher.build_host(cfg)
            initial_parameters = {key: value.detach().cpu().clone() for key, value in initial_host.named_parameters()}
            del initial_host

            base = student(base_module.WeightedStudent, "baseline")
            base_done = base.fit(cfg.num_epochs, log_every=3)
            captures = {}
            real_hinge = patched_module.hinge_per_anchor

            def capture_first_hinge(*args, **kwargs):
                row, column = real_hinge(*args, **kwargs)
                if "row_value" not in captures:
                    captures["row_value"] = row.detach().cpu().clone()
                    captures["column_value"] = column.detach().cpu().clone()

                    def row_hook(gradient):
                        captures["row_gradient"] = gradient.detach().cpu().clone()

                    def column_hook(gradient):
                        captures["column_gradient"] = gradient.detach().cpu().clone()

                    row.register_hook(row_hook)
                    column.register_hook(column_hook)
                return row, column

            patched_module.hinge_per_anchor = capture_first_hinge
            patched = student(patched_module.WeightedStudent, "patched")
            try:
                patched_done = patched.fit(cfg.num_epochs, log_every=3)
            finally:
                patched_module.hinge_per_anchor = real_hinge

            class StopPatchedAtEpochOne(patched_module.WeightedStudent):
                def _save(self, host, opt, epoch, best, log, name="last.pt"):
                    super()._save(host, opt, epoch, best, log, name=name)
                    if name == "last.pt" and epoch == 1:
                        (self.out / "STOP_REQUESTED").write_text("synthetic verification stop\n")

            interrupted = student(StopPatchedAtEpochOne, "resumed")
            interrupted_result = interrupted.fit(cfg.num_epochs, log_every=3)
            stop_ok = (
                interrupted_result.get("status") == "INTERRUPTED"
                and interrupted_result.get("epochs_run") == 2
                and interrupted_result.get("target_epochs") == 4
                and interrupted_result.get("global_step") == 8
                and not (interrupted.out / "done.json").exists()
                and (interrupted.out / "last.pt").is_file()
            )
            report["checks"]["stop_at_saved_epoch_boundary"] = {"status": "PASS" if stop_ok else "FAIL", "returned": interrupted_result}
            (interrupted.out / "STOP_REQUESTED").unlink()
            resumed = student(patched_module.WeightedStudent, "resumed")
            resumed_done = resumed.fit(cfg.num_epochs, log_every=3)

            class LegacyFixtureBoundary(Exception):
                pass

            class StopLegacyAtEpochOne(base_module.WeightedStudent):
                def _save(self, host, opt, epoch, best, log, name="last.pt"):
                    super()._save(host, opt, epoch, best, log, name=name)
                    if name == "last.pt" and epoch == 1:
                        raise LegacyFixtureBoundary("synthetic interruption after original epoch checkpoint")

            legacy = student(StopLegacyAtEpochOne, "legacy_resumed")
            try:
                legacy.fit(cfg.num_epochs, log_every=3)
            except LegacyFixtureBoundary:
                pass
            legacy_ck = torch.load(legacy.out / "last.pt", map_location="cpu", weights_only=False)
            if "global_step" in legacy_ck:
                raise AssertionError("legacy fixture unexpectedly has a patched checkpoint field")
            legacy_resumed = student(patched_module.WeightedStudent, "legacy_resumed")
            legacy_done = legacy_resumed.fit(cfg.num_epochs, log_every=3)

            paths = {"baseline": base.out, "patched": patched.out, "resumed": resumed.out, "legacy_resumed": legacy_resumed.out}
            checkpoints = {name: torch.load(path / "last.pt", map_location="cpu", weights_only=False) for name, path in paths.items()}
            for name in ("patched", "resumed", "legacy_resumed"):
                for field in ("model", "opt", "rng", "best"):
                    report["checks"][f"baseline_vs_{name}_{field}"] = _compare_tree(checkpoints["baseline"][field], checkpoints[name][field], torch, field)
                report["checks"][f"baseline_vs_{name}_scientific_history"] = _compare_tree(
                    _scientific_history(checkpoints["baseline"]["log"]),
                    _scientific_history(checkpoints[name]["log"]), torch, "history",
                )
            bindings = {
                name: (
                    checkpoints[name].get("execution_identity_sha256") == sha256_file(paths[name] / "identity.json")
                    and checkpoints[name].get("run_manifest_sha256") == sha256_file(paths[name] / "run_manifest.json")
                )
                for name in ("patched", "resumed")
            }
            report["checks"]["checkpoint_execution_and_manifest_binding"] = {
                "status": "PASS" if all(bindings.values()) else "FAIL",
                "identified_fixtures": bindings,
                "legacy_compatibility_fixture_has_identity": (legacy_resumed.out / "identity.json").exists(),
            }
            rejected_checkpoints = {}
            for name, checkpoint in (("wrong_identity", checkpoints["resumed"]), ("production_legacy", legacy_ck)):
                wrong = student(patched_module.WeightedStudent, name)
                torch.save(checkpoint, wrong.out / "last.pt")
                try:
                    wrong.fit(cfg.num_epochs, log_every=3)
                    rejected_checkpoints[name] = {"rejected": False}
                except ValueError as exc:
                    rejected_checkpoints[name] = {"rejected": "execution identity" in str(exc), "reason": str(exc)}
            report["checks"]["copied_wrong_unit_and_legacy_production_checkpoint_refused"] = {
                "status": "PASS" if all(c["rejected"] for c in rejected_checkpoints.values()) else "FAIL",
                "cases": rejected_checkpoints,
            }
            expected_updates = len(batch_indices) * cfg.num_epochs
            counts = {name: int(checkpoints[name]["global_step"]) for name in ("patched", "resumed", "legacy_resumed")}
            telemetry_ok = all(count == expected_updates for count in counts.values())
            for name in counts:
                log = checkpoints[name]["log"]
                telemetry_ok = telemetry_ok and all(
                    ("global_step" not in rec or rec["global_step"] == (rec["epoch"] + 1) * len(batch_indices))
                    and rec["steps_in_epoch"] == len(batch_indices)
                    for rec in log["epochs"]
                )
                for rec in log["steps"]:
                    # Existing legacy rows retain their schema; their step values
                    # are checked by the exact baseline-history comparison.
                    if "step_in_epoch" in rec:
                        telemetry_ok = telemetry_ok and rec["step"] == rec["global_step"] == rec["epoch"] * len(batch_indices) + rec["step_in_epoch"]
            report["checks"]["actual_update_and_epoch_telemetry"] = {
                "status": "PASS" if telemetry_ok else "FAIL", "expected": expected_updates,
                "checkpoint_global_steps": counts,
                "legacy_sparse_rows_before_resume": len(legacy_ck["log"]["steps"]),
                "legacy_actual_updates_before_resume": sum(e["steps_in_epoch"] for e in legacy_ck["log"]["epochs"]),
                "legacy_epoch_records_without_global_step": sum("global_step" not in e for e in legacy_ck["log"]["epochs"]),
            }
            # Compare the actual fp32 autograd coefficients with an independent
            # fp64 expression. Division versus reciprocal multiplication can
            # differ by an fp32 rounding unit; tensor-trajectory comparisons
            # above remain strictly exact.
            expected_coefficients = patched.v[batch_indices[0]].astype(np.float32).astype(np.float64) / patched.mean_v_full
            row_difference = np.abs(captures["row_gradient"].numpy().astype(np.float64) - expected_coefficients)
            column_difference = np.abs(captures["column_gradient"].numpy().astype(np.float64) - expected_coefficients)
            coefficient_tolerance = 4 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(expected_coefficients))
            row_error = float(row_difference.max())
            column_error = float(column_difference.max())
            coefficient_ok = bool(np.all(row_difference <= coefficient_tolerance) and np.all(column_difference <= coefficient_tolerance))
            report["checks"]["actual_first_sgr_batch_weight_coefficients"] = {
                "status": "PASS" if coefficient_ok else "FAIL",
                "meaning": "d(loss)/d(each row/column hinge) equals v_i / mean_v_full_pool; SUM, detached weights",
                "batch_indices": batch_indices[0].tolist(),
                "batch_weights": patched.v[batch_indices[0]].tolist(),
                "mean_v_full_pool": patched.mean_v_full,
                "expected_coefficients": expected_coefficients.tolist(),
                "reference_dtype": "float64 after the trainer's explicit float32 weight conversion",
                "allowed_absolute_error_per_coefficient": coefficient_tolerance.tolist(),
                "max_abs_row_gradient_error": row_error,
                "max_abs_column_gradient_error": column_error,
                "zero_weight_pairs_retained_in_batch_pool": bool(np.any(patched.v == 0.0) and len(patched.rows) == 30),
            }
            prefixes = ("img_enc.", "txt_enc.", "sim_enc.")
            changed_modules = {
                prefix.rstrip("."): any(not torch.equal(value, checkpoints["baseline"]["model"][name]) for name, value in initial_parameters.items() if name.startswith(prefix))
                for prefix in prefixes
            }
            report["checks"]["actual_sgr_parameters_updated"] = {"status": "PASS" if all(changed_modules.values()) else "FAIL", "changed_modules": changed_modules}
            epochs = checkpoints["patched"]["log"]["epochs"]
            schedule_ok = (
                [e["phase"] for e in epochs] == ["warmup_mean_hinge", "warmup_mean_hinge", "max_violation", "max_violation"]
                and [e["lr"] for e in epochs] == [cfg.learning_rate] * 3 + [cfg.learning_rate * 0.1]
                and [len(i) for i in batch_indices] == [8, 8, 8, 6]
            )
            report["checks"]["warmup_hard_lr_and_nonfull_last_batch"] = {
                "status": "PASS" if schedule_ok else "FAIL",
                "phases": [e["phase"] for e in epochs], "learning_rates": [e["lr"] for e in epochs],
                "batch_sizes": [len(i) for i in batch_indices],
            }
            last_hash = sha256_file(patched.out / "last.pt")
            build_host = patched_module.build_host

            def refuse_fresh_host(*args, **kwargs):
                raise AssertionError("completed fit tried to create a fresh host")

            patched_module.build_host = refuse_fresh_host
            try:
                repeated_done = patched.fit(cfg.num_epochs, log_every=3)
                rejected_other_target = False
                try:
                    patched.fit(cfg.num_epochs + 1, log_every=3)
                except ValueError:
                    rejected_other_target = True
            finally:
                patched_module.build_host = build_host
            done_ok = repeated_done == patched_done and rejected_other_target and last_hash == sha256_file(patched.out / "last.pt")
            report["checks"]["completion_is_idempotent_and_invalid_target_refused"] = {"status": "PASS" if done_ok else "FAIL", "fresh_host_called": False, "checkpoint_unchanged": last_hash == sha256_file(patched.out / "last.pt")}
            report["completion_best"] = {"baseline": base_done["best"], "patched": patched_done["best"], "resumed": resumed_done["best"], "legacy_resumed": legacy_done["best"]}
            report["checkpoint_payload_note"] = "Model/optimizer/RNG tensors and scientific histories are compared; whole .pt bytes differ because telemetry and serialization identities differ."
        report["status"] = "PASS" if all(check["status"] == "PASS" for check in report["checks"].values()) else "FAIL"
    except Exception as exc:
        report["status"] = "FAIL"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=8)
    finally:
        if torch is not None and previous_settings is not None:
            torch.use_deterministic_algorithms(previous_settings["deterministic"], warn_only=previous_settings["warn_only"])
            torch.backends.cudnn.benchmark = previous_settings["benchmark"]
            torch.backends.cudnn.deterministic = previous_settings["cudnn_deterministic"]
            torch.backends.cuda.matmul.allow_tf32 = previous_settings["matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = previous_settings["cudnn_tf32"]
            torch.set_num_threads(previous_settings["num_threads"])
        if added_sys_path is not None:
            sys.path.remove(added_sys_path)
        for name in set(sys.modules) - module_keys_before:
            if name == "v9" or name.startswith("v9.") or name.startswith("v9_pkg_ref_"):
                sys.modules.pop(name, None)
    return _finish(report, out_path)


def _finish(report, out_path):
    report["finished_utc"] = utc_now()
    write_json(out_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = verify_runtime(args.runtime, args.baseline, args.out, args.device)
    print(json.dumps({"status": result["status"], "report": str(Path(args.out).resolve()), "reason": result.get("reason")}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
