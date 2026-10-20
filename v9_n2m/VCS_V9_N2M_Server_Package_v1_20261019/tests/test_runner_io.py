"""Torch-free runner identity, locking, interruption and cost regressions.

These tests exercise orchestration/IO, never a student model.  Every synthetic
experiment and checkpoint is temporary and explicitly labelled as a fixture.
"""
from __future__ import annotations

import copy
import math
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v9n2m import runner
from v9n2m.common import read_json, sha256_file, write_json


class ManifestCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.original = {"n_pool": 145000, "n_A": 14500, "n_U": 130500,
                         "sum_w_U": 79848.8919540997, "mean_v_full_pool": 0.6506820134765496,
                         "ess_U": 105233.07692884997, "weight_lock_sha256": "a" * 64,
                         "seed": 901, "lr_mult": 1.0,
                         "cfg": {**runner.SCIENTIFIC_CONFIG, "vocab_size": 8481, "to_verify": []},
                         "formal": False, "loss": "synthetic manifest fixture"}

    def test_one_ulp_derived_diagnostics_do_not_reject_resume(self):
        changed = copy.deepcopy(self.original)
        for field in ("sum_w_U", "mean_v_full_pool", "ess_U"):
            changed[field] = math.nextafter(changed[field], math.inf)
            self.assertNotEqual(changed[field], self.original[field])
        self.assertTrue(runner.manifests_compatible(self.original, changed))

    def test_weight_hash_and_cfg_are_exact_even_for_tiny_lr_change(self):
        for field in ("weight_lock_sha256", "cfg"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.original)
                if field == "cfg":
                    changed[field]["learning_rate"] = math.nextafter(changed[field]["learning_rate"], math.inf)
                else:
                    changed[field] = "b" * 64
                self.assertFalse(runner.manifests_compatible(self.original, changed))

    def test_material_derived_change_missing_field_and_nonfinite_are_rejected(self):
        changed = copy.deepcopy(self.original)
        changed["mean_v_full_pool"] += 1e-6
        self.assertFalse(runner.manifests_compatible(self.original, changed))
        changed = copy.deepcopy(self.original)
        changed.pop("ess_U")
        self.assertFalse(runner.manifests_compatible(self.original, changed))
        for value in (math.nan, math.inf):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.original)
                changed["ess_U"] = value
                self.assertFalse(runner.manifests_compatible(self.original, changed))

    def test_recipe_validation_rejects_architecture_and_unresolved_fields(self):
        for field, value in (("embed_size", 512), ("sgr_step", 1), ("warmup_epochs", 0),
                             ("to_verify", ["margin"])):
            with self.subTest(field=field):
                cfg = dict(self.original["cfg"])
                cfg[field] = value
                with self.assertRaises(ValueError):
                    runner.check_config(cfg)


class ProcessLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="n2m-process-lock-test-")
        self.out = Path(self.tmp.name) / "unit"

    def tearDown(self):
        self.tmp.cleanup()

    def probe_from_another_process(self):
        code = """import sys
from v9n2m.runner import unit_lock
try:
    with unit_lock(sys.argv[1]):
        print('ACQUIRED')
except RuntimeError:
    print('BLOCKED')
"""
        environment = os.environ.copy()
        source = str(Path(runner.__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        completed = subprocess.run([sys.executable, "-c", code, str(self.out)],
                                   capture_output=True, text=True, env=environment,
                                   timeout=10, check=True)
        return completed.stdout.strip()

    def test_other_process_is_excluded_until_lock_released(self):
        with runner.unit_lock(self.out):
            self.assertEqual(self.probe_from_another_process(), "BLOCKED")
        self.assertEqual(self.probe_from_another_process(), "ACQUIRED")

    def test_exception_releases_process_lock(self):
        with self.assertRaisesRegex(RuntimeError, "fixture exception"):
            with runner.unit_lock(self.out):
                raise RuntimeError("fixture exception")
        self.assertEqual(self.probe_from_another_process(), "ACQUIRED")


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="n2m-runner-boundary-test-")
        self.root = Path(self.tmp.name)
        self.experiment_path = self.root / "experiment.json"
        self.unit = {"unit_id": "s901__ConstantU", "seed": 901, "arm": "ConstantU", "action": "RUN", "order": 0,
                     "output_dir": str(self.root / "runs/s901__ConstantU"),
                     "observed_path": str(self.root / "unread.observed.json"), "observed_sha256": "d" * 64,
                     "weights_path": str(self.root / "unread.csv"), "weights_sha256": "e" * 64,
                     "weight_lock_path": str(self.root / "unread.csv.lock.json"), "weight_lock_sha256": "f" * 64}
        self.exp = {"units": [self.unit], "protocol_sha256": "a" * 64, "base_freeze_sha256": "b" * 64,
                    "source_identity": {"effective_digest": "c" * 64}, "new_full_runs": 6,
                    "target_environment": {}, "runtime_dir": str(self.root / "unread_runtime"),
                    "base_handoff": str(self.root / "unread_baseline"), "inputs": {},
                    "fixture_note": "SYNTHETIC ORCHESTRATION TEST; NOT AN EXPERIMENT"}
        write_json(self.experiment_path, self.exp)
        self.exp_patch = mock.patch.object(runner, "verify_experiment", return_value=self.exp)
        self.exp_patch.start()
        self.env_patch = mock.patch.dict(os.environ, {"SLURM_JOB_ID": "SYNTHETIC_TEST_ONLY"})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.exp_patch.stop()
        self.tmp.cleanup()

    @property
    def out(self):
        return Path(self.unit["output_dir"])

    def write_matching_identity(self):
        identity = runner._unit_identity(self.exp, self.unit, self.experiment_path)
        write_json(self.out / "identity.json", identity)
        return identity

    def test_orphan_scientific_artifacts_are_never_adopted(self):
        for filename in ("last.pt", "best.pt", "done.json", "run_manifest.json", "train_log.json", "full_result.json"):
            with self.subTest(filename=filename):
                self.unit["output_dir"] = str(self.root / "orphans" / filename.replace(".", "_"))
                self.out.mkdir(parents=True)
                orphan = self.out / filename
                orphan.write_bytes(b"SYNTHETIC_ORPHAN_MUST_NOT_BE_ADOPTED")
                before = orphan.read_bytes()
                with mock.patch.object(runner, "configure_precision") as torch_boundary:
                    with self.assertRaisesRegex(ValueError, "Unidentified scientific artifacts"):
                        runner.run_unit(self.experiment_path, self.unit["unit_id"])
                    torch_boundary.assert_not_called()
                self.assertEqual(orphan.read_bytes(), before)
                self.assertFalse((self.out / "identity.json").exists())
                self.assertFalse((self.out / "attempts").exists())

    def test_wrong_existing_identity_is_refused_before_torch(self):
        identity = self.write_matching_identity()
        identity["experiment_sha256"] = "0" * 64
        write_json(self.out / "identity.json", identity)
        with mock.patch.object(runner, "configure_precision") as torch_boundary:
            with self.assertRaisesRegex(ValueError, "different unit/experiment"):
                runner.run_unit(self.experiment_path, self.unit["unit_id"])
            torch_boundary.assert_not_called()

    def test_corrupt_completion_is_not_reported_already_complete(self):
        self.write_matching_identity()
        (self.out / "full_result.json").write_text("{corrupted completion")
        with mock.patch.object(runner, "configure_precision") as torch_boundary:
            with self.assertRaises(ValueError):
                runner.run_unit(self.experiment_path, self.unit["unit_id"])
            torch_boundary.assert_not_called()
        self.assertFalse((self.out / "attempts").exists())

    def test_unallocated_training_is_refused_without_creating_a_run(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(runner, "configure_precision") as torch_boundary:
            with self.assertRaisesRegex(RuntimeError, "allocated Slurm GPU job"):
                runner.run_unit(self.experiment_path, self.unit["unit_id"])
            torch_boundary.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_predecessor_check_binds_the_full_experiment_identity(self):
        predecessor = {**self.unit, "unit_id": "s901__VCS", "arm": "VCS", "order": 0,
                       "output_dir": str(self.root / "runs/s901__VCS")}
        self.unit["order"] = 1
        self.exp["units"] = [self.unit, predecessor]
        with mock.patch.object(runner, "validate_complete_unit", return_value={"status": "COMPLETE"}) as validator:
            runner._check_predecessors(self.exp, self.unit, self.experiment_path)
        validator.assert_called_once()
        expected = validator.call_args.kwargs["expected_identity"]
        self.assertEqual(expected, runner._unit_identity(self.exp, predecessor, self.experiment_path))
        self.assertEqual(expected["experiment_sha256"], sha256_file(self.experiment_path))
        self.assertEqual(expected["weights_sha256"], predecessor["weights_sha256"])
        self.assertTrue(validator.call_args.kwargs["check_binary"])
        with mock.patch.object(runner, "validate_complete_unit", side_effect=ValueError("wrong experiment")):
            with self.assertRaisesRegex(ValueError, "finish first"):
                runner._check_predecessors(self.exp, self.unit, self.experiment_path)

    def test_preflight_failure_has_attempt_cost_but_no_complete_receipt(self):
        with mock.patch.object(runner, "configure_precision", side_effect=RuntimeError("synthetic preflight failure")), \
                mock.patch.object(runner.time, "monotonic", side_effect=[100.0, 105.25]):
            with self.assertRaisesRegex(RuntimeError, "synthetic preflight failure"):
                runner.run_unit(self.experiment_path, self.unit["unit_id"])
        paths = list((self.out / "attempts").glob("*.json"))
        self.assertEqual(len(paths), 1)
        attempt = read_json(paths[0])
        self.assertEqual(attempt["status"], "FAILED")
        self.assertEqual(attempt["elapsed_s"], 5.25)
        self.assertEqual(attempt["job_id"], "SYNTHETIC_TEST_ONLY")
        self.assertFalse((self.out / "full_result.json").exists())

    def test_termination_records_interruption_preserves_checkpoint_and_handlers(self):
        self.write_matching_identity()
        last = self.out / "last.pt"
        last.write_bytes(b"SYNTHETIC_CHECKPOINT_IO_ONLY")
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}

        def terminate_during_preflight():
            # Invoke the installed callback directly; do not send an OS signal.
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)

        with mock.patch.object(runner, "configure_precision", side_effect=terminate_during_preflight), \
                mock.patch.object(runner.time, "monotonic", side_effect=[100.0, 103.0]):
            with self.assertRaises(InterruptedError):
                runner.run_unit(self.experiment_path, self.unit["unit_id"])
        attempt = read_json(next((self.out / "attempts").glob("*.json")))
        self.assertEqual(attempt["status"], "INTERRUPTED")
        self.assertEqual(attempt["elapsed_s"], 3.0)
        self.assertEqual(attempt["checkpoint_at_start_sha256"], sha256_file(last))
        self.assertEqual(last.read_bytes(), b"SYNTHETIC_CHECKPOINT_IO_ONLY")
        self.assertTrue((self.out / "STOP_REQUESTED").exists())
        self.assertFalse((self.out / "full_result.json").exists())
        self.assertEqual({sig: signal.getsignal(sig) for sig in previous_handlers}, previous_handlers)

    def test_old_stop_request_is_archived_not_discarded_on_new_attempt(self):
        self.out.mkdir(parents=True)
        marker = self.out / "STOP_REQUESTED"
        marker.write_text("previous allocation stop request\n")
        with mock.patch.object(runner, "configure_precision", side_effect=RuntimeError("stop before Torch")):
            with self.assertRaisesRegex(RuntimeError, "stop before Torch"):
                runner.run_unit(self.experiment_path, self.unit["unit_id"])
        self.assertFalse(marker.exists())
        archived = list((self.out / "attempts").glob("*.prior_stop_request.txt"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), "previous allocation stop request\n")


class AttemptAndOrderTests(unittest.TestCase):
    def test_unknown_attempt_duration_is_not_invented_as_zero(self):
        with tempfile.TemporaryDirectory(prefix="n2m-attempt-cost-test-") as tmp:
            root = Path(tmp)
            for aid, elapsed, status in (("completed", 10.5, "SUCCEEDED"), ("interrupted", 2.5, "INTERRUPTED"), ("lost", None, "RUNNING")):
                write_json(root / "attempts" / f"{aid}.json",
                           {"attempt_id": aid, "elapsed_s": elapsed, "status": status})
            report = runner.attempts_summary(root)
            self.assertEqual(report["attempts"], 3)
            self.assertEqual(report["attempt_wall_s_known"], 13.0)
            self.assertEqual(report["missing_duration_attempts"], ["lost"])
            self.assertFalse(report["total_duration_complete"])
            self.assertEqual(len(report["entries"]), 3)

    def test_seed_order_is_frozen_and_stops_after_graceful_interruption(self):
        # Seed 903: Shuffle -> VCS -> Constant. The input list is not sorted.
        units = [{"unit_id": "constant", "seed": 903, "order": 2},
                 {"unit_id": "shuffle", "seed": 903, "order": 0},
                 {"unit_id": "vcs", "seed": 903, "order": 1}]
        with mock.patch.object(runner, "verify_experiment", return_value={"units": units}), \
                mock.patch.object(runner, "run_unit", side_effect=[{"status": "COMPLETE"}, {"status": "INTERRUPTED"}]) as run, \
                mock.patch("builtins.print"):
            result = runner.run_seed("SYNTHETIC_UNUSED_PATH", 903)
        self.assertEqual(result["status"], "INTERRUPTED")
        self.assertEqual([call.args[1] for call in run.call_args_list], ["shuffle", "vcs"])

    def test_final_epoch_stop_request_does_not_start_next_arm(self):
        units = [{"unit_id": str(k), "seed": 901, "order": k} for k in range(3)]
        with mock.patch.object(runner, "verify_experiment", return_value={"units": units}), \
                mock.patch.object(runner, "run_unit", return_value={"status": "COMPLETE_STOP_REQUESTED"}) as run, \
                mock.patch("builtins.print"):
            result = runner.run_seed("SYNTHETIC_UNUSED_PATH", 901)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result["status"], "INTERRUPTED")


if __name__ == "__main__":
    unittest.main()
