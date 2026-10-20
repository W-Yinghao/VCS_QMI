"""Torch-free identity/receipt tests; checkpoint bytes are synthetic IO fixtures.

No test here claims that synthetic checkpoint bytes are a trained model.  GPU
and Torch numerical checks belong to the separate server verification command.
Set V9N2M_BASELINE_ROOT to the original handoff's source directory to enable
additional read-only checks of the actual frozen N2F handoff.
"""
from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v9n2m.common import canonical_sha256, read_json, sha256_file, write_json
from v9n2m.identity import (
    closeout_census, expected_steps, pinned_freeze, validate_complete_unit,
    validate_vcs_reference, verify_source,
)
from v9n2m.runner import SCIENTIFIC_CONFIG
from v9n2m.runtime import bootstrap_runtime, environment_differences, ENVIRONMENT_KEYS, verify_runtime_files


def _complete_io_fixture(root, identity=None):
    """Consistent receipts around unmistakable dummy bytes, not Torch objects."""
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(identity or {"seed": 901, "method": "VCS_Weight", "scale": "sum"})
    manifest = {"seed": identity["seed"], "n_pool": 145000, "n_A": 14500, "n_U": 130500,
                "lr_mult": 1.0, "weight_lock_sha256": "a" * 64,
                "cfg": {**SCIENTIFIC_CONFIG, "vocab_size": 8481, "teacher_epochs": 5, "to_verify": []}}
    epochs = []
    for epoch in range(40):
        rsum = float(400 + min(epoch, 38))  # epochs 38/39 tie: 38 must be selected.
        dev = {key: rsum / 6 for key in ("i2t_R1", "i2t_R5", "i2t_R10", "t2i_R1", "t2i_R5", "t2i_R10")}
        dev.update(rsum=rsum, mR=rsum / 6)
        epochs.append({"epoch": epoch, "steps_in_epoch": 1133, "dev": dev})
    write_json(root / "identity.json", identity)
    write_json(root / "run_manifest.json", manifest)
    write_json(root / "train_log.json", {"epochs": epochs, "steps": []})
    for name in ("best.pt", "last.pt"):
        (root / name).write_bytes(b"SYNTHETIC_IO_FIXTURE_NOT_A_MODEL\n" + name.encode())
    done = {"best": {"epoch": 38, "rsum": 438.0}, "epochs_run": 40, "global_step": 45320,
            "manifest_sha256": sha256_file(root / "run_manifest.json")}
    write_json(root / "done.json", done)
    result = {"identity": identity, "done": done, "full_epochs": 40,
              "n_pool": 145000, "n_U": 130500, "test_read": False,
              "fixture_note": "SYNTHETIC IO ONLY; NOT A TRAINING RESULT",
              "artifact_sha256": {name: sha256_file(root / name) for name in
                                  ("best.pt", "last.pt", "train_log.json", "run_manifest.json", "identity.json")}}
    write_json(root / "full_result.json", result)
    return result, manifest


class CompletionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="n2m-identity-test-")
        self.root = Path(self.tmp.name)
        self.unit = self.root / "synthetic_unit"
        self.result, self.manifest = _complete_io_fixture(self.unit)

    def tearDown(self):
        self.tmp.cleanup()

    def update_log(self, change):
        log = read_json(self.unit / "train_log.json")
        change(log)
        write_json(self.unit / "train_log.json", log)
        self.result["artifact_sha256"]["train_log.json"] = sha256_file(self.unit / "train_log.json")
        write_json(self.unit / "full_result.json", self.result)

    def test_complete_receipts_derive_all_45320_updates_and_earliest_tie(self):
        result = validate_complete_unit(self.unit, expected_identity={"seed": 901, "scale": "sum"})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["updates"], 45320)
        self.assertEqual(result["best_epoch"], 38)
        self.assertEqual(len(result["verified_artifacts"]), 5)

    def test_last_partial_batch_is_kept_but_singleton_is_dropped(self):
        self.assertEqual(expected_steps(145000), 1133)
        self.assertEqual(expected_steps(128), 1)
        self.assertEqual(expected_steps(129), 1)
        self.assertEqual(expected_steps(130), 2)

    def test_missing_binary_never_counts_as_verified_complete(self):
        (self.unit / "best.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_complete_unit(self.unit, check_binary=True)
        result = validate_complete_unit(self.unit, check_binary=False)
        self.assertEqual(result["status"], "METADATA_COMPLETE_CHECKPOINTS_NOT_VERIFIED")
        self.assertEqual(set(result["unverified_artifacts"]), {"best.pt", "last.pt"})

    def test_corrupted_binary_or_json_is_rejected(self):
        for filename in ("best.pt", "train_log.json"):
            with self.subTest(filename=filename):
                content = (self.unit / filename).read_bytes()
                (self.unit / filename).write_bytes(content + b"altered")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    validate_complete_unit(self.unit)
                (self.unit / filename).write_bytes(content)

    def test_even_rehashed_incomplete_or_duplicate_epochs_are_rejected(self):
        for mutation in (lambda x: x["epochs"].pop(),
                         lambda x: x["epochs"][20].update(epoch=19),
                         lambda x: x["epochs"][10].update(steps_in_epoch=1132)):
            with self.subTest(mutation=mutation):
                self.result, self.manifest = _complete_io_fixture(self.unit)
                self.update_log(mutation)
                with self.assertRaisesRegex(ValueError, "40 complete epochs|Incomplete epoch"):
                    validate_complete_unit(self.unit)

    def test_later_tie_cannot_replace_prespecified_earliest_checkpoint(self):
        self.result["done"]["best"]["epoch"] = 39
        write_json(self.unit / "full_result.json", self.result)
        with self.assertRaisesRegex(ValueError, "Best/earliest-tie"):
            validate_complete_unit(self.unit)

    def test_done_receipt_and_manifest_are_bound(self):
        self.result["done"]["manifest_sha256"] = "0" * 64
        write_json(self.unit / "full_result.json", self.result)
        with self.assertRaisesRegex(ValueError, "different run manifest"):
            validate_complete_unit(self.unit)
        self.result, _ = _complete_io_fixture(self.unit)
        done = read_json(self.unit / "done.json")
        done["epochs_run"] = 39
        write_json(self.unit / "done.json", done)
        with self.assertRaisesRegex(ValueError, "different done receipt"):
            validate_complete_unit(self.unit)

    def test_staged_valid_result_can_be_verified_without_publication(self):
        (self.unit / "full_result.json").unlink()
        verification = validate_complete_unit(self.unit, result_override=self.result)
        self.assertEqual(verification["status"], "COMPLETE")
        self.assertIsNone(verification["result_sha256"])
        self.assertFalse((self.unit / "full_result.json").exists())

    def test_invalid_staged_result_does_not_publish_a_complete_receipt(self):
        (self.unit / "full_result.json").unlink()
        candidate = copy.deepcopy(self.result)
        candidate["done"]["best"]["rsum"] = 999.0
        with self.assertRaisesRegex(ValueError, "Best/earliest-tie"):
            validate_complete_unit(self.unit, result_override=candidate)
        self.assertFalse((self.unit / "full_result.json").exists())

    def test_manifest_cannot_hash_an_artifact_outside_its_unit(self):
        outside = self.root / "outside.bin"
        outside.write_bytes(b"unrelated")
        self.result["artifact_sha256"]["../outside.bin"] = sha256_file(outside)
        write_json(self.unit / "full_result.json", self.result)
        with self.assertRaisesRegex(ValueError, "outside unit directory"):
            validate_complete_unit(self.unit)

    def test_test_read_and_wrong_expected_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected seed"):
            validate_complete_unit(self.unit, expected_identity={"seed": 902})
        self.result["test_read"] = True
        write_json(self.unit / "full_result.json", self.result)
        with self.assertRaisesRegex(ValueError, "TEST role"):
            validate_complete_unit(self.unit)


class HistoricalVcsIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="n2m-historical-identity-test-")
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "FROZEN.json").write_text("synthetic frozen receipt\n")
        self.unit = self.root / "old_vcs"
        self.identity = {"method": "VCS_Weight", "scale": "sum", "seed": 901,
                         "protocol_sha256": "b" * 64,
                         "freeze_sha256": sha256_file(self.base / "FROZEN.json"),
                         "learning_rate_multiplier": 1.0, "design_rho": 0.4}
        self.result, self.manifest = _complete_io_fixture(self.unit, self.identity)
        csv = "/synthetic/old/rho0.4/s901/weights/VCS_Weight.csv"
        self.freeze = {"old_output": "/synthetic/old", "input_sha256": {csv: "a" * 64},
                       "code_sha256": {"PROTOCOL_CN.md": "b" * 64}}
        self.patch = mock.patch("v9n2m.identity.pinned_freeze", return_value=self.freeze)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_strict_historical_vcs_recipe_and_weight_identity_pass(self):
        result = validate_vcs_reference(self.unit, self.base, 901)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["weight_sha256"], "a" * 64)

    def test_wrong_historical_weight_count_and_lr_are_rejected(self):
        for field, value in (("weight_lock_sha256", "c" * 64), ("n_A", 14000),
                             ("n_U", 130499), ("n_pool", 144999), ("lr_mult", 0.5)):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest[field] = value
                write_json(self.unit / "run_manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "Historical VCS manifest"):
                    validate_vcs_reference(self.unit, self.base, 901)
        write_json(self.unit / "run_manifest.json", self.manifest)

    def test_changed_host_architecture_or_training_schedule_is_rejected(self):
        for field, value in (("embed_size", 512), ("warmup_epochs", 0), ("batch_size", 64),
                             ("learning_rate", 0.0001), ("to_verify", ["margin"])):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["cfg"][field] = value
                write_json(self.unit / "run_manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "Frozen host|unresolved recipe"):
                    validate_vcs_reference(self.unit, self.base, 901)

    def test_wrong_scale_protocol_or_seed_cannot_be_historical_vcs(self):
        for field, value in (("scale", "mean"), ("method", "BCE_Weight"), ("seed", 902),
                             ("protocol_sha256", "d" * 64), ("design_rho", 0.2)):
            with self.subTest(field=field):
                result = copy.deepcopy(self.result)
                result["identity"][field] = value
                write_json(self.unit / "full_result.json", result)
                with self.assertRaisesRegex(ValueError, "Historical VCS identity"):
                    validate_vcs_reference(self.unit, self.base, 901)


class RuntimeSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="n2m-runtime-identity-test-")
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "runtime"
        mapping = {}
        for rel, text in (("src/v9/n2/train.py", "# SYNTHETIC SOURCE IDENTITY FIXTURE\n"),
                          ("src/v9/n2/weights.py", "# SYNTHETIC WEIGHTS IDENTITY FIXTURE\n"),
                          ("HOST_LOCK.yaml", "synthetic: true\n")):
            p = self.runtime / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            mapping[rel] = sha256_file(p)
        self.receipt = {"effective_sha256": mapping, "effective_digest": canonical_sha256(mapping)}
        write_json(self.runtime / "RUNTIME.json", self.receipt)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_source_set_is_verified_and_normal_pycache_is_ignored(self):
        cache = self.runtime / "src/v9/n2/__pycache__/train.cpython-312.pyc"
        cache.parent.mkdir()
        cache.write_bytes(b"SYNTHETIC_CACHE_NOT_EXECUTED")
        self.assertEqual(verify_runtime_files(self.runtime), self.receipt)

    def test_unlisted_package_cannot_shadow_frozen_weights_module(self):
        shadow = self.runtime / "src/v9/n2/weights/__init__.py"
        shadow.parent.mkdir()
        shadow.write_text("raise RuntimeError('synthetic shadow must never import')\n")
        with self.assertRaisesRegex(ValueError, "Unregistered or missing importable"):
            verify_runtime_files(self.runtime)

    def test_unlisted_extension_module_is_rejected(self):
        shadow = self.runtime / "src/v9/n2/weights.so"
        shadow.write_bytes(b"SYNTHETIC_NOT_A_SHARED_LIBRARY")
        with self.assertRaisesRegex(ValueError, "Unregistered or missing importable"):
            verify_runtime_files(self.runtime)

    def test_modified_registered_source_is_rejected(self):
        (self.runtime / "src/v9/n2/train.py").write_text("# altered scientific loop\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_runtime_files(self.runtime)

    def test_source_symlink_escape_is_rejected_even_with_matching_bytes(self):
        original = self.runtime / "src/v9/n2/train.py"
        outside = self.root / "external.py"
        outside.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escaping path"):
            verify_runtime_files(self.runtime)

    def test_runtime_file_map_digest_is_bound(self):
        self.receipt["effective_digest"] = "0" * 64
        write_json(self.runtime / "RUNTIME.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "file-map digest"):
            verify_runtime_files(self.runtime)

    def test_missing_historical_environment_is_not_assumed_equal(self):
        target = {key: "synthetic" for key in ENVIRONMENT_KEYS}
        self.assertEqual(environment_differences(target, target), {})
        historical = dict(target)
        historical.pop("driver")
        differences = environment_differences(historical, target)
        self.assertEqual(differences["driver"]["reason"], "historical_or_target_value_not_recorded")
        altered = {**target, "torch": "different"}
        self.assertEqual(set(environment_differences(target, altered)), {"torch"})


@unittest.skipUnless(os.environ.get("V9N2M_BASELINE_ROOT"),
                     "NOT_RUN: set V9N2M_BASELINE_ROOT to the real frozen handoff/source for read-only baseline checks")
class ActualBaselineIdentityTests(unittest.TestCase):
    def setUp(self):
        source = Path(os.environ["V9N2M_BASELINE_ROOT"]).resolve()
        self.base = source if (source / "FROZEN.json").exists() else source.parent

    def test_actual_source_census_and_named_inputs(self):
        verified = verify_source(self.base)
        self.assertEqual(verified["files_checked"], 47)
        freeze = pinned_freeze(self.base)
        for seed in (901, 902, 903):
            parent = Path(freeze["old_output"]) / "rho0.4" / f"s{seed}"
            for rel in ("pairs.observed_pairs.json", "pairs.noise_manifest.json",
                        "weights/VCS_Weight.csv", "weights/VCS_Weight.csv.lock.json"):
                self.assertIn(str(parent / rel), freeze["input_sha256"])
        self.assertEqual(len(freeze["cache_names"]), 2)
        census = closeout_census(self.base, check_binary=False)
        self.assertEqual(census["expected"], 33)
        self.assertEqual(census["complete"], 0)
        self.assertFalse(census["ready_for_new_training"])
        self.assertEqual(census["metadata_complete"] + sum(r["status"] == "INCOMPLETE_OR_INVALID" for r in census["units"]), 33)

    def test_actual_historical_vcs_recipe_matches(self):
        for seed, group in ((901, "901_1"), (902, "902_1")):
            with self.subTest(seed=seed):
                result = validate_vcs_reference(self.base / "results" / group / "VCS_Weight_sum", self.base, seed)
                self.assertEqual(result["status"], "PASS")

    def test_actual_bootstrap_changes_only_pinned_train_file(self):
        with tempfile.TemporaryDirectory(prefix="n2m-real-baseline-bootstrap-test-") as tmp:
            out = Path(tmp) / "new_round"
            first = bootstrap_runtime(self.base, out)
            self.assertEqual(first["changed_source_files"], ["src/v9/n2/train.py"])
            self.assertEqual(len(first["effective_sha256"]), 33)
            self.assertEqual(bootstrap_runtime(self.base, out), first)
            self.assertEqual(verify_runtime_files(out / "runtime"), first)


if __name__ == "__main__":
    unittest.main()
