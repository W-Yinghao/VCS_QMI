"""Metadata-only regression tests. Never call Slurm or load scientific data."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from control import (POLICY_ID, capacity, digest, exit_receipt, gpu_count, parsed_job,
                     read, ref, report_ok, science_equivalence, sha, write)

SOURCE = Path(os.environ.get("V10_READY_SOURCE", Path(__file__).resolve().parents[2]/"ssl_pilot_v10_remaining"))
spec = importlib.util.spec_from_file_location("guard_under_test", SOURCE/"src/v10/supplement/contract.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        self.family = "MI_native"
        self.inventory = {"fixture.py": "abc"}
        self.policy = {"policy_id": POLICY_ID, "validated_supplement_overlap": True,
                       "v9_closeout_independent_obligation": True, "normal_gpu_limit": 8}
        self.policy_path = self.p/"POLICY.json"; write(self.policy_path, self.policy)
        self.active_path = self.p/"handoff/ACTIVE_SCHEDULING_POLICY.json"
        write(self.active_path, {"policy_id": POLICY_ID, "policy": str(self.policy_path)})
        parent = self.p/"parent"; write(parent/"PROTOCOL.json", {"fixture": True})
        acceptance = {}
        for stage in ("T01", "T02", "RUNNER"):
            path = self.p/(stage+".json")
            write(path, {"status": "PASS", "family": self.family, "code_sha256": digest(self.inventory)})
            acceptance[stage] = ref(path)
        write(self.p/"DATA.json", {"fixture": True})
        self.release = {"status": "FROZEN_VALIDATED", "source_files": self.inventory,
                        "code_sha256": digest(self.inventory), "parent_protocol_sha256": sha(parent/"PROTOCOL.json"),
                        "families": {self.family: {"candidates": C.candidates(self.family), "schedule": C.schedule(self.family)}},
                        "acceptance": {self.family: acceptance}, "scheduling_authorization": ref(self.policy_path),
                        "data_lock": ref(self.p/"DATA.json"), "comir_source": ref(self.p/"DATA.json")}
        self.path = self.p/"RELEASE.json"
        for target, value in (("PROJECT", self.p), ("PARENT", parent),
                              ("PARENT_PROTOCOL_SHA256", sha(parent/"PROTOCOL.json"))):
            context = patch.object(C, target, value); context.start(); self.addCleanup(context.stop)
        context = patch.object(C, "source_inventory", return_value=self.inventory)
        context.start(); self.addCleanup(context.stop)

    def run_guard(self):
        write(self.path, self.release)
        with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture", "V10_SUPPLEMENT_EXPECT_RELEASE": sha(self.path)}):
            return C.require_release(self.path, self.family)

    def test_authorized_overlap_needs_no_v9_result(self):
        self.assertEqual(self.run_guard(), self.release)
        self.assertFalse((self.p/"V9_CLOSEOUT.json").exists())

    def test_missing_method_acceptance_still_rejected(self):
        for key, value in (("status", "FAIL"), ("family", "NMI_native"), ("code_sha256", "different")):
            with self.subTest(key=key):
                original = read(self.p/"T02.json")
                write(self.p/"T02.json", {**original, key: value})
                self.release["acceptance"][self.family]["T02"] = ref(self.p/"T02.json")
                with self.assertRaisesRegex(RuntimeError, "Acceptance is not PASS"):
                    self.run_guard()
                write(self.p/"T02.json", original)

    def test_superseded_active_policy_rejected(self):
        write(self.active_path, {"policy_id": "later_policy", "policy": str(self.policy_path)})
        with self.assertRaisesRegex(RuntimeError, "Active policy"):
            self.run_guard()

    def test_unauthorized_overlap_rejected(self):
        write(self.policy_path, {**self.policy, "validated_supplement_overlap": False})
        self.release["scheduling_authorization"] = ref(self.policy_path)
        with self.assertRaisesRegex(RuntimeError, "Active policy"):
            self.run_guard()

    def test_changed_policy_pin_rejected(self):
        write(self.policy_path, {**self.policy, "extra": "unreviewed"})
        with self.assertRaisesRegex(RuntimeError, "authorization changed"):
            self.run_guard()

    def test_pause_still_rejected(self):
        write(self.p/"handoff/v10_parallel8_20261018/PAUSE_SUBMISSIONS", {"fixture": True})
        with self.assertRaisesRegex(RuntimeError, "pauses new V10"):
            self.run_guard()

    def test_legacy_release_still_requires_complete_v9(self):
        del self.release["scheduling_authorization"]
        for count in (32, 33):
            p = self.p/"V9_CLOSEOUT.json"; write(p, {"status": "PASS", "units": list(range(count))})
            self.release["v9_closeout"] = ref(p)
            if count == 33: self.run_guard()
            else:
                with self.assertRaisesRegex(RuntimeError, "V9 full closeout"):
                    self.run_guard()

    def test_draft_and_wrong_source_cannot_be_released(self):
        self.release["status"] = "DRAFT"
        with self.assertRaisesRegex(RuntimeError, "not been validated/frozen"):
            self.run_guard()
        self.release["status"] = "FROZEN_VALIDATED"
        self.release["code_sha256"] = "other"
        with self.assertRaisesRegex(RuntimeError, "source identity"):
            self.run_guard()


class CapacityTests(unittest.TestCase):
    @staticmethod
    def row(i, state="PENDING", qos="normal", gres="gres/gpu:1", name="v10s_c0"):
        return {"id": str(i), "name": name, "state": state, "qos": qos, "gres": gres, "reason": "fixture"}

    def test_gpu_zero_cpu_and_other_qos(self):
        self.assertEqual(gpu_count("gres/gpu:0"), 0)
        self.assertEqual(gpu_count("gres/gpu:l40s:2"), 2)
        self.assertTrue(capacity([self.row(i, qos="runfill") for i in range(40)]))
        self.assertTrue(capacity([self.row(i, gres="gres/gpu:0") for i in range(8)]))

    def test_expanded_pending_and_normal_caps(self):
        self.assertTrue(capacity([self.row(i) for i in range(7)]))
        self.assertFalse(capacity([self.row(i) for i in range(8)]))
        self.assertFalse(capacity([self.row(i, gres="N/A", name="cpu") for i in range(29)]))
        self.assertFalse(capacity([self.row(i, "RUNNING") for i in range(16)]))

    def test_ambiguous_submission_never_accepted(self):
        for rc, output in ((1, ""), (0, ""), (0, "123\n124\n")):
            with self.assertRaises(RuntimeError):
                parsed_job(types.SimpleNamespace(returncode=rc, stdout=output))
        self.assertEqual(parsed_job(types.SimpleNamespace(returncode=0, stdout="123;cluster\n")), "123")

    def test_dispatch_matrix_and_no_duplicate_submission(self):
        import dispatch as D
        units = D.matrix(C)
        self.assertEqual(len(units), 36)
        self.assertEqual(sum(u["phase"] == "final" for u in units), 15)
        self.assertEqual(sum(len(C.candidates(f)) for f in C.FAMILIES), 19)
        self.assertEqual(len({u["key"] for u in units}), 36)
        with patch.object(D.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                D.submit({"status": "SUBMITTED", "job": "123"}, {}, {})
            run.assert_not_called()

    def test_failed_gate_blocks_only_its_own_family(self):
        import dispatch as D
        ledger = {"units": D.matrix(C)}
        by_key = {u["key"]: u for u in ledger["units"]}
        by_key['gate_0']['status'] = 'FAILED_REVIEW_REQUIRED'
        by_key['gate_1']['status'] = 'COMPLETE_VERIFIED'
        D.advance_waiters(ledger)
        self.assertEqual(by_key['full_0']['status'], 'BLOCKED_UPSTREAM_FAILURE')
        self.assertEqual(by_key['cal_0']['status'], 'BLOCKED_UPSTREAM_FAILURE')
        self.assertEqual(by_key['final_0_1001']['status'], 'BLOCKED_UPSTREAM_FAILURE')
        self.assertEqual(by_key['full_1']['status'], 'READY')
        self.assertEqual(by_key['cal_1']['status'], 'WAIT_FULL')
        self.assertEqual(by_key['gate_2']['status'], 'READY')

    def test_full_success_releases_only_that_methods_cal(self):
        import dispatch as D
        ledger = {"units": D.matrix(C)}
        by_key = {u["key"]: u for u in ledger["units"]}
        by_key['full_0']['status'] = 'COMPLETE_VERIFIED'
        D.advance_waiters(ledger)
        self.assertEqual(by_key['cal_0']['status'], 'READY_TO_RELEASE')
        self.assertEqual(by_key['cal_1']['status'], 'WAIT_FULL')
        self.assertEqual(by_key['final_0_1001']['status'], 'WAIT_CAL')


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)

    def test_science_change_cannot_use_scheduling_migration(self):
        name = "src/v10/supplement/contract.py"
        old, new = self.p/"old", self.p/"new"
        for root in (old, new): (root/name).parent.mkdir(parents=True)
        (old/name).write_text("def recipe():\n    return 3\ndef require_release():\n    return 0\n")
        (new/name).write_text("def recipe():\n    return 3\ndef require_release():\n    return 1\n")
        result = science_equivalence(old, new, {name: sha(old/name)}, {name: sha(new/name)})
        self.assertEqual(result["status"], "PASS")
        (new/name).write_text("def recipe():\n    return 4\ndef require_release():\n    return 1\n")
        with self.assertRaisesRegex(RuntimeError, "Method recipes"):
            science_equivalence(old, new, {name: sha(old/name)}, {name: sha(new/name)})

    def test_wrong_validation_job_or_hash_cannot_pass(self):
        p = self.p/"report.json"
        base = {"status": "PASS", "family": "MI_native", "code_sha256": "code", "job": "10",
                "validation_manifest_sha256": "manifest", "held_test_read": False, "dev_read": False}
        write(p, base); report_ok(p, "MI_native", "code", "manifest", "10")
        for key, value in (("job", "11"), ("code_sha256", "other"), ("dev_read", True), ("status", "FAIL")):
            write(p, {**base, key: value})
            with self.assertRaises(RuntimeError): report_ok(p, "MI_native", "code", "manifest", "10")

    def test_failed_or_missing_receipt_never_proves_completion(self):
        p = self.p/"exit.json"
        base = {"job": "10", "exit_code": 0, "release_sha256": "pin", "allocated_seconds": 12}
        write(p, base); exit_receipt(p, "10", "pin", "release_sha256")
        for key, value in (("exit_code", 1), ("job", "11"), ("release_sha256", "other"), ("allocated_seconds", -1)):
            write(p, {**base, key: value})
            with self.assertRaises(RuntimeError): exit_receipt(p, "10", "pin", "release_sha256")


if __name__ == "__main__":
    unittest.main(verbosity=2)
