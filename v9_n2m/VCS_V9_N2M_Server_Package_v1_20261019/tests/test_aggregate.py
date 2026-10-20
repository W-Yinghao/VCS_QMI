"""Synthetic artifact and resume-location tests; no trained model is implied."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v9n2m.aggregate import RECALLS, collapse_diagnostic, summarize_experiment, verify_analysis_metadata
from v9n2m.common import ARMS, PACKAGE_ROOT, SEEDS, read_json, sha256_file, write_json


def epoch_record(epoch, rsum=400.0):
    return {"epoch": epoch, "steps_in_epoch": 1133, "dev": {**{k: rsum/6 for k in RECALLS}, "rsum": rsum, "mR": rsum/6},
            "lr": .0002 if epoch < 30 else .00002, "phase": "warmup_mean_hinge" if epoch < 5 else "max_violation",
            "mean_loss": 20., "mean_u_risk_unnormalised": .2, "seconds": (epoch + 1) * 10.}


def collapse_log(resume=None, low_epochs=(5, 6), high_risk=True, explicit=False):
    records = [epoch_record(e, 100. if e in low_epochs else 400.) for e in range(40)]
    samples = []
    counter = 0
    for e in range(40):
        if resume == e:
            counter = len(samples)  # Exactly the frozen original trainer bug.
        for k in range(1, 1134):
            counter += 1
            if counter % 20 == 0:
                row = {"epoch": e, "step": counter, "u_risk_unnormalised": .4 if high_risk and e in low_epochs and k > 933 else .2}
                if explicit:
                    row["step_in_epoch"] = k
                samples.append(row)
    out = {"epochs": records, "steps": samples}
    if resume is not None:
        out["resumed_at"] = [resume]
    return out


class CollapseTests(unittest.TestCase):
    def test_exact_final_200_updates(self):
        m = collapse_diagnostic(collapse_log())
        self.assertTrue(m["severe_collapse"])
        e5 = next(r for r in m["epochs"] if r["epoch"] == 5)
        self.assertEqual(e5["n_samples"], 10)
        self.assertTrue(min(e5["sample_positions_in_epoch"]) > 933)
        self.assertEqual(e5["fraction_U_risk_ge_0p39"], 1)

    def test_legacy_resume_counter_is_located_correctly(self):
        log = collapse_log(resume=5)
        self.assertLess(next(r["step"] for r in log["steps"] if r["epoch"] == 5), 1000)
        m = collapse_diagnostic(log)
        self.assertTrue(m["severe_collapse"])
        for e in (5, 6):
            rec = next(r for r in m["epochs"] if r["epoch"] == e)
            self.assertTrue(rec["sample_coverage_valid"])
            self.assertEqual(rec["n_samples"], 10)
            self.assertIn("reconstructed_legacy_counter_and_resume", rec["location_methods"])

    def test_explicit_locations_survive_different_counter(self):
        m = collapse_diagnostic(collapse_log(resume=5, explicit=True))
        self.assertTrue(m["severe_collapse"])
        self.assertIn("explicit_step_in_epoch", m["epochs"][0]["location_methods"])

    def test_low_dev_alone_does_not_satisfy_collapse(self):
        self.assertFalse(collapse_diagnostic(collapse_log(high_risk=False))["severe_collapse"])
        self.assertFalse(collapse_diagnostic(collapse_log(low_epochs=(5, 7)))["severe_collapse"])

    def test_missing_tail_sample_is_unknown_not_false(self):
        log = collapse_log()
        final_e5 = [i for i, r in enumerate(log["steps"]) if r["epoch"] == 5][-1]
        del log["steps"][final_e5]
        m = collapse_diagnostic(log)
        self.assertIsNone(m["severe_collapse"])
        self.assertEqual(m["status"], "UNKNOWN")

    def test_incomplete_trajectory_never_certifies_no_collapse(self):
        log = collapse_log(low_epochs=())
        log["epochs"] = log["epochs"][:7]
        self.assertIsNone(collapse_diagnostic(log)["severe_collapse"])


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "experiment.json"
        self.exp = {"schema_version": 1, "protocol_sha256": sha256_file(PACKAGE_ROOT / "config/protocol.json"),
                    "analysis_parameters": read_json(PACKAGE_ROOT / "config/protocol.json")["analysis"], "base_freeze_sha256": "b"*64,
                    "units": [{"unit_id": f"s{s}__{a}", "seed": s, "arm": a, "action": "RUN",
                               "output_dir": str(self.root / f"s{s}__{a}"), "weights_sha256": "w"*64}
                              for s in SEEDS for a in ARMS]}
        self.freeze_experiment()
        for u in self.exp["units"]:
            self.make_unit(u)

    def freeze_experiment(self):
        write_json(self.path, self.exp)
        (self.root / "EXPERIMENT.sha256").write_text(sha256_file(self.path) + "  experiment.json\n")

    def make_unit(self, u):
        out = Path(u["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        offset = {"VCS": 10., "ConstantU": 0., "ShuffleVCS": 3.}[u["arm"]]
        log = {"epochs": [epoch_record(e, 300 + 2*e + offset) for e in range(40)], "steps": []}
        identity = {"unit_id": u["unit_id"], "seed": u["seed"], "method": u["arm"], "arm": u["arm"], "scale": "sum",
                    "protocol_sha256": self.exp["protocol_sha256"], "experiment_sha256": sha256_file(self.path),
                    "weights_sha256": u["weights_sha256"]}
        manifest = {"n_pool": 145000, "n_A": 14500, "n_U": 130500, "seed": u["seed"], "lr_mult": 1.,
                    "weight_lock_sha256": u["weights_sha256"], "cfg": {"batch_size": 128, "num_epochs": 40, "warmup_epochs": 5,
                     "learning_rate": .0002, "margin": .2, "grad_clip": 2., "lr_update": 30}}
        write_json(out / "identity.json", identity)
        write_json(out / "train_log.json", log)
        write_json(out / "run_manifest.json", manifest)
        # Synthetic opaque bytes are sufficient to test integrity bookkeeping.
        (out / "best.pt").write_bytes(b"synthetic-not-a-torch-checkpoint-best")
        (out / "last.pt").write_bytes(b"synthetic-not-a-torch-checkpoint-last")
        result = {"identity": identity, "done": {"epochs_run": 40, "global_step": 45320, "best": {"epoch": 39, "rsum": 378+offset}, "seconds": 400.},
                  "hardware": {"gpu": "SYNTHETIC"}, "n_pool": 145000, "n_U": 130500, "full_epochs": 40, "test_read": False,
                  "artifact_sha256": {name: sha256_file(out/name) for name in ("identity.json", "run_manifest.json", "train_log.json", "best.pt", "last.pt")}}
        write_json(out / "full_result.json", result)
        write_json(out / "attempts" / "a.json", {"status": "SUCCEEDED", "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-01-01T00:06:40Z", "elapsed_s": 400., "job_id": "synthetic"})

    def summarize(self):
        return summarize_experiment(self.path, self.root / "readout")

    def refresh_log_hash(self, unit):
        out = Path(unit["output_dir"])
        r = read_json(out / "full_result.json")
        r["artifact_sha256"]["train_log.json"] = sha256_file(out / "train_log.json")
        write_json(out / "full_result.json", r)

    def test_complete_matrix_produces_only_two_prespecified_contrasts(self):
        report = self.summarize()
        self.assertEqual(report["n_complete_verified_units"], 9)
        self.assertEqual(len(report["contrasts"]), 2)
        self.assertEqual([c["mean_difference"] for c in report["contrasts"]], [10, 7])
        self.assertTrue(report["investment_rule"]["pass"])
        self.assertTrue(all(c["p_value"] is None and c["confidence_interval"] is None for c in report["contrasts"]))
        self.assertTrue((self.root / "readout" / "epoch_trajectories.csv").exists())

    def test_incomplete_unit_never_uses_its_current_best(self):
        u = self.exp["units"][-1]
        out = Path(u["output_dir"])
        (out / "full_result.json").unlink()
        log = read_json(out / "train_log.json")
        log["epochs"] = log["epochs"][:12]
        write_json(out / "train_log.json", log)
        report = self.summarize()
        row = next(r for r in report["units"] if r["unit_id"] == u["unit_id"])
        self.assertIsNone(row["primary_rsum"])
        self.assertIsNotNone(row["current_best_diagnostic"])
        contrast = report["contrasts"][1]
        self.assertEqual(contrast["n_eligible_pairs"], 2)
        self.assertIsNone(contrast["mean_difference"])
        self.assertIsNone(contrast["investment_rule_pass"])

    def test_checkpoint_identity_mismatch_is_invalid(self):
        u = self.exp["units"][0]
        (Path(u["output_dir"]) / "best.pt").write_bytes(b"changed")
        row = self.summarize()["units"][0]
        self.assertEqual(row["status"], "INVALID")
        self.assertIsNone(row["primary_rsum"])
        self.assertIn("HASH_MISMATCH", row["artifact_verification"].values())

    def test_step_count_and_six_recall_consistency_are_enforced(self):
        u = self.exp["units"][0]
        out = Path(u["output_dir"])
        log = read_json(out / "train_log.json")
        log["epochs"][4]["steps_in_epoch"] = 1132
        log["epochs"][8]["dev"]["i2t_R1"] += 1
        write_json(out / "train_log.json", log)
        self.refresh_log_hash(u)
        row = self.summarize()["units"][0]
        self.assertEqual(row["status"], "INVALID")
        self.assertTrue(any("45320" in e for e in row["errors"]))
        self.assertTrue(any("six Recalls" in e for e in row["errors"]))

    def test_best_tie_must_choose_earlier_epoch(self):
        u = self.exp["units"][0]
        out = Path(u["output_dir"])
        log = read_json(out / "train_log.json")
        log["epochs"][38]["dev"] = dict(log["epochs"][39]["dev"])
        write_json(out / "train_log.json", log)
        self.refresh_log_hash(u)
        row = self.summarize()["units"][0]
        self.assertEqual(row["status"], "INVALID")
        self.assertTrue(any("earliest tie" in e for e in row["errors"]))

    def test_unknown_attempt_duration_not_zero(self):
        u = self.exp["units"][0]
        out = Path(u["output_dir"])
        write_json(out / "attempts" / "b.json", {"status": "INTERRUPTED", "start_utc": "2026-01-02T00:00:00Z", "job_id": "synthetic2"})
        cost = self.summarize()["units"][0]["cost"]
        self.assertFalse(cost["duration_complete"])
        self.assertEqual(cost["known_attempt_seconds"], 400)
        self.assertEqual(cost["unknown_or_unclosed_attempts"], 1)

    def test_historical_reference_preserves_provenance(self):
        u = self.exp["units"][0]
        out = Path(u["output_dir"])
        u["action"] = "REUSE"
        u["reference_source_dir"] = str(out)
        self.freeze_experiment()
        # Bind the other eight newly generated result identities to new manifest.
        for other in self.exp["units"][1:]:
            self.make_unit(other)
        identity = read_json(out / "identity.json")
        identity.pop("arm")
        identity["method"] = "VCS_Weight"
        identity["freeze_sha256"] = self.exp["base_freeze_sha256"]
        write_json(out / "identity.json", identity)
        result = read_json(out / "full_result.json")
        result["identity"] = identity
        result["artifact_sha256"]["identity.json"] = sha256_file(out / "identity.json")
        write_json(out / "full_result.json", result)
        row = self.summarize()["units"][0]
        self.assertEqual(row["status"], "COMPLETE_VERIFIED")
        self.assertEqual(row["provenance"], "historical_frozen_VCS")

    def test_metadata_only_checkpoints_are_not_primary_results(self):
        u = self.exp["units"][0]
        out = Path(u["output_dir"])
        result = read_json(out / "full_result.json")
        result["checkpoints_not_materialized"] = True
        write_json(out / "full_result.json", result)
        (out / "best.pt").unlink()
        (out / "last.pt").unlink()
        row = self.summarize()["units"][0]
        self.assertEqual(row["status"], "COMPLETE_METADATA_ONLY")
        self.assertFalse(row["complete_for_primary"])
        self.assertIsNone(row["primary_rsum"])
        self.assertIsNotNone(row["current_best_diagnostic"])

    def test_reuse_result_is_bound_to_frozen_file_digest(self):
        u = self.exp["units"][0]
        out = Path(u["output_dir"])
        u["action"] = "REUSE"
        u["reference_source_dir"] = str(out)
        identity = read_json(out / "identity.json")
        identity["freeze_sha256"] = self.exp["base_freeze_sha256"]
        write_json(out / "identity.json", identity)
        result = read_json(out / "full_result.json")
        result["identity"] = identity
        result["artifact_sha256"]["identity.json"] = sha256_file(out / "identity.json")
        write_json(out / "full_result.json", result)
        u["reference_result_sha256"] = sha256_file(out / "full_result.json")
        self.freeze_experiment()
        for other in self.exp["units"][1:]:
            self.make_unit(other)
        result["changed_after_freeze"] = True
        write_json(out / "full_result.json", result)
        row = self.summarize()["units"][0]
        self.assertEqual(row["status"], "INVALID")
        self.assertTrue(any("reference_result_sha256" in e for e in row["errors"]))

    def test_malformed_unit_does_not_abort_remaining_summary(self):
        u = self.exp["units"][0]
        out = Path(u["output_dir"])
        write_json(out / "train_log.json", {"epochs": None, "steps": None, "resumed_at": None})
        self.refresh_log_hash(u)
        report = self.summarize()
        self.assertEqual(report["n_complete_verified_units"], 8)
        self.assertEqual(report["units"][0]["status"], "INVALID")
        self.assertTrue(any("schema" in e for e in report["units"][0]["errors"]))

    def test_done_manifest_digest_must_match_actual_manifest(self):
        u = self.exp["units"][0]
        out = Path(u["output_dir"])
        result = read_json(out / "full_result.json")
        result["done"]["manifest_sha256"] = "0" * 64
        write_json(out / "full_result.json", result)
        row = self.summarize()["units"][0]
        self.assertEqual(row["status"], "INVALID")
        self.assertTrue(any("done.manifest_sha256" in e for e in row["errors"]))

    def test_analysis_rejects_a_changed_experiment_receipt(self):
        self.exp["unrecorded_change"] = True
        write_json(self.path, self.exp)  # Deliberately do not refresh the receipt.
        with self.assertRaisesRegex(ValueError, "EXPERIMENT.sha256"):
            self.summarize()

    def test_analysis_requires_receipt_but_not_original_training_files(self):
        self.exp["runtime_dir"] = "/does/not/exist/runtime"
        self.exp["inputs"] = {"data": "/does/not/exist/features", "vocab": "/does/not/exist/vocab"}
        self.freeze_experiment()
        for u in self.exp["units"]:
            self.make_unit(u)
        report = self.summarize()
        self.assertEqual(report["n_complete_verified_units"], 9)
        self.assertFalse(report["analysis_identity"]["original_training_inputs_read"])
        (self.root / "EXPERIMENT.sha256").unlink()
        with self.assertRaises(FileNotFoundError):
            self.summarize()

    def test_analysis_parameters_cannot_drift_from_the_frozen_protocol(self):
        self.exp["analysis_parameters"]["investment_rule"]["mean_rsum_difference_each_at_least"] = 0
        self.freeze_experiment()
        with self.assertRaisesRegex(ValueError, "analysis_parameters"):
            self.summarize()

    def test_investment_threshold_is_read_from_verified_protocol_parameters(self):
        # A separate synthetic protocol fixture demonstrates that the readout
        # consumes the frozen number; it does not modify the real package.
        package = self.root / "synthetic_package"
        protocol = read_json(PACKAGE_ROOT / "config/protocol.json")
        protocol["analysis"]["investment_rule"]["mean_rsum_difference_each_at_least"] = 11.0
        pp = package / "config/protocol.json"
        write_json(pp, protocol)
        self.exp["protocol_sha256"] = sha256_file(pp)
        self.exp["analysis_parameters"] = protocol["analysis"]
        self.freeze_experiment()
        for u in self.exp["units"]:
            self.make_unit(u)
        with patch("v9n2m.aggregate.PACKAGE_ROOT", package):
            report = self.summarize()
        self.assertFalse(report["investment_rule"]["pass"])
        self.assertEqual(report["investment_rule"]["frozen_parameters"]["mean_rsum_difference_each_at_least"], 11.0)
        self.assertEqual([c["mean_difference"] for c in report["contrasts"]], [10, 7])

    def test_incomplete_matrix_manifest_cannot_silently_shrink(self):
        self.exp["units"].pop()
        self.freeze_experiment()
        with self.assertRaisesRegex(ValueError, "three arms"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
