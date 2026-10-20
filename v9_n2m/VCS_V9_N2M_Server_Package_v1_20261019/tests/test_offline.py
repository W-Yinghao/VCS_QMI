"""Synthetic scorer checks; these do not use real data or a trained model."""
import csv
import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from v9n2m.common import read_json, sha256_file, write_json
from v9n2m.controls import generate_controls
from v9n2m.offline import probability_metrics, score_weights, score_fixed_losses


class ProbabilityMetricTests(unittest.TestCase):
    def test_ties_receive_half_roc_credit_and_grouped_ap(self):
        m = probability_metrics([0, 1, 0, 1], [.1, .5, .5, .9])
        self.assertAlmostEqual(m["auroc"], .875)
        self.assertAlmostEqual(m["auprc_average_precision"], 5 / 6)
        # Swapping differently labelled rows within a tie cannot change metrics.
        other = probability_metrics([0, 0, 1, 1], [.1, .5, .5, .9])
        self.assertEqual(m["auroc"], other["auroc"])
        self.assertEqual(m["auprc_average_precision"], other["auprc_average_precision"])

    def test_constant_is_half_auc_and_prevalence_ap(self):
        m = probability_metrics([1, 1, 1, 0], [.6] * 4)
        self.assertEqual(m["auroc"], .5)
        self.assertEqual(m["auprc_average_precision"], .75)
        self.assertAlmostEqual(m["ess"], 4)
        self.assertAlmostEqual(m["residual_noise_mass_fraction"], .25)

    def test_label_polarity_and_retained_mass(self):
        m = probability_metrics([1, 0], [.9, .2])
        self.assertEqual(m["auroc"], 1)
        self.assertAlmostEqual(m["clean_mean_retention"], .9)
        self.assertAlmostEqual(m["mismatched_mean_retention"], .2)
        self.assertAlmostEqual(m["residual_noise_mass_fraction"], .2 / 1.1)
        self.assertAlmostEqual(m["brier"], .025)

    def test_fixed_bins_include_endpoints_and_empty_bins(self):
        m = probability_metrics([0, 1, 0, 1], [0, .1, .9, 1])
        self.assertEqual([r["n"] for r in m["calibration"]], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertIsNone(m["calibration"][2]["clean_fraction"])
        self.assertEqual(m["logloss_n_clipped"], 2)
        self.assertTrue(math.isfinite(m["logloss"]))

    def test_single_class_and_zero_mass_are_explicit(self):
        m = probability_metrics([0, 0], [0, 0])
        self.assertIsNone(m["auroc"])
        self.assertIsNone(m["auprc_average_precision"])
        self.assertIsNone(m["residual_noise_mass_fraction"])
        self.assertEqual(m["ess"], 0)
        m = probability_metrics([1, 1], [.3, .8])
        self.assertIsNone(m["auroc"])
        self.assertEqual(m["auprc_average_precision"], 1)

    def test_invalid_numeric_values_rejected(self):
        for w in ([float("nan")], [float("inf")], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                probability_metrics([1], w)
        with self.assertRaises(ValueError):
            probability_metrics([.5], [.5])


class OfflineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.observed = self.root / "pairs.observed_pairs.json"
        self.labels = self.root / "eval_only" / "pairs.eval_only.json"
        self.vcs = self.root / "VCS_Weight.csv"
        self.lock = Path(str(self.vcs) + ".lock.json")
        self.controls = self.root / "controls"
        self.noise = self.root / "pairs.noise_manifest.json"
        rows = [{"pair_id": "a0", "caption_id": "99", "observed_image_id": "99", "role": "critic_fit"}]
        labels = {"a0": {"original_image_id": "99", "mismatched": False}}
        scores = np.linspace(-1, 1, 10)
        a, b, rho = 2.5, -.2, .4
        t = np.tanh(a * scores + b)
        w = (1-rho)*(1+t) / ((1-rho)*(1+t)+rho*(1-t))
        for i in range(10):
            image = str(i // 5)
            rows.append({"pair_id": f"p{i}", "caption_id": str(i), "observed_image_id": image, "role": "U"})
            mismatch = i in (0, 2, 5, 7)
            labels[f"p{i}"] = {"original_image_id": str(1-int(image)) if mismatch else image, "mismatched": mismatch}
        write_json(self.observed, rows)
        write_json(self.labels, labels)
        with self.vcs.open("w", newline="") as handle:
            wr = csv.writer(handle)
            wr.writerow(["pair_id", "w", "score", "rho_hat", "fit_hash", "reference_hash"])
            for i in range(10):
                wr.writerow([f"p{i}", f"{w[i]:.10f}", f"{scores[i]:.10f}", f"{rho:.8f}", "synthetic", "synthetic"])
        write_json(self.lock, {"sha256": sha256_file(self.vcs), "n": 10,
                              "record": {"method": "VCS_Weight", "critic": {"a": a, "b": b}, "rho_used": rho,
                                         "rho": {"raw": rho, "projected": rho, "a": .6, "b": -.2, "c": .28, "denominator": .8}}})
        mapping = hashlib.sha256("\n".join(f"{r['pair_id']}\t{r['observed_image_id']}" for r in rows).encode()).hexdigest()
        write_json(self.noise, {"design_rho": .4, "realized_rho": .4, "n_pairs": 11, "n_U_pairs": 10,
                               "n_mismatched": 4, "n_A_mismatched": 0, "mapping_sha256": mapping,
                               "observed_sha256": sha256_file(self.observed), "eval_only_sha256": sha256_file(self.labels)})
        generate_controls(self.observed, self.vcs, self.lock, 901, self.controls, expected_n_pool=11, expected_n_u=10)

    def score(self, **kwargs):
        return score_weights(self.observed, self.labels, self.vcs, self.lock, self.controls, self.root / "readout", **kwargs)

    def test_end_to_end_frozen_score_and_oracle_rho(self):
        report = self.score(noise_manifest_path=self.noise)
        self.assertEqual(report["arms"]["VCS"]["n_clean"], 6)
        self.assertEqual(report["arms"]["VCS"]["n_mismatched"], 4)
        self.assertAlmostEqual(report["rho"]["realized"], .4)
        self.assertTrue(report["production_reconstruction"]["consistent"])
        self.assertTrue(report["production_reconstruction"]["inverse_T_interval_check"]["all_inside"])
        self.assertTrue(report["same_T_oracle_rho"]["available"])
        self.assertTrue(report["same_T_oracle_rho"]["same_critic_no_refit"])
        self.assertAlmostEqual(report["arms"]["ConstantU"]["ess"], 10)
        self.assertAlmostEqual(report["arms"]["ConstantU"]["auroc"], .5)
        self.assertAlmostEqual(report["arms"]["ConstantU"]["auprc_average_precision"], .6)
        self.assertEqual(read_json(self.root / "readout" / "offline_metrics.json")["source"], report["source"])
        self.assertTrue((self.root / "readout" / "calibration_10bins.csv").is_file())

    def test_wrong_map_fails_before_evaluator_labels_are_read(self):
        rows = read_json(self.observed)
        rows[1]["observed_image_id"] = "3"
        write_json(self.observed, rows)
        with patch("v9n2m.offline._observed_and_labels", side_effect=AssertionError("label access happened")):
            with self.assertRaisesRegex(ValueError, "source identity"):
                self.score()

    def test_changed_control_lock_fails_before_labels(self):
        lock = self.controls / "ConstantU.csv.lock.json"
        obj = read_json(lock)
        obj["record"]["seed"] = 999
        write_json(lock, obj)
        with patch("v9n2m.offline._observed_and_labels", side_effect=AssertionError("label access happened")):
            with self.assertRaisesRegex(ValueError, "control CSV/lock"):
                self.score()

    def test_label_polarity_must_match_actual_mapping(self):
        labels = read_json(self.labels)
        labels["p0"]["mismatched"] = False
        write_json(self.labels, labels)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.score()

    def test_wrong_manifest_is_rejected(self):
        manifest = read_json(self.noise)
        manifest["realized_rho"] = .5
        write_json(self.noise, manifest)
        with self.assertRaisesRegex(ValueError, "realized_rho"):
            self.score(noise_manifest_path=self.noise)

    def fixed_export(self, *, include_hash=True, include_batches=True):
        path = self.root / "exported_fixed_losses.csv"
        rows = read_json(self.observed)
        indices = [0, 1, 2]
        with path.open("w", newline="") as handle:
            wr = csv.writer(handle)
            wr.writerow(["batch_id", "pair_id", "ell"])
            wr.writerows([[0, rows[i]["pair_id"], i + 1] for i in indices])
        meta = {"schema_version": 1, "seed": 901, "checkpoint_sha256": "1"*64,
                "experiment_sha256": "2"*64, "observed_sha256": sha256_file(self.observed),
                "batch_definition": "synthetic first fixed batch", "ell_definition": "synthetic row+column losses",
                "same_negative_pool": True, "weights_applied_during_export": False,
                "evaluator_labels_read": False, "test_read": False}
        if include_hash:
            meta["loss_csv_sha256"] = sha256_file(path)
        if include_batches:
            meta["batches"] = [{"batch_id": 0, "n": 3, "pool_indices": indices}]
        write_json(str(path)+".meta.json", meta)
        return path, meta

    def test_fixed_export_hash_and_ordered_batch_binding(self):
        path, meta = self.fixed_export()
        report = score_fixed_losses(self.observed, self.labels, self.controls, path, self.root / "risk",
                                    expected_experiment_sha256="2"*64, expected_checkpoint_sha256="1"*64)
        self.assertTrue(report["identity"]["csv_identity_verified"])
        self.assertTrue(report["identity"]["batch_identity_verified"])
        self.assertTrue(report["identity"]["experiment_identity_verified"])
        self.assertTrue(report["identity"]["checkpoint_identity_compared_with_expected"])
        self.assertFalse(report["scientific_training_complete"])

    def test_fixed_export_wrong_csv_digest_fails_before_labels(self):
        path, meta = self.fixed_export()
        with path.open("a") as handle:
            handle.write("0,p2,1\n")
        with patch("v9n2m.offline._observed_and_labels", side_effect=AssertionError("label access happened")):
            with self.assertRaisesRegex(ValueError, "loss_csv_sha256"):
                score_fixed_losses(self.observed, self.labels, self.controls, path, self.root / "risk")

    def test_fixed_export_without_csv_hash_is_explicitly_unverified(self):
        path, meta = self.fixed_export(include_hash=False)
        report = score_fixed_losses(self.observed, self.labels, self.controls, path, self.root / "risk")
        self.assertFalse(report["identity"]["csv_identity_verified"])
        self.assertTrue(any("no loss_csv_sha256" in w for w in report["warnings"]))
        self.assertFalse(report["scientific_training_complete"])

    def test_fixed_export_wrong_batch_order_fails_before_labels(self):
        path, meta = self.fixed_export()
        meta["batches"][0]["pool_indices"] = [0, 2, 1]
        write_json(str(path)+".meta.json", meta)
        with patch("v9n2m.offline._observed_and_labels", side_effect=AssertionError("label access happened")):
            with self.assertRaisesRegex(ValueError, "pair order"):
                score_fixed_losses(self.observed, self.labels, self.controls, path, self.root / "risk")

    def test_fixed_export_wrong_observed_source_fails_before_labels(self):
        path, meta = self.fixed_export()
        rows = read_json(self.observed)
        rows[1]["observed_image_id"] = "5"
        write_json(self.observed, rows)
        with patch("v9n2m.offline._observed_and_labels", side_effect=AssertionError("label access happened")):
            with self.assertRaisesRegex(ValueError, "differs from frozen controls"):
                score_fixed_losses(self.observed, self.labels, self.controls, path, self.root / "risk")

    def test_fixed_export_binds_requested_experiment_and_checkpoint(self):
        path, meta = self.fixed_export()
        for kwargs in ({"expected_experiment_sha256": "3"*64}, {"expected_checkpoint_sha256": "3"*64}):
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                score_fixed_losses(self.observed, self.labels, self.controls, path, self.root / "risk", **kwargs)

    def test_optional_fixed_loss_reuses_same_anchor_losses(self):
        loss_path = self.root / "frozen_losses.csv"
        with loss_path.open("w", newline="") as handle:
            wr = csv.writer(handle)
            wr.writerow(["batch_id", "pair_id", "ell"])
            wr.writerows([["b0", "a0", 2], ["b0", "p0", 1], ["b0", "p1", 3]])
        write_json(str(loss_path) + ".meta.json", {"checkpoint_sha256": "1" * 64,
                    "observed_sha256": sha256_file(self.observed), "batch_definition": "synthetic ordered batch",
                    "ell_definition": "synthetic nonnegative losses", "same_negative_pool": True})
        report = score_fixed_losses(self.observed, self.labels, self.controls, loss_path, self.root / "risk")
        self.assertEqual(len(report["batches"]), 4)
        oracle = next(r for r in report["batches"] if r["arm"] == "CleanOracle_EVAL_ONLY")
        self.assertAlmostEqual(oracle["sum_objective"], 5 / (7 / 11))
        self.assertTrue(report["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
