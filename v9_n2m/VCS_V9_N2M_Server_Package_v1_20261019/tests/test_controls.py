"""Synthetic mathematical and identity checks; no training/GPU claims."""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from v9n2m.common import read_json, sha256_file, sort_id, write_csv, write_json
from v9n2m.controls import generate_controls, load_observed, load_weights


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="v9n2m-controls-test-")
        self.root = Path(self.tmp.name)
        self.observed = self.root / "pairs.observed_pairs.json"
        self.csv = self.root / "VCS_Weight.csv"
        self.lock = self.root / "VCS_Weight.csv.lock.json"
        self.out = self.root / "controls"
        self.make_fixture()

    def tearDown(self):
        self.tmp.cleanup()

    def make_fixture(self, images=None, weights=None):
        images = images if images is not None else ["10", "2", "x", "-3", "01", "1", "0", "z"]
        rows = [{"pair_id": f"a{j}", "caption_id": str(j), "observed_image_id": "anchor", "role": "teacher_warmup"}
                for j in range(5)]
        urows = [{"pair_id": f"p{100 + k * 5 + j}", "caption_id": str(100 + k * 5 + j),
                  "observed_image_id": image_id, "role": "U"}
                 for k, image_id in enumerate(images) for j in range(5)]
        # Input row order deliberately differs from image-block and caption order.
        rows.extend(urows[::-1])
        self.rows = rows
        self.ids = [row["pair_id"] for row in rows if row["role"] == "U"]
        if weights is None:
            weights = np.linspace(0.012345678901234, 0.987654321098765, len(self.ids), dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        write_json(self.observed, rows)
        write_csv(self.csv, [{"pair_id": pair_id, "w": format(float(weight), ".17g"),
                             "score": format(k / 100.0, ".10f"), "rho_hat": "0.4",
                             "fit_hash": "fit", "reference_hash": "reference"}
                            for k, (pair_id, weight) in enumerate(zip(self.ids, self.weights))],
                  ["pair_id", "w", "score", "rho_hat", "fit_hash", "reference_hash"])
        self.write_source_lock()

    def write_source_lock(self, **record_overrides):
        write_json(self.lock, {"file": str(self.csv), "sha256": sha256_file(self.csv), "n": len(self.ids),
                               "record": {"method": "VCS_Weight", "rho_strategy": "estimated",
                                          "deployable": True, **record_overrides}})

    def generate(self, **kwargs):
        return generate_controls(self.observed, self.csv, self.lock, 901, self.out,
                                 expected_n_pool=len(self.rows), expected_n_u=len(self.ids), **kwargs)

    def arm_values(self, manifest, arm):
        entry = manifest["arms"][arm]
        weights = load_weights(entry["csv"], self.ids, entry["lock"])
        return np.asarray([weights[pair_id] for pair_id in self.ids], dtype=np.float64)

    def test_controls_preserve_mass_and_exact_shuffle_multiset(self):
        manifest = self.generate()
        vcs, constant, shuffled = [self.arm_values(manifest, arm) for arm in ("VCS", "ConstantU", "ShuffleVCS")]
        np.testing.assert_array_equal(vcs, self.weights)
        np.testing.assert_array_equal(constant, np.full(len(vcs), np.mean(vcs, dtype=np.float64)))
        np.testing.assert_array_equal(np.sort(shuffled), np.sort(vcs))
        self.assertEqual(sha256_file(self.csv), manifest["arms"]["VCS"]["sha256"])
        for arm in ("ConstantU", "ShuffleVCS"):
            self.assertTrue(math.isclose(manifest["arms"][arm]["total_mass_U"], float(vcs.sum()), rel_tol=1e-12))
            self.assertTrue(math.isclose(manifest["arms"][arm]["mean_v_full_pool"], (5 + vcs.sum()) / len(self.rows), rel_tol=1e-12))
        self.assertAlmostEqual(manifest["arms"]["ConstantU"]["ess_U"], len(self.ids), places=10)
        self.assertGreater(manifest["arms"]["ConstantU"]["ess_U"], manifest["arms"]["VCS"]["ess_U"])
        self.assertAlmostEqual(manifest["arms"]["ShuffleVCS"]["ess_U"], manifest["arms"]["VCS"]["ess_U"], places=10)
        self.assertFalse(manifest["label_fields_read"])

    def test_prescribed_permutation_direction_and_block_positions(self):
        manifest = self.generate()
        pi = np.load(self.out / "permutation.npy", allow_pickle=False)
        expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([901, 20261019, 9101]))).permutation(8)
        np.testing.assert_array_equal(pi, expected)
        image_ids = sorted({r["observed_image_id"] for r in self.rows if r["role"] == "U"}, key=sort_id)
        blocks = [sorted([r for r in self.rows if r["role"] == "U" and r["observed_image_id"] == image_id],
                         key=lambda row: sort_id(row["caption_id"])) for image_id in image_ids]
        source = load_weights(self.csv)
        target = load_weights(manifest["arms"]["ShuffleVCS"]["csv"])
        for k, block in enumerate(blocks):
            for j, row in enumerate(block):
                self.assertEqual(target[row["pair_id"]], source[blocks[int(pi[k])][j]["pair_id"]])
        self.assertEqual(pi.dtype, np.dtype("int64"))
        self.assertEqual(manifest["permutation"]["sha256"], sha256_file(self.out / "permutation.npy"))

    def test_fixed_points_are_allowed_even_for_single_block(self):
        self.make_fixture(images=["u"])
        manifest = self.generate()
        self.assertEqual(manifest["permutation"]["n_fixed_points"], 1)
        self.assertEqual(manifest["permutation"]["fixed_point_fraction"], 1.0)
        np.testing.assert_array_equal(self.arm_values(manifest, "VCS"), self.arm_values(manifest, "ShuffleVCS"))

    def test_zero_and_constant_inputs_have_honest_ess_and_correlation(self):
        for value in (0.0, 0.3, 1.0):
            with self.subTest(value=value):
                self.out = self.root / ("out" + str(value))
                self.make_fixture(images=["u", "v"], weights=np.full(10, value))
                manifest = self.generate()
                expected_ess = 0.0 if value == 0 else 10.0
                for arm in manifest["arms"].values():
                    self.assertAlmostEqual(arm["ess_U"], expected_ess)
                # Constant vectors have no mathematically defined correlation.
                self.assertIsNone(manifest["permutation"]["weight_pearson_r"])

    def test_idempotent_generation_preserves_every_published_byte(self):
        manifest = self.generate()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.out.iterdir()}
        again = self.generate()
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.out.iterdir()}
        self.assertEqual(manifest, again)
        self.assertEqual(before, after)

    def test_changed_valid_source_is_refused_not_overwritten(self):
        self.generate()
        old_output = (self.out / "manifest.json").read_bytes()
        contents = self.csv.read_text()
        self.csv.write_text(contents.replace("fit", "new_fit"))
        self.write_source_lock()
        with self.assertRaisesRegex(ValueError, "different source/configuration"):
            self.generate()
        self.assertEqual((self.out / "manifest.json").read_bytes(), old_output)

    def test_changed_seed_cannot_reuse_existing_control_directory(self):
        self.generate()
        with self.assertRaisesRegex(ValueError, "different source/configuration"):
            generate_controls(self.observed, self.csv, self.lock, 902, self.out,
                              expected_n_pool=len(self.rows), expected_n_u=len(self.ids))

    def test_tampered_output_or_permutation_is_detected(self):
        self.generate()
        permutation_path = self.out / "permutation.npy"
        pi = np.load(permutation_path, allow_pickle=False)
        np.save(permutation_path, pi[::-1], allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "Saved permutation"):
            self.generate()

    def test_tampered_summary_statistic_is_detected(self):
        manifest = self.generate()
        manifest["arms"]["ConstantU"]["mean_v_full_pool"] = 0.123
        write_json(self.out / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "inconsistent mean_v_full_pool"):
            self.generate()

    def test_output_csv_tampering_fails_before_training(self):
        manifest = self.generate()
        target = Path(manifest["arms"]["ConstantU"]["csv"])
        target.write_text(target.read_text() + "extra,0.1\n")
        with self.assertRaisesRegex(ValueError, "differs from its manifest"):
            self.generate()

    def test_no_silent_pair_shuffle_fallback(self):
        rows = read_json(self.observed)
        urows = [r for r in rows if r["role"] == "U"]
        urows[0]["observed_image_id"] = urows[-1]["observed_image_id"]
        write_json(self.observed, rows)
        with self.assertRaisesRegex(ValueError, "No silent pair-shuffle fallback"):
            self.generate()
        self.assertFalse(self.out.exists())

    def test_eval_labels_cannot_enter_observed_or_weight_schema(self):
        for field in ("original_image_id", "mismatched", "clean", "caption_text"):
            with self.subTest(field=field):
                rows = read_json(self.observed)
                rows[0][field] = False
                write_json(self.observed, rows)
                with self.assertRaisesRegex(ValueError, "evaluator labels are forbidden"):
                    load_observed(self.observed)
                rows[0].pop(field)
                write_json(self.observed, rows)
        bad_csv = self.root / "leaked.csv"
        write_csv(bad_csv, [{"pair_id": "p1", "w": "0.5", "mismatched": "true"}])
        with self.assertRaisesRegex(ValueError, "forbidden evaluator columns"):
            load_weights(bad_csv)

    def test_eval_only_path_and_disguised_symlink_are_rejected(self):
        directory = self.root / "eval_only"
        directory.mkdir()
        forbidden = directory / "copy.observed_pairs.json"
        forbidden.write_bytes(self.observed.read_bytes())
        alias = self.root / "innocent.json"
        alias.symlink_to(forbidden)
        for path in (forbidden, alias):
            with self.subTest(path=str(path)), self.assertRaisesRegex(ValueError, "Evaluator-only path"):
                load_observed(path)

    def test_observed_duplicate_pair_and_caption_ids_are_rejected(self):
        for field in ("pair_id", "caption_id"):
            with self.subTest(field=field):
                rows = read_json(self.observed)
                rows[1][field] = rows[0][field]
                path = self.root / (field + ".json")
                write_json(path, rows)
                with self.assertRaisesRegex(ValueError, "Duplicate " + field):
                    load_observed(path)

    def test_observed_cross_role_image_is_rejected(self):
        rows = read_json(self.observed)
        rows[-1]["observed_image_id"] = "anchor"
        write_json(self.observed, rows)
        with self.assertRaisesRegex(ValueError, "crosses roles"):
            load_observed(self.observed)

    def test_bad_observed_schema_and_ids_fail_with_value_error(self):
        examples = [{"unexpected": "object"}, [], [None],
                    [{**self.rows[0], "role": []}],
                    [{**self.rows[0], "pair_id": True}],
                    [{**self.rows[0], "caption_id": " 12"}]]
        for k, rows in enumerate(examples):
            with self.subTest(k=k):
                path = self.root / f"bad_{k}.json"
                write_json(path, rows)
                with self.assertRaises(ValueError):
                    load_observed(path)

    def test_weight_duplicate_missing_extra_id_and_nonfinite_checks(self):
        cases = [("pair_id,w\np1,0.1\np1,0.2\n", None, "Duplicate"),
                 ("pair_id,w\np1,0.1\n", ["p1", "p2"], "coverage"),
                 ("pair_id,w\np1,0.1\np2,0.2\n", ["p1"], "coverage"),
                 ("pair_id,w\np1,nan\n", None, "finite"),
                 ("pair_id,w\np1,inf\n", None, "finite"),
                 ("pair_id,w\np1,-0.1\n", None, "finite"),
                 ("pair_id,w\np1,1.1\n", None, "finite"),
                 ("pair_id,w,w\np1,0.1,0.2\n", None, "unique columns"),
                 ("pair_id,w\np1,0.1,surplus\n", None, "column count")]
        for k, (contents, expected, message) in enumerate(cases):
            with self.subTest(k=k):
                path = self.root / f"bad_{k}.csv"
                path.write_text(contents)
                with self.assertRaisesRegex(ValueError, message):
                    load_weights(path, expected)

    def test_wrong_lock_hash_and_row_count_are_rejected(self):
        lock = read_json(self.lock)
        lock["sha256"] = "0" * 64
        write_json(self.lock, lock)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.generate()
        self.write_source_lock()
        lock = read_json(self.lock)
        lock["n"] += 1
        write_json(self.lock, lock)
        with self.assertRaisesRegex(ValueError, "row count"):
            self.generate()

    def test_nonproduction_or_oracle_source_lock_is_rejected(self):
        for change in ({"method": "BCE_Weight"}, {"rho_strategy": "oracle"}, {"deployable": False}):
            with self.subTest(change=change):
                self.write_source_lock(**change)
                with self.assertRaises(ValueError):
                    self.generate()
        self.assertFalse(self.out.exists())

    def test_partial_output_and_source_overlap_are_not_overwritten(self):
        self.out.mkdir()
        sentinel = self.out / "notes.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.generate()
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "source input"):
            generate_controls(self.observed, self.csv, self.lock, 901, self.root,
                              expected_n_pool=len(self.rows), expected_n_u=len(self.ids))

    def test_production_count_default_does_not_accept_mini_fixture(self):
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            generate_controls(self.observed, self.csv, self.lock, 901, self.out)

    def test_generator_does_not_use_numpy_global_random_state(self):
        np.random.seed(1)
        before = np.random.get_state()
        self.generate()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
