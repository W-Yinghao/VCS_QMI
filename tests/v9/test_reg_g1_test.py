"""V9-G1 sealed-test read: REG-P2's perturbation generator reproduced exactly (new tags only change the seed), the TIFF header parser against REG-P1's
reader on a validation file, every guard (prepare / register / freeze), the predictor table, and the readings path (g1_aggregate.main over test records)
end to end on synthetic results.  No test file is opened here; the real roots are never written."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TMP = tempfile.mkdtemp(prefix="v9g1test_")
os.environ["V9_REG_ROOT"] = TMP + "/g1"; os.environ["V9_REG_TEST_ROOT"] = TMP + "/test_root"; os.environ["V9_REG_TEST_FIXTURE_ROOT"] = TMP + "/fixture"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from v9.reg import g1_test as T  # noqa: E402

P, Q, G = T.P, T.Q, T.G


class Perturbations(unittest.TestCase):
    def test_reg_p2_tags_reproduce_reg_p2(self):
        cores = ["1B_B5", "2B_H8", "x", "test-core-42"]
        if (T.P1_ROOT / "manifest.json").exists():
            cores += sorted(P.load_manifest(T.P1_ROOT, False)["roles"])[:20]
        for c in cores:
            self.assertEqual(T.perturbations_tagged(c, *T.P2_TAGS), Q.perturbations_p2(c))

    def test_test_tags_same_distribution_new_draws(self):
        for c in ("1B_A1", "2B_C7", "1B_Z9"):
            a, b = T.perturbations_tagged(c, *T.TEST_TAGS), Q.perturbations_p2(c); self.assertNotEqual(a, b); self.assertEqual(len(a), P.N_PERT)
            for i, (th, ty, tx) in enumerate(a):
                lo, hi = P.STRATA[i % 3]; self.assertLessEqual(abs(th), Q.THMAX); self.assertTrue(lo - 1e-9 <= np.hypot(ty, tx) <= hi + 1e-9)


class Inventory(unittest.TestCase):
    def test_tiff_header_matches_reg_p1_reader(self):
        f = sorted((P.DATA / "validation1").glob("*_WB.tif"))
        if not f:
            self.skipTest("validation files absent")
        b = f[0].read_bytes(); h = T.tiff_header(b); a = P.read_tiff(f[0])
        self.assertEqual((h["h"], h["w"]), a.shape[:2]); self.assertEqual(h["spp"], a.shape[2] if a.ndim == 3 else 1); self.assertEqual(h["comp"], 1); self.assertEqual(h["bits"], 8 * a.dtype.itemsize)
        self.assertEqual(T.tiff_header(b"MM\x00*" + b"\x00" * 10)["w"], None)


class Guards(unittest.TestCase):
    def test_prepare_guards(self):
        root = Path(TMP) / "fx"; root.mkdir(exist_ok=True)
        with self.assertRaises(SystemExit):
            T.prepare(root, Path(TMP) / "some.zip", {}, formal=True)              # formal reads only the sealed archive into the test root
        with self.assertRaises(SystemExit):
            T.prepare(T.TEST_ROOT, Path(TMP) / "some.zip", {}, formal=False)       # the fixture never writes the test root
        with self.assertRaises(SystemExit):
            T.prepare(root, T.TEST_ZIP, {}, formal=False)                           # the fixture never opens the sealed archive
        (root / "manifest.json").write_text("{}")
        with self.assertRaises(SystemExit):
            T.prepare(root, Path(TMP) / "some.zip", {}, formal=False)               # prepared once
        self.assertFalse(T.TEST_ROOT.exists())

    def test_register_refuses_overwrite_and_changed_predictor(self):
        root = Path(TMP) / "rg"; (root / "results").mkdir(parents=True, exist_ok=True); (root / "results" / "mi.json").write_text("{}")
        preds = {"mi": {"method": "mi", "file": "", "g1_result": "", "file_sha256": "", "g1_result_sha256": ""}}
        with self.assertRaises(SystemExit):
            T.register("mi", root, {}, preds)
        f = Path(TMP) / "pred.pt"; f.write_bytes(b"abc")
        preds["vcs_x"] = {"method": "vcs", "file": str(f), "file_sha256": "0" * 64, "g1_result": "", "g1_result_sha256": ""}
        with self.assertRaises(SystemExit):
            T.register("vcs_x", root, {}, preds)

    def test_require_frozen_refuses_missing_or_outside(self):
        with self.assertRaises(SystemExit):
            T.require_frozen([Path(TMP) / "nope.yaml"])
        f = Path(TMP) / "p.yaml"; f.write_text("frozen_files: {}\n")
        with self.assertRaises(SystemExit):
            T.require_frozen([f])

    def test_ledger_keeps_first_read_time(self):
        led = Path(TMP) / "ledger.csv"; T.ledger_update(led, read_utc="2026-10-15T00:00:00Z", prepare_job="1")
        row = T.ledger_update(led, read_utc="2026-10-16T00:00:00Z", prepare_job="2"); self.assertEqual(row["read_utc"], "2026-10-15T00:00:00Z"); self.assertEqual(row["prepare_job"], "1;2")


class Predictors(unittest.TestCase):
    def test_table(self):
        if not (T.G1_ROOT / "units").exists():
            self.skipTest("G1 units absent")
        rows = T.predictor_rows(); self.assertEqual(len(rows), 54); self.assertEqual(len({r["predictor_id"] for r in rows}), 54)
        for r in rows:
            if r["method"] in ("vcs", "logistic", "infonce"):
                self.assertTrue(r["choice"].startswith("ckpt_")); self.assertTrue(Path(r["file"]).exists())
            elif r["method"] == "kernel":
                self.assertIn(json.load(open(r["g1_result"]))["chosen_mult"], (0.5, 1.0, 2.0))
            elif r["method"] == "comir":
                self.assertEqual(Path(r["file"]).name, "comir.pt")


class Readings(unittest.TestCase):
    def test_aggregate_end_to_end_on_synthetic_results(self):
        root = Path(TMP) / "agg"; (root / "results").mkdir(parents=True, exist_ok=True); g = np.random.default_rng(0); preds = {}
        units = [(u, m, f, e, s) for u, m, f, e, s in G.all_units() + G.kernel_units()] + [("mi", "mi", None, None, None), ("nmi", "nmi", None, None, None)]
        for u, m, f, e, s in units:
            preds[u] = {"predictor_id": u, "method": m, "fit_fraction": f, "train_mismatch": e, "seed": s}
            rec = {f"c{i}": [{"err_full": float(g.exponential(30)), "rot_err_deg": 0.1, "trans_err_full": 1.0, "F_hat": 0.0, "F_true": 0.1, "seconds": 0.5, "theta_hat": 0.0} for _ in range(3)] for i in range(6)}
            res = {"test": rec, "train": {"realised_mismatch_fraction": e, "fit": ["a"]}}; Q.summarise(res, "test"); json.dump(res, open(root / "results" / f"{u}.json", "w"))
        j = T.aggregate(root, root / "out", preds, split_label="synthetic")
        self.assertEqual(len(j["primary"]), 4); self.assertEqual(j["missing"], []); self.assertTrue((root / "out" / "g1_test_tables.md").exists()); self.assertEqual(j["n_results"], 54)
        v = T.dev_view({"test": {"a": []}, "test_success42": 0.5, "x": 1}); self.assertEqual(v, {"dev": {"a": []}, "dev_success42": 0.5, "x": 1})


if __name__ == "__main__":
    unittest.main()
