"""V9-G2 tests: g1 is reused unmodified (source byte-identical before / after activate(), hooks restored), unit-level stratified FIT sampling on
the EPFL manifest (scenes, categories; merged units stay together), the EPFL state hash, the r6 REG-P3 reuse rule, the REG-P3-shaped unit root,
the smoke-root builder (reads no DEV window), the thresholds (err_full 42 / 9 = 10.5 / 2.25 native px) and the unit table."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TMP = tempfile.mkdtemp(prefix="v9g2_")
os.environ["V9_REG_EPFL_ROOT"] = TMP; os.environ.setdefault("V9_REG_ROOT", TMP + "/g1")                 # g1 must never see the live G1 root from a test
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from v9.reg import g1 as G1  # noqa: E402
from v9.reg import g2 as G  # noqa: E402

G1_FILE = Path(G1.__file__)


def synthetic_manifest():
    """An EPFL-shaped manifest: 3 categories × 8 scenes; two near-duplicate merges (one unit with 2 windows); roles FIT except 2 DEV + 1 CAL per category."""
    windows, units, pairs, roles = {}, {}, {}, {}
    for cat in ("country", "field", "water"):
        for i in range(8):
            pid = f"{cat}/{i:04d}"; w = f"{cat}_{i:04d}"; unit = pid if i != 1 else f"{cat}/0000+{cat}/0001"
            if i == 0:
                unit = f"{cat}/0000+{cat}/0001"
            windows[w] = {"pair": pid, "unit": unit, "category": cat, "y0": 0, "x0": 0}; pairs[pid] = {"sha256_rgb": hashlib.sha256(f"r{pid}".encode()).hexdigest(), "sha256_nir": hashlib.sha256(f"n{pid}".encode()).hexdigest()}
            units.setdefault(unit, {"role": "FIT" if i >= 3 else ("DEV" if i < 2 else "CAL"), "category": cat, "pairs": []}); units[unit]["pairs"].append(pid)
            roles[w] = units[unit]["role"]
    return {"roles": roles, "windows": windows, "pairs": pairs, "units": units, "size": 512, "down": 1}


class G1Untouched(unittest.TestCase):
    def test_activate_restores_hooks_and_source(self):
        before = G1_FILE.read_bytes(); hooks = {k: getattr(G1, k) for k in G._HOOKS}; h0 = G1.code_hash()
        with G.activate(synthetic_manifest(), TMP) as g:
            self.assertIs(g, G1); self.assertEqual(G1.unit_root("x"), G.V9_ROOT / "units" / "x"); self.assertEqual(G1.P1_ROOT, Path(TMP)); self.assertIsNot(G1.sample_fit, hooks["sample_fit"])
        for k, v in hooks.items():
            self.assertIs(getattr(G1, k), v)
        self.assertEqual(G1_FILE.read_bytes(), before); self.assertEqual(G1.code_hash(), h0)

    def test_shared_module_objects_for_the_kernel_monkeypatch(self):
        self.assertIs(G.G, G1); self.assertIs(G.P3.P, G.P); self.assertIs(G.P3.Q, G.Q); self.assertIs(G1.P, G.P3.P)
        orig = G.P.fit_kernel; G.P.fit_kernel = lambda *a, **k: "patched"
        try:
            self.assertEqual(G.P3.P.fit_kernel(), "patched")                    # register_one's P.fit_kernel sees the patch
        finally:
            G.P.fit_kernel = orig

    def test_g1_unit_ids_and_counts_unchanged(self):
        self.assertEqual(len(G1.all_units()), 48); self.assertTrue(all(u[0].startswith("V9-G1-") for u in G1.all_units()))
        self.assertEqual(len(G.all_units()), 48); self.assertTrue(all(u[0].startswith("V9-G2-") for u in G.all_units())); self.assertEqual(len(G.kernel_units()), 4)


class Sampling(unittest.TestCase):
    def setUp(self):
        self.m = synthetic_manifest(); self.fit = sorted(w for w, r in self.m["roles"].items() if r == "FIT")

    def test_units_not_windows_and_merged_units_stay_together(self):
        wins, counts = G.sample_fit_units(self.fit, 0.5, 901, self.m); wu = G.window_unit(self.m)
        units = set(wu[w] for w in wins); n_units = len(set(wu[w] for w in self.fit))
        self.assertEqual(sum(counts.values()), round(0.5 * n_units)); self.assertEqual(len(units), sum(counts.values()))
        for u in units:                                                           # every window of a chosen unit is included
            self.assertEqual(sorted(w for w in self.fit if wu[w] == u), sorted(w for w in wins if wu[w] == u))

    def test_stratified_by_category_and_seeded(self):
        wins, counts = G.sample_fit_units(self.fit, 0.25, 901, self.m); self.assertEqual(set(counts), {"country", "field", "water"})
        self.assertEqual(G.sample_fit_units(self.fit, 0.25, 901, self.m)[0], wins); self.assertNotEqual(G.sample_fit_units(self.fit, 0.25, 902, self.m)[0], wins)
        self.assertEqual(G.sample_fit_units(self.fit, 1.0, 901, self.m)[0], self.fit)

    def test_real_manifest_counts(self):
        if not (G.P3_ROOT / "manifest.json").exists():
            self.skipTest("REG-P3 manifest absent")
        m = G.load_manifest(G.P3_ROOT, False); fit = G.fit_full_windows(m, G.P3_ROOT); self.assertEqual(len(fit), 336)
        wins, counts = G.sample_fit_units(fit, 0.25, 901, m); self.assertEqual(sum(counts.values()), round(0.25 * 330)); self.assertEqual(len(counts), 9)
        self.assertEqual(sum(G.sample_fit_units(fit, 1.0, 903, m)[1].values()), 330)


class HashAndReuse(unittest.TestCase):
    def test_state_hash_sensitivity(self):
        m = synthetic_manifest(); fit = sorted(w for w, r in m["roles"].items() if r == "FIT")
        h = lambda **k: G.state_hash_epfl(m, fit, k.get("seed", 901), k.get("frac", 1.0), k.get("eps", 0.0), "vcs", 3000)
        self.assertEqual(h(), h()); self.assertNotEqual(h(), h(seed=902)); self.assertNotEqual(h(), h(eps=0.2)); self.assertNotEqual(h(), h(frac=0.25))
        self.assertNotEqual(G.state_hash_epfl(m, fit[:4], 901, 0.25, 0.0, "vcs", 3000), h())
        self.assertNotEqual(h(), G1.state_hash({"files": {}, "size": 512, "down": 1}, [], 901, 1.0, 0.0, "vcs", 3000))

    def test_r6_reuse_only_identical(self):
        m = synthetic_manifest(); fit = sorted(w for w, r in m["roles"].items() if r == "FIT")
        self.assertIsNone(G.r6_reusable(m, fit, "vcs", 1.0, 0.0, 901, TMP)); self.assertIsNone(G.r6_reusable(m, fit, "vcs", 0.25, 0.0, 0, TMP)); self.assertIsNone(G.r6_reusable(m, fit, "vcs", 1.0, 0.2, 0, TMP))
        if (G.P3_ROOT / "runs" / "vcs_s0" / "train.json").exists():
            mm = G.load_manifest(G.P3_ROOT, False); r = G.r6_reusable(mm, sorted(json.load(open(G.P3_ROOT / "runs" / "vcs_s0" / "train.json"))["fit"]), "vcs", 1.0, 0.0, 0)
            self.assertIsNotNone(r); self.assertTrue(r["run"].endswith("vcs_s0")); self.assertIsNone(G.r6_reusable(mm, G.fit_full_windows(mm, G.P3_ROOT), "vcs", 1.0, 0.0, 901))


class Layout(unittest.TestCase):
    def test_unit_root_symlinks_p3_layout(self):
        uid = G.unit_id("vcs", 0.25, 0.2, 901); u = G.ensure_layout(uid, "vcs", 901, TMP)
        self.assertTrue(str(u).startswith(TMP)); self.assertTrue((u / "p3" / "cache").is_symlink()); self.assertTrue((u / "p3" / "manifest.json").is_symlink()); self.assertTrue((u / "p3" / "perturbations.json").is_symlink())
        self.assertEqual((u / "p3" / "runs" / "vcs_s901").resolve(), (u / "run").resolve()); self.assertTrue((u / "p3" / "results").is_dir())
        self.assertEqual(G.result_path(uid, "vcs", 901), u / "p3" / "results" / "vcs_s901.json"); self.assertEqual(G.result_path(uid, "kernel", 901, True).name, "kernel_s901_smoke.json")

    def test_smoke_root_reads_no_dev_window(self):
        if not (G.P3_ROOT / "manifest.json").exists():
            self.skipTest("REG-P3 root absent")
        m = G.load_manifest(G.P3_ROOT, False); sel = G.P3.smoke_subset(m["roles"], m["windows"])
        self.assertEqual(sorted(sel.values()).count("FIT"), 4); self.assertEqual(sorted(sel.values()).count("CAL"), 2); self.assertEqual(sorted(sel.values()).count("DEV"), 2)
        for w, r in sel.items():
            self.assertNotEqual(m["roles"][w], "DEV")                             # the smoke "DEV" windows are relabelled CAL windows


class Thresholds(unittest.TestCase):
    def test_native_thresholds(self):
        t = G.THRESHOLDS_NATIVE; self.assertAlmostEqual(t["primary_px"] * t["err_full_factor"], 42.0); self.assertAlmostEqual(t["strict_px"] * t["err_full_factor"], 9.0)
        self.assertIn("10.5 / 2.25", G.P3.ERR_UNIT)

    def test_mismatch_radii_as_g1(self):
        self.assertEqual(G.PATCH_RADIUS, G1.PATCH_RADIUS); self.assertEqual(G.PATCH_RADIUS["critic"], float(G1.P.RF_HALF))


class UnitTable(unittest.TestCase):
    def test_table_rows(self):
        if not (G.P3_ROOT / "manifest.json").exists():
            self.skipTest("REG-P3 root absent")
        rows = G.unit_table(); self.assertEqual(len(rows), 52); self.assertEqual(len(set(r["unit_id"] for r in rows)), 52)
        self.assertTrue(all(r["dataset"] == "EPFL-RGB-NIR" for r in rows)); self.assertTrue(all(not r["reuse"].startswith("r6:") for r in rows))
        q = [r for r in rows if r["fit_fraction"] == 0.25 and r["method"] == "vcs"]; self.assertTrue(all(r["n_fit_units"] == round(0.25 * 330) for r in q))


if __name__ == "__main__":
    unittest.main()
