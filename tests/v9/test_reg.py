"""V9-G1 tests: specimen-stratified FIT sampling, the mismatch generator (exact count, exclusion radius, same draws across objectives, no RNG use at
ε = 0), the r6 reuse rule and state hash, REG-P2 metric definitions, the unit-root layout, and the kernel fit's ε = 0 equality with REG-P1's."""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TMP = tempfile.mkdtemp(prefix="v9g1_")
os.environ["V9_REG_ROOT"] = TMP
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from v9.reg import g1 as G  # noqa: E402

FIT32 = ['1B_A2', '1B_A5', '1B_B2', '1B_C7', '1B_D5', '1B_E3', '1B_E4', '1B_E5', '1B_F9', '1B_G3', '1B_G5', '1B_H2', '1B_I1', '1B_I7', '1B_J6', '1B_K7', '1B_L2', '1B_M2',
         '2B_A2', '2B_A3', '2B_B1', '2B_B8', '2B_C3', '2B_C7', '2B_D7', '2B_E5', '2B_F4', '2B_G2', '2B_G4', '2B_H4', '2B_H8', '2B_H9']


class Sampling(unittest.TestCase):
    def test_fraction_counts_by_slide(self):
        fit, counts = G.sample_fit(FIT32, 0.25, 901)
        self.assertEqual(len(fit), 8); self.assertEqual(sum(counts.values()), 8); self.assertEqual(set(counts), {"1B", "2B"})
        self.assertTrue(set(fit) <= set(FIT32)); self.assertEqual(sorted(counts.values()), [3, 5] if counts["1B"] == 5 else [4, 4])
        for s, n in counts.items():                                                # largest remainder: 18 × 0.25 = 4.5, 14 × 0.25 = 3.5 -> 4 + 4 or 5 + 3
            self.assertIn(n, (3, 4, 5))

    def test_full_fraction_is_identity_and_seeds_differ(self):
        self.assertEqual(G.sample_fit(FIT32, 1.0, 901)[0], sorted(FIT32))
        a, b = G.sample_fit(FIT32, 0.25, 901)[0], G.sample_fit(FIT32, 0.25, 902)[0]
        self.assertEqual(a, G.sample_fit(FIT32, 0.25, 901)[0]); self.assertNotEqual(a, b)

    def test_units_are_cores_not_patches(self):
        for c in G.sample_fit(FIT32, 0.25, 903)[0]:
            self.assertRegex(c, r"^[12]B_[A-M]\d+$")


class Mismatch(unittest.TestCase):
    def test_exact_count_and_radius(self):
        g = np.random.default_rng(0); yx = g.uniform(16, 143, (128, 2)); obs, n = G.mismatch_positions(np.random.default_rng(1), yx, 16, 143, 0.2, 30.0)
        self.assertEqual(n, round(0.2 * 128)); moved = np.any(obs != yx, 1); self.assertEqual(int(moved.sum()), n)
        d = np.hypot(*(obs[moved] - yx[moved]).T); self.assertTrue(np.all(d >= 30.0)); self.assertTrue(np.all((obs >= 16) & (obs <= 143)))

    def test_eps_zero_leaves_rng_untouched(self):
        g = np.random.default_rng(5); before = g.bit_generator.state; obs, n = G.mismatch_positions(g, np.zeros((10, 2)), 0, 1, 0.0, 1.0)
        self.assertEqual(n, 0); self.assertEqual(g.bit_generator.state, before)

    def test_same_draws_for_every_objective(self):
        yx = np.random.default_rng(2).uniform(16, 143, (64, 2))
        a = G.mismatch_positions(np.random.default_rng(77), yx, 16, 143, 0.2, 30.0)[0]; b = G.mismatch_positions(np.random.default_rng(77), yx, 16, 143, 0.2, 30.0)[0]
        np.testing.assert_array_equal(a, b)
        src = __import__("inspect").getsource(G.cmd_train); self.assertLess(src.index("mismatch_positions"), src.index('if objective == "infonce"'))

    def test_displaced_rejects_near(self):
        g = np.random.default_rng(3)
        for _ in range(200):
            y, x = G.displaced(g, (60.0, 60.0), 16, 143, 30.0); self.assertGreaterEqual(math.hypot(y - 60, x - 60), 30.0)


class ReuseRule(unittest.TestCase):
    MAN = {"files": {c: {"sha256_bf": "a" * 64, "sha256_shg": "b" * 64} for c in FIT32}, "size": 512, "down": 4}

    def test_state_hash_sensitivity(self):
        h = lambda **k: G.state_hash(self.MAN, FIT32, k.get("seed", 901), k.get("frac", 1.0), k.get("eps", 0.0), "vcs", 3000)
        self.assertEqual(h(), h()); self.assertNotEqual(h(), h(seed=902)); self.assertNotEqual(h(), h(eps=0.2)); self.assertNotEqual(h(), h(frac=0.25))
        self.assertNotEqual(G.state_hash(self.MAN, FIT32[:8], 901, 0.25, 0.0, "vcs", 3000), h())

    def test_r6_reuse_only_identical(self):
        self.assertIsNone(G.r6_reusable(self.MAN, FIT32, "vcs", 1.0, 0.0, 901)); self.assertIsNone(G.r6_reusable(self.MAN, FIT32, "vcs", 0.25, 0.0, 0)); self.assertIsNone(G.r6_reusable(self.MAN, FIT32, "vcs", 1.0, 0.2, 0))
        if (G.P1_ROOT / "runs" / "vcs_s0" / "train.json").exists():
            r = G.r6_reusable(G.P.load_manifest(G.P1_ROOT, False), sorted(json.load(open(G.P1_ROOT / "runs" / "vcs_s0" / "train.json"))["fit"]), "vcs", 1.0, 0.0, 0)
            self.assertIsNotNone(r); self.assertTrue(r["run"].endswith("vcs_s0"))

    def test_unit_table_counts(self):
        units = G.all_units(); self.assertEqual(len(units), 48); self.assertEqual(len(set(u[0] for u in units)), 48); self.assertEqual(len(G.kernel_units()), 4)


class Metrics(unittest.TestCase):
    def test_corner_error_scale(self):
        Q = G.Q; self.assertAlmostEqual(Q.corner_error_full(0.0, (0.0, 0.0), 0.0, (0.0, 0.0)), 0.0)
        self.assertAlmostEqual(Q.corner_error_full(0.0, (10.5, 0.0), 0.0, (0.0, 0.0)), 42.0, places=6)          # 10.5 px at 512 = 42 full-res px
        self.assertAlmostEqual(Q.corner_error_full(4.09, (0.0, 0.0), 0.0, (0.0, 0.0)) / 42.0, 1.0, places=2)   # pure rotation 4.09° ~ 42 px
        self.assertAlmostEqual(Q.corner_error_full(0.0, (2.25, 0.0), 0.0, (0.0, 0.0)), 9.0, places=6)

    def test_success_thresholds(self):
        recs = {"a": [{"err_full": 41.9}, {"err_full": 42.0}], "b": [{"err_full": 8.0}, {"err_full": 100.0}], "b#profile": {}}
        self.assertAlmostEqual(G.P.success(recs, 42.0), 0.5); self.assertAlmostEqual(G.P.success(recs, 9.0), 0.25)

    def test_holm(self):
        from v9.reg import g1_aggregate as A
        np.testing.assert_allclose(A.holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])   # step-down: 0.03 x 3 = 0.09 carries to the next


class Layout(unittest.TestCase):
    def test_unit_root_symlinks(self):
        uid = G.unit_id("vcs", 0.25, 0.2, 901); u = G.ensure_layout(uid, "vcs", 901)
        self.assertTrue((u / "p1" / "cache").is_symlink()); self.assertTrue((u / "p1" / "manifest.json").is_symlink()); self.assertTrue((u / "p2" / "perturbations.json").is_symlink())
        self.assertEqual((u / "p1" / "runs" / "vcs_s901").resolve(), (u / "run").resolve()); self.assertTrue(str(u).startswith(TMP))


class KernelFit(unittest.TestCase):
    def test_eps0_equals_reg_p1(self):
        root = G.P1_ROOT / "smoke"
        if not (root / "manifest_smoke.json").exists():
            self.skipTest("REG-P1 smoke cache absent")
        torch, nn, F = G.P.torch_mods(); dev = torch.device("cpu"); m = G.P.load_manifest(root, True)
        fit = sorted(c for c, r in m["roles"].items() if r == "FIT" and (root / "cache" / f"{c}.npz").exists())[:2]
        a = G.P.fit_kernel(torch, F, root, fit, 1.0, 0, dev, m_feat=64, n_per=1024); b = G.fit_kernel_g1(torch, F, root, fit, 1.0, 0, dev, 0.0, m_feat=64, n_per=1024)
        torch.testing.assert_close(a["w"], b["w"]); self.assertEqual(a["h"], b["h"])
        c = G.fit_kernel_g1(torch, F, root, fit, 1.0, 0, dev, 0.2, m_feat=64, n_per=1024); self.assertFalse(torch.allclose(a["w"], c["w"]))


if __name__ == "__main__":
    unittest.main()


class RunGuards(unittest.TestCase):
    """scripts/v9/reg_g1.py run: smoke units live under <root>/smoke; a formal run refuses a run dir made with other steps / epochs or orphan checkpoints."""

    def _args(self, unit, smoke=False, force=False):
        import argparse
        return argparse.Namespace(unit=unit, smoke=smoke, force=force, steps=3000, n_pert=2, comir_epochs=0)

    def test_refuses_to_mix_smoke_run_with_formal(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v9")); import reg_g1 as CLI
        uid = G.unit_id("vcs", 0.25, 0.2, 902); u = G.ensure_layout(uid, "vcs", 902); (u / "run").mkdir(exist_ok=True)
        json.dump({"steps": 200}, open(u / "run" / "train.json", "w"))
        with self.assertRaises(SystemExit) as cm:
            CLI.cmd_run(self._args(uid))
        self.assertIn("refusing to mix", str(cm.exception))
        os.remove(u / "run" / "train.json"); (u / "run" / "ckpt_00050.pt").write_bytes(b"x")
        with self.assertRaises(SystemExit) as cm:
            CLI.cmd_run(self._args(uid))
        self.assertIn("checkpoints without train.json", str(cm.exception))

    def test_smoke_root_is_separate(self):
        """A smoke call redirects the unit root to <root>/smoke before any layout or training (training itself is stubbed out here)."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v9")); import reg_g1 as CLI
        root0, train0, seen = G.V9_ROOT, G.cmd_train, {}

        def stub(*a, **k):
            seen["root"] = G.V9_ROOT; raise RuntimeError("stop-before-training")
        G.cmd_train = stub
        try:
            with self.assertRaises(RuntimeError):
                CLI.cmd_run(self._args(G.unit_id("vcs", 1.0, 0.0, 903), smoke=True))
        finally:
            G.cmd_train = train0; G.V9_ROOT = root0
        self.assertEqual(seen["root"], root0 / "smoke"); self.assertFalse((root0 / "units" / G.unit_id("vcs", 1.0, 0.0, 903)).exists())
