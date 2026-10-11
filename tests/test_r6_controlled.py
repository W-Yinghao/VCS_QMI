"""R6-02 controlled validations: REG (scripts/r6_reg_controlled.py) and GAIN (scripts/r6_gain_controlled.py) against identities and brute force."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "r6" / "package_r6_20261011" / "reference"))
import r6_gain_controlled as G  # noqa: E402
import r6_reg_controlled as RG  # noqa: E402
import r6_reference as REF  # noqa: E402


def scene(seed, phi_star, **kw):
    c = {"relation": "nonmonotone", **kw}; rng = np.random.default_rng(seed); U, V = RG.make_scene(rng, np.asarray(phi_star, float), c)
    return U, V, RG.positions(rng), c


class Registration(unittest.TestCase):
    def test_bilinear_values_gradient_and_guard(self):
        img = np.random.default_rng(0).normal(size=(12, 12)); ys = np.array([3.0, 4.25]); xs = np.array([5.0, 6.6])
        v, gy, gx = RG.bilinear(img, ys, xs); self.assertAlmostEqual(v[0], img[3, 5])
        h = 1e-6; fy = (RG.bilinear(img, ys + h, xs)[0] - RG.bilinear(img, ys - h, xs)[0]) / (2 * h); self.assertAlmostEqual(gy[1], fy[1], places=5)
        with self.assertRaises(ValueError):
            RG.bilinear(img, np.array([11.5]), np.array([2.0]))                              # never reads outside the image

    def test_sign_convention(self):
        U, V, pos, c = scene(1, (3.0, -2.0), relation="monotone", sig_a=0.0, sig_b=0.0)
        best = max(((dy, dx) for dy in RG.GRID for dx in RG.GRID), key=lambda d: RG.F_classic("corr", U, V, pos, np.array(d, float)))
        self.assertEqual(best, (3, -2))                                                       # V(x + φ*) realigns with U(x)

    def test_fixed_support_guard(self):
        U, V, pos, c = scene(2, (0.0, 0.0))
        for d in ((-8, -8), (8, 8), (-8, 8)):
            RG.samples(U, V, pos, np.array(d, float))                                         # the interior support is valid for every candidate
        full = RG.positions(np.random.default_rng(3), full=True)
        with self.assertRaises(ValueError):
            RG.samples(U, V, full, np.array([8.0, 8.0]))                                      # full-image samples would read outside: refused

    def test_q_recomputed_and_finite_difference(self):
        U, V, pos, c = scene(4, (1.4, -0.7)); rng = np.random.default_rng(5)
        tr = [scene(10 + i, (0.0, 0.0)) for i in range(3)]
        uP = np.concatenate([u[p[0].astype(int), p[1].astype(int)] for u, v, p, _ in tr]); vP = np.concatenate([v[p[0].astype(int), p[1].astype(int)] for u, v, p, _ in tr])
        vQ = np.concatenate([v[p[0][p[2]].astype(int), p[1][p[2]].astype(int)] for u, v, p, _ in tr]); crit = RG.PolyCritic(uP, vP, uP, vQ, "vcs", iters=300)
        _, (_, _, _), (vQ0, _, _), _ = RG.samples(U, V, pos, np.array([0.0, 0.0])); _, _, (vQ1, _, _), _ = RG.samples(U, V, pos, np.array([2.0, 1.0]))
        self.assertFalse(np.allclose(vQ0, vQ1))                                               # the reference term moves with φ
        x = np.array([1.13, -0.41]); F0, gr = RG.F_critic(crit, U, V, pos, x, grad=True); h = 1e-5
        fd = [(RG.F_critic(crit, U, V, pos, x + d) - RG.F_critic(crit, U, V, pos, x - d)) / (2 * h) for d in (np.array([h, 0]), np.array([0, h]))]
        self.assertTrue(np.allclose(gr, fd, rtol=1e-4, atol=1e-7))

    def test_position_permutation_invariance(self):
        U, V, pos, c = scene(6, (0.5, 0.5)); oracle = RG.OracleCritic(c); ys, xs, perm = pos
        sigma = np.random.default_rng(7).permutation(len(ys)); inv = np.argsort(sigma)
        pos2 = (ys[sigma], xs[sigma], inv[perm[sigma]])                                        # relabel samples, same pairs
        a = RG.F_critic(oracle, U, V, pos, np.array([0.5, 0.5])); b = RG.F_critic(oracle, U, V, pos2, np.array([0.5, 0.5]))
        self.assertAlmostEqual(a, b, places=12)                                                # the critic sees local values only

    def test_oracle_bounded_and_peaks_at_truth(self):
        U, V, pos, c = scene(8, (2.0, 1.0)); oracle = RG.OracleCritic(c)
        self.assertLessEqual(np.abs(oracle.eta).max(), 1.0)
        self.assertGreater(RG.F_critic(oracle, U, V, pos, np.array([2.0, 1.0])), RG.F_critic(oracle, U, V, pos, np.array([-3.0, 4.0])))


class Gain(unittest.TestCase):
    def test_exact_delta(self):
        want = {"redundant": 0.0, "irrelevant": 0.0, "xor": 1.0, "context": 0.5, "context_noise": 0.5}
        for name, d in want.items():
            self.assertAlmostEqual(G.exact(name, 0.0)["delta"], d)
            self.assertAlmostEqual(G.exact(name, 0.1)["delta"], d * 0.8 ** 2)                   # label noise shrinks η by (1 − 2ε)
            e = G.exact(name, 0.1); self.assertAlmostEqual(e["S_full"] - e["S_cheap"], e["risk_cheap"] - e["risk_full"])

    def test_context_noise_local_gain(self):
        d = G.exact("context_noise", 0.0)["d_u"]
        for key, val in d.items():
            self.assertAlmostEqual(val, 1.0 if key.startswith("(1") else 0.0)                 # K = 1 benefits, K = 0 does not

    def test_cheap_uncertainty_cannot_separate_context_noise(self):
        rng = np.random.default_rng(0); k, x, v, c = G.sample("context_noise", 20000, 0.0, rng)
        a = G.fit(G.feats(k, x), c, "square")(G.feats(k, x)); u = 1 - a ** 2
        self.assertLess(abs(u[k == 0].mean() - u[k == 1].mean()), 0.05)                        # both strata look equally uncertain

    def test_identity_and_counterexample(self):
        rng = np.random.default_rng(1); c = rng.choice([-1.0, 1.0], 200); a = rng.uniform(-1, 1, 200); b = rng.uniform(-1, 1, 200)
        z, r, h = REF.risk_components(c, a, b); self.assertTrue(np.allclose(h, z + r))
        e = REF.finite_model_utility(REF.make_task("redundant"), {-1: -1.0, 1: 1.0}, {(-1, -1): 0.0, (1, 1): 0.0})
        self.assertEqual((e["square"], e["actual_gain"], e["residual"]), (1.0, -1.0, -2.0))

    def test_policy_guard_and_budgets(self):
        with self.assertRaises(ValueError):
            REF.validate_policy_inputs(["cheap_input", "expensive_observation"])
        c = np.array([1.0, -1, 1, -1]); a = np.zeros(4); b = c.copy(); tie = np.arange(4.0)
        self.assertAlmostEqual(G.budget_risk(c, a, b, np.ones(4), 0.0, tie), 1.0); self.assertAlmostEqual(G.budget_risk(c, a, b, np.ones(4), 1.0, tie), 0.0)
        self.assertAlmostEqual(G.budget_risk(c, a, b, np.array([0, 1, 0, 1.0]), 0.5, tie), 0.5)

    def test_interaction_fit_recovers_xor(self):
        rng = np.random.default_rng(2); k, x, v, c = G.sample("xor", 40000, 0.1, rng)
        for fam in ("square", "logistic"):
            b = G.fit(G.feats(k, x, v, "interaction"), c, fam)(G.feats(k, x, v, "interaction"))
            self.assertLess(np.abs(b - 0.8 * x * v).max(), 0.03, fam)                           # E[C | X, V] = (1 − 2ε) X V


if __name__ == "__main__":
    unittest.main()
