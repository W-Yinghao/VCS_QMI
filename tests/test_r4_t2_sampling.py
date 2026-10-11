"""R4-T2: the fixed-critic readouts against brute force (scripts/r4_t2_sampling.py; package r4 docs/03 §4–6)."""
import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
import r4_t2_sampling as T2  # noqa: E402


def small_law(seed=0, shape=(2, 2)):
    g = np.random.default_rng(seed); P = g.dirichlet(np.ones(shape[0] * shape[1])).reshape(shape); Q = np.outer(P.sum(1), P.sum(0))
    return P, Q, np.tanh(g.normal(0, 0.8, shape))


class UStatistic(unittest.TestCase):
    def test_pair_form(self):
        g = np.random.default_rng(1); Tm = np.tanh(g.normal(size=(7, 7))); n = 7
        k = lambda i, j: 0.5 * (T2.a_plus(Tm[i, i]) + T2.a_plus(Tm[j, j])) + 0.5 * (T2.a_minus(Tm[i, j]) + T2.a_minus(Tm[j, i]))
        u = np.mean([k(i, j) for i, j in itertools.combinations(range(n), 2)])
        self.assertAlmostEqual(T2.j_hat(Tm), u, places=12)

    def test_exact_variance_by_enumeration(self):
        P, Q, T = small_law(2); w, xs, ys = T2.grid_states(P); n = 3; m = len(w); vals, prob = [], []
        for tup in itertools.product(range(m), repeat=n):
            s = np.array(tup); vals.append(T2.j_hat(T[xs[s][:, None], ys[s][None, :]])); prob.append(np.prod(w[s]))
        vals, prob = np.array(vals), np.array(prob); mean = float(prob @ vals); var = float(prob @ (vals - mean) ** 2)
        mu, z1, eh2 = T2.exact_hoeffding(P, T)
        self.assertAlmostEqual(mean, mu, places=12)
        self.assertAlmostEqual(mu, float((P * T2.a_plus(T)).sum() + (Q * T2.a_minus(T)).sum()), places=12)   # E k_T = J(T)
        self.assertAlmostEqual(var, T2.var_exact(z1, eh2, n), places=12)

    def test_delete_one(self):
        g = np.random.default_rng(3); Tm = np.tanh(g.normal(size=(9, 9))); a = T2.a_plus(np.diag(Tm).copy()); B = T2.a_minus(Tm); np.fill_diagonal(B, 0)
        U = T2.delete_one(a, B)
        for i in range(9):
            keep = np.delete(np.arange(9), i); self.assertAlmostEqual(U[i], T2.j_hat(Tm[np.ix_(keep, keep)]), places=12)


class SparseShifts(unittest.TestCase):
    def setUp(self):
        g = np.random.default_rng(4); self.n = 7; self.Tm = np.tanh(g.normal(size=(7, 7)))
        self.a = T2.a_plus(np.diag(self.Tm).copy()); B = T2.a_minus(self.Tm); np.fill_diagonal(B, 0); idx = np.arange(7)
        self.V = np.stack([B[idx, (idx + l) % 7] for l in range(1, 7)]); self.B = B

    def test_conditional_variance_formula(self):
        v = self.V.mean(1); n = self.n
        for K in (1, 2, 3, 6):
            means = [v[list(c)].mean() for c in itertools.combinations(range(n - 1), K)]
            self.assertAlmostEqual(float(np.var(means)), (1 - K / (n - 1)) * float(v.var(ddof=1)) / K, places=12)
        self.assertAlmostEqual(self.a.sum() / n + v.mean(), T2.j_hat(self.Tm), places=12)            # all n − 1 shifts = the full estimator

    def test_incomplete_delete_one(self):
        S = np.array([0, 2, 4]); n = self.n; UK = T2.incomplete_delete_one(self.a, self.V, S)
        for i in range(n):
            pairs = [(k, (k + l + 1) % n) for l in S for k in range(n) if k != i and (k + l + 1) % n != i]
            want = np.delete(self.a, i).mean() + np.mean([self.B[k, j] for k, j in pairs])
            self.assertAlmostEqual(UK[i], want, places=12)

    def test_readout_formula_wiring(self):
        r = T2.cross_readouts(self.Tm, np.random.default_rng(0)); v = self.V.mean(1)
        self.assertAlmostEqual(r["var_shift_means"], float(v.var(ddof=1)), places=12)
        self.assertAlmostEqual(r["K4"]["cond_var"], (1 - 4 / 6) * r["var_shift_means"] / 4, places=12)
        self.assertAlmostEqual(r["J"], T2.j_hat(self.Tm), places=12)


class Identities(unittest.TestCase):
    def test_gap(self):
        P1 = T2.base_table(); P, Q = T2.mixture(P1, 0.6); S, eta = T2.s_eta(P, Q)
        T = np.tanh(np.random.default_rng(5).normal(size=P.shape)); J = float((P * T2.a_plus(T)).sum() + (Q * T2.a_minus(T)).sum())
        self.assertAlmostEqual(S - J, float(((P + Q) / 2 * (T - eta) ** 2).sum()), places=12)
        self.assertAlmostEqual(float((P * T2.a_plus(eta)).sum() + (Q * T2.a_minus(eta)).sum()), S, places=12)   # J(η) = S

    def test_bounded_difference(self):
        P, Q, T = small_law(6, (4, 5)); w, xs, ys = T2.grid_states(P); g = np.random.default_rng(7); n = 10; worst = 0.0
        for _ in range(20):
            s = g.choice(len(w), n); base = T2.j_hat(T[xs[s][:, None], ys[s][None, :]])
            for i in range(n):
                for st in range(len(w)):
                    s2 = s.copy(); s2[i] = st; worst = max(worst, abs(T2.j_hat(T[xs[s2][:, None], ys[s2][None, :]]) - base))
        self.assertLessEqual(worst, 8.0 / n + 1e-12)

    def test_rho_calibration(self):
        P1 = T2.base_table()
        for target in (0.05, 0.40):
            rho = T2.rho_for(P1, target); S = T2.s_eta(*T2.mixture(P1, rho))[0]
            self.assertTrue(abs(S - target) < 1e-9 or rho == 0.95, (target, rho, S))

    def test_image_local_exact(self):
        P, Q, T = small_law(8); w, xs, ys = T2.grid_states(P); tot = 0.0
        for r in T2.R_IMG:
            acc = 0.0
            for tup in itertools.product(range(len(w)), repeat=r):
                s = np.array(tup); Tg = T[xs[s][:, None], ys[s][None, :]]
                acc += np.prod(w[s]) * (T2.a_plus(np.diag(Tg)).mean() + T2.a_minus(Tg).mean())
            tot += acc / len(T2.R_IMG)
        self.assertAlmostEqual(tot, T2.mu_image_local(P, Q, T), places=12)


class Generators(unittest.TestCase):
    def test_pmi_broadcast(self):
        from vcs_estim.data import setting_from_mi
        for name in ("gaussian", "xor_mixture"):
            s = setting_from_mi(name, 0.5, d=3); x, y = s.sample(6, torch.Generator().manual_seed(0)); x, y = x.double(), y.double()
            M = s.pmi(x[:, None, :], y[None, :, :])
            self.assertTrue(torch.allclose(torch.diagonal(M), s.pmi(x, y), atol=1e-12), name)
            self.assertTrue(torch.allclose(M[1, 4], s.pmi(x[1:2], y[4:5])[0], atol=1e-12), name)


if __name__ == "__main__":
    unittest.main()
