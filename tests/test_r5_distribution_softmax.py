"""r5 R5-30: production 'distribution_softmax' (src/vcs_vl/pairlaw.py) against the r5 reference
(r5/package_r5_20261011/reference/r5_reference.py) and the package's required checks (docs/03 §8.7): reference value / gradient; single valid box
= set softmax; D2 identity L_D − L_E = log k + KL(U_A ‖ ρ_A); never-positive real proposals in the denominator (the old conditional_softmax
branch drops them); extreme logits; padding values and extra padding change nothing; empty queries; per-image weighting; exact chunk accumulation."""
import sys
import unittest
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "r5" / "package_r5_20261011" / "reference"))
from vcs_vl.pairlaw import padded_batch_loss  # noqa: E402
import r5_reference as REF  # noqa: E402


def batch(seed, B=3, R_=7, W=4, pad=2, score_nu=False):
    """Random compatibility with padding rows, a never-positive real proposal (0), an empty query (W − 1) and multi-valid queries."""
    g = torch.Generator().manual_seed(seed)
    real = torch.ones(B, R_, dtype=torch.bool); real[:, R_ - pad:] = False
    A = (torch.rand(B, R_, W, generator=g) < 0.35).double(); A[~real] = 0; A[:, 0, :] = 0; A[:, :, W - 1] = 0; A[:, 1, 0] = 1
    nu = torch.rand(B, R_, generator=g, dtype=torch.float64) + 0.05 if score_nu else None
    p, q = REF.make_laws(A, real, nu)
    f = torch.randn(B, R_, W, generator=g, dtype=torch.float64) * 1.5
    return f, p, q, real, A


def loss_and_grad(f, p, q, objective):
    f = f.clone().requires_grad_(); L = padded_batch_loss(f, p, q, objective); L.backward(); return float(L), f.grad


class Reference(unittest.TestCase):
    def test_distribution_matches_reference(self):
        for seed in range(5):
            for score_nu in (False, True):
                f, p, q, real, _ = batch(seed, score_nu=score_nu); L, g = loss_and_grad(f, p, q, "distribution_softmax")
                self.assertAlmostEqual(L, float(REF.batch_loss(f, p, q, real, "distribution_softmax")), places=12)
                self.assertTrue(torch.allclose(g, REF.analytic_f_gradient(f, p, q, real, "distribution_softmax") / f.shape[0], atol=1e-12))

    def test_r4_set_softmax_matches_reference(self):
        for seed in range(5):
            f, p, q, real, _ = batch(seed); L, g = loss_and_grad(f, p, q, "set_softmax")
            self.assertAlmostEqual(L, float(REF.batch_loss(f, p, q, real, "set_softmax")), places=12)
            self.assertTrue(torch.allclose(g, REF.analytic_f_gradient(f, p, q, real, "set_softmax") / f.shape[0], atol=1e-12))


class Requirements(unittest.TestCase):
    def test_single_valid_box_equals_set_softmax(self):
        f, p, q, real, A = batch(1); A1 = torch.zeros_like(A); A1[:, 1, :-1] = 1; A1[:, 2, 1] = 0   # exactly one valid box per covered query
        p, q = REF.make_laws(A1, real)
        Ld, gd = loss_and_grad(f, p, q, "distribution_softmax"); Le, ge = loss_and_grad(f, p, q, "set_softmax")
        self.assertAlmostEqual(Ld, Le, places=12); self.assertTrue(torch.allclose(gd, ge, atol=1e-12))

    def test_d2_identity(self):
        for seed in range(4):
            f, p, q, real, _ = batch(seed)
            for b in range(f.shape[0]):
                sl = slice(b, b + 1); d = REF.uniform_set_decomposition(f[sl], p[sl], q[sl], real[sl])
                Ld = float(padded_batch_loss(f[sl], p[sl], q[sl], "distribution_softmax")); Le = float(padded_batch_loss(f[sl], p[sl], q[sl], "set_softmax"))
                self.assertAlmostEqual(Ld - Le, float(d["per_image_extra"][0]), places=10)

    def test_never_positive_real_proposal_in_denominator(self):
        f, p, q, real, _ = batch(2); f2 = f.clone(); f2[:, 0, :] += 3.0                       # proposal 0 is real but never valid
        self.assertNotAlmostEqual(float(padded_batch_loss(f, p, q, "distribution_softmax")), float(padded_batch_loss(f2, p, q, "distribution_softmax")), places=6)
        self.assertAlmostEqual(float(padded_batch_loss(f, p, q, "conditional_softmax")), float(padded_batch_loss(f2, p, q, "conditional_softmax")), places=12)

    def test_padding_values_and_extra_padding(self):
        f, p, q, real, _ = batch(3); L, g = loss_and_grad(f, p, q, "distribution_softmax"); f2 = f.clone(); f2[~real] = 1e4
        self.assertAlmostEqual(L, float(padded_batch_loss(f2, p, q, "distribution_softmax")), places=12)
        self.assertEqual(float(g[~real].abs().max()), 0.0)
        z = torch.zeros(f.shape[0], 3, f.shape[2], dtype=f.dtype)
        self.assertAlmostEqual(L, float(padded_batch_loss(torch.cat([f, z + 50.0], 1), torch.cat([p, z], 1), torch.cat([q, z], 1), "distribution_softmax")), places=12)

    def test_extreme_logits_and_empty_query(self):
        f, p, q, real, _ = batch(4); f = f.sign() * 60.0; L, g = loss_and_grad(f, p, q, "distribution_softmax")
        self.assertTrue(torch.isfinite(torch.tensor(L)) and torch.isfinite(g).all()); self.assertEqual(float(g[:, :, -1].abs().max()), 0.0)   # empty query

    def test_per_image_weighting_and_chunks(self):
        f, p, q, real, _ = batch(5, B=4); full = float(padded_batch_loss(f, p, q, "distribution_softmax"))
        per = [float(padded_batch_loss(f[b:b + 1], p[b:b + 1], q[b:b + 1], "distribution_softmax")) for b in range(4)]
        self.assertAlmostEqual(full, sum(per) / 4, places=12)                                   # every image weighs 1 / B
        fa = f.clone().requires_grad_(); padded_batch_loss(fa, p, q, "distribution_softmax").backward()
        fc = f.clone().requires_grad_(); (padded_batch_loss(fc[:3], p[:3], q[:3], "distribution_softmax") * 0.75 + padded_batch_loss(fc[3:], p[3:], q[3:], "distribution_softmax") * 0.25).backward()
        self.assertTrue(torch.allclose(fa.grad, fc.grad, atol=1e-14))                          # the trainer's exact chunk accumulation


if __name__ == "__main__":
    unittest.main()
