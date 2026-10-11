"""r5 follow-up: the query-normalized critic (src/vcs_vl/pairlaw.py query_normalize; objectives vcs_qnorm, balanced_logistic_qnorm)."""
import math
import sys
import unittest
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "r5" / "package_r5_20261011" / "reference"))
from vcs_vl.pairlaw import padded_batch_loss, query_normalize  # noqa: E402
import r5_reference as REF  # noqa: E402

OBJ = ("vcs_qnorm", "balanced_logistic_qnorm")


def batch(seed, score_nu=False, B=3, R_=7, W=4, pad=2):
    g = torch.Generator().manual_seed(seed)
    real = torch.ones(B, R_, dtype=torch.bool); real[:, R_ - pad:] = False
    A = (torch.rand(B, R_, W, generator=g) < 0.35).double(); A[~real] = 0; A[:, 0, :] = 0; A[:, :, W - 1] = 0; A[:, 1, 0] = 1
    nu = torch.rand(B, R_, generator=g, dtype=torch.float64) + 0.05 if score_nu else None
    p, q = REF.make_laws(A, real, nu)
    return torch.randn(B, R_, W, generator=g, dtype=torch.float64) * 1.5, p, q, real


class Normalizer(unittest.TestCase):
    def test_constraint_and_offset(self):
        for score_nu in (False, True):
            f, p, q, real = batch(0, score_nu); gq = query_normalize(f, q); pi = q.sum(1); active = pi > 0
            nu = q / pi.clamp_min(1e-30)[:, None, :]
            mass = (nu * torch.exp(2 * gq) * real[:, :, None]).sum(1)
            self.assertTrue(torch.allclose(mass[active], torch.ones_like(mass[active]), atol=1e-12))
            shift = torch.randn(f.shape[0], 1, f.shape[2], dtype=f.dtype)
            self.assertTrue(torch.allclose(query_normalize(f + shift, q)[real[:, :, None].expand_as(f)], gq[real[:, :, None].expand_as(f)], atol=1e-12))
            for o in OBJ:
                self.assertAlmostEqual(float(padded_batch_loss(f, p, q, o)), float(padded_batch_loss(f + shift, p, q, o)), places=12)

    def test_uniform_closed_form(self):
        f, p, q, real = batch(1); gq = query_normalize(f, q)
        for b in range(f.shape[0]):
            m = int(real[b].sum()); fr = f[b, :m]
            want = fr - 0.5 * torch.logsumexp(2 * fr, 0, keepdim=True) + 0.5 * math.log(m)
            self.assertTrue(torch.allclose(gq[b, :m], want, atol=1e-12))

    def test_direct_loss(self):
        for score_nu in (False, True):
            f, p, q, real = batch(2, score_nu); gq = query_normalize(f, q)
            for o in OBJ:
                per = []
                for b in range(f.shape[0]):
                    t = torch.tanh(gq[b])
                    per.append(float((0.5 * (p[b] * (1 - t) ** 2 + q[b] * (1 + t) ** 2)).sum()) if o == "vcs_qnorm"
                               else float((p[b] * torch.nn.functional.softplus(-2 * gq[b]) + q[b] * torch.nn.functional.softplus(2 * gq[b])).sum()))
                self.assertAlmostEqual(float(padded_batch_loss(f, p, q, o)), sum(per) / len(per), places=12)

    def test_stationary_at_the_unchanged_optimum(self):
        g = torch.Generator().manual_seed(3); R_, W = 6, 3
        pc = torch.distributions.Dirichlet(torch.ones(R_, dtype=torch.float64)).sample((W,)).T                     # p(r | w) > 0, columns sum to 1
        nu = torch.rand(R_, generator=g, dtype=torch.float64) + 0.1; nu = nu / nu.sum(); pi = torch.full((W,), 1.0 / W, dtype=torch.float64)
        p, q = (pc * pi)[None], (nu[:, None] * pi)[None]
        for o in OBJ:
            f = (0.5 * torch.log(pc / nu[:, None]) + 0.7)[None].requires_grad_()                               # 2f = log p/ν + const
            padded_batch_loss(f, p, q, o).backward(); self.assertLess(float(f.grad.abs().max()), 1e-12, o)

    def test_candidates_are_coupled(self):
        f, p, q, real = batch(4); f2 = f.clone(); f2[:, 2, :] += 1.0
        g1, g2 = query_normalize(f, q), query_normalize(f2, q)
        self.assertTrue((g2[:, 3, :] < g1[:, 3, :]).all())                                                      # raising one candidate lowers the others

    def test_padding_empty_query_extreme(self):
        f, p, q, real = batch(5)
        for o in OBJ:
            fa = f.clone().requires_grad_(); L = padded_batch_loss(fa, p, q, o); L.backward()
            self.assertEqual(float(fa.grad[~real].abs().max()), 0.0); self.assertEqual(float(fa.grad[:, :, -1].abs().max()), 0.0)   # padding, empty query
            f2 = f.clone(); f2[~real] = 1e4; self.assertAlmostEqual(float(L), float(padded_batch_loss(f2, p, q, o)), places=12)
            fe = (f.sign() * 60.0).requires_grad_(); Le = padded_batch_loss(fe, p, q, o); Le.backward()
            self.assertTrue(math.isfinite(float(Le)) and torch.isfinite(fe.grad).all())


if __name__ == "__main__":
    unittest.main()
