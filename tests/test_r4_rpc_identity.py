"""R4-T0: value / gradient identity between balanced RPC F_{1,1,1} (Tsai et al., ICLR 2021) and our VCS implementations.

F_{a,b,g}(T) = E_P T - a E_Q T - (b/2) E_P T^2 - (g/2) E_Q T^2.  With a = b = g = 1, F = J.
  synthetic benchmark (src/vcs_estim/objectives.py):  vcs_loss(fp, fn) = -F_{1,1,1}(tanh fp, tanh fn)    on empirical means
  VL (src/vcs_vl/pairlaw.py):  padded_batch_loss(f, p, q, "vcs") = mean over images of (1 - F_{1,1,1}) with per-image laws summing to one;
                               padded_j(f, p, q) = F_{1,1,1} per image.
Values and autograd gradients must coincide on the same output tensor; other (a, b, g) must not (negative control).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "r4" / "package_r4_20261010" / "reference"))
from vcs_estim.objectives import js_match_loss, vcs_loss  # noqa: E402
from vcs_vl.pairlaw import padded_batch_loss, padded_j  # noqa: E402
import vcs_r4_reference as ref  # noqa: E402


def rpc_emp(tp, tn, a=1.0, b=1.0, g=1.0):
    return tp.mean() - a * tn.mean() - 0.5 * b * (tp * tp).mean() - 0.5 * g * (tn * tn).mean()


def rpc_law(t, p, q, a=1.0, b=1.0, g=1.0):
    return (p * t).sum((1, 2)) - a * (q * t).sum((1, 2)) - 0.5 * (b * p + g * q).mul(t * t).sum((1, 2))


class SyntheticObjective(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.fp = (torch.randn(257, generator=g, dtype=torch.float64) * 1.5).requires_grad_()
        self.fn = (torch.randn(511, generator=g, dtype=torch.float64) * 1.5).requires_grad_()

    def test_value(self):
        F = rpc_emp(torch.tanh(self.fp), torch.tanh(self.fn))
        self.assertAlmostEqual(float(vcs_loss(self.fp, self.fn)), float(-F), places=12)

    def test_gradient(self):
        g1 = torch.autograd.grad(vcs_loss(self.fp, self.fn), (self.fp, self.fn))
        g2 = torch.autograd.grad(-rpc_emp(torch.tanh(self.fp), torch.tanh(self.fn)), (self.fp, self.fn))
        for x, y in zip(g1, g2):
            self.assertTrue(torch.allclose(x, y, atol=1e-14, rtol=0))

    def test_other_rpc_parameters_differ(self):
        F = rpc_emp(torch.tanh(self.fp), torch.tanh(self.fn), a=1.0, b=1.0, g=0.6)
        self.assertGreater(abs(float(vcs_loss(self.fp, self.fn)) + float(F)), 1e-3)

    def test_js_is_not_rpc(self):
        F = rpc_emp(torch.tanh(self.fp), torch.tanh(self.fn))
        self.assertGreater(abs(float(js_match_loss(self.fp, self.fn)) + float(F)), 1e-3)


class VLObjective(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1); n, Rr, W = 5, 4, 6
        self.f = (torch.randn(n, Rr, W, generator=g, dtype=torch.float64) * 2).requires_grad_()
        p = torch.rand(n, Rr, W, generator=g, dtype=torch.float64); q = torch.rand(n, Rr, W, generator=g, dtype=torch.float64)
        mask = torch.ones(n, Rr, W, dtype=torch.float64); mask[0, 3, :] = 0; mask[2, :, 5] = 0      # padding as in variable-size scenes
        p, q = p * mask, q * mask
        self.p = p / p.sum((1, 2), keepdim=True); self.q = q / q.sum((1, 2), keepdim=True)

    def test_value(self):
        t = torch.tanh(self.f)
        self.assertTrue(torch.allclose(padded_j(self.f, self.p, self.q), rpc_law(t, self.p, self.q), atol=1e-13, rtol=0))
        self.assertAlmostEqual(float(padded_batch_loss(self.f, self.p, self.q, "vcs")), float((1 - rpc_law(t, self.p, self.q)).mean()), places=12)

    def test_gradient(self):
        g1 = torch.autograd.grad(padded_batch_loss(self.f, self.p, self.q, "vcs"), self.f)[0]
        g2 = torch.autograd.grad((1 - rpc_law(torch.tanh(self.f), self.p, self.q)).mean(), self.f)[0]
        self.assertTrue(torch.allclose(g1, g2, atol=1e-14, rtol=0))

    def test_matches_package_reference(self):
        t = torch.tanh(self.f).detach().numpy()
        for i in range(t.shape[0]):
            p, q = self.p[i].numpy().ravel(), self.q[i].numpy().ravel()
            self.assertAlmostEqual(ref.rpc(t[i].ravel(), p, q), float(padded_j(self.f, self.p, self.q)[i]), places=12)
            self.assertAlmostEqual(ref.common_j(t[i].ravel(), p, q), ref.rpc(t[i].ravel(), p, q), places=12)

    def test_optimum_is_eta(self):
        p, q = self.p[0].numpy().ravel(), self.q[0].numpy().ravel()
        eta = ref.oracle(p, q)["eta"]
        self.assertTrue(np.allclose(ref.rpc_oracle(p, q, 1, 1, 1), eta))
        self.assertAlmostEqual(ref.common_j(eta, p, q), ref.oracle(p, q)["S"], places=12)


if __name__ == "__main__":
    unittest.main()
