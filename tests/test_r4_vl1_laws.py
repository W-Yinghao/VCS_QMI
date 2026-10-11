"""R4-VL1: the proposal-reference laws and the set-event softmax (src/vcs_vl/pairlaw.py 'set_softmax')."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
from vcs_vl.pairlaw import padded_batch_loss, padded_j  # noqa: E402
from r4_vl1_finetune import laws_from_A  # noqa: E402


class Laws(unittest.TestCase):
    def test_laws(self):
        A = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)   # 4 proposals x 3 queries; query 2 has no valid proposal
        p, q, n_valid, n_invalid = laws_from_A(A)
        self.assertEqual((n_valid, n_invalid), (2, 1))
        self.assertAlmostEqual(p.sum(), 1.0, places=6); self.assertAlmostEqual(q.sum(), 1.0, places=6)
        self.assertTrue(np.allclose(p[:, 2], 0) and np.allclose(q[:, 2], 0))                  # excluded query carries no weight
        self.assertTrue(np.allclose(p[:, 0], [0.25, 0.25, 0, 0]))                            # pi = 1/2 over the two valid queries, uniform over positives
        self.assertTrue(np.allclose(q[:, 0], 0.5 / 4))                                       # pi * nu, nu uniform over all 4 proposals
        self.assertTrue(np.allclose(p.sum(0), q.sum(0)))                                     # same query marginal for P and Q


class SetSoftmax(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.f = torch.randn(1, 5, 3, generator=g, dtype=torch.float64).requires_grad_()
        A = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        p, q, _, _ = laws_from_A(A); self.p = torch.from_numpy(p)[None].double(); self.q = torch.from_numpy(q)[None].double(); self.A = A

    def test_matches_direct_formula(self):
        loss = padded_batch_loss(self.f, self.p, self.q, "set_softmax")
        g = 2 * self.f[0]; direct = 0.0
        for w in range(3):
            pos = np.where(self.A[:, w] > 0)[0]
            if len(pos) == 0:
                continue
            direct += -0.5 * (torch.logsumexp(g[pos, w], 0) - torch.logsumexp(g[:, w], 0))
        self.assertAlmostEqual(float(loss), float(direct), places=10)
        grads = torch.autograd.grad(loss, self.f)[0]; self.assertTrue(torch.isfinite(grads).all())
        self.assertTrue(torch.allclose(grads[0, :, 2], torch.zeros(5, dtype=torch.float64)))  # the excluded query gets no gradient

    def test_single_positive_equals_conditional_softmax(self):
        A = np.array([[1, 0], [0, 0], [0, 1]], dtype=np.float32); p, q, _, _ = laws_from_A(A)
        f = torch.randn(1, 3, 2, dtype=torch.float64); p, q = torch.from_numpy(p)[None].double(), torch.from_numpy(q)[None].double()
        self.assertAlmostEqual(float(padded_batch_loss(f, p, q, "set_softmax")), float(padded_batch_loss(f, p, q, "conditional_softmax")), places=10)

    def test_padding_and_vcs_js_ignore_invalid(self):
        f = torch.zeros(2, 5, 3, dtype=torch.float64); P = torch.zeros(2, 5, 3, dtype=torch.float64); Q = torch.zeros_like(P)
        P[0], Q[0] = self.p[0], self.q[0]; P[1, :3, :2], Q[1, :3, :2] = self.p[0, :3, :2] / self.p[0, :3, :2].sum(), self.q[0, :3, :2] / self.q[0, :3, :2].sum()
        for obj in ("vcs", "balanced_logistic", "set_softmax"):
            self.assertTrue(math.isfinite(float(padded_batch_loss(f, P, Q, obj))))
        self.assertTrue(torch.allclose(padded_j(f, P, Q), torch.zeros(2, dtype=torch.float64)))   # T = 0 -> J = 0 for both images


if __name__ == "__main__":
    unittest.main()
