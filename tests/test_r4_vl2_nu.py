"""R4-VL2: the detector-score reference ν (laws_from_A with weights, pad_laws' log ν) and the posterior-corrected rank log ν + 2 α cos."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
from vcs_vl.pairlaw import padded_batch_loss  # noqa: E402
from r4_vl1_finetune import acc, laws_from_A, pad_laws  # noqa: E402


class NuLaws(unittest.TestCase):
    A = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)          # 4 proposals x 3 queries; query 2 has no valid proposal
    s = np.array([0.9, 0.5, 0.05, 0.3])

    def test_uniform_default_unchanged(self):
        p0, q0, *_ = laws_from_A(self.A); p1, q1, *_ = laws_from_A(self.A, np.ones(4))
        self.assertTrue(np.array_equal(p0, p1)); self.assertTrue(np.allclose(q0, q1, rtol=1e-7, atol=0))

    def test_score_reference(self):
        p, q, n_valid, _ = laws_from_A(self.A, self.s); p0, *_ = laws_from_A(self.A)
        self.assertEqual(n_valid, 2); self.assertTrue(np.array_equal(p, p0))                  # P does not depend on ν
        self.assertAlmostEqual(float(q.sum()), 1.0, places=6)
        self.assertTrue(np.allclose(q[:, 0], 0.5 * self.s / self.s.sum()))                   # q = π(w) ν(r)
        self.assertTrue(np.allclose(q[:, 2], 0)); self.assertTrue(np.allclose(p.sum(0), q.sum(0)))

    def test_pad_laws_logn(self):
        recs = []
        for m, k in ((3, 2), (5, 1)):
            A = np.zeros((m, k), np.float32); A[0, :] = 1; nu = np.arange(1, m + 1, dtype=np.float64); p, q, nv, _ = laws_from_A(A, nu)
            recs.append({"boxes": [None] * m, "texts": ["t"] * k, "p": p, "q": q, "A": A, "A75": A, "logn": np.log(nu / nu.sum()).astype(np.float32)})
        d = pad_laws(recs)
        self.assertTrue(torch.allclose(d["logn"][0, :3], torch.from_numpy(recs[0]["logn"]))); self.assertTrue(torch.all(d["logn"][0, 3:] == 0))
        for r in recs:
            del r["logn"]
        self.assertNotIn("logn", pad_laws(recs))                                             # uniform runs: no correction term


class CorrectedRank(unittest.TestCase):
    """At the optimum of either estimator objective exp(2 f) = p / q = p(r | w) / ν(r): log ν + 2 f ranks by the posterior, f alone by p / ν."""
    pc = np.array([[0.5, 0.1], [0.3, 0.2], [0.15, 0.3], [0.05, 0.4]])                         # p(r | w), 4 proposals x 2 queries (all > 0)
    nu = np.array([0.05, 0.9, 0.5, 0.3]) / 1.75

    def laws(self):
        p = torch.tensor(self.pc * 0.5)[None]; q = torch.tensor(np.outer(self.nu, [0.5, 0.5]))[None]
        return p, q

    def test_stationary_point(self):
        p, q = self.laws(); fstar = 0.5 * torch.log(p / q)
        for obj in ("vcs", "balanced_logistic"):
            f = fstar.clone().requires_grad_(); padded_batch_loss(f, p, q, obj).backward()
            self.assertLess(float(f.grad.abs().max()), 1e-12, obj)
            for eps in (1e-3, -1e-3):                                                         # a minimum, not a saddle
                self.assertGreater(float(padded_batch_loss(fstar + eps, p, q, obj)), float(padded_batch_loss(fstar, p, q, obj)), obj)

    def test_acc_uses_correction(self):
        p, q = self.laws(); fstar = 0.5 * torch.log(p / q); alpha = 7.0; C = (fstar / alpha).float()   # critic f = alpha cos (+ b, constant per query)
        A = torch.zeros(1, 4, 2); A[0, 0, 0] = 1; A[0, 3, 1] = 1                              # hits at the posterior argmax of each query
        d = {"rmask": torch.ones(1, 4, dtype=torch.bool), "wmask": torch.ones(1, 2, dtype=torch.bool), "logn": torch.tensor(np.log(self.nu), dtype=torch.float32)[None]}
        self.assertEqual(acc(C, d, A, alpha=alpha)["query"], 1.0)
        self.assertEqual(acc(C, d, A)["query"], 0.5)                                          # plain f picks argmax p / ν = proposal 0 for query 1
        d_u = {k: v for k, v in d.items() if k != "logn"}
        self.assertEqual(acc(C, d_u, A, alpha=alpha)["query"], acc(C, d_u, A)["query"])        # no log ν: alpha is ignored
        self.assertEqual(acc(C, d, A, K=2, alpha=alpha)["query"], 0.5)                       # top-K keeps the first K proposals only


if __name__ == "__main__":
    unittest.main()
