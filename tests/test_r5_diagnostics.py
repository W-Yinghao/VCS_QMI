"""r5 diagnostics (scripts/r5_diagnostics.py, scripts/r5_fusion.py): score-level gradients against autograd of the production losses
(src/vcs_vl/pairlaw.py), KL(U_A ‖ ρ_A) against the r5 reference, λ tie-breaking, nested prefix / determinism / target tie-break, NA for zero
gradient denominators, and the repeats → query → image → images averaging order."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "r5" / "package_r5_20261011" / "reference"))
from vcs_vl.pairlaw import padded_batch_loss  # noqa: E402
import r5_diagnostics as D  # noqa: E402
import r5_fusion as FU  # noqa: E402
import r5_reference as REF  # noqa: E402

LOSSES = ("vcs", "balanced_logistic", "set_softmax", "distribution_softmax")


def query(seed, n=9, k=3, score=True):
    g = np.random.default_rng(seed); valid = np.zeros(n, bool); valid[g.choice(n, k, replace=False)] = True
    nu = g.uniform(0.05, 1.0, n) if score else np.ones(n); nu = nu / nu.sum(); f = g.normal(0, 1.2, n)
    return f, valid, valid / k, nu


class Gradients(unittest.TestCase):
    def test_against_autograd(self):
        for seed in range(6):
            for loss in LOSSES:
                f, valid, p, q = query(seed, score=loss in ("vcs", "balanced_logistic"))
                ft = torch.tensor(f)[None, :, None].requires_grad_(); padded_batch_loss(ft, torch.tensor(p)[None, :, None], torch.tensor(q)[None, :, None], loss).backward()
                T = np.tanh(f); rho_s = np.exp(2 * f - (2 * f).max()); rho_s /= rho_s.sum()
                gp, gq, g = D.score_gradients(loss, p, q, T, rho_s, valid)
                self.assertTrue(np.allclose(g, ft.grad[0, :, 0].numpy(), atol=1e-12), loss); self.assertTrue(np.allclose(gp + gq, g))

    def test_invalid_p_component_is_zero(self):
        f, valid, p, q = query(1)
        for loss in ("vcs", "balanced_logistic"):
            gp, gq, g = D.score_gradients(loss, p, q, np.tanh(f), None, valid)
            self.assertTrue(np.all(gp[~valid] == 0) and np.allclose(g[~valid], gq[~valid]))


class Probabilities(unittest.TestCase):
    def test_kl_matches_reference_and_k1(self):
        for seed in range(5):
            f, valid, p, q = query(seed, k=3, score=False); rho = np.exp(2 * f); rho /= rho.sum(); kl, h = D.kl_uniform(rho, valid)
            ft = torch.tensor(f)[None, :, None]; pt = torch.tensor(p)[None, :, None]; qt = torch.tensor(q)[None, :, None]
            ref = REF.uniform_set_decomposition(ft, pt, qt, torch.ones(1, len(f), dtype=torch.bool))
            self.assertAlmostEqual(kl, float(ref["kl_uniform_to_legal"][0, 0]), places=10); self.assertGreaterEqual(kl, -1e-12)
        f, valid, p, q = query(0, k=1, score=False); rho = np.exp(2 * f); rho /= rho.sum(); self.assertAlmostEqual(D.kl_uniform(rho, valid)[0], 0.0, places=12)

    def test_model_arrays_rank(self):
        cp = np.random.default_rng(2).normal(size=(6, 2)); det = np.linspace(0.9, 0.1, 6)
        f, T, dep_u, rho_u, rho_s, logn = D.model_arrays(cp, det, "uniform", "vcs", (10.0, -2.5)); self.assertTrue(np.array_equal(dep_u.argmax(0), cp.argmax(0)))
        f, T, dep_s, rho, rho_s, logn = D.model_arrays(cp, det, "score", "vcs", (10.0, -2.5))
        self.assertTrue(np.allclose(dep_s - dep_s.max(0), (20 * cp + np.log(det)[:, None]) - (20 * cp + np.log(det)[:, None]).max(0)))
        self.assertTrue(np.allclose(rho.sum(0), 1) and np.allclose(rho_s.sum(0), 1))


class Selection(unittest.TestCase):
    def test_lambda_ties_take_smallest(self):
        self.assertEqual(REF.choose_cal_lambda({0.0: 0.5, 0.25: 0.6, 1.0: 0.6, 4.0: 0.55}), 0.25)

    def test_nested_prefix_and_determinism(self):
        ids = [f"{r:03d}" for r in range(40)]; a = REF.nested_invalid_ids("img:0", ids, 3, D.NESTED); b = REF.nested_invalid_ids("img:0", ids, 3, D.NESTED)
        self.assertEqual(a, b)
        for c in D.NESTED:
            self.assertEqual(a[str(c)], a["all"][:c])
        self.assertNotEqual(REF.nested_invalid_ids("img:0", ids, 4, D.NESTED)["all"], a["all"])

    def test_target_tie_break_and_hits(self):
        P = np.array([[0, 0, 10, 10]] * 3 + [[50, 50, 60, 60]] * 40, float); meta = [{"image_id": 7, "prop_boxes": P, "det_score": np.r_[0.5, 0.9, 0.9, np.full(40, 0.2)],
                                                                                   "target": np.array([0]), "ref_cat": np.array([1]), "ref_boxes": np.array([[0, 0, 10, 10.0]])}]
        G = [{"iou": np.r_[np.ones(3), np.zeros(40)][None, :], "etype": [["valid"] * 3 + ["no_matching_annotation"] * 40]}]
        items = D.nested_manifest(meta, G); self.assertEqual(len(items), 1); self.assertEqual(items[0]["target"], 1)   # IoU tie -> det score desc -> index asc
        dep = np.r_[0.0, 1.0, 0.0, np.linspace(-1, 2, 40)][:, None]; h = D.nested_hits(items, [dep])[0]
        self.assertEqual(h["all"], 0.0); self.assertTrue(all(0.0 <= h[str(c)] <= 1.0 for c in D.NESTED)); self.assertGreaterEqual(h["1"], h["31"])


class Averaging(unittest.TestCase):
    def test_query_then_image_then_images(self):
        items = [{"image": 0}, {"image": 0}, {"image": 1}]; vals = [{"x": 1.0}, {"x": 0.0}, {"x": 1.0}]
        per = D.per_image_from_queries(items, vals, 3, "x"); self.assertTrue(np.allclose(per[:2], [0.5, 1.0]) and np.isnan(per[2]))
        self.assertAlmostEqual(D.macro_mean(per), 0.75)                                         # not the query mean 2/3

    def test_grad_fraction_na(self):
        self.assertTrue(math.isnan(D.frac(0.0, 0.0))); self.assertEqual(D.frac(1.0, 4.0), 0.25)

    def test_fusion_accuracy_counts_uncovered(self):
        meta = [{"det_score": np.array([0.9, 0.1]), "target": np.array([0, 0])}, {"det_score": np.array([]), "target": np.array([0])}]
        cps = [np.array([[0.1, 0.5], [0.2, 0.4]]), np.zeros((0, 1))]; valid = [np.array([[True, True], [False, False]]), np.zeros((0, 1), bool)]
        self.assertAlmostEqual(FU.accuracy(cps, meta, valid, 10.0, 0.0), (0.5 + 0.0) / 2)       # query 0 picks proposal 1 (invalid); empty image = 0
        self.assertAlmostEqual(FU.accuracy(cps, meta, valid, 10.0, 4.0), (1.0 + 0.0) / 2)       # the detector score flips query 0


if __name__ == "__main__":
    unittest.main()
