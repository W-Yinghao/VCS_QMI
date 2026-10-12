"""R7-30 D1-20 selectors (scripts/r7_d1_selectors.py, r7_d1_train.py): masking semantics, relaxations, objectives, end-to-end training on a
tiny synthetic task, CAL selection, bit-exact resume, dynamic budget.  CPU, small dimensions."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r7_d1_selectors as SEL  # noqa: E402

TINY = dict(emb_dim=8, vis_dim=6, n_vis=16, grid=4, d=16, heads=2, layers=1, ffn=32, dropout=0.0, budgets=(2, 4, 8, 12), q_per_step=16, masks_per_q=3,
            warmup=5)


def synth(n=48, seed=0, M=12):
    g = np.random.default_rng(seed); feats = torch.tensor(g.normal(size=(n, 16, 6)), dtype=torch.float32)
    C = np.where(feats[:, 0, 0].numpy() > 0, 1, -1)                                               # the label lives in visual token 0
    keep = np.zeros((n, M, 16), dtype=bool); h = np.zeros((n, M), np.float32); d = np.zeros((n, M), np.float32)
    for i in range(n):
        for m in range(M):
            k = (2, 4, 8, 12)[m % 4]; keep[i, m, g.permutation(16)[:k]] = True
            b = 0.8 * C[i]; a = b if keep[i, m, 0] else 0.0; h[i, m] = (C[i] - a) ** 2 - (C[i] - b) ** 2; d[i, m] = (b - a) ** 2
    q_ids = g.integers(1, 30, size=(n, 5)); q_len = np.full(n, 5); emb = torch.tensor(g.normal(size=(30, 8)), dtype=torch.float32)
    return SEL.TupleData(feats, np.arange(n), q_ids, q_len, emb, keep, C, h, d, "cpu")


class Semantics(unittest.TestCase):
    def test_dropped_tokens_change_nothing(self):
        torch.manual_seed(0); cfg = SEL.Cfg(**TINY); m = SEL.Selector(cfg, "vcs").eval()
        q = torch.randn(3, 5, 8); pad = torch.zeros(3, 5, dtype=torch.bool); pad[0, 3:] = True; v = torch.randn(3, 16, 6)
        keep = torch.zeros(3, 16, dtype=torch.bool); keep[:, :5] = True; v2 = v.clone(); v2[:, 5:] += 100.0; q2 = q.clone(); q2[0, 3:] += 50.0
        with torch.no_grad():
            p1 = m.encode(q, pad, v, SEL.hard_bias(keep))[1]; p2 = m.encode(q2, pad, v2, SEL.hard_bias(keep))[1]
        self.assertTrue(torch.allclose(p1, p2, atol=1e-5))                                            # dropped visual tokens and pads are invisible

    def test_no_question_ablation_ignores_question(self):
        torch.manual_seed(0); cfg = SEL.Cfg(**{**TINY, "no_question": True}); m = SEL.Selector(cfg, "vcs").eval()
        v = torch.randn(2, 16, 6); pad = torch.zeros(2, 5, dtype=torch.bool); keep = torch.ones(2, 16, dtype=torch.bool)
        with torch.no_grad():
            a = m.encode(torch.randn(2, 5, 8), pad, v, SEL.hard_bias(keep))[1]; b = m.encode(torch.randn(2, 5, 8), pad, v, SEL.hard_bias(keep))[1]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_topk_relaxations(self):
        g = torch.randn(5, 16); k = torch.tensor([2, 4, 8, 12, 16]); hk = SEL.hard_topk(g, k)
        self.assertEqual(hk.sum(1).tolist(), k.tolist())
        for r in range(5):
            self.assertTrue(set(torch.topk(g[r], int(k[r])).indices.tolist()) == set(torch.nonzero(hk[r]).flatten().tolist()))
        soft = SEL.relaxed_topk(g, k, 1e-4, torch.Generator().manual_seed(0)); self.assertTrue(torch.allclose(soft.sum(1), k.float(), atol=1e-2))

    def test_vcs_readout_zero_at_full_mask(self):
        torch.manual_seed(1); cfg = SEL.Cfg(**TINY); m = SEL.Selector(cfg, "logistic").eval(); q = torch.randn(2, 5, 8); v = torch.randn(2, 16, 6)
        pad = torch.zeros(2, 5, dtype=torch.bool); keep = torch.ones(2, 16, dtype=torch.bool)
        with torch.no_grad():
            pb = m.encode(q, pad, v, torch.zeros(2, 16))[1]; pa = m.encode(q, pad, v, SEL.hard_bias(keep))[1]
        self.assertTrue(torch.allclose(m.readout(pb, pa), torch.zeros(2), atol=1e-6))


class Schedules(unittest.TestCase):
    def test_warmup_then_cosine(self):
        lr = [SEL.cosine_lr(s, 100, 1.0, 10) for s in range(1, 101)]
        self.assertAlmostEqual(lr[0], 0.1); self.assertAlmostEqual(lr[9], 1.0); self.assertTrue(all(a >= b for a, b in zip(lr[9:], lr[10:])))
        self.assertAlmostEqual(lr[-1], 0.0, places=12)

    def test_pca_token_subsample_seeded(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cache", R / "scripts" / "r7_d1_cache.py"); C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(C)
        a, b, c = C.token_subsample("2354786"), C.token_subsample("2354786"), C.token_subsample("2354787")
        self.assertEqual(len(a), 64); self.assertEqual(len(set(a.tolist())), 64); self.assertTrue(np.array_equal(a, b)); self.assertFalse(np.array_equal(a, c))


class Training(unittest.TestCase):
    def test_end_to_end_all_objectives(self):
        fit, cal = synth(48, 0), synth(24, 1)
        for obj in ("vcs", "logistic", "direct"):
            cfg = SEL.Cfg(**TINY, steps_a=40, steps_b=10, eval_every=10, lr=3e-3, seed=0)
            model, log = SEL.train(cfg, obj, fit, cal, (1.0, 1.0))
            self.assertEqual([e["step"] for e in log["evals"]], [10, 20, 30, 40]); self.assertIn(log["selected_step"], (10, 20, 30, 40))
            best = max(e["cal_spearman_r_h"] for e in log["evals"]); self.assertAlmostEqual(log["selected_cal_spearman"], best, places=12)
            self.assertEqual(set(log["soft_hard_gap"]), {"2", "4", "8", "12"})
            dyn = SEL.dynamic_tau(model, cal, cfg, 0.5); self.assertLessEqual(abs(dyn["mean_kept_fraction"] - 0.5), 0.26)

    def test_resume_is_bit_exact(self):
        fit, cal = synth(32, 2), synth(16, 3); cfg = SEL.Cfg(**TINY, steps_a=40, steps_b=8, eval_every=10, lr=3e-3, seed=5)
        straight, _ = SEL.train(cfg, "vcs", fit, cal, (1.0, 1.0))
        with tempfile.TemporaryDirectory() as t:
            rp = Path(t) / "unit.resume.pt"
            SEL.train(cfg, "vcs", fit, cal, (1.0, 1.0), resume_path=rp, stop_after=20)                    # interrupted after the step-20 save
            resumed, log = SEL.train(cfg, "vcs", fit, cal, (1.0, 1.0), resume_path=rp)
        self.assertEqual(log["resumed_at"], [20])
        for (k1, v1), (k2, v2) in zip(straight.state_dict().items(), resumed.state_dict().items()):
            self.assertTrue(torch.equal(v1, v2), k1)


if __name__ == "__main__":
    unittest.main()
