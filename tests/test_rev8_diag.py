"""REV8 diagnostics (scripts/rev8_diag.py): the soft-mask reimplementation equals the frozen relaxed top-k (LLaVA and Qwen paths, edge budgets);
the guard; end-to-end units on tiny synthetic LLaVA / Qwen hosts (variable N with padding); dropout-free sanity (train mode == eval mode);
the summary statistics.  CPU, small dimensions."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r7_d1_40_selectors as S4  # noqa: E402
import r7_d1_selectors as SEL  # noqa: E402
import rev8_diag as RD  # noqa: E402

TINY = dict(emb_dim=8, vis_dim=6, n_vis=16, grid=4, d=16, heads=2, layers=1, ffn=32, budgets=(2, 4, 8, 12))
TINY40 = dict(emb_dim=8, vis_dim=6, d=16, heads=2, layers=1, ffn=32, max_rows=8, max_cols=8)
FLAGS = {"between_budget_fraction": 0.5, "dominance_factor": 2.0, "legacy_z": 4.0, "distortion_ratio": 0.25, "gate_jaccard": 0.8}


def synth_llava(n=24, seed=0):
    g = np.random.default_rng(seed); feats = torch.tensor(g.normal(size=(n // 2, 16, 6)), dtype=torch.float32); img_of = np.arange(n) // 2
    keep = np.zeros((n, 12, 16), dtype=bool)
    for i in range(n):
        for mech in range(3):
            order = g.permutation(16)
            for j, k in enumerate((2, 4, 8, 12)):
                keep[i, mech * 4 + j, order[:k]] = True                                              # nested per mechanism, as D1-10
    C = np.where(g.uniform(size=n) < 0.5, 1, -1); h = g.uniform(-1, 1, size=(n, 12)).astype(np.float32); d = g.uniform(0, 1, size=(n, 12)).astype(np.float32)
    emb = torch.tensor(g.normal(size=(30, 8)), dtype=torch.float32)
    return SEL.TupleData(feats, img_of, g.integers(1, 30, size=(n, 5)), np.full(n, 5), emb, keep, C, h, d, "cpu")


def synth_qwen(n=20, seed=0):
    g = np.random.default_rng(seed); n_img = n // 2; ntok = g.integers(9, 21, size=n_img); cols = np.full(n_img, 4); nmax = int(ntok.max())
    feats = torch.tensor(g.normal(size=(int(ntok.sum()), 6)), dtype=torch.float16); offsets = np.concatenate([[0], np.cumsum(ntok)[:-1]])
    img_of = np.arange(n) // 2; rank = np.full((n, 3, nmax), S4.PAD_RANK, np.int16)
    for i in range(n):
        for mech in range(3):
            rank[i, mech, :ntok[img_of[i]]] = g.permutation(ntok[img_of[i]])
    kk = np.array([[max(1, int(np.floor(f * ntok[img_of[i]] + 0.5))) for f in (0.125, 0.25, 0.5, 0.75)] for i in range(n)], np.int32)
    C = np.where(g.uniform(size=n) < 0.5, 1, -1); emb = torch.tensor(g.normal(size=(40, 8)), dtype=torch.bfloat16)
    return S4.TupleData40(feats, offsets, ntok, cols, img_of, g.integers(1, 40, size=(n, 6)), g.integers(3, 7, size=n), emb, rank, kk, C,
                          g.uniform(-1, 1, size=(n, 12)), g.uniform(0, 1, size=(n, 12)), "cpu", 8, 8)


class SynthLlava(RD.LlavaHost):
    def __init__(self, data, objective, dropout):
        self.torch, self.SEL, self.dev, self.data, self.obj, self.coverage = torch, SEL, "cpu", data, objective, {}
        self.cfg0 = SEL.Cfg(**TINY, dropout=dropout)

    def load(self, tag):
        torch.manual_seed(5); self.cfg = self.cfg0; self.model = SEL.Selector(self.cfg, self.obj).eval(); self.budget_keys = [str(k) for k in self.cfg.budgets]
        return self.obj


class SynthQwen(RD.QwenHost):
    def __init__(self, data, objective, dropout):
        self.torch, self.S4, self.SEL, self.dev, self.data, self.obj, self.coverage = torch, S4, SEL, "cpu", data, objective, {}
        self.cfg0 = S4.Cfg40(**TINY40, dropout=dropout)

    def load(self, tag):
        torch.manual_seed(5); self.cfg = self.cfg0; self.model = S4.Selector40(self.cfg, self.obj).eval(); self.budget_keys = [f"{f:g}" for f in self.cfg.fractions]
        return self.obj


def logged(H, gap=None):
    H.load("t"); r = H.predict_r(np.arange(H.data.n)); return {"selected_cal_spearman": SEL.spearman(r.reshape(-1), H.data.h.cpu().numpy().reshape(-1)), "soft_hard_gap": gap}


class SoftMatchesFrozen(unittest.TestCase):
    def test_llava_relaxed_topk_bit_exact(self):
        torch.manual_seed(0); g = torch.randn(5, 40); k = torch.tensor([1, 5, 20, 39, 40])
        for tau in (0.1, 1.0):
            a = SEL.relaxed_topk(g, k, tau, torch.Generator().manual_seed(7)); b = RD.soft_from_logits(g + RD.gumbel(g.shape, torch.Generator().manual_seed(7), "cpu"), k, tau)
            self.assertTrue(torch.equal(a, b))

    def test_qwen_relaxed_topk40_bit_exact_with_padding(self):
        torch.manual_seed(1); g = torch.randn(4, 30); valid = torch.arange(30)[None, :] < torch.tensor([30, 25, 12, 7])[:, None]; k = torch.tensor([3, 25, 12, 2])
        for tau in (0.1, 1.0):
            a = S4.relaxed_topk40(g, k, tau, valid, torch.Generator().manual_seed(3))
            b = RD.soft_from_logits(g + RD.gumbel(g.shape, torch.Generator().manual_seed(3), "cpu"), k, tau, valid)
            self.assertTrue(torch.equal(a, b))

    def test_noise_free_soft_tends_to_hard(self):
        torch.manual_seed(2); g = torch.randn(6, 20) * 5; k = torch.full((6,), 7)
        m = RD.soft_from_logits(g, k, 1e-4); self.assertTrue(torch.equal(m > 0.5, SEL.hard_topk(g, k)))


class Units(unittest.TestCase):
    def run_host(self, H):
        out, meta = RD.run_unit(H, "t", logged(H)); n, nb = H.data.n, 4
        self.assertEqual(out["r"].shape, (n, 12)); self.assertEqual(out["B_risk"].shape, (6, n, nb)); self.assertEqual(out["C_Lmean"].shape, (n, 12))
        self.assertFalse(H.model.training)
        return out, meta

    def test_dropout_free_train_mode_equals_eval(self):
        for H in (SynthLlava(synth_llava(), "vcs", 0.0), SynthQwen(synth_qwen(), "vcs", 0.0), SynthQwen(synth_qwen(), "direct", 0.0)):
            out, _ = self.run_host(H)
            self.assertTrue(np.allclose(out["C_Lmean"], out["r"], atol=1e-5)); self.assertTrue(np.allclose(out["C_Lvar"], 0, atol=1e-8))
            self.assertTrue(np.allclose(out["C_jac_gate"], 1.0))

    def test_dropout_is_seen_in_train_mode(self):
        out, _ = self.run_host(SynthLlava(synth_llava(), "logistic", 0.5))
        self.assertGreater(out["C_Lvar"].max(), 0); self.assertGreater(out["C_TAvar"].max(), 0); self.assertLess(out["C_jac_gate"].min(), 1.0)

    def test_eval_cells_reproduce_frozen_hard_readout(self):
        """HD at each budget equals the frozen dynamic-grid readout (risk_grid on Qwen; the hard top-k readout on LLaVA)."""
        H = SynthQwen(synth_qwen(), "logistic", 0.1); out, _ = self.run_host(H); Rg = S4.risk_grid(H.model, H.data, H.cfg)
        self.assertTrue(np.allclose(out["B_risk"][RD.CELLS.index("HD")], Rg[:, :4], atol=1e-5))

    def test_soft_cells_are_frozen_soft_hard_gap_shape(self):
        H = SynthLlava(synth_llava(), "vcs", 0.1); out, _ = self.run_host(H)
        self.assertTrue(np.all(np.isfinite(out["B_risk"]))); self.assertTrue(np.all(out["B_jac_hd_hn"] <= 1.0))
        self.assertTrue(np.all(np.abs(out["B_over_half_minus_k"][0, 0]) <= 16))

    def test_guard_stops_on_mismatch(self):
        H = SynthLlava(synth_llava(), "vcs", 0.1); lg = logged(H); lg["selected_cal_spearman"] += 0.01
        with self.assertRaises(SystemExit) as cm:
            RD.run_unit(H, "t", lg)
        self.assertEqual(cm.exception.code, 3)

    def test_smoke_subset(self):
        H = SynthLlava(synth_llava(), "direct", 0.1); out, meta = RD.run_unit(H, "t", logged(H), smoke_n=5)
        self.assertEqual(out["B_risk"].shape[1], 5); self.assertEqual(out["r"].shape[0], H.data.n); self.assertEqual(meta["n_sub"], 5)


class Summaries(unittest.TestCase):
    def test_avg_rank_and_spearman(self):
        self.assertEqual(list(RD.avg_rank([1, 2, 2, 3])), [0, 1.5, 1.5, 3]); self.assertAlmostEqual(RD.spearman_avg([1, 2, 3], [3, 4, 9]), 1.0)
        self.assertIsNone(RD.spearman_avg([1, 1, 1], [1, 2, 3]))

    def test_pairwise(self):
        sc, cnt, ties = RD.pairwise(np.array([[1., 2., 3.], [1., 1., 2.], [3., 2., 1.]]), np.array([[1., 2., 3.], [1., 2., 3.], [1., 1., 2.]]))
        self.assertEqual(list(sc), [3, 2.5, 0]); self.assertEqual(list(cnt), [3, 3, 2]); self.assertEqual(list(ties), [0, 0, 1])

    def test_group_boot(self):
        est, lo, hi = RD.group_boot(np.array([0, 0, 1, 2]), np.array([1., 1., 0., 2.]), np.array([2., 2., 1., 2.]))
        self.assertAlmostEqual(est, 4 / 7); self.assertLessEqual(lo, est); self.assertGreaterEqual(hi, est)

    def test_summarize_and_flags_run(self):
        for H in (SynthLlava(synth_llava(), "vcs", 0.1), SynthQwen(synth_qwen(), "direct", 0.1)):
            gap = {k: {"soft": 0.0, "hard": 0.0, "gap": 0.0} for k in (["2", "4", "8", "12"] if H.name == "llava" else ["0.125", "0.25", "0.5", "0.75"])}
            out, meta = RD.run_unit(H, "t", logged(H, gap)); meta.update({"scale": "synth", "gpu": "cpu"})
            S = RD.summarize_unit(out, meta); f = RD.flags(S, {"flags": FLAGS})
            self.assertIn("R8_02_dominant", f); self.assertLess(S["B_R8_02"]["by_budget"][meta["budget_keys"][0]]["identity_residual_max"], 1e-5)
            self.assertAlmostEqual(S["A_R8_04"]["pooled_spearman_frozen"], meta["guard"]["rho"])


if __name__ == "__main__":
    unittest.main()
