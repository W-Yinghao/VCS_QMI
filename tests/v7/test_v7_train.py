"""V7 M runner pieces (scripts/v7/v7_common.py, v7_train.py): the element losses equal the package reference (incl. 1 - 4 L = J for VCS and
micro-batch additivity), span pooling counts each element once, offsets -> tokens, prior correction, Brier / hierarchical means, the FIT prior
of the spanned population, ElementData / Batcher on synthetic manifests, and forward / losses / evaluate on a tiny model built from LAVIS's
own Q-Former class (gradients reach the Q-Former and the heads, never the frozen vision encoder).  CPU."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "v7"))
import v7_common as VC  # noqa: E402
import v7_train as VT  # noqa: E402

REF = Path("/home/infres/yinwang/CS_QMI/outputs/V7/package_ref/reference")


def ref():
    if not (REF / "v7_math.py").exists():
        return None
    sys.path.insert(0, str(REF)); import v7_math, v7_metrics  # noqa: E401
    return v7_math, v7_metrics


class Losses(unittest.TestCase):
    def test_match_reference_and_J(self):
        r = ref()
        if r is None:
            self.skipTest("package reference not available")
        vm, _ = r; z = torch.tensor([-0.8, 0.1, 0.6, 1.4], dtype=torch.float64); y = torch.tensor([0, 0.4, 1, 0.6], dtype=torch.float64)
        w = torch.tensor([0.5, 2, 1, 3], dtype=torch.float64); pi = float((w * y).sum() / w.sum())
        for obj in ("vcs", "logistic", "l1"):
            ours = VC.element_loss(z, y, w, pi, float(w.sum()), obj); theirs = vm.matched_torch_loss(z, y, w, pi, float(w.sum()), obj)
            self.assertAlmostEqual(float(ours), float(theirs), places=6)
        L = float(VC.element_loss(z, y, w, pi, float(w.sum()), "vcs")); self.assertAlmostEqual(1 - 4 * L, vm.common_j(np.tanh(z.numpy() / 2), y.numpy(), w.numpy()), places=6)

    def test_microbatch_additive(self):
        z = torch.randn(10, requires_grad=True); y = torch.rand(10); w = torch.rand(10) + 0.1; n = float(w.sum())
        for obj in VC.OBJECTIVES:
            full = VC.element_loss(z, y, w, 0.4, n, obj); g = torch.autograd.grad(full, z)[0]
            parts = VC.element_loss(z[:3], y[:3], w[:3], 0.4, n, obj) + VC.element_loss(z[3:], y[3:], w[3:], 0.4, n, obj)
            g2 = torch.autograd.grad(parts, z)[0]; self.assertTrue(torch.allclose(full, parts, atol=1e-6) and torch.allclose(g, g2, atol=1e-6))

    def test_bad_prior(self):
        with self.assertRaises(ValueError):
            VC.element_loss(torch.zeros(1), torch.zeros(1), torch.ones(1), 0.0, 1.0, "vcs")


class Readout(unittest.TestCase):
    def test_prior_correction(self):
        p, q = np.array([0.8, 0.2]), np.array([0.3, 0.7]); z = np.log(p / q)
        for pi in (0.1, 0.5, 0.9):
            self.assertTrue(np.allclose(VC.natural_probability(z, pi), pi * p / (pi * p + (1 - pi) * q)))

    def test_brier_and_hierarchy(self):
        y = np.array([0, 0.2, 0.5, 1]); p = np.array([0.1, 0.1, 0.6, 0.9])
        self.assertTrue(np.allclose(VC.expected_brier(p, y), (p - y) ** 2 + y * (1 - y)))
        m, n = VC.hierarchical_mean(["p1"] * 11 + ["p2"], ["i1"] * 10 + ["i2", "i3"], [1] * 10 + [0, 0]); self.assertEqual((m, n), (0.25, 2))

    def test_spearman_ties_and_constant(self):
        from scipy.stats import spearmanr
        a, b = [1, 1, 2, 3, 3], [2, 1, 1, 3, 3]; self.assertAlmostEqual(VC.avg_rank_spearman(a, b), spearmanr(a, b).statistic)
        self.assertTrue(np.isnan(VC.avg_rank_spearman([1, 1, 1], [1, 2, 3])))


class Tokens(unittest.TestCase):
    def test_token_spans_and_pooling(self):
        offs = [(0, 0), (0, 3), (4, 7), (8, 11), (0, 0)]
        self.assertEqual(VC.token_spans(offs, [(4, 11)]), [2, 3]); self.assertEqual(VC.token_spans(offs, [(0, 3), (8, 11)]), [1, 3]); self.assertEqual(VC.token_spans(offs, [(20, 22)]), [])
        z = torch.tensor([[0., 1., 2., 3., 9.], [5., 5., 5., 5., 5.]]); owner = torch.tensor([0, 0, 1])
        idx = torch.tensor([[1, 2], [3, 0], [1, 2]]); m = torch.tensor([[True, True], [True, False], [True, True]])
        self.assertTrue(torch.allclose(VC.pool_logits(z, owner, idx, m), torch.tensor([1.5, 3.0, 5.0])))


def synth_manifests(n_prompts=6, per=2):
    imgs, by_img = [], {}
    roles = ["FIT"] * (n_prompts - 2) + ["CAL_SELECT", "CAL_SELECT"]
    for p in range(n_prompts):
        prompt = f"a red cat and a blue dog number {p}"
        for k in range(per if p % 2 == 0 else 1):
            iid = f"G/{p:03d}_{k}.png"
            imgs.append({"image_id": iid, "prompt_id": f"{p:03d}", "prompt": prompt, "role": roles[p], "overall_mean": 1 + (p + k) % 5})
            by_img[iid] = [{"element_id": f"{iid}::cat", "y": 1.0, "spans": [[6, 9]]}, {"element_id": f"{iid}::blue dog", "y": 0.5 * (k + p % 2), "spans": [[16, 24]]}]
    return imgs, by_img


class Data(unittest.TestCase):
    def setUp(self):
        from transformers import BertTokenizerFast
        if not VC.BERT_DIR.exists():
            self.skipTest("local bert-base-uncased not available")
        self.tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))

    def test_prior_and_rows(self):
        imgs, by_img = synth_manifests(); pi = VT.fit_prior_spanned(imgs, by_img)
        fit = [i for i in imgs if i["role"] == "FIT"]; per = {}
        for i in fit:
            per.setdefault(i["prompt_id"], []).append(i)
        exp = np.mean([np.mean([np.mean([e["y"] for e in by_img[i["image_id"]]]) for i in v]) for v in per.values()]); self.assertAlmostEqual(pi, exp)
        d = VT.ElementData(imgs, by_img, "FIT", self.tok, 128); self.assertEqual(len(d), len(fit))
        r = d.rows[0]; toks = [self.tok.convert_ids_to_tokens([r["ids"][t] for t in e["tok"]]) for e in r["elements"]]
        self.assertEqual(toks, [["cat"], ["blue", "dog"]]); self.assertAlmostEqual(sum(x["w"] for x in d.rows), len(d.rows))

    def test_too_long_is_an_error(self):
        imgs, by_img = synth_manifests()
        with self.assertRaises(SystemExit):
            VT.ElementData(imgs, by_img, "FIT", self.tok, 5)

    def test_batcher(self):
        from PIL import Image
        imgs, by_img = synth_manifests(); d = VT.ElementData(imgs, by_img, "FIT", self.tok, 128)
        proc = lambda im: torch.from_numpy(np.asarray(im.resize((8, 8)), dtype=np.float32)).permute(2, 0, 1) / 255  # noqa: E731
        b = VT.Batcher(d, lambda iid: Image.new("RGB", (16, 16)), proc)([0, 1, 2])
        self.assertEqual(b["image"].shape, (3, 3, 8, 8)); self.assertEqual(len(b["y"]), 6); self.assertEqual(b["owner"].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertTrue(torch.allclose(b["w_e"][:2].sum(), b["w_img"][0])); self.assertEqual(float(b["content"][0, 0]), 0.0)


class TinyModel(torch.nn.Module):
    """The attributes VT.forward uses, built from LAVIS's Q-Former class and FGA's mask head at toy sizes."""

    def __init__(self):
        super().__init__()
        VC.import_lavis()
        from lavis.models.blip2_models.Qformer import BertLMHeadModel
        from lavis.models.blip2_models.fga_blip2 import MLP
        from transformers import BertConfig
        cfg = BertConfig(vocab_size=30524, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
        cfg.encoder_width, cfg.add_cross_attention, cfg.cross_attention_freq, cfg.query_length = 16, True, 2, 32
        self.Qformer = BertLMHeadModel(cfg); self.query_tokens = torch.nn.Parameter(torch.randn(1, 32, 32) * 0.02)
        self.visual_encoder = FakeVision()
        for p in self.visual_encoder.parameters():
            p.requires_grad_(False)
        self.ln_vision = torch.nn.LayerNorm(16); self.itm_head = torch.nn.Linear(32, 2); self.mask_proj = MLP(32)


class FakeVision(torch.nn.Module):
    """[B, 3, H, W] -> [B, 30, 16] patch features (a stand-in for the frozen EVA ViT-g)."""

    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((2, 5)); self.lin = torch.nn.Linear(10, 160)

    def forward(self, x):
        return self.lin(self.pool(x).flatten(2)).view(x.shape[0], -1, 16)


class Forward(unittest.TestCase):
    def test_forward_losses_eval(self):
        if not VC.BERT_DIR.exists():
            self.skipTest("local bert-base-uncased not available")
        from PIL import Image
        from transformers import BertTokenizerFast
        torch.manual_seed(0); m = TinyModel(); ve = m.visual_encoder
        tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR)); imgs, by_img = synth_manifests(); d = VT.ElementData(imgs, by_img, "FIT", tok, 128)
        proc = lambda im: torch.from_numpy(np.asarray(im.resize((8, 8)), dtype=np.float32)).permute(2, 0, 1) / 255  # noqa: E731
        src = lambda iid: Image.new("RGB", (16, 16), color=(hash(iid) % 255, 10, 200))  # noqa: E731
        b = VT.Batcher(d, src, proc)(list(range(len(d)))); pi = VT.fit_prior_spanned(imgs, by_img)
        for obj in VC.OBJECTIVES:
            m.zero_grad(); loss, parts = VT.losses(m, b, "cpu", obj, pi); loss.backward(); self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(m.Qformer.bert.encoder.layer[0].attention.self.query.weight.grad); self.assertIsNotNone(m.mask_proj.layers[0].weight.grad)
            self.assertTrue(all(p.grad is None for p in ve.parameters()))
        met, el, ov = VT.evaluate(m, VT.ElementData(imgs, by_img, "CAL_SELECT", tok, 128), src, proc, "cpu", pi, batch=2)
        self.assertEqual(met["n_elements"], 6); self.assertTrue(np.isfinite(met["natural_brier_macro"])); self.assertTrue(m.training)


if __name__ == "__main__":
    unittest.main()
