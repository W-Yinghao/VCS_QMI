"""r7 R7-20 fixtures: D3 finite laws (scripts/r7_fixtures_d3.py) against the package reference (r7/package_r7_20261011/reference), the D1 / D2
metric and grouping helpers (scripts/r7_vl_common.py, scripts/r7_fixtures_d2.py).  Main env; the TextVQA m4c test runs only where llava imports."""
import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import r7_fixtures_d2 as D2  # noqa: E402
import r7_fixtures_d3 as D3  # noqa: E402
import r7_vl_common as V  # noqa: E402
import vcs_vl_reference as REF  # noqa: E402


class FiniteLaws(unittest.TestCase):
    def test_laws_and_count(self):
        self.assertEqual(sum(len(v) for v in D3.LEVELS.values()), 13)
        for k, lams in D3.LEVELS.items():
            for lam in lams:
                q = D3.q_cond(k, lam); self.assertTrue(np.allclose(q.sum(1), 1.0)); self.assertTrue((q >= 0).all())

    def test_marginal_shift_by_family(self):
        p = D3.joint(D3.cond_law()).sum(0)
        for k, lams in D3.LEVELS.items():
            for lam in lams:
                shift = np.abs(p - D3.joint(D3.q_cond(k, lam)).sum(0)).sum()
                if k in ("clean", "cond_mismatch"):
                    self.assertLess(shift, 1e-14, (k, lam))                                # image marginal kept
                else:
                    self.assertGreater(shift, 1e-3, (k, lam))                              # image marginal moves

    def test_truth_matches_reference_and_monotone(self):
        p = D3.joint(D3.cond_law()); self.assertEqual(D3.truth(p, D3.joint(D3.q_cond("clean", 0.0)))[1], 0.0)
        for k, lams in D3.LEVELS.items():
            if k == "clean":
                continue
            ss = []
            for lam in lams:
                q = D3.joint(D3.q_cond(k, lam)); eta, s = D3.truth(p, q); ss.append(s)
                g, per = REF.conditional_s(D3.cond_law(), D3.q_cond(k, lam), np.full(D3.C, 1.0 / D3.C))
                self.assertAlmostEqual(s, g, places=12)
                s_c, j_c = D3.per_condition(D3.cond_law(), D3.q_cond(k, lam), eta)
                self.assertTrue(np.allclose(s_c, per)); self.assertTrue(np.allclose(s_c, j_c))                   # J_c(η) = S_c
                self.assertAlmostEqual(D3.pop_j(p, q, eta), s, places=12)                                       # J(η) = S
                t = np.random.default_rng(1).uniform(-1, 1, size=eta.shape)                                   # S − J(T) = E_M (T − η)²
                self.assertAlmostEqual(s - D3.pop_j(p, q, t), float(np.sum(0.5 * (p + q) * (t - eta) ** 2)), places=12)
            self.assertTrue(all(np.diff(ss) > 0), k)

    def test_smoke_cell(self):
        r = D3.one_cell(("cond_mismatch", 0.8, 8, 0, True)); e = r["estimators"]
        self.assertAlmostEqual(e["oracle"]["posterior_mse"], 0.0, places=12); self.assertAlmostEqual(e["null"]["posterior_mse"], r["S_true"], places=12)
        for k in ("vcs", "logistic", "kernel"):
            self.assertGreaterEqual(e[k]["posterior_mse"], -1e-12)

    def test_mmd_unbiased_identical(self):
        x = np.random.default_rng(0).normal(size=(64, 3)); self.assertLess(abs(D3.mmd2(x, x.copy(), 1.0)), 0.05)


class Helpers(unittest.TestCase):
    def test_finite_risk_identity(self):
        g = np.random.default_rng(0); c = g.integers(0, 2, 50).astype(float); a, b = g.uniform(size=50), g.uniform(size=50)
        h, d, res = REF.finite_risk_delta(c, a, b); self.assertTrue(np.allclose(h - d, res)); self.assertTrue((h < 0).any())

    def test_official_parsers(self):
        self.assertEqual(V.gqa_correct("Yes.", "yes"), 1.0); self.assertEqual(V.gqa_correct("left side", "left"), 0.0)
        self.assertEqual(V.pope_yes("No, there is not."), 0); self.assertEqual(V.pope_yes("Yes"), 1); self.assertEqual(V.pope_yes("There is no dog."), 0)
        m = V.pope_metrics([1, 1, 0, 0], [1, 0, 0, 1]); self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (1, 1, 1, 1)); self.assertEqual(m["f1"], 0.5)

    def test_auroc_matches_pairwise(self):
        g = np.random.default_rng(2); s = g.integers(0, 5, 40).astype(float); y = g.integers(0, 2, 40)
        pairs = [(a, b) for a, b in itertools.product(np.where(y == 1)[0], np.where(y == 0)[0])]
        ref = np.mean([1.0 if s[a] > s[b] else 0.5 if s[a] == s[b] else 0.0 for a, b in pairs])
        self.assertAlmostEqual(D2.auroc(s, y), ref, places=12)

    def test_grouped_folds_and_logreg(self):
        groups = [i // 3 for i in range(300)]; f = D2.grouped_folds(groups, 5)
        for gi in set(groups):
            self.assertEqual(len({f[i] for i in range(300) if groups[i] == gi}), 1)
        X = D2.ngram_matrix(["the red cat"] * 50 + ["a blue dog"] * 50); y = np.r_[np.ones(50), np.zeros(50)]
        scorer, _ = D2.logreg(X, y); self.assertEqual(D2.auroc(scorer(X), y), 1.0)

    def test_pick_deterministic_grouped(self):
        groups = {f"img{i}": [{"_key": f"q{i}_{j}"} for j in range(4)] for i in range(30)}
        a, b = V._pick(groups, "t", 25, 2), V._pick(groups, "t", 25, 2)
        self.assertEqual([x["_key"] for x in a], [x["_key"] for x in b]); self.assertEqual(len(a), 25)

    def test_cluster_bootstrap(self):
        lo, hi = V.cluster_bootstrap(np.r_[np.ones(50), np.zeros(50)], [i // 2 for i in range(100)], reps=500); self.assertTrue(lo < 0.5 < hi)

    @unittest.skipUnless(importlib.util.find_spec("llava") is not None, "llava (official SparseVLM / LLaVA code) not importable in this env")
    def test_textvqa_soft_accuracy(self):
        sc = V.textvqa_scorer(); ans = ["two"] * 3 + ["2"] * 2 + ["three"] * 5
        self.assertAlmostEqual(sc("2", ans), 1.0); self.assertAlmostEqual(sc("Three.", ans), 1.0); self.assertAlmostEqual(sc("four", ans), 0.0)


class FakeHost:
    """Stands in for the GPU host: tensor-valued outputs and counters (as SparseVLM produces), optional crash at the k-th answer."""
    sparse = True

    def __init__(self, fail_at=None):
        import torch
        self.torch, self.load_seconds, self.yes_ids, self.no_ids, self.calls, self.fail_at = torch, torch.tensor(2.5), [1], [2], 0, fail_at

    def reset_sparse_counters(self):
        pass

    def sparse_counters(self):
        return {"equal_visual_tokens": self.torch.tensor(192.0), "official_tflops_per_sample": self.torch.tensor(2.08)}

    def answer(self, prompt, image_file, max_new_tokens=128):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("simulated crash")
        return {"text": "Yes" if image_file else "No", "p_yes": self.torch.tensor(0.9), "p_no": 0.05, "wall_ms": self.torch.tensor([1.0])[0],
                "vision_ms": 0.0, "selector_ms": 0.0, "prefill_ms": 1.0, "decode_ms": 1.0, "decode_steps": 1, "prompt_tokens": 9}


class ResumeAndSerialization(unittest.TestCase):
    """The serialisation / resume path of the GPU runs, exercised on CPU with a tiny fixture and a fake host."""

    def _fixture(self, root, names=("gqa", "pope")):
        import json
        fx = root / "fixture"; fx.mkdir(parents=True, exist_ok=True)
        for name in names:
            rows = [{"fixture": name, "qid": f"{name}{i}", "image_id": f"COCO_val2014_{i // 2:012d}", "image_file": str(root / "x.jpg"), "question": "q?",
                     "prompt": "q?" + V.SHORT, "answer": "yes" if i % 2 else "no", "binary": True, "label": i % 2, "category": "adversarial"} for i in range(6)]
            (fx / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_d1_crash_resume_tensor_json(self):
        import json
        import tempfile
        from argparse import Namespace
        import r7_fixtures_d1 as D1
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._fixture(root); a = Namespace(out=str(root), config="sparsevlm", retain=192, version="1_0", fixtures=["gqa", "pope"], warmup=1)
            h1 = FakeHost(fail_at=1 + 6 + 1 + 3)                                        # crash on the 3rd POPE answer
            with self.assertRaises(RuntimeError):
                D1.cmd_run(a, host_factory=lambda a_, o: (h1, lambda it: it["prompt"]))
            pred, run = root / "d1" / "pred_sparsevlm192.jsonl", root / "d1" / "run_sparsevlm192.json"
            self.assertEqual(len(V.read_jsonl(pred)), 8)                                 # 6 GQA + 2 POPE survived the crash
            self.assertEqual(list(json.loads(run.read_text())["fixtures"]), ["gqa"])
            with open(pred, "a") as fh:
                fh.write('{"fixture": "pope", "qid": "po')                              # torn tail
            h2 = FakeHost()
            D1.cmd_run(a, host_factory=lambda a_, o: (h2, lambda it: it["prompt"]))
            self.assertEqual(h2.calls, 1 + 4)                                           # warm-up + the 4 missing POPE items only
            rows = V.read_jsonl(pred); keys = [(r["fixture"], r["qid"]) for r in rows]
            self.assertEqual(len(keys), 12); self.assertEqual(len(set(keys)), 12)
            info = json.loads(run.read_text())
            self.assertEqual(info["fixtures"]["pope"]["resumed_items"], 2); self.assertEqual(info["fixtures"]["pope"]["items_timed"], 4)
            self.assertIsInstance(info["fixtures"]["pope"]["equal_visual_tokens"], float); self.assertIsInstance(rows[0]["p_yes"], float)
            def boom(a_, o):
                raise AssertionError("host must not load when everything is on disk")
            self.assertEqual(D1.cmd_run(a, host_factory=boom), 0)

    def test_d2_crash_resume(self):
        import json
        import tempfile
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._fixture(root, ("pope",)); a = Namespace(out=str(root))
            h1 = FakeHost(fail_at=3 + 6 + 2)                                            # crash on the 2nd text-only answer
            with self.assertRaises(RuntimeError):
                D2.cmd_pope(a, host_factory=lambda: h1)
            self.assertEqual(len(V.read_jsonl(root / "d2" / "pope_preds.jsonl")), 7)
            h2 = FakeHost(); D2.cmd_pope(a, host_factory=lambda: h2)
            self.assertEqual(h2.calls, 3 + 5)
            res = json.loads((root / "d2" / "pope_summary.json").read_text())
            self.assertEqual(res["conditions"]["image"]["accuracy"], 50.0); self.assertEqual(res["conditions"]["text_only"]["accuracy"], 50.0)


if __name__ == "__main__":
    unittest.main()
