"""V7 evaluators (scripts/v7/v7_score.py): Platt recovery, per-prompt AUROC with ties / single class, ECE, paired prompt bootstrap.  CPU."""
import sys
import unittest
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "v7"))
import v7_score as VS  # noqa: E402


class Score(unittest.TestCase):
    def test_platt_recovers(self):
        rng = np.random.default_rng(0); z = rng.normal(0, 2, 20000); p = 1 / (1 + np.exp(-(0.6 * z - 0.4)))
        a, b = VS.platt_fit(z, p); self.assertAlmostEqual(a, 0.6, places=3); self.assertAlmostEqual(b, -0.4, places=3)

    def test_auroc(self):
        v, n = VS.auroc_macro(["p", "p", "p", "q", "q"], [0.9, 0.1, 0.5, 0.2, 0.3], [1.0, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual((v, n), (1.0, 1))                                                    # prompt q has one class -> excluded
        v2, _ = VS.auroc_macro(["p", "p"], [0.5, 0.5], [1.0, 0.0]); self.assertEqual(v2, 0.5)
        self.assertEqual(VS.auroc_macro(["p"], [0.5], [0.5]), (None, 0))

    def test_ece(self):
        self.assertAlmostEqual(VS.ece([0.25] * 4, [0, 0, 0, 1]), 0.0); self.assertAlmostEqual(VS.ece([1.0, 1.0], [0, 0]), 1.0)

    def test_paired_boot_sign(self):
        a = {str(i): float(i) for i in range(30)}; b = {k: v + 1 for k, v in a.items()}
        r = VS.paired_boot(a, b, reps=200); self.assertEqual(r["delta"], -1.0); self.assertEqual(r["ci95"], [-1.0, -1.0]); self.assertEqual(r["n_prompts"], 30)


class Rank(unittest.TestCase):
    def setUp(self):
        import tempfile
        if not VS.REF_DIR.exists():
            self.skipTest("package reference copy not available")
        self.d = Path(tempfile.mkdtemp()); rng = np.random.default_rng(1); self.labels = []; preds = {"good": [], "bad": [], "base": []}
        for g in VS.GENS:
            for p in range(12):
                h = rng.uniform(1, 5, 9)
                for c in range(9):
                    iid = f"{g}/{p:05d}_{c + 1:02d}.jpeg"; self.labels.append({"prompt_id": f"{p:05d}", "generator_id": g, "image_id": iid, "candidate_index": c, "human_score": float(h[c])})
                    for name, sc in (("good", h[c]), ("bad", -h[c]), ("base", float(rng.uniform()))):
                        preds[name].append({"prompt_id": f"{p:05d}", "generator_id": g, "image_id": iid, "score": float(sc)})
        self.files = {}
        for name, rows in preds.items():
            f = self.d / f"{name}.jsonl"; f.write_text("\n".join(__import__("json").dumps(r) for r in rows)); self.files[name] = str(f)
        (self.d / "labels.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in self.labels))
        (self.d / "methods.json").write_text(__import__("json").dumps({"M2_E": [self.files["good"]] * 2, "M1_E": [self.files["bad"]] * 2, "B0": [self.files["base"]], "B1": [self.files["base"]]}))

    def args(self, out, allow=True):
        a = ["rank", "--methods", str(self.d / "methods.json"), "--labels", str(self.d / "labels.jsonl"), "--out", str(out), "--comparators", "M1_E", "B0", "B1"]
        return a + (["--allow-scoring"] if allow else [])

    def test_refuses_without_switch_and_scores(self):
        with self.assertRaises(SystemExit):
            VS.main(self.args(self.d / "r0.json", allow=False))
        VS.main(self.args(self.d / "r1.json")); r = __import__("json").loads((self.d / "r1.json").read_text())
        self.assertEqual(len(r["primary"]), 6); self.assertTrue(all(v["delta"] > 0 for v in r["primary"].values()))
        self.assertTrue(all(v["p_holm"] >= v["p_approx"] for v in r["primary"].values()))
        self.assertEqual(r["methods"]["M2_E"]["DALLE_3"]["N9"]["regret"], [0.0, 0.0])
        with self.assertRaises(SystemExit):
            VS.main(self.args(self.d / "r1.json"))                                              # never overwrites

    def test_join_rejects_missing(self):
        rows = VS.VC.read_jsonl(self.files["good"])[:-1]
        with self.assertRaises(ValueError):
            VS.pools(rows, self.labels)


if __name__ == "__main__":
    unittest.main()
