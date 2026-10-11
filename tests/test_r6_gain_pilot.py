"""r6 GAIN-P1: identities, role split, U-only policy inputs, budget routing, losses (scripts/r6_gain_pilot.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r6_gain_pilot as G  # noqa: E402
import r6_gain_controlled as C  # noqa: E402


def probs(rng, n, k):
    x = rng.normal(size=(n, k)); e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


class Identities(unittest.TestCase):
    def test_h_minus_z(self):
        rng = np.random.default_rng(0); pa, pb = probs(rng, 500, 10), probs(rng, 500, 10); y = rng.integers(0, 10, 500)
        z, h = G.gain_targets(pa, pb, y); e = np.eye(10)[y]
        self.assertTrue(np.allclose(h - z, 2 * ((pb - pa) * (e - pb)).sum(1), atol=1e-12))
        self.assertTrue(np.allclose(h, G.brier_vec(pa, y) - G.brier_vec(pb, y), atol=1e-12))

    def test_controlled_truths_unchanged(self):                                       # the exact gains the pilot's identities rest on
        for name, want in (("redundant", 0.0), ("irrelevant", 0.0), ("xor", 1.0), ("context", 0.5)):
            self.assertAlmostEqual(C.exact(name, 0.0)["delta"], want, places=12)

    def test_family_losses(self):
        y = torch.tensor([0, 3, 9]); logits = torch.full((3, 10), -30.0); logits[torch.arange(3), y] = 30.0
        self.assertLess(float(G.family_loss(logits, y, "brier")), 1e-12)
        rnd = torch.randn(3, 10)
        self.assertAlmostEqual(float(G.family_loss(rnd, y, "ce")), float(torch.nn.functional.cross_entropy(rnd, y)), places=6)
        p = rnd.softmax(1).numpy().astype(float)
        self.assertAlmostEqual(float(G.family_loss(rnd, y, "brier")), float(G.brier_vec(p, y.numpy()).mean()), places=5)


class Roles(unittest.TestCase):
    def test_grouped_disjoint_deterministic(self):
        groups = np.repeat(np.arange(101), 3); r1, r2 = G.make_roles(groups, seed=5), G.make_roles(groups, seed=5)
        allidx = np.concatenate(list(r1.values()))
        self.assertEqual(len(allidx), len(groups)); self.assertEqual(len(np.unique(allidx)), len(groups))
        for k in r1:
            self.assertTrue(np.array_equal(r1[k], r2[k]))
        owner = {}
        for k, v in r1.items():
            for g in np.unique(groups[v]):
                self.assertNotIn(g, owner); owner[g] = k                              # no group split across roles
        self.assertEqual(len(np.unique(groups[r1["FIT_TASK"]])), 61)

    def test_saved_roles_disjoint_and_hashed(self):
        import hashlib
        import json
        for ds in ("avmnist", "cremad"):
            p = G.roles_path(ds)
            if not p.exists():
                continue
            r = json.load(open(p)); idx = {k: np.array(v["indices"], dtype=np.int64) for k, v in r["roles"].items()}; allidx = np.concatenate(list(idx.values()))
            self.assertEqual(len(allidx), len(np.unique(allidx))); self.assertEqual(len(allidx), r["n"]); self.assertEqual(int(allidx.max()), r["n"] - 1)
            for k, v in idx.items():
                self.assertEqual(hashlib.sha256(v.tobytes()).hexdigest(), r["roles"][k]["sha256"])
            groups = np.array(r["groups"]); owner = {}                            # stored groups (audio clip / actor) never span roles
            self.assertEqual(len(groups), r["n"]); self.assertEqual(len(np.unique(groups)), r["n_groups"])
            for k, v in idx.items():
                for g in np.unique(groups[v]):
                    self.assertNotIn(g, owner); owner[g] = k
            if ds == "cremad":                                                     # stored groups are the actors in the file names
                self.assertTrue(np.array_equal(G.groups_and_labels("cremad")[0], groups))

    def test_clip_groups_are_byte_exact(self):
        v = np.zeros((5, 4, 4), np.uint8); v[1] = 7; v[3] = 7; v[4, 0, 0] = 1
        g = G.avmnist_clip_groups(v)
        self.assertEqual(g[1], g[3]); self.assertEqual(g[0], g[2]); self.assertEqual(len(np.unique(g)), 3)

    def test_test_split_never_referenced(self):
        src = (R / "scripts" / "r6_gain_pilot.py").read_text()
        for bad in ('"test_data.npy"', "'test_data.npy'", '"test_labels.npy"', "'test_labels.npy'"):
            self.assertNotIn(bad, src)


class Policy(unittest.TestCase):
    def test_policy_inputs_do_not_depend_on_v(self):
        torch.manual_seed(0); m = G.build_system("avmnist", 10).eval(); u = torch.randn(8, 1, 28, 28)
        v1, v2 = torch.randn(8, 1, 112, 112), torch.randn(8, 1, 112, 112)
        with torch.no_grad():
            h1, la1 = m.cheap(u); h2, la2 = m.cheap(u); b1, b2 = m.full(h1, v1), m.full(h2, v2)
        X1 = G.policy_features(h1.numpy(), la1.softmax(1).numpy()); X2 = G.policy_features(h2.numpy(), la2.softmax(1).numpy())
        self.assertTrue(np.array_equal(X1, X2)); self.assertFalse(torch.allclose(b1, b2))    # V changes b, never the policy input
        self.assertEqual(X1.shape[1], 128 + 10)


class Routing(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1); self.n = 1000; self.pa, self.pb = probs(rng, self.n, 10), probs(rng, self.n, 10); self.y = rng.integers(0, 10, self.n)
        self.s = rng.normal(size=self.n)

    def test_budget_rate_and_endpoints(self):
        out = G.budget_curve(self.s, self.s, self.pa, self.pb, self.y, budgets=(0.1, 0.25, 0.5, 0.75))
        for b, c in out.items():
            self.assertAlmostEqual(c["rate"], float(b), delta=1.5 / self.n)
        b0, _ = G.route(np.zeros(self.n, bool), self.pa, self.pb, self.y); b1, _ = G.route(np.ones(self.n, bool), self.pa, self.pb, self.y)
        self.assertTrue(np.allclose(b0, G.brier_vec(self.pa, self.y))); self.assertTrue(np.allclose(b1, G.brier_vec(self.pb, self.y)))

    def test_topk_exact_count_label_free_seeded_ties(self):
        score = np.round(self.s, 1)                                                  # many ties
        y2 = (self.y + 3) % 10
        for beta in (0.1, 0.25, 0.5, 0.75):
            k = G.n_acquire(beta, self.n)
            self.assertEqual(k, int(np.ceil(beta * self.n)))
        c1 = G.topk_curve(score, self.pa, self.pb, self.y, tie_seed=11); c2 = G.topk_curve(score, self.pa, self.pb, y2, tie_seed=11)
        c3 = G.topk_curve(score, self.pa, self.pb, self.y, tie_seed=12)
        for b in c1:
            self.assertEqual(int(c1[b]["acquire"].sum()), G.n_acquire(float(b), self.n))
            self.assertTrue(np.array_equal(c1[b]["acquire"], c2[b]["acquire"]))      # labels never change the selection
            self.assertGreaterEqual(score[c1[b]["acquire"]].min(), score[~c1[b]["acquire"]].max())
        self.assertFalse(all(np.array_equal(c1[b]["acquire"], c3[b]["acquire"]) for b in c1))   # ties follow the seed

    def test_bootstraps(self):
        rng = np.random.default_rng(0); gi = np.repeat(np.arange(20), 5); D = np.full((4, 100), 0.3)
        tw = G.two_way_boot(D, gi, rng, nboot=200); cb = G.cluster_boot(D.mean(0), gi, rng, nboot=200)
        self.assertAlmostEqual(tw["ci95"][0], 0.3); self.assertAlmostEqual(tw["ci95"][1], 0.3); self.assertAlmostEqual(cb["ci95"][0], 0.3)
        D2 = rng.normal(size=(4, 100)) + np.arange(4)[:, None]                        # seed effects widen the two-way interval
        self.assertGreater(np.diff(G.two_way_boot(D2, gi, rng, 500)["ci95"])[0], np.diff(G.cluster_boot(D2.mean(0), gi, rng, 500)["ci95"])[0])

    def test_random_exact_count(self):
        out = G.random_curve(self.n, self.pa, self.pb, self.y, seed=0, budgets=(0.25,), repeats=3)
        self.assertEqual(out["0.25"]["rate"], 0.25)

    def test_oracle_h_is_best_at_each_budget(self):
        z, h = G.gain_targets(self.pa, self.pb, self.y)
        o = G.budget_curve(h, h, self.pa, self.pb, self.y, budgets=(0.25,)); r = G.random_curve(self.n, self.pa, self.pb, self.y, 0, budgets=(0.25,), repeats=20)
        self.assertLessEqual(o["0.25"]["brier"], r["0.25"]["brier"])


if __name__ == "__main__":
    unittest.main()
