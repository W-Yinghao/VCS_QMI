"""R4-E1 solver check (scripts/r4_e1_solver.py): the CAL readout equals P85's native VCS value, the J-on-JS-coordinate loss, the point readouts
equal BM.posterior_readouts, the population identity J(T) = S − E_M (T − η)², the selection rule, and identical branch starts / batch streams."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
from vcs_estim import benchmark as BM  # noqa: E402
from vcs_estim.critics import JointMLP  # noqa: E402
from vcs_estim.data import setting_from_mi  # noqa: E402
from vcs_estim.estimators import estimate, scores  # noqa: E402
import r4_e1_solver as E1  # noqa: E402


def small_role(n=48, d=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return BM.Role(torch.randn(n, d, generator=g, dtype=torch.float64), torch.randn(n, d, generator=g, dtype=torch.float64),
                   torch.randn(n, d, generator=g, dtype=torch.float64), torch.randn(n, d, generator=g, dtype=torch.float64))


class Readouts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3); self.critic = JointMLP(20, 20); self.role = small_role()

    def test_select_common_equals_p85_native_vcs(self):
        common, native = E1.inbatch_values(self.critic, self.role, 1.0, "vcs")
        a, b, how, *_ = BM._native_on(self.critic, "vcs", "inbatch", self.role, E1.DEV, BM.EVAL_BLOCK, None)
        self.assertEqual(common, native); self.assertAlmostEqual(common, BM.combine(a, b, how), places=12)

    def test_js_native_and_common_coordinate(self):
        common, native = E1.inbatch_values(self.critic, self.role, 0.5, "js")
        a, b, how, *_ = BM._native_on(self.critic, "js", "inbatch", self.role, E1.DEV, BM.EVAL_BLOCK, None)
        self.assertAlmostEqual(native, BM.combine(a, b, how), places=12)
        with torch.no_grad():
            fp, fn = scores(self.critic, BM._dev(self.role.xp, E1.DEV), BM._dev(self.role.yp, E1.DEV), "inbatch", None)
        tp, tq = torch.tanh(0.5 * fp.double()), torch.tanh(0.5 * fn.double())
        self.assertAlmostEqual(common, float((tp - tp ** 2 / 2).mean() + (-tq - tq ** 2 / 2).mean()), places=12)

    def test_fit_readout_first_anchors(self):
        sub = BM.Role(self.role.xp[:16], self.role.yp[:16], self.role.xq[:16], self.role.yq[:16])
        self.assertAlmostEqual(E1.inbatch_values(self.critic, self.role, 0.5, None, n_anchor=16)[0], E1.inbatch_values(self.critic, sub, 0.5, None)[0], places=12)

    def test_eval_point_equals_posterior_readouts(self):
        g = torch.Generator().manual_seed(1); tp, tn = torch.rand(500, generator=g, dtype=torch.float64) * 2 - 1, torch.rand(500, generator=g, dtype=torch.float64) * 2 - 1
        ep, en = torch.rand(500, generator=g, dtype=torch.float64) * 2 - 1, torch.rand(500, generator=g, dtype=torch.float64) * 2 - 1
        tr = {"S": 0.4, "bayes_denominator_resolvable": True}; ref = BM.posterior_readouts(tp, tn, ep, en, tr, torch.Generator().manual_seed(0), 5)
        pt = E1.eval_point(tp, tn, ep, en, 0.4)
        for k in ("J_eval", "J_eval_minus_S", "posterior_mse"):
            self.assertAlmostEqual(pt[k], ref[k], places=12, msg=k)

    def test_population_identity(self):
        """With M uniform over atoms, P = M(1 + η), Q = M(1 − η): J(T) = E_M η² − E_M (T − η)² for any T (the common-readout identity)."""
        g = np.random.default_rng(0); eta = g.uniform(-0.9, 0.9, 200); eta = np.concatenate([eta, -eta]); M = np.full(len(eta), 1 / len(eta))
        P, Q = M * (1 + eta), M * (1 - eta); T = g.uniform(-1, 1, len(eta))
        J = float((P * (T - T ** 2 / 2)).sum() + (Q * (-T - T ** 2 / 2)).sum()); self.assertAlmostEqual(P.sum(), 1.0); self.assertAlmostEqual(Q.sum(), 1.0)
        self.assertAlmostEqual(J, float((M * eta ** 2).sum() - (M * (T - eta) ** 2).sum()), places=12)


class Losses(unittest.TestCase):
    def test_loss_fn(self):
        g = torch.Generator().manual_seed(2); fp, fn = torch.randn(8, generator=g), torch.randn(8, 7, generator=g)
        self.assertTrue(torch.equal(E1.loss_fn("J", 1.0, fp, fn), estimate("vcs", fp, fn)[0]))
        tp, tq = torch.tanh(fp / 2), torch.tanh(fn / 2)
        self.assertAlmostEqual(float(E1.loss_fn("J", 0.5, fp, fn)), float(-((tp - tp ** 2 / 2).mean() + (-tq - tq ** 2 / 2).mean())), places=6)
        self.assertTrue(torch.equal(E1.loss_fn("js", 0.5, fp, fn), estimate("js", fp, fn)[0]))
        with self.assertRaises(ValueError):
            E1.loss_fn("infonce", 1.0, fp, fn)

    def test_js_and_J_share_the_optimum_coordinate(self):
        """For one pair with masses (p, q) both fit criteria are stationary at u* = log(p / q), where tanh(u*/2) = (p − q)/(p + q): JS directly,
        J on T = tanh(u/2) through T* = (p − q)/(p + q)."""
        p, q = 0.7, 0.2; ustar = math.log(p / q); self.assertAlmostEqual(math.tanh(ustar / 2), (p - q) / (p + q), places=14)
        for crit in ("js", "J"):
            u = torch.tensor(ustar, dtype=torch.float64, requires_grad=True)
            if crit == "js":
                L = p * torch.nn.functional.softplus(-u) + q * torch.nn.functional.softplus(u)
            else:
                t = torch.tanh(u / 2); L = -(p * (t - t ** 2 / 2) + q * (-t - t ** 2 / 2))
            L.backward(); self.assertLess(abs(float(u.grad)), 1e-12, crit)
            for eps in (1e-3, -1e-3):                                                           # a minimum, not a maximum
                v = torch.tensor(ustar + eps, dtype=torch.float64)
                Lv = (p * torch.nn.functional.softplus(-v) + q * torch.nn.functional.softplus(v)) if crit == "js" else \
                    -(p * (torch.tanh(v / 2) - torch.tanh(v / 2) ** 2 / 2) + q * (-torch.tanh(v / 2) - torch.tanh(v / 2) ** 2 / 2))
                self.assertGreater(float(Lv), float(L), crit)


class Selection(unittest.TestCase):
    def test_earliest_on_ties_and_nonfinite(self):
        c = [{"select_common_J": v} for v in (0.1, 0.5, float("nan"), 0.5, 0.4, float("inf"))]
        self.assertEqual(E1.select_index(c[:5]), 1)
        self.assertEqual(E1.select_index(c), 1)                                                # +inf is non-finite: never selected


class Branching(unittest.TestCase):
    def setUp(self):
        setting = setting_from_mi("gaussian", 6.0, 20); roles, _ = BM.build_roles(setting, 20, 20, 64, 0, smoke=True); tr = BM.truths(setting, 20, roles["TRUTH"])
        E = roles["EVAL"]; eta = (BM.eta_of(setting, E.xp, E.yp, 20), BM.eta_of(setting, E.xq, E.yq, 20)); self.ctx = E1.Ctx(roles, tr, eta, True)

    def fresh(self, lr=1e-3, seed=0):
        torch.manual_seed(seed * 7919 + 17); c = JointMLP(20, 20); return c, torch.optim.Adam(c.parameters(), lr=lr), torch.Generator().manual_seed(seed * 1000 + 7)

    def test_branch_batch_stream_and_identical_starts(self):
        c, o, g = self.fresh(); curve, states, gn, lv = [], {}, [], []; saved = {}

        def keep(step, cc, oo, gg):
            if step == 10:
                saved.update(state={k: v.clone() for k, v in cc.state_dict().items()}, g=gg.get_state().clone())
        E1.train(self.ctx, c, o, g, "js", 0.5, "js", 0, 20, curve, states, gn, lv, keep)
        self.assertEqual([r["update"] for r in curve], [10, 20]); self.assertEqual(len(gn), 20)
        ends = []
        for objective in ("J", "J", "js"):                                                    # two identical branches + the control
            torch.manual_seed(1); b = JointMLP(20, 20); b.load_state_dict(saved["state"]); gb = torch.Generator(); gb.set_state(saved["g"])
            self.assertEqual(E1.checksum(b.state_dict()), E1.checksum(saved["state"]))
            E1.train(self.ctx, b, torch.optim.Adam(b.parameters(), lr=1e-3), gb, objective, 0.5, "js", 10, 20, [], {}, [], [])
            ends.append(E1.checksum(b.state_dict()))
        self.assertEqual(ends[0], ends[1]); self.assertNotEqual(ends[0], ends[2])
        g1 = torch.Generator().manual_seed(7); idx = [torch.randint(0, 64, (E1.B,), generator=g1) for _ in range(20)]
        g2 = torch.Generator().manual_seed(7)
        for _ in range(10):
            torch.randint(0, 64, (E1.B,), generator=g2)
        g3 = torch.Generator(); g3.set_state(g2.get_state())
        self.assertTrue(all(torch.equal(idx[10 + i], torch.randint(0, 64, (E1.B,), generator=g3)) for i in range(10)))

    def test_direct_loop_matches_benchmark_train_neural(self):
        """The recorded loop takes the same updates as P85's train_neural (its seeds, draws and loss call, replayed line by line): identical weights."""
        c, o, g = self.fresh(5e-4); E1.train(self.ctx, c, o, g, "J", 1.0, "vcs", 0, 20, [], {}, [], [])
        torch.manual_seed(0 * 7919 + 17); r = JointMLP(20, 20); ro = torch.optim.Adam(r.parameters(), lr=5e-4); rg = torch.Generator().manual_seed(0 * 1000 + 7)
        F = self.ctx.roles["FIT"]; Fx, Fy = BM._dev(F.xp, E1.DEV), BM._dev(F.yp, E1.DEV)
        for _ in range(20):
            ip = torch.randint(0, F.n, (E1.B,), generator=rg); fp, fn = scores(r, Fx[ip], Fy[ip], "inbatch", rg); loss, _ = estimate("vcs", fp, fn)
            ro.zero_grad(set_to_none=True); loss.backward(); ro.step()
        self.assertTrue(all(torch.equal(c.state_dict()[k], r.state_dict()[k]) for k in r.state_dict()))


if __name__ == "__main__":
    unittest.main()
