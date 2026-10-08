"""VL1: the package reference tests (VCS-VL-SERVER-v1-20261008 tests/test_vl_pair_reference.py, ported unchanged against src/vcs_vl/pairlaw.py)
plus checks of the batched loss and the RefCOCOg adapter."""
import unittest
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_vl.pairlaw import *  # noqa: F401,F403
from vcs_vl import refcocog as RG

class PairReferenceTests(unittest.TestCase):
    def setUp(self):
        self.a=np.array([[1,1,0],[0,1,1]],dtype=float)
        self.p,self.q=pair_laws(self.a)
    def test_mass(self):
        self.assertAlmostEqual(self.p.sum(),1);self.assertAlmostEqual(self.q.sum(),1)
    def test_uniform_region(self):
        np.testing.assert_allclose(self.p.sum(1),.5)
    def test_q_is_product(self):
        np.testing.assert_allclose(self.q,np.outer(self.p.sum(1),self.p.sum(0)))
    def test_q_keeps_positives(self):
        self.assertGreater(self.q[self.a>0].sum(),0)
    def test_gap(self):
        rng=np.random.default_rng(0);t=np.tanh(rng.normal(size=self.p.shape))
        eta,s=eta_and_s(self.p,self.q)
        self.assertAlmostEqual(s-j_readout(self.p,self.q,t),np.sum(.5*(self.p+self.q)*(eta-t)**2),12)
    def test_oracle(self):
        eta,s=eta_and_s(self.p,self.q)
        self.assertAlmostEqual(j_readout(self.p,self.q,eta),s,12)
    def test_diagonal_formula(self):
        for m in [2,4,8,16]:
            for r in [0,.2,.6,1]:
                p,q,s=diagonal_channel(m,r);self.assertAlmostEqual(eta_and_s(p,q)[1],s,12)
    def test_candidate_count(self):
        for m in [2,4,16]:
            p,q,s=diagonal_channel(m,1);self.assertAlmostEqual(s,(m-1)/(m+1),12)
    def test_zero_critic(self):
        self.assertEqual(j_readout(self.p,self.q,np.zeros_like(self.p)),0)
    def test_pushforward_collision(self):
        p,q,_=diagonal_channel(4,.6)
        pp,qq=pushforward(p,q,np.zeros_like(p))
        self.assertAlmostEqual(eta_and_s(pp,qq)[1],0)
    def test_injective_map(self):
        pp,qq=pushforward(self.p,self.q,np.arange(self.p.size).reshape(self.p.shape))
        self.assertAlmostEqual(eta_and_s(pp,qq)[1],eta_and_s(self.p,self.q)[1])
    def test_prior_correction(self):
        p=np.array([[.36,.44],[.14,.06]])
        q=np.outer(p.sum(1),p.sum(0));f=.5*np.log(p/q)
        sc=posterior_rank_score(f,p.sum(1))
        self.assertNotEqual(np.argmax(f[:,0]),np.argmax(p[:,0]))
        np.testing.assert_array_equal(sc.argmax(0),p.argmax(0))
    def test_monotone_scaling(self):
        rng=np.random.default_rng(3);s=rng.uniform(-1,1,(7,8))
        np.testing.assert_array_equal(s.argmax(0),(3*(s-.3)).argmax(0))
    def test_prior_logit_conversion(self):
        eta,_=eta_and_s(self.p,self.q);r=self.p/self.q
        valid=r>0
        for pi in [.5,1/9]:
            u=torch.from_numpy(np.log(r[valid])+np.log(pi/(1-pi)))
            np.testing.assert_allclose(probability_to_common_t(u,pi).numpy(),eta[valid],atol=1e-12)
    def test_vcs_equals_one_minus_j(self):
        f=torch.randn(*self.p.shape,dtype=torch.float64)
        val=image_losses(f,self.p,self.q,'vcs').item()
        self.assertAlmostEqual(val,1-j_readout(self.p,self.q,f.tanh().numpy()),12)
    def test_two_loss_gradient(self):
        f=torch.zeros(*self.p.shape,dtype=torch.float64,requires_grad=True)
        g1=torch.autograd.grad(image_losses(f,self.p,self.q,'vcs'),f)[0]
        g2=torch.autograd.grad(image_losses(f,self.p,self.q,'balanced_logistic'),f)[0]
        torch.testing.assert_close(g1,g2,atol=1e-12,rtol=1e-12)
    def test_gradcheck(self):
        f=torch.randn(*self.p.shape,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:image_losses(x,self.p,self.q,'vcs'),(f,)))
    def test_equal_image_weights(self):
        p2,q2,_=diagonal_channel(4,.8)
        fs=[torch.full(self.p.shape,.1,dtype=torch.float64),torch.full(p2.shape,.4,dtype=torch.float64)]
        direct=sum(image_losses(f,p,q,'vcs') for f,(p,q) in zip(fs,[(self.p,self.q),(p2,q2)]))/2
        torch.testing.assert_close(batch_loss(fs,[(self.p,self.q),(p2,q2)],'vcs'),direct)
    def test_softmax_multi_positive(self):
        f=torch.zeros(*self.p.shape,dtype=torch.float64)
        self.assertAlmostEqual(image_losses(f,self.p,self.q,'conditional_softmax').item(),np.log(2),12)
    def test_reject_empty_row(self):
        with self.assertRaises(ValueError):pair_laws(np.array([[1,0],[0,0]]))
    def test_reject_bad_prior(self):
        with self.assertRaises(ValueError):pair_laws(self.a,np.array([.8,.8]))
    def test_no_image_leakage(self):
        assert_image_disjoint({'fit':['x','x'],'eval':['y']})
        with self.assertRaises(ValueError):assert_image_disjoint({'fit':['x'],'eval':['x']})



class BatchedLossTests(unittest.TestCase):
    def _scenes(self):
        rng = np.random.default_rng(5); out = []
        for m, k in [(2, 3), (4, 5), (3, 7)]:
            a = np.zeros((m, k))
            for j in range(k):
                a[rng.integers(m), j] = 1
            for r in range(m):
                if a[r].sum() == 0:
                    a[r, rng.integers(k)] = 1
            for j in range(k):
                if a[:, j].sum() == 0:
                    a[rng.integers(m), j] = 1
            out.append(pair_laws(a))
        return out

    def test_padded_equals_reference(self):
        laws = self._scenes(); R = max(p.shape[0] for p, _ in laws); W = max(p.shape[1] for p, _ in laws)
        g = torch.Generator().manual_seed(0)
        fs = [torch.randn(*p.shape, generator=g, dtype=torch.float64) for p, _ in laws]
        F_ = torch.zeros(len(laws), R, W, dtype=torch.float64); P_ = torch.zeros_like(F_); Q_ = torch.zeros_like(F_)
        for i, (f, (p, q)) in enumerate(zip(fs, laws)):
            F_[i, :f.shape[0], :f.shape[1]] = f; P_[i, :p.shape[0], :p.shape[1]] = torch.from_numpy(p); Q_[i, :q.shape[0], :q.shape[1]] = torch.from_numpy(q)
        for obj in ("vcs", "balanced_logistic", "conditional_softmax"):
            Fv = F_.clone().requires_grad_(True); fl = [f.clone().requires_grad_(True) for f in fs]
            a = padded_batch_loss(Fv, P_, Q_, obj); b = batch_loss(fl, laws, obj)
            self.assertAlmostEqual(float(a.detach()), float(b.detach()), 12)
            ga = torch.autograd.grad(a, Fv)[0]; gb = torch.autograd.grad(b, fl)
            for i, g_ in enumerate(gb):
                torch.testing.assert_close(ga[i, :g_.shape[0], :g_.shape[1]], g_)
                pad = torch.ones_like(ga[i], dtype=torch.bool); pad[:g_.shape[0], :g_.shape[1]] = False
                self.assertTrue(bool((ga[i][pad] == 0).all()))   # padding entries receive exactly zero gradient
        Jp = padded_j(F_, P_, Q_)
        for i, (f, (p, q)) in enumerate(zip(fs, laws)):
            self.assertAlmostEqual(float(Jp[i]), j_readout(p, q, f.tanh().numpy()), 12)


class RefCocogAdapterTests(unittest.TestCase):
    def _scene(self, iid, split, n_ref, n_expr=2):
        refs = [RG.Obj(10 * iid + r, (1.0 * r, 2.0, 5.0, 6.0), 1, 30.0, 0, [{"sent_id": 100 * iid + 10 * r + e, "raw": f"o{r} e{e}", "n_tokens": 2}
                                                                       for e in range(n_expr)]) for r in range(n_ref)]
        return RG.Scene(iid, split, 50, 40, str(9000 + iid), refs, [RG.Obj(999, (0, 0, 3, 3), 2, 9.0, 0)])

    def test_compatibility_and_law(self):
        s = self._scene(1, "train", 3)
        A, anns, sents = s.compatibility()
        self.assertEqual(A.shape, (3, 6)); np.testing.assert_array_equal(A.sum(0), np.ones(6)); np.testing.assert_array_equal(A.sum(1), 2 * np.ones(3))
        p, q = pair_laws(A); np.testing.assert_allclose(p.sum(1), 1 / 3); self.assertGreater(q[A > 0].sum(), 0)

    def test_roles_disjoint_deterministic(self):
        sc = [self._scene(i, "train", 2) for i in range(200)] + [self._scene(1000 + i, "val", 2) for i in range(10)] + [self._scene(2000, "test", 2)]
        r1, r2 = RG.dev_roles(sc), RG.dev_roles(list(reversed(sc)))
        self.assertEqual(r1, r2)
        c = {k: sum(1 for v in r1.values() if v == k) for k in ("FIT", "CAL", "DEV", "VAL_OFFICIAL", "TEST_CLOSED")}
        self.assertEqual(c, {"FIT": 160, "CAL": 20, "DEV": 20, "VAL_OFFICIAL": 10, "TEST_CLOSED": 1})
        roles = {}
        for k, v in r1.items():
            roles.setdefault(v, []).append(str(k))
        assert_image_disjoint(roles)
        pre = RG.fit_prefixes(sc, r1, ns=(10, 50, 1000))
        self.assertEqual(pre[50][:10], pre[10]); self.assertEqual(len(pre[1000]), 160)

    def test_box_and_flickr(self):
        self.assertEqual(RG.clip_box((-5, 2, 20, 10), 10, 20), (0.0, 2.0, 10.0, 12.0))
        self.assertIsNone(RG.clip_box((9.5, 2, 3, 3), 10, 20))
        self.assertEqual(RG.flickr_id_of("http://farm9.staticflickr.com/8308/7908210548_33e532d119_z.jpg"), "7908210548")
        self.assertIsNone(RG.flickr_id_of(None))


if __name__ == "__main__":
    unittest.main()


class PriorCorrectionPaddedTests(unittest.TestCase):
    """VL1-11: with the exact half-log-ratio f = ½ log(p/q), ranking by 2 f + log p_G(r) recovers argmax_r p_G(r | w) (O1 §3.4); under the
    region-uniform law the correction is a constant per image, so corrected and raw Top-1 coincide."""

    def _laws(self, A, prior):
        p, q = pair_laws(A, prior); f = 0.5 * np.log(np.where(p > 0, p, 1e-300) / q); return p, q, f

    def test_phrase_uniform_prior_changes_ranking(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts")); import vl1_10_fit as V
        A = np.array([[1, 1, 1, 0], [0, 0, 0, 1.0]]); prior = A.sum(1) / A.sum()           # region 0 has 3 aliases, region 1 one
        p, q, f = self._laws(A, prior)
        self.assertNotAlmostEqual(prior[0], prior[1])
        corrected = (2 * f + np.log(prior)[:, None]).argmax(0); np.testing.assert_array_equal(corrected, p.argmax(0))
        rec = {"image_id": 1, "U": torch.zeros(2, 4), "Ud": torch.zeros(0, 4), "V": torch.zeros(4, 4), "A": torch.tensor(A, dtype=torch.float32),
               "P": torch.tensor(p, dtype=torch.float32), "Q": torch.tensor(q, dtype=torch.float32), "target": torch.tensor(A.argmax(0))}
        ft = torch.tensor(f, dtype=torch.float32); ft[~torch.isfinite(ft)] = -50.0
        out = V.evaluate(lambda U, Vv: ft[None], [rec], "vcs")
        self.assertEqual(out["top1_query_prior_corrected"], 1.0)
        # raw f ranks phrase 3 (column 3) correctly but the correction cannot hurt the exact posterior ranking
        self.assertLessEqual(out["top1_query"], out["top1_query_prior_corrected"])

    def test_region_uniform_correction_is_identity(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts")); import vl1_10_fit as V
        rng = np.random.default_rng(0); A = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]); p, q = pair_laws(A)
        f = torch.tensor(rng.normal(size=A.shape), dtype=torch.float32)
        rec = {"image_id": 2, "U": torch.zeros(3, 4), "Ud": torch.zeros(0, 4), "V": torch.zeros(4, 4), "A": torch.tensor(A, dtype=torch.float32),
               "P": torch.tensor(p, dtype=torch.float32), "Q": torch.tensor(q, dtype=torch.float32), "target": torch.tensor(A.argmax(0))}
        out = V.evaluate(lambda U, Vv: f[None], [rec], "vcs")
        self.assertEqual(out["top1_query"], out["top1_query_prior_corrected"])
