import unittest,sys,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
import vcs_vl_reference as v

class FiniteDefinitions(unittest.TestCase):
    def setUp(self):
        self.p=np.array([.5,.3,.2]);self.q=np.array([.1,.5,.4]);self.t=np.array([.2,-.4,.1])
    def test_01_optimum(self):
        t,s=v.eta_s(self.p,self.q);self.assertAlmostEqual(v.j(self.p,self.q,t),s)
    def test_02_gap(self):
        t,s=v.eta_s(self.p,self.q);self.assertAlmostEqual(s-v.j(self.p,self.q,self.t),np.sum(.5*(self.p+self.q)*(self.t-t)**2))
    def test_03_risk(self):self.assertAlmostEqual(v.j(self.p,self.q,self.t),1-v.common_risk(self.p,self.q,self.t))
    def test_04_identity(self):self.assertEqual(v.eta_s(self.p,self.p)[1],0)
    def test_05_disjoint(self):self.assertEqual(v.eta_s(np.array([1.,0]),np.array([0.,1]))[1],1)
    def test_06_symmetry(self):self.assertAlmostEqual(v.eta_s(self.p,self.q)[1],v.eta_s(self.q,self.p)[1])
    def test_07_measure_permutation(self):self.assertAlmostEqual(v.eta_s(self.p,self.q)[1],v.eta_s(self.p[::-1],self.q[::-1])[1])
    def test_08_pushforward(self):
        p,q=v.pushforward(self.p,self.q,np.array([0,0,1]));self.assertLessEqual(v.eta_s(p,q)[1],v.eta_s(self.p,self.q)[1]+1e-12)
    def test_09_zero_mass(self):
        t,s=v.eta_s(np.array([1.,0]),np.array([1.,0]));self.assertTrue(np.isfinite(t).all());self.assertEqual(s,0)
    def test_10_invalid_law(self):
        with self.assertRaises(ValueError):v.law([1,1])
    def test_11_invalid_critic(self):
        with self.assertRaises(ValueError):v.j(self.p,self.q,[0,0,2])
    def test_12_cs(self):self.assertAlmostEqual(v.cs_transform(1/3),2*math.log(2))
    def test_13_cs_endpoint(self):self.assertTrue(math.isinf(v.cs_transform(1)))
    def test_14_no_empirical_clipping(self):
        self.assertLess(v.j(self.p,self.q,-self.t),v.eta_s(self.p,self.q)[1])
        with self.assertRaises(ValueError):v.cs_transform(-.01)
    def test_15_conditional_average(self):
        p=np.array([[.7,.3],[.1,.9]]);q=np.array([[.2,.8],[.6,.4]]);w=np.array([.3,.7]);s,parts=v.conditional_s(p,q,w);self.assertAlmostEqual(s,w@parts)
    def test_16_transform_not_average(self):
        p=np.array([[.5,.5],[.99,.01]]);q=np.array([[.5,.5],[.01,.99]]);s,parts=v.conditional_s(p,q,[.5,.5]);self.assertNotAlmostEqual(v.cs_transform(s),.5*sum(v.cs_transform(x) for x in parts))

class RiskAndPosterior(unittest.TestCase):
    def test_17_redundant(self):
        c=np.array([-1,-1,1,1]);w=np.ones(4)/4; a=v.task_posterior(c,w,np.array([0,0,1,1]));b=v.task_posterior(c,w,np.arange(4));self.assertEqual(np.mean((b-a)**2),0)
    def test_18_complement(self):
        c=np.array([1,-1,-1,1]);w=np.ones(4)/4; a=v.task_posterior(c,w,np.array([0,0,1,1]));b=v.task_posterior(c,w,np.arange(4));self.assertEqual(np.mean((b-a)**2),1)
    def test_19_bayes_risk_identity(self):
        c=np.array([1,-1,-1,1]);a=np.zeros(4);b=c.copy();h,d,r=v.finite_risk_delta(c,a,b);np.testing.assert_allclose(h,d);np.testing.assert_allclose(r,0)
    def test_20_finite_residual(self):
        c=np.array([-1,1]);a=np.array([0,.2]);b=np.array([.5,.8]);h,d,r=v.finite_risk_delta(c,a,b);np.testing.assert_allclose(h,d+r)
    def test_21_disagreement_not_gain(self):
        h,d,_=v.finite_risk_delta(np.array([1.]),np.array([.9]),np.array([-.9]));self.assertLess(h[0],0);self.assertGreater(d[0],0)
    def test_22_shared_logistic_posterior(self):
        p=np.array([.3,.7]);q=np.array([.8,.2]);t,_=v.eta_s(p,q);np.testing.assert_allclose(2*p/(p+q)-1,t)
    def test_23_prior_conversion(self):np.testing.assert_allclose(v.probability_at_prior(np.zeros(3),.1),.1)
    def test_24_prior_endpoints(self):np.testing.assert_allclose(v.probability_at_prior(np.array([-1.,1.]),.3),[0.,1.])
    def test_25_brier_scale(self):
        y=np.array([0.,1.,1.]);d=np.array([.2,.7,.4]);self.assertAlmostEqual(np.mean(((2*y-1)-(2*d-1))**2),4*v.binary_brier(y,d))
    def test_26_stable_sigmoid(self):np.testing.assert_allclose(v.sigmoid(np.array([-1000,0,1000.])),[0,.5,1])
    def test_27_set_distribution(self):
        f=np.array([.5,-.8,1.2,2.]);mask=np.array([1,0,1,0],bool);d,s,k=v.positive_loss_decomposition(f,mask);self.assertAlmostEqual(d,s+math.log(2)+k)
    def test_28_single_positive(self):
        d,s,k=v.positive_loss_decomposition(np.array([1.,2.,3.]),np.array([0,1,0]));self.assertAlmostEqual(d,s);self.assertAlmostEqual(k,0)
    def test_29_budget(self):self.assertEqual(v.earliest_budget([.5,.1,.01,0],[1,2,3,4],.11),1)
    def test_30_no_feasible(self):
        with self.assertRaises(ValueError):v.earliest_budget([.5,.1],[1,2],.01)

class MetricAndGroupingFixtures(unittest.TestCase):
    def test_31_tiou(self):self.assertAlmostEqual(v.time_iou([0,2],[1,3]),1/3)
    def test_32_tiou_zero(self):self.assertEqual(v.time_iou([0,1],[2,3]),0)
    def test_33_tiou_invalid(self):
        with self.assertRaises(ValueError):v.time_iou([1,1],[1,2])
    def test_34_mapr(self):self.assertAlmostEqual(v.fixture_ap_at_r(['a','x','b'],{'a','b'}),.5)
    def test_35_mapr_duplicates(self):
        with self.assertRaises(ValueError):v.fixture_ap_at_r(['a','a'],{'a'})
    def test_36_riskcoverage(self):
        c,r=v.risk_coverage(np.array([0,1,1]),np.array([.9,.8,.7]));np.testing.assert_allclose(c,[1/3,2/3,1]);np.testing.assert_allclose(r,[0,.5,2/3])
    def test_37_group_splits(self):v.assert_group_disjoint({'FIT':['a','a'],'CAL':['b'],'TEST':['c']})
    def test_38_detect_group_leak(self):
        with self.assertRaises(ValueError):v.assert_group_disjoint({'FIT':['a'],'TEST':['a']})
    def test_39_ess_uniform(self):self.assertEqual(v.effective_sample_size(np.ones(7)),7)
    def test_40_ess_concentration(self):self.assertAlmostEqual(v.effective_sample_size(np.array([1,0,0])),1)
    def test_41_support_failure(self):
        with self.assertRaises(ValueError):v.effective_sample_size(np.zeros(4))

if __name__=='__main__':unittest.main(verbosity=2)
