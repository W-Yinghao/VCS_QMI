import sys, unittest, math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from r6_reference import *

class TestDiscrepancy(unittest.TestCase):
    def setUp(self):
        self.p=np.array([.1,.2,.3,.4]);self.q=np.array([.4,.3,.2,.1]);self.t=np.array([-.3,-.2,.1,.2])
    def test_gap(self):
        e,s=eta_s(self.p,self.q)
        self.assertAlmostEqual(s-j_value(self.p,self.q,self.t),float(np.sum((self.p+self.q)/2*(self.t-e)**2)))
    def test_rpc_equal(self):self.assertAlmostEqual(j_value(self.p,self.q,self.t),rpc_value(self.p,self.q,self.t))
    def test_rpc_non_equal(self):self.assertNotAlmostEqual(j_value(self.p,self.q,self.t),rpc_value(self.p,self.q,self.t,gamma=.5))
    def test_optimum(self):
        e,s=eta_s(self.p,self.q);self.assertAlmostEqual(j_value(self.p,self.q,e),s)
    def test_equal_distributions(self):self.assertEqual(eta_s(self.p,self.p)[1],0.)
    def test_disjoint(self):self.assertEqual(eta_s([1,0],[0,1])[1],1.)
    def test_cs_zero(self):self.assertEqual(cs_transform(0),0.)
    def test_cs_boundary(self):self.assertTrue(math.isinf(cs_transform(1)))
    def test_cs_negative_rejected(self):
        with self.assertRaises(ValueError):cs_transform(-.1)
    def test_bad_mass(self):
        with self.assertRaises(ValueError):law([1,1])
    def test_unbounded_t(self):
        with self.assertRaises(ValueError):j_value(self.p,self.q,np.ones(4)*1.01)

class TestEvidence(unittest.TestCase):
    def test_redundant_zero(self):self.assertEqual(evidence_summary(make_task('redundant'))['delta'],0)
    def test_irrelevant_zero(self):self.assertEqual(evidence_summary(make_task('irrelevant'))['delta'],0)
    def test_xor_one(self):self.assertEqual(evidence_summary(make_task('xor'))['delta'],1)
    def test_context_half(self):self.assertEqual(evidence_summary(make_task('context'))['delta'],.5)
    def test_context_local(self):
        e=evidence_summary(make_task('context'))
        for (k,x),v in e['conditional_gain'].items():self.assertEqual(k,v)
    def test_risk_equals_delta(self):
        for n in ('redundant','irrelevant','xor','context'):
            e=evidence_summary(make_task(n));self.assertAlmostEqual(e['risk_cheap']-e['risk_full'],e['delta'])
    def test_s_equals_delta(self):
        for n in ('redundant','irrelevant','xor','context'):
            e=evidence_summary(make_task(n));self.assertAlmostEqual(e['S_full']-e['S_cheap'],e['delta'])
    def test_nonbalanced_rejected(self):
        with self.assertRaises(ValueError):evidence_summary([State(1.,'u','v',1)])
    def test_bad_label_rejected(self):
        with self.assertRaises(ValueError):evidence_summary([State(1.,'u','v',0)])
    def test_finite_residual(self):
        rng=np.random.default_rng(42);c=rng.choice([-1,1],100);a=rng.uniform(-1,1,100);b=rng.uniform(-1,1,100)
        z,r,h=risk_components(c,a,b);np.testing.assert_allclose(h,z+r,atol=1e-14)
    def test_false_positive_gain(self):
        e=finite_model_utility(make_task('redundant'),{-1:-1,1:1},{(-1,-1):0,(1,1):0})
        self.assertEqual(e['square'],1);self.assertEqual(e['actual_gain'],-1);self.assertEqual(e['residual'],-2)
    def test_oracle_policy(self):
        e=oracle_policy_risk(make_task('context'),.3);self.assertEqual(e['task_risk'],0);self.assertAlmostEqual(e['total'],.15)
    def test_oracle_no_value(self):self.assertEqual(oracle_policy_risk(make_task('redundant'),.3)['acquisition_cost'],0)
    def test_xor_constant_value(self):
        e=evidence_summary(make_task('xor'));self.assertTrue(all(v==1 for v in e['conditional_gain'].values()))

class TestRegistration(unittest.TestCase):
    def test_sensor_truth(self):
        for rho in (0,.1,.6,1,-.6):
            p,q,s=binary_sensor_law(rho);self.assertAlmostEqual(eta_s(p,q)[1],s)
    def test_sensor_zero(self):self.assertEqual(binary_sensor_law(0)[2],0)
    def test_perfect_binary_not_one(self):self.assertAlmostEqual(binary_sensor_law(1)[2],1/3)
    def test_shift_sign(self):self.assertEqual(shift_demo()['recovered_shift'],3)
    def test_shift_zero(self):self.assertEqual(shift_demo(true_shift=0)['recovered_shift'],0)
    def test_shift_negative(self):self.assertEqual(shift_demo(true_shift=-2)['recovered_shift'],-2)
    def test_rho_reject(self):
        with self.assertRaises(ValueError):binary_sensor_law(2.)

class TestDeployment(unittest.TestCase):
    def test_policy_allowed(self):self.assertTrue(validate_policy_inputs(['cheap_features','cheap_confidence']))
    def test_policy_no_v(self):
        with self.assertRaises(ValueError):validate_policy_inputs(['cheap_input','expensive_observation'])
    def test_policy_no_logits(self):
        with self.assertRaises(ValueError):validate_policy_inputs(['full_predictor_logits'])
    def test_cost(self):self.assertEqual(total_cost(1,.1,4,[0,1]),3.1)
    def test_cost_reject(self):
        with self.assertRaises(ValueError):total_cost(1,0,4,[2])
    def test_groups(self):self.assertTrue(assert_group_disjoint({'a':['x','x'],'b':['y']}))
    def test_group_leak(self):
        with self.assertRaises(ValueError):assert_group_disjoint({'a':['x'],'b':['x']})
    def test_hash_split_stable(self):
        a=stable_group_roles(['a','b','c','a'],'s');b=stable_group_roles(['c','b','a'],'s');self.assertEqual(a,b)
    def test_hash_split_no_leak(self):self.assertTrue(assert_group_disjoint(stable_group_roles([str(i) for i in range(200)],'s')))

if __name__=='__main__':unittest.main()
