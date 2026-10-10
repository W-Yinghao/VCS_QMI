from __future__ import annotations
import itertools, json, math, sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
import vcs_r4_reference as v

class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.p=np.array([.45,.25,.20,.10]); self.q=np.array([.1,.2,.3,.4]); self.t=np.array([.4,-.1,.5,-.7])
    def test_01_probability_rejects_bad_mass(self):
        with self.assertRaises(ValueError): v.probability([.2,.2])
    def test_02_probability_rejects_negative(self):
        with self.assertRaises(ValueError): v.probability([-.2,1.2])
    def test_03_gap_identity(self):
        o=v.oracle(self.p,self.q)
        self.assertAlmostEqual(o['S']-v.common_j(self.t,self.p,self.q),float(np.sum(o['M']*(self.t-o['eta'])**2)),places=13)
    def test_04_risk_equals_one_minus_j(self):
        risk=.5*np.sum(self.p*(1-self.t)**2+self.q*(1+self.t)**2)
        self.assertAlmostEqual(risk,1-v.common_j(self.t,self.p,self.q),places=13)
    def test_05_rpc_balanced_identity(self):
        self.assertAlmostEqual(v.rpc(self.t,self.p,self.q),v.common_j(self.t,self.p,self.q),places=14)
    def test_06_general_rpc_gap(self):
        a,b,g=.3,.8,1.7; star=v.rpc_oracle(self.p,self.q,a,b,g)
        gap=v.rpc(star,self.p,self.q,a,b,g)-v.rpc(self.t,self.p,self.q,a,b,g)
        self.assertAlmostEqual(gap,float(.5*np.sum((b*self.p+g*self.q)*(self.t-star)**2)),places=13)
    def test_07_loss_gradients(self):
        for f in [-4.,-.5,.1,3.]:
            for c in [-1,1]:
                eps=1e-6;t=np.tanh(f)
                l1=lambda x:.5*(np.tanh(x)-c)**2
                l2=lambda x:np.logaddexp(0.,-2*c*x)
                self.assertAlmostEqual((l1(f+eps)-l1(f-eps))/(2*eps),(t-c)*(1-t*t),places=8)
                self.assertAlmostEqual((l2(f+eps)-l2(f-eps))/(2*eps),t-c,places=8)
    def test_08_tanh_sigmoid_coordinate(self):
        f=np.linspace(-8,8,33)
        np.testing.assert_allclose(np.tanh(f),2/(1+np.exp(-2*f))-1,atol=1e-14)
    def test_09_cs_transform(self):
        o=v.oracle(self.p,self.q); pm=self.p/o['M'];qm=self.q/o['M'];m=o['M']
        direct=-np.log(np.sum(m*pm*qm)**2/(np.sum(m*pm**2)*np.sum(m*qm**2)))
        self.assertAlmostEqual(direct,v.mixture_cs(o['S']),places=13)
    def test_10_cs_does_not_silently_clip(self):
        for x in [-.01,1,1.01]:
            with self.assertRaises(ValueError):v.mixture_cs(x)
    def test_11_contamination_posterior(self):
        eta=v.oracle(self.p,self.q)['eta']
        for r in [0,.1,.4,.8]:
            ep=v.oracle((1-r)*self.p+r*self.q,self.q)['eta']
            np.testing.assert_allclose(ep,v.contaminate_eta(eta,r),atol=1e-14)
    def test_12_contamination_inverse(self):
        e=np.linspace(-1,1,71)
        for r in [0,.1,.5,.9]:
            np.testing.assert_allclose(v.recover_eta(v.contaminate_eta(e,r),r),e,atol=1e-13)
    def test_13_inverse_derivative_max(self):
        r=.5;u=v.contaminate_eta(np.linspace(-1,1,10001),r)
        d=(1-r)/(1-r+r*u)**2
        self.assertAlmostEqual(d.max(),(1+r)**2/(1-r),places=12)
    def test_14_contamination_rpc_correction(self):
        for r in [0,.1,.4]:
            pr=(1-r)*self.p+r*self.q
            self.assertAlmostEqual((1-r)*v.common_j(self.t,self.p,self.q),v.rpc(self.t,pr,self.q,1,1,1-2*r),places=13)
    def test_15_cross_u_equals_direct(self):
        t=np.tanh(np.random.default_rng(0).normal(size=(8,8)))
        self.assertAlmostEqual(v.u_mean(v.cross_pair_kernel(t)),v.direct_cross_j(t),places=13)
    def test_16_exact_variance_by_enumeration(self):
        p=np.array([.2,.3,.5]);k=np.array([[.1,.5,-.2],[.5,.8,-.5],[-.2,-.5,.2]])
        n=4;us=[];ws=[]
        for idx in itertools.product(range(3),repeat=n):
            us.append(v.u_mean(k[np.ix_(idx,idx)]));ws.append(np.prod(p[list(idx)]))
        us,ws=np.array(us),np.array(ws);mean=np.sum(us*ws);var=np.sum(ws*(us-mean)**2)
        theory=v.finite_u_variance(k,p,n)
        self.assertAlmostEqual(mean,theory['mean'],places=13)
        self.assertAlmostEqual(var,theory['variance'],places=13)
    def test_17_degenerate_variance(self):
        k=np.array([[1.,-1.],[-1.,1.]])
        out=v.finite_u_variance(k,[.5,.5],10)
        self.assertAlmostEqual(out['first_projection_variance'],0.)
        self.assertAlmostEqual(out['variance'],2/(10*9))
    def test_18_delete_one_formula(self):
        t=np.tanh(np.random.default_rng(4).normal(size=(9,9)));k=v.cross_pair_kernel(t)
        out=v.delete_one_variance(k)
        explicit=np.array([v.u_mean(np.delete(np.delete(k,i,axis=0),i,axis=1)) for i in range(len(k))])
        np.testing.assert_allclose(out['leave_one_out'],explicit,atol=1e-13)
    def test_19_jackknife_rejects_n2(self):
        with self.assertRaises(ValueError):v.delete_one_variance(np.eye(2))
    def test_20_radius_and_lcb(self):
        self.assertAlmostEqual(v.bounded_radius(100,.05,'cross_pair_mcd'),2*v.bounded_radius(100,.05,'image_local'))
        self.assertEqual(v.oracle_lcb(-.1,.2),0.)
    def test_21_shift_variance_enumeration(self):
        arr=np.array([1.,.2,-.7,2.,-.4]);k=3
        draws=np.array([arr[list(s)].mean() for s in itertools.combinations(range(5),k)])
        self.assertAlmostEqual(draws.var(),v.shift_sampling_variance(arr,k),places=14)
        self.assertAlmostEqual(v.shift_sampling_variance(arr,5),0.)
    def test_22_group_macro_not_pair_micro(self):
        self.assertAlmostEqual(v.group_macro([1,1,1,0],['a','a','a','b']),.5)
    def test_23_group_bootstrap_reproducible(self):
        a=v.group_bootstrap([1,0,1],['a','a','b'],20,4);b=v.group_bootstrap([1,0,1],['a','a','b'],20,4)
        np.testing.assert_array_equal(a,b)
    def test_24_proposal_mass_and_overlap(self):
        a=np.array([[1,0],[0,1],[0,0]])
        p,q=v.proposal_laws(a)
        self.assertAlmostEqual(p.sum(),1.)
        self.assertAlmostEqual(q.sum(),1.)
        self.assertTrue(np.all(q[a>0]>0))
        self.assertEqual(p[2].sum(),0)
        self.assertGreater(q[2].sum(),0)
    def test_25_proposal_q_not_automatically_product(self):
        p,q=v.proposal_laws([[1,0],[0,1],[0,0]])
        prod=p.sum(1)[:,None]*p.sum(0)[None,:]
        self.assertFalse(np.allclose(q,prod))
    def test_26_invalid_closed_query(self):
        with self.assertRaises(ValueError):v.proposal_laws([[1,0],[0,0]])
    def test_27_ranking_prior_correction(self):
        cond=np.array([.4,.6]);nu=np.array([.1,.9]);f=.5*np.log(cond/nu)
        self.assertEqual(int(f.argmax()),0)
        self.assertEqual(int(v.corrected_rank(f,nu).argmax()),1)
    def test_28_uniform_prior_ranking(self):
        cond=np.array([.2,.5,.3]);nu=np.ones(3)/3
        self.assertEqual(int(v.corrected_rank(.5*np.log(cond/nu),nu).argmax()),int(cond.argmax()))
    def test_29_misleading_native_readout(self):
        row={'native':{'value':1.1},'posterior':{'J_eval':.7,'R_eval':.3,'posterior_mse':.1}}
        self.assertEqual(v.read_common_fields(row)['common_J'],.7)
    def test_30_missing_common_readout_not_native(self):
        row={'native':{'value':1.1},'posterior':{}}
        self.assertIsNone(v.read_common_fields(row)['common_J'])
    def test_31_common_risk_consistency(self):
        with self.assertRaises(ValueError):v.read_common_fields({'posterior':{'J_eval':.7,'R_eval':.5}})
    def test_32_paired_error_changes(self):
        o=v.matched_error_changes([.1,.2],[.2,.3],[.05,.1],[.25,.3])
        np.testing.assert_allclose(o['degradation_difference'],[-.1,-.1],atol=1e-15)
    def test_33_gradient_rpc_identity_torch(self):
        import torch
        z=torch.tensor([.2,-.6,.9,.1],dtype=torch.float64,requires_grad=True)
        p=torch.tensor(self.p,dtype=torch.float64);q=torch.tensor(self.q,dtype=torch.float64);t=z.tanh()
        j=(p*(t-.5*t*t)+q*(-t-.5*t*t)).sum()
        f=(p*t-q*t-.5*p*t*t-.5*q*t*t).sum()
        g1=torch.autograd.grad(j,z,retain_graph=True)[0];g2=torch.autograd.grad(f,z)[0]
        torch.testing.assert_close(g1,g2,atol=1e-13,rtol=1e-13)
    def test_34_task_graph_is_acyclic(self):
        path=Path(__file__).resolve().parents[1]/'machine/tasks.json';d=json.loads(path.read_text());tasks=d['tasks'];seen=set()
        for t in tasks:
            self.assertTrue(set(t['depends_on'])<=seen);seen.add(t['id'])
        self.assertFalse(d['auto_launch'])
    def test_35_posteriors_can_be_equal_without_label_recovery(self):
        # Independence gives zero dependence even after every label is erased.
        p=np.ones((2,2))/4;q=p.copy()
        self.assertAlmostEqual(v.oracle(p,q)['S'],v.oracle([1.],[1.])['S'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
