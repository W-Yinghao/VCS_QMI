from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from r5_reference import (OBJECTIVES, make_laws, per_image_loss, batch_loss, j_per_image,
    analytic_f_gradient, uniform_set_decomposition, corrected_scores, query_attention,
    nested_invalid_ids, choose_cal_lambda, image_macro_hit, single_positive_counterexample)

torch.set_num_threads(1)
DT=torch.float64


def fixture():
    # Three real candidates in image 0; four in image 1. Third query is uncovered.
    a=torch.tensor([[[1,0,0],[0,1,0],[0,1,0],[0,0,0]],
                    [[1,0,0],[1,0,0],[0,1,0],[0,0,0]]], dtype=DT)
    rm=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
    p,q=make_laws(a,rm)
    gen=torch.Generator().manual_seed(121)
    f=torch.randn(a.shape,generator=gen,dtype=DT)*.7
    return f,p,q,rm


class TestLaws(unittest.TestCase):
    def test_unit_mass_and_query_marginal(self):
        _,p,q,_=fixture()
        self.assertTrue(torch.allclose(p.sum((1,2)),torch.ones(2,dtype=DT)))
        self.assertTrue(torch.allclose(q.sum((1,2)),torch.ones(2,dtype=DT)))
        self.assertTrue(torch.allclose(p.sum(1),q.sum(1)))

    def test_never_positive_region_retained(self):
        _,p,q,rm=fixture()
        self.assertEqual(float(p[1,3,:].sum()),0)
        self.assertGreater(float(q[1,3,:].sum()),0)
        self.assertTrue(bool(rm[1,3]))

    def test_q_includes_valid_candidates(self):
        _,p,q,_=fixture()
        self.assertTrue((q[p>0]>0).all())

    def test_zero_query_mass(self):
        _,p,q,_=fixture()
        self.assertEqual(float(p[:,:,2].sum()+q[:,:,2].sum()),0)

    def test_score_nu_correct(self):
        _,p,_,rm=fixture(); a=(p>0).to(DT)
        w=torch.tensor([[1.,2.,3.,99.],[4.,3.,2.,1.]],dtype=DT)
        _,q=make_laws(a,rm,w)
        self.assertAlmostEqual(float(q[0,1,:].sum()),2/6)
        self.assertEqual(float(q[0,3,:].sum()),0)

    def test_zero_reference_weight_rejected(self):
        _,p,_,rm=fixture(); w=rm.to(DT);w[0,0]=0
        with self.assertRaises(ValueError):make_laws((p>0).to(DT),rm,w)

    def test_all_uncovered_image_rejected(self):
        with self.assertRaises(ValueError):
            make_laws(torch.zeros((1,2,3),dtype=DT),torch.ones((1,2),dtype=torch.bool))

    def test_no_proposal_image_rejected(self):
        with self.assertRaises(ValueError):
            make_laws(torch.zeros((1,2,1),dtype=DT),torch.zeros((1,2),dtype=torch.bool))

    def test_nonbinary_A_rejected(self):
        with self.assertRaises(ValueError):
            make_laws(torch.ones((1,2,1),dtype=DT)*.5,torch.ones((1,2),dtype=torch.bool))


class TestLosses(unittest.TestCase):
    def test_vcs_one_minus_j(self):
        f,p,q,rm=fixture()
        self.assertTrue(torch.allclose(per_image_loss(f,p,q,rm,'vcs'),1-j_per_image(f,p,q,rm),atol=1e-12))

    def test_all_analytical_gradients(self):
        for obj in OBJECTIVES:
            with self.subTest(objective=obj):
                f,p,q,rm=fixture();f.requires_grad_(True)
                grad=torch.autograd.grad(per_image_loss(f,p,q,rm,obj).sum(),f)[0]
                expected=analytic_f_gradient(f.detach(),p,q,rm,obj)
                self.assertTrue(torch.allclose(grad,expected,atol=1e-11,rtol=1e-11),obj)

    def test_mean_gradient_scaling(self):
        f,p,q,rm=fixture();f.requires_grad_(True)
        g=torch.autograd.grad(batch_loss(f,p,q,rm,'vcs'),f)[0]
        self.assertTrue(torch.allclose(g,analytic_f_gradient(f,p,q,rm,'vcs')/2,atol=1e-12))

    def test_single_positive_equal_losses_and_gradients(self):
        a=torch.tensor([[[1,0],[0,1],[0,0]]],dtype=DT); rm=torch.ones((1,3),dtype=torch.bool)
        p,q=make_laws(a,rm);f=torch.tensor([[[.2,.8],[-.4,.1],[1.,-1.]]],dtype=DT)
        self.assertTrue(torch.allclose(batch_loss(f,p,q,rm,'distribution_softmax'),batch_loss(f,p,q,rm,'set_softmax'),atol=1e-12))
        self.assertTrue(torch.allclose(analytic_f_gradient(f,p,q,rm,'distribution_softmax'),analytic_f_gradient(f,p,q,rm,'set_softmax'),atol=1e-12))

    def test_multiple_positive_decomposition(self):
        f,p,q,rm=fixture();d=uniform_set_decomposition(f,p,q,rm)
        left=per_image_loss(f,p,q,rm,'distribution_softmax')-per_image_loss(f,p,q,rm,'set_softmax')
        self.assertTrue(torch.allclose(left,d['per_image_extra'],atol=1e-12))
        self.assertTrue((d['kl_uniform_to_legal']>=-1e-12).all())

    def test_k_one_kl_zero(self):
        f,p,q,rm=fixture();d=uniform_set_decomposition(f,p,q,rm)
        self.assertTrue(torch.allclose(d['kl_uniform_to_legal'][d['k']==1],torch.zeros_like(d['kl_uniform_to_legal'][d['k']==1]),atol=1e-12))

    def test_set_allows_concentration(self):
        a=torch.tensor([[[1],[1],[0]]],dtype=DT);rm=torch.ones((1,3),dtype=torch.bool);p,q=make_laws(a,rm)
        f=torch.tensor([[[5.],[0.],[-5.]]],dtype=DT)
        self.assertLess(float(batch_loss(f,p,q,rm,'set_softmax')),1e-6)
        self.assertGreater(float(batch_loss(f,p,q,rm,'distribution_softmax')),4.0)

    def test_extreme_logits_finite(self):
        f,p,q,rm=fixture(); f=torch.where(f>0,100.*torch.ones_like(f),-100.*torch.ones_like(f))
        for obj in OBJECTIVES:
            with self.subTest(obj=obj):
                x=f.clone().requires_grad_(True);loss=batch_loss(x,p,q,rm,obj);g=torch.autograd.grad(loss,x)[0]
                self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(g).all())
                self.assertTrue(torch.isfinite(analytic_f_gradient(x,p,q,rm,obj)).all())

    def test_padding_and_empty_query_unchanged(self):
        f,p,q,rm=fixture(); altered=f.clone(); altered[0,3,:]=float('nan'); altered[:,:,2]=float('nan')
        for obj in OBJECTIVES:
            self.assertTrue(torch.allclose(batch_loss(f,p,q,rm,obj),batch_loss(altered,p,q,rm,obj),atol=1e-12),obj)

    def test_padding_gradient_zero(self):
        for obj in OBJECTIVES:
            f,p,q,rm=fixture();f.requires_grad_(True);g=torch.autograd.grad(batch_loss(f,p,q,rm,obj),f)[0]
            self.assertEqual(float(g[0,3,:].abs().sum()+g[:,:,2].abs().sum()),0)

    def test_never_positive_logit_in_denominator(self):
        f,p,q,rm=fixture();g=f.clone();g[1,3,:2]+=20
        for obj in ('distribution_softmax','set_softmax'):
            self.assertGreater(float(per_image_loss(g,p,q,rm,obj)[1]),float(per_image_loss(f,p,q,rm,obj)[1])+1.)

    def test_shift_invariance_softmax(self):
        f,p,q,rm=fixture()
        shift=torch.tensor([[[1.,2.,3.]],[[4.,5.,6.]]],dtype=DT)
        for obj in ('distribution_softmax','set_softmax'):
            self.assertTrue(torch.allclose(batch_loss(f,p,q,rm,obj),batch_loss(f+shift,p,q,rm,obj),atol=1e-12))

    def test_image_average_not_query_pool(self):
        f,p,q,rm=fixture()
        for obj in OBJECTIVES:
            separate=sum(float(batch_loss(f[i:i+1],p[i:i+1],q[i:i+1],rm[i:i+1],obj)) for i in range(2))/2
            self.assertAlmostEqual(float(batch_loss(f,p,q,rm,obj)),separate,12)

    def test_exact_chunk_gradient(self):
        f,p,q,rm=fixture()
        for obj in OBJECTIVES:
            x=f.clone().requires_grad_(True); g1=torch.autograd.grad(batch_loss(x,p,q,rm,obj),x)[0]
            y=f.clone().requires_grad_(True)
            loss=sum(per_image_loss(y[i:i+1],p[i:i+1],q[i:i+1],rm[i:i+1],obj).sum()/2 for i in range(2))
            g2=torch.autograd.grad(loss,y)[0]
            self.assertTrue(torch.allclose(g1,g2,atol=1e-12))

    def test_nonuniform_positive_decomposition_rejected(self):
        f,p,q,rm=fixture();p=p.clone();p[0,1,1]=.1;p[0,2,1]=.4
        with self.assertRaises(ValueError):uniform_set_decomposition(f,p,q,rm)

    def test_invalid_active_logit_rejected(self):
        f,p,q,rm=fixture();f[0,0,0]=float('nan')
        with self.assertRaises(ValueError):batch_loss(f,p,q,rm,'vcs')

    def test_unknown_loss_rejected(self):
        f,p,q,rm=fixture()
        with self.assertRaises(ValueError):batch_loss(f,p,q,rm,'wrong')

    def test_bad_law_normalization_rejected(self):
        f,p,q,rm=fixture()
        with self.assertRaises(ValueError):batch_loss(f,p*2,q,rm,'set_softmax')


class TestDiagnostics(unittest.TestCase):
    def test_reference_correction_recovers_p(self):
        p=torch.tensor([.1,.25,.65],dtype=DT);nu=torch.tensor([.7,.2,.1],dtype=DT)
        f=(.5*(p/nu).log())[None,:,None]
        score=corrected_scores(f,nu[None,:],torch.ones((1,3),dtype=torch.bool))[0,:,0]
        self.assertTrue(torch.allclose(torch.softmax(score,0),p,atol=1e-12))

    def test_reference_weight_scale_invariant(self):
        _,p,_,rm=fixture();f=torch.randn(p.shape,dtype=DT);w=rm.to(DT)
        self.assertTrue(torch.equal(corrected_scores(f,w,rm),corrected_scores(f,10*w,rm)))

    def test_corrected_padding(self):
        f,p,q,rm=fixture();v=rm.to(DT);f[0,3,:]=1000
        sc=corrected_scores(f,v,rm)
        self.assertTrue(torch.isneginf(sc[0,3]).all())

    def test_attention_fraction_and_gain(self):
        f=torch.tensor([1.,2.,-1.],dtype=DT);p=torch.tensor([1.,0.,0.],dtype=DT);q=torch.ones(3,dtype=DT)/3
        d=query_attention(f,p,q,f,'balanced_logistic',1,alpha=3)
        self.assertEqual(d['selected_count'],1);self.assertAlmostEqual(d['q_fraction_invalid'],.5)
        self.assertAlmostEqual(d['abs_cosine_gradient'],3*d['abs_f_gradient'])
        self.assertGreater(d['gradient_fraction_invalid'],.5)

    def test_all_saturated_invalid_no_divide_zero(self):
        f=torch.tensor([100.,-100.,-100.],dtype=DT);p=torch.tensor([1.,0.,0.],dtype=DT);q=torch.ones(3,dtype=DT)/3
        d=query_attention(f,p,q,f,'vcs',5)
        self.assertIsNone(d['gradient_fraction_invalid'])

    def test_no_invalid_attention(self):
        f=torch.tensor([.1,.2],dtype=DT);p=torch.ones(2,dtype=DT)/2;q=p.clone()
        d=query_attention(f,p,q,f,'vcs',1)
        self.assertEqual(d['invalid_count'],0);self.assertIsNone(d['gradient_fraction_invalid'])

    def test_nested_reproducible_and_nested(self):
        ids=[str(i) for i in range(40)];a=nested_invalid_ids('qA',ids,0);b=nested_invalid_ids('qA',list(reversed(ids)),0)
        self.assertEqual(a,b)
        self.assertEqual(a['3'][:1],a['1']);self.assertEqual(a['31'][:15],a['15'])
        self.assertEqual(set(a['all']),set(ids))

    def test_nested_different_repeat(self):
        ids=[str(i) for i in range(40)]
        self.assertNotEqual(nested_invalid_ids('q',ids,0)['31'],nested_invalid_ids('q',ids,1)['31'])

    def test_nested_strict_support(self):
        with self.assertRaises(ValueError):nested_invalid_ids('q',[str(i) for i in range(10)],0)

    def test_nested_duplicate_rejected(self):
        with self.assertRaises(ValueError):nested_invalid_ids('q',['a']*40,0)

    def test_cal_tie_smallest_lambda(self):
        self.assertEqual(choose_cal_lambda({0:.5,.5:.7,1.:.7,2.:.6}),.5)

    def test_invalid_cal_rejected(self):
        with self.assertRaises(ValueError):choose_cal_lambda({0:float('nan')})

    def test_uncovered_query_counted(self):
        scores=np.array([[1.,3.],[0.,2.]]);a=np.array([[1,0],[0,0]])
        self.assertEqual(image_macro_hit([(scores,a)]),.5)

    def test_no_proposal_image_counted(self):
        self.assertEqual(image_macro_hit([(np.empty((0,2)),np.empty((0,2)))]),0)

    def test_image_macro_not_query_average(self):
        rec=[(np.zeros((1,1)),np.ones((1,1))), (np.zeros((1,3)),np.zeros((1,3)))]
        self.assertEqual(image_macro_hit(rec),.5)

    def test_finite_counterexample(self):
        d=single_positive_counterexample()
        self.assertTrue(d['wrong_top1']);self.assertGreater(d['J'],.9)
        self.assertAlmostEqual(d['gap'],d['gap_closed_form'],12)


if __name__=='__main__':unittest.main()
