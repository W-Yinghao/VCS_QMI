import numpy as np
import pytest
import torch
from reference.torch_losses import balanced_fit_loss,detached_weighted_loss

@pytest.mark.parametrize('seed',range(5))
def test_vcs_identity_and_gradient(seed):
    g=torch.Generator().manual_seed(seed)
    p=torch.randn(11,generator=g,dtype=torch.float64,requires_grad=True)
    q=torch.randn(13,generator=g,dtype=torch.float64,requires_grad=True)
    tp=torch.tanh(p/2);tq=torch.tanh(q/2)
    J=(tp-.5*tp**2).mean()+(-tq-.5*tq**2).mean()
    l=balanced_fit_loss(p,q,'vcs')
    assert torch.allclose(l,1-J,atol=1e-12)
    gg=torch.autograd.grad(l,(p,q));ga=torch.autograd.grad(1-J,(p,q))
    assert all(torch.allclose(a,b,atol=1e-12) for a,b in zip(gg,ga))

def test_weight_detach_and_scale():
    loss=torch.tensor([2.,3.],requires_grad=True);w=torch.tensor([.2,.8],requires_grad=True)
    o=detached_weighted_loss(loss,w,.5);o.backward()
    assert w.grad is None and torch.allclose(loss.grad,torch.tensor([.2,.8]))

@pytest.mark.parametrize('kind',['vcs','bce'])
def test_extreme_logits_finite(kind):
    p=torch.tensor([-100.,100.],requires_grad=True);q=torch.tensor([100.,-100.],requires_grad=True)
    l=balanced_fit_loss(p,q,kind);l.backward()
    assert torch.isfinite(l) and torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()

def test_bce_perfect_limit():
    assert balanced_fit_loss(torch.tensor([40.]),torch.tensor([-40.]),'bce')<1e-10

def test_weight_invalid():
    with pytest.raises(ValueError):detached_weighted_loss(torch.tensor([1.]),torch.tensor([-1.]),.5)
