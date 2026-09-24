"""CPU tests of formulas and gradient routing; not benchmark evidence."""
import math
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reference.ssl_core import (
    PairCritic, cyclic_negative_indices, joint_step_fp32, rpc_score_reference,
    simclr_nt_xent, vcs_from_scores, vcs_pair_loss, vicreg_loss,
)


def test_j_risk_identity():
    torch.manual_seed(11)
    p = torch.randn(17, dtype=torch.float64).tanh()
    q = torch.randn(43, dtype=torch.float64).tanh()
    s = vcs_from_scores(p, q)
    torch.testing.assert_close(s['J_raw'], 1 - s['R_binary'], atol=1e-12, rtol=1e-12)


def test_quadratic_not_square_of_mean():
    p = torch.tensor([-0.5, 0.5], dtype=torch.float64)
    s = vcs_from_scores(p, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(s['J_raw'], torch.tensor(-0.125, dtype=torch.float64))


def test_duplicate_negatives_do_not_reweight_q():
    p = torch.tensor([0.1, 0.7, -0.2], requires_grad=True)
    q = torch.tensor([-0.2, -0.6, 0.3], requires_grad=True)
    a = vcs_from_scores(p, q)['loss']
    b = vcs_from_scores(p, q.repeat(4))['loss']
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a, (p, q), retain_graph=True)
    gb = torch.autograd.grad(b, (p, q))
    for x, y in zip(ga, gb): torch.testing.assert_close(x, y)


def test_perfect_scores_and_loss_sign():
    s = vcs_from_scores(torch.ones(5), -torch.ones(11))
    assert s['J_raw'].item() == 1.0
    assert s['R_binary'].item() == 0.0
    assert s['loss'].item() == -1.0


def test_zero_scores():
    s = vcs_from_scores(torch.zeros(4), torch.zeros(9))
    assert s['J_raw'].item() == 0.0
    assert s['R_binary'].item() == 1.0


def test_wrong_extremes_negative_j_is_not_clipped():
    s = vcs_from_scores(-torch.ones(4), torch.ones(4))
    assert s['J_raw'].item() == -3.0
    assert s['loss'].item() == 3.0


def test_score_derivative_signs():
    p = torch.tensor([0.2, 0.3], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([-0.1, 0.4, -0.2], dtype=torch.float64, requires_grad=True)
    gp, gq = torch.autograd.grad(vcs_from_scores(p, q)['loss'], (p, q))
    torch.testing.assert_close(gp, (p.detach() - 1) / p.numel())
    torch.testing.assert_close(gq, (q.detach() + 1) / q.numel())


def test_rpc_111_value_and_gradient_identity():
    torch.manual_seed(3)
    p = torch.randn(9, dtype=torch.float64, requires_grad=True)
    q = torch.randn(18, dtype=torch.float64, requires_grad=True)
    a, b = vcs_from_scores(p, q)['J_raw'], rpc_score_reference(p, q)
    torch.testing.assert_close(a, b)
    for x, y in zip(torch.autograd.grad(a, (p, q), retain_graph=True),
                    torch.autograd.grad(b, (p, q))):
        torch.testing.assert_close(x, y)


def test_discrete_oracle_and_gap():
    p = torch.tensor([0.9, 0.1], dtype=torch.float64)
    q = torch.tensor([0.6, 0.4], dtype=torch.float64)
    m, eta = (p+q)/2, (p-q)/(p+q)
    S = (m*eta.square()).sum()
    torch.testing.assert_close(S, torch.tensor(0.12, dtype=torch.float64))
    for t in [eta, torch.tensor([0.1, -0.4], dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]:
        j = (p*t).sum()-(q*t).sum()-.5*(p*t.square()).sum()-.5*(q*t.square()).sum()
        torch.testing.assert_close(S-j, (m*(t-eta).square()).sum(), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('b,k', [(2,1),(3,2),(32,1),(17,16)])
def test_pairs_have_no_self_or_duplicate_partner(b, k):
    ind, shift = cyclic_negative_indices(b, k, generator=torch.Generator().manual_seed(7))
    assert ind.shape == (k,b)
    assert (ind != torch.arange(b)[None,:]).all()
    assert len(torch.unique(shift)) == k
    for row in ind: assert torch.equal(row.sort().values, torch.arange(b))


@pytest.mark.parametrize('b,k', [(1,1),(4,0),(4,4)])
def test_bad_pair_counts_rejected(b,k):
    with pytest.raises(ValueError): cyclic_negative_indices(b,k)


def test_dedicated_pair_rng_is_reproducible():
    g = torch.Generator().manual_seed(29)
    state = g.get_state()
    a = cyclic_negative_indices(32,3,generator=g)[0]
    torch.randn(100)  # unrelated global random activity
    g.set_state(state)
    b = cyclic_negative_indices(32,3,generator=g)[0]
    assert torch.equal(a,b)


def test_all_nonzero_shifts_cover_off_diagonal():
    b=7
    pairs,_ = cyclic_negative_indices(b,b-1,generator=torch.Generator().manual_seed(2))
    counts = torch.zeros(b,b,dtype=torch.long)
    for row in pairs: counts[torch.arange(b),row] += 1
    assert torch.equal(counts,1-torch.eye(b,dtype=torch.long))


def test_critic_pointwise_and_bounded():
    torch.manual_seed(3)
    c = PairCritic(4,16)
    x,y = torch.randn(6,4),torch.randn(6,4)
    a=c(x,y)
    b=torch.cat([c(x[i:i+1],y[i:i+1]) for i in range(len(x))])
    torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)
    assert (a.abs() <= 1).all()
    assert not any(isinstance(m,(nn.BatchNorm1d,nn.Dropout)) for m in c.modules())


def test_negative_branch_gradients_reach_both_views():
    torch.manual_seed(8)
    x=torch.randn(8,4,requires_grad=True); y=torch.randn(8,4,requires_grad=True)
    c=PairCritic(4,16)
    ind,_=cyclic_negative_indices(8,1,generator=torch.Generator().manual_seed(0))
    q=c(x,y[ind[0]])
    (q.mean()+0.5*q.square().mean()).backward()
    assert x.grad is not None and x.grad.abs().sum()>0
    assert y.grad is not None and y.grad.abs().sum()>0


def test_end_to_end_modules_receive_gradients_and_update():
    torch.manual_seed(9)
    enc=nn.Sequential(nn.Linear(6,12),nn.ReLU())
    proj=nn.Sequential(nn.Linear(12,8),nn.ReLU(),nn.Linear(8,4))
    critic=PairCritic(4,16)
    modules=(enc,proj,critic)
    params=[p for m in modules for p in m.parameters()]
    old=[[p.detach().clone() for p in m.parameters()] for m in modules]
    opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=0)
    x,y=torch.randn(12,6),torch.randn(12,6)
    out=joint_step_fp32(enc,proj,critic,opt,x,y,generator=torch.Generator().manual_seed(5))
    assert math.isfinite(out['J_raw'])
    for m,before in zip(modules,old):
        assert sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)>0
        assert any(not torch.equal(a,p.detach()) for a,p in zip(before,m.parameters()))


def test_estimator_only_detach_blocks_encoder_gradients():
    torch.manual_seed(10)
    x=torch.randn(8,4,requires_grad=True); y=torch.randn(8,4,requires_grad=True)
    c=PairCritic(4,16)
    stats,_=vcs_pair_loss(x.detach(),y.detach(),c,generator=torch.Generator().manual_seed(0))
    stats['loss'].backward()
    assert x.grad is None and y.grad is None
    assert any(p.grad is not None for p in c.parameters())


def test_batch_size_one_refused_by_pair_loss():
    with pytest.raises(ValueError): vcs_pair_loss(torch.ones(1,4),torch.ones(1,4),PairCritic(4,16))


def test_half_scores_reduce_in_float32():
    s=vcs_from_scores(torch.tensor([0.1,0.2],dtype=torch.float16),torch.tensor([-0.1],dtype=torch.float16))
    assert s['J_raw'].dtype==torch.float32


def test_empty_or_integer_scores_rejected():
    with pytest.raises(ValueError): vcs_from_scores(torch.zeros(0),torch.ones(1))
    with pytest.raises(TypeError): vcs_from_scores(torch.ones(1,dtype=torch.long),torch.ones(1))


def test_simclr_matches_manual_denominators():
    torch.manual_seed(1)
    x=torch.randn(4,3,dtype=torch.float64); y=torch.randn(4,3,dtype=torch.float64)
    z=F.normalize(torch.cat((x,y)),dim=1)
    logits=z@z.T/0.2
    terms=[]
    for i in range(8):
        allowed=torch.tensor([j for j in range(8) if j!=i])
        terms.append(-logits[i,(i+4)%8]+torch.logsumexp(logits[i,allowed],dim=0))
    torch.testing.assert_close(simclr_nt_xent(x,y),torch.stack(terms).mean())


def test_simclr_all_identical_reference():
    x=torch.ones(5,3,dtype=torch.float64)
    torch.testing.assert_close(simclr_nt_xent(x,x),torch.tensor(math.log(9),dtype=torch.float64))


def test_vicreg_collapsed_input_has_variance_penalty():
    x=torch.zeros(8,4,requires_grad=True)
    v=vicreg_loss(x,x)
    assert v['invariance'].item()==0
    assert v['covariance'].item()==0
    assert v['variance'].item()>0.9
    v['loss'].backward()
    assert torch.isfinite(x.grad).all()


def test_eval_frozen_encoder_statistics_unchanged():
    torch.manual_seed(0)
    enc=nn.Sequential(nn.Linear(4,8),nn.BatchNorm1d(8),nn.ReLU()).eval()
    for p in enc.parameters():p.requires_grad_(False)
    before={k:v.clone() for k,v in enc.state_dict().items()}
    head=nn.Linear(8,3)
    x=torch.randn(16,4); labels=torch.arange(16)%3
    with torch.no_grad():h=enc(x)
    F.cross_entropy(head(h),labels).backward()
    assert head.weight.grad is not None
    assert all(p.grad is None for p in enc.parameters())
    for key,val in enc.state_dict().items(): assert torch.equal(before[key],val)
