import numpy as np
import pytest
from reference.core import *

@pytest.mark.parametrize('seed',range(8))
def test_gap_and_moments(seed):
    g=np.random.default_rng(seed);p=g.dirichlet(np.ones(9));q=g.dirichlet(np.ones(9));t=g.uniform(-1,1,9)
    d=finite_target(p,q);e=d['eta'];m=d['m'];S=d['S']
    assert finite_j(p,q,t)==pytest.approx(S-m@((t-e)**2),abs=1e-12)
    assert p@e==pytest.approx(S,abs=1e-12)
    assert q@e==pytest.approx(-S,abs=1e-12)
    assert 0<=S<=1

@pytest.mark.parametrize('rho',[0,.01,.1,.3,.5,.9,1])
def test_mixture_and_risk(rho):
    p=np.array([.6,.3,.1]);q=np.array([.1,.2,.7]);e=finite_target(p,q)['eta'];r=mixture_mass(p,q,rho)
    rr=rho_from_means(p@e,q@e,r@e)
    assert rr.raw==pytest.approx(rho,abs=1e-12)
    w=clean_posterior(e,rho)
    assert r@w==pytest.approx(1-rho)
    ell=np.array([.3,.9,.1])
    assert r@(w*ell)==pytest.approx((1-rho)*(p@ell))

@pytest.mark.parametrize('scale,offset',[(2,3),(-2,4),(.01,-.4),(5,0)])
def test_affine_invariance(scale,offset):
    x=rho_from_means(.6,-.3,.15).raw
    z=rho_from_means(scale*.6+offset,scale*(-.3)+offset,scale*.15+offset).raw
    assert x==pytest.approx(z)

def test_unidentified_point():
    assert rho_from_means(.4,.4,.2).raw is None

def test_ratio_outside_kept_raw():
    r=rho_from_means(.5,-.5,.8)
    assert r.raw<0 and r.projected==0 and r.status=='outside_parameter_space'

@pytest.mark.parametrize('rho',[0,.1,.3,.5,.8,1])
def test_zero_radius_recovers(rho):
    a,b=.5,-.2;c=(1-rho)*a+rho*b
    z=interval_from_radii(a,b,c,0,0,0)
    assert z.lower==pytest.approx(rho,abs=1e-12)
    assert z.upper==pytest.approx(rho,abs=1e-12)

def test_full_when_nondiscriminating():
    z=interval_from_radii(.1,.1,.1,.05,.05,.05)
    assert z.status=='full_parameter_space'

def test_empty_when_incompatible():
    z=interval_from_radii(.1,.2,.9,.01,.01,.01)
    assert z.status=='empty'

@pytest.mark.parametrize('seed',range(5))
def test_feasible_set_against_grid(seed):
    rng=np.random.default_rng(seed);a,b,c=rng.uniform(-1,1,3);es=rng.uniform(.01,.3,3)
    z=interval_from_radii(a,b,c,*es);grid=np.linspace(0,1,10001)
    feasible=np.abs(c-(1-grid)*a-grid*b)<=es[2]+(1-grid)*es[0]+grid*es[1]+1e-12
    if feasible.any():
        assert z.lower<=grid[feasible][0]+1e-12 and z.upper>=grid[feasible][-1]-1e-12
    elif z.status!='empty':
        assert z.upper-z.lower<.0001

def test_ci_contract():
    with pytest.raises(ValueError):rho_confidence_set([.1],[.2],[.15])
    with pytest.raises(ValueError):rho_confidence_set([2],[.2],[.1],assumptions_verified=True)

def test_small_n_interval_is_wide():
    z=rho_confidence_set([.8],[-.8],[0],assumptions_verified=True)
    assert z.status=='full_parameter_space'

@pytest.mark.parametrize('rho',[.1,.5,.9])
def test_ci_contains_population_mean_case(rho):
    p=np.full(1000,.6);q=np.full(1000,-.4);r=np.full(1000,(1-rho)*.6-rho*.4)
    z=rho_confidence_set(p,q,r,assumptions_verified=True)
    assert z.lower<=rho<=z.upper

def test_representation_sufficiency_counterexample():
    # Z constant; latent clean C determines raw-data loss. Perfect w_Z is not sufficient.
    mass=np.array([.5,.5]);C=np.array([1.,0.]);ell=np.array([1.,0.]);wZ=np.array([.5,.5])
    lhs=mass@(wZ*ell)-mass@(C*ell)
    cov=mass@(C*ell)-(mass@C)*(mass@ell)
    assert lhs==pytest.approx(-cov) and abs(lhs)>.2

def test_weight_metric_limits():
    assert effective_sample_size([1,1,1])==pytest.approx(3)
    assert effective_sample_size([1,0,0])==pytest.approx(1)
    assert effective_sample_size([0,0])==0
    assert brier([0,1],[0,1])==0
    assert logloss([0,1],[0,1])<1e-10

@pytest.mark.parametrize('mass',[0,.5,1,2.4,4])
def test_rank_mass(mass):
    w=rank_mass_weights([.8,.8,.4,.1],mass)
    assert w.sum()==pytest.approx(mass) and np.all((w>=0)&(w<=1))
    assert w[0]>=w[1]>=w[2]>=w[3]

def test_roles_and_donors():
    ok=[{'role':'P','cluster_id':'p','units':['i1']},{'role':'Q','cluster_id':'q','units':['i2','i3']}]
    assert role_disjointness(ok)
    with pytest.raises(ValueError):role_disjointness(ok+[{'role':'R','cluster_id':'r','units':['i3']}])
    with pytest.raises(ValueError):role_disjointness([ok[0],ok[0]])

def test_multiscore_shift_not_identified():
    r=mean_ci_vector_consistency([.5,.2],[-.5,-.2],[0,0],.5)
    assert np.allclose(r,0)

@pytest.mark.parametrize('bad',[[-.1,1.1],[.1,.2],[float('nan'),1],[0,0]])
def test_mass_validation(bad):
    with pytest.raises(ValueError):probability_mass(bad)

@pytest.mark.parametrize('t,rho',[([1.1],.2),([0],-1),([-1],0),([1],1)])
def test_posterior_invalid(t,rho):
    with pytest.raises(ValueError):clean_posterior(t,rho)

def test_retrieval_multiple_positives():
    s=np.array([[.9,.8,.1,.2],[.2,.1,.8,.9]])
    z=retrieval_recall(s,['a','b'],['a','a','b','b'])
    assert z['rsum']==600 and z['mR']==100

def test_retrieval_ties():
    z=retrieval_recall(np.zeros((2,2)),['a','b'],['a','b'],ks=(1,))
    assert z['i2t_R1']==50 and z['t2i_R1']==50

def test_retrieval_missing_mapping():
    with pytest.raises(ValueError):retrieval_recall([[1,2]],['a'],['a','b'])
