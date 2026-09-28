"""Run local, synthetic validation for the documentation reference.

Usage: python support/validate_spec.py
No server access, neural training, dataset downloads, or real-task evaluation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from bounded_regression_core import (j_from_scores, balanced_squared_risk,
                                     residual_step, fit_small_simplex, dictionary_moments)


def run() -> dict:
    rng = np.random.default_rng(20260928)
    checks: dict[str, float | int | bool] = {}
    max_risk = max_gain = max_moments = max_concavity = max_kkt = 0.0
    residual_boundaries: set[int] = set()
    for _ in range(32):
        p = np.tanh(rng.normal(size=(71, 4)))
        q = np.tanh(rng.normal(size=(113, 4)))
        for j in range(4):
            max_risk = max(max_risk, abs(j_from_scores(p[:,j],q[:,j])
                             + balanced_squared_risk(p[:,j],q[:,j])-1))
        st = residual_step(p[:,0], q[:,0], p[:,1], q[:,1])
        t, u = (1-st.coefficient)*p[:,0]+st.coefficient*p[:,1], (1-st.coefficient)*q[:,0]+st.coefficient*q[:,1]
        max_gain = max(max_gain, abs(j_from_scores(t,u)-j_from_scores(p[:,0],q[:,0])-st.predicted_gain))
        fit = fit_small_simplex(p,q)
        max_kkt = max(max_kkt,fit.kkt_gap)
        assert fit.objective >= max(j_from_scores(p[:,j],q[:,j]) for j in range(4))-1e-10
        max_moments = max(max_moments, abs(fit.objective-j_from_scores(p@fit.weights,q@fit.weights)))
        assert np.max(np.abs(p@fit.weights)) <= 1+1e-12
        w=rng.dirichlet(np.ones(4))
        disagreement=.5*np.mean((p*p)@w-(p@w)**2)+.5*np.mean((q*q)@w-(q@w)**2)
        mixture_j=sum(w[j]*j_from_scores(p[:,j],q[:,j]) for j in range(4))+disagreement
        max_concavity=max(max_concavity,abs(j_from_scores(p@w,q@w)-mixture_j))
    # Deliberately cover no direction, rejected direction, and full step.
    zero=residual_step([.2,.4],[-.1,-.3],[.2,.4],[-.1,-.3])
    assert zero.zero_direction and zero.coefficient == 0
    bad=residual_step([.8,.7],[-.6,-.7],[-.8,-.7],[.6,.7])
    good=residual_step([0,0],[0,0],[.8,.7],[-.6,-.7])
    assert bad.coefficient == 0 and good.coefficient == 1
    same=fit_small_simplex(np.full((3,3),.4),np.full((7,3),-.2))
    assert same.kkt_gap < 1e-10
    # Known finite distribution, exact weights.
    p=rng.dirichlet(np.ones(12)).reshape(4,3)
    q=p.sum(1,keepdims=True)*p.sum(0,keepdims=True); m=.5*(p+q)
    eta=(p-q)/(p+q); S=float(np.sum(m*eta**2))
    T=np.tanh(rng.normal(size=p.shape))
    jw=lambda pp,qq,t: float(np.sum(pp*(t-.5*t*t))+np.sum(qq*(-t-.5*t*t)))
    exact_gap=abs(S-jw(p,q,T)-np.sum(m*(eta-T)**2))
    g=1+T
    rulsif=abs(.5*np.sum(m*g*g)-np.sum(p*g)+.5+.5*jw(p,q,T))
    pa=p.reshape(2,2,3).sum(1);qa=q.reshape(2,2,3).sum(1)
    ma=.5*(pa+qa);etaa=(pa-qa)/(pa+qa);sa=float(np.sum(ma*etaa**2))
    nested=abs(S-sa-np.sum(m*(eta-np.repeat(etaa,2,axis=0))**2))
    noise_error=0.
    for eps in (0.,.1,.4,.8):
        pe=(1-eps)*p+eps*q
        etae=(pe-q)/(pe+q)
        pred=(1-eps)*eta/(1-eps*eta)
        inv=etae/(1-eps+eps*etae)
        recover=(np.sum(pe*(T-.5*T*T))-np.sum(q*(T+.5*(1-2*eps)*T*T)))/(1-eps)
        noise_error=max(noise_error,float(np.max(abs(etae-pred))),float(np.max(abs(inv-eta))),abs(recover-jw(p,q,T)))
    graderr=0.
    for c in (-1.,1.):
        for f in (-3.,-1.,0.,1.,3.):
            h=1e-5;t=np.tanh(f)
            fV=lambda x:.5*(c-np.tanh(x))**2
            fJ=lambda x:np.logaddexp(0.,-2*c*x)
            graderr=max(graderr,abs((t-c)*(1-t*t)-(fV(f+h)-fV(f-h))/(2*h)),abs((t-c)-(fJ(f+h)-fJ(f-h))/(2*h)))
    checks.update(max_risk_identity_error=max_risk,max_residual_gain_error=max_gain,
                  max_simplex_moment_error=max_moments,max_mixture_identity_error=max_concavity,
                  max_simplex_kkt_gap=max_kkt,exact_population_gap_error=float(exact_gap),
                  rulsif_identity_error=float(rulsif),nested_increment_error=float(nested),
                  known_mismatch_identity_error=noise_error,max_finite_difference_gradient_error=graderr,
                  random_dictionary_cases=32,zero_and_boundary_steps=True,degenerate_dictionary=True)
    assert all(checks[k] < 1e-9 for k in checks if 'error' in k)
    assert max_kkt < 1e-8
    for invalid in ([1.1],[float('nan')]):
        try:
            j_from_scores(invalid,[0.])
        except ValueError:
            pass
        else:
            raise AssertionError('invalid scores not rejected')
    checks['invalid_scores_rejected']=True
    return {'seed':20260928,'status':'passed','scope':'Synthetic identities and fixed-output NumPy fitting only. No neural/server/SSL experiments.',
            'checks':checks,'not_tested':['torch gradient paths','dataset identity manifests','online SSL','real task improvements','neural residual training','observation-scale performance']}

if __name__ == '__main__':
    result=run()
    Path(__file__).with_name('validation_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
