"""Independent mathematical checks for the VCS-QMI discussion.
These are synthetic checks, not new SSL or repository experiments.
Run: python independent_checks.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEED = 20260928
rng = np.random.default_rng(SEED)

def j_value(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> float:
    return float(np.sum(p * (t - 0.5*t*t)) + np.sum(q * (-t - 0.5*t*t)))

# A finite strictly-positive joint distribution and its actual product marginals.
p = rng.dirichlet(np.ones(12)*0.7).reshape(4, 3)
q = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
m = (p+q)/2
eta = (p-q)/(p+q)
s = float(np.sum(m*eta**2))
t0 = np.tanh(np.arctanh(eta) + 0.8)
u = np.tanh(np.arctanh(eta) - 0.5)
r = u-t0
A = float(np.sum(m*(eta-t0)*r))
B = float(np.sum(m*r*r))
lam = float(np.clip(A/B, 0.0, 1.0))
t_mix = t0 + lam*r
j0, j1, jm = [j_value(p,q,t) for t in (t0,u,t_mix)]
mixture_identity = jm - ((1-lam)*j0 + lam*j1 + lam*(1-lam)*B)
gain_identity = (jm-j0) - (2*lam*A-lam*lam*B)
gap_identity = s-jm-float(np.sum(m*(eta-t_mix)**2))

# RuLSIF at alpha=1/2 with the same function g=1+T, without mismatched penalties.
g = 1+t_mix
rulsif_loss = float(0.5*np.sum(m*g*g)-np.sum(p*g))
rulsif_identity = rulsif_loss - (-0.5-0.5*jm)

# Coarsen X from four values to two: A=(floor(X/2),Y), B=(X,Y).
p_a = p.reshape(2,2,3).sum(1)
q_a = q.reshape(2,2,3).sum(1)
m_a = (p_a+q_a)/2
eta_a = (p_a-q_a)/(p_a+q_a)
s_a = float(np.sum(m_a*eta_a**2))
eta_a_lift = np.repeat(eta_a,2,axis=0)
residual_value = float(np.sum(m*(eta-eta_a_lift)**2))
nested_identity = s-s_a-residual_value

# Exact deterministic example: X=Y uniform on n states.
deterministic = [{"n": n,"S": (n-1)/(n+1)} for n in (10,100,1000,45000)]

# Gradients w.r.t. f, when T=tanh(f), with .5 squared loss and binary log loss.
gradients=[]
for t in (-0.99,-0.9,0.0,0.9,0.99):
    f = float(np.arctanh(t))
    eps = 1e-5
    sq = lambda z: 0.5*(1-np.tanh(z))**2
    log = lambda z: np.logaddexp(0.0,-2*z)
    dv = (t-1)*(1-t*t)
    dl = t-1
    gradients.append({"T":t,"C":1,"squared_grad":dv,"logloss_grad":dl,
        "ratio":1-t*t,"squared_fd_error":abs(dv-(sq(f+eps)-sq(f-eps))/(2*eps)),
        "logloss_fd_error":abs(dl-(log(f+eps)-log(f-eps))/(2*eps))})

# Independent Gaussian oracle calculation: d=20, I=2,4,6,8,10 nats.
# Only sums ||x||^2, ||e||^2 and x.e are needed. Draw independently for P and Q.
n=300_000
d=20

def stats(n:int,d:int):
    a=[];b=[];c=[]
    for start in range(0,n,25_000):
        size=min(25_000,n-start)
        x=rng.normal(size=(size,d)); e=rng.normal(size=(size,d))
        a.append(np.sum(x*x,axis=1));b.append(np.sum(e*e,axis=1));c.append(np.sum(x*e,axis=1))
    return tuple(np.concatenate(v) for v in (a,b,c))
ap,bp,cp=stats(n,d)
aq,bq,cq=stats(n,d)
gaussian=[]
for info in (2.,4.,6.,8.,10.):
    rho=np.sqrt(1-np.exp(-2*info/d))
    lp=info+0.5*rho*rho*(ap-bp)+rho*np.sqrt(1-rho*rho)*cp
    lq=info+(2*rho*cq-rho*rho*(aq+bq))/(2*(1-rho*rho))
    tp=np.tanh(lp/2);tq=np.tanh(lq/2)
    a=tp-0.5*tp**2;b=-tq-0.5*tq**2
    jhat=float(a.mean()+b.mean())
    se=float(np.sqrt(a.var(ddof=1)/n+b.var(ddof=1)/n))
    js_p=-np.logaddexp(0,-lp);js_q=-np.logaddexp(0,lq)
    js=float(np.log(2)+0.5*(js_p.mean()+js_q.mean()))
    gaussian.append({"dimension":d,"Shannon_MI_nats":info,"rho":float(rho),"oracle_S_J_MC":jhat,
        "MC_standard_error":se,"oracle_JS_nats_MC":js,
        "mixture_mean_1_minus_T_squared":float(1-0.5*(np.mean(tp**2)+np.mean(tq**2)))})

out={"seed":SEED,"scope":"Synthetic math checks only. No learned neural estimator, CIFAR or server jobs executed.",
 "finite_example":{"S":s,"J_T0":j0,"J_U":j1,"lambda":lam,"J_mixture":jm,
 "error_mixture_concavity_identity":mixture_identity,"error_exact_step_gain":gain_identity,
 "error_regression_gap_identity":gap_identity,"error_RuLSIF_equivalence":rulsif_identity,
 "S_coarse":s_a,"S_full_minus_coarse":s-s_a,"squared_critic_difference":residual_value,
 "error_nested_identity":nested_identity},"deterministic_S":deterministic,
 "gradient_check":gradients,"gaussian_oracle":{"samples_per_distribution":n,"rows":gaussian}}
for key,value in out["finite_example"].items():
    if key.startswith("error_"):
        assert abs(value)<1e-12,(key,value)
assert max(v["squared_fd_error"] for v in gradients)<1e-8
assert max(v["logloss_fd_error"] for v in gradients)<1e-8
path=Path(__file__).with_name("independent_checks.json")
path.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
