"""V10 v0.1 CPU reference. No natural-data training or performance benchmark.
A/B are deterministic simplex features. Independent base pairs are required
for an off-diagonal U-statistic interpretation. No n*n score matrix in moments.
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize

torch.set_num_threads(1)
D = torch.float64

def product_moments(a, b, w, off=False):
    if a.ndim != 2 or b.ndim != 2 or w.shape != (a.shape[1], b.shape[1]):
        raise ValueError('Need A[n,r], B[m,s], W[r,s]')
    n,m=len(a),len(b)
    first=a.sum(0)@w@b.sum(0)
    second=torch.sum(w*((a.T@a)@w@(b.T@b)))
    denominator=n*m
    if off:
        if n!=m or n<2: raise ValueError('Off-diagonal needs paired rows n=m>=2')
        diag=((a@w)*b).sum(1)
        first=first-diag.sum(); second=second-diag.square().sum()
        denominator=n*(n-1)
    return first/denominator,second/denominator

def objective(a,b,w,off=True,ridge=0.):
    if len(a)!=len(b): raise ValueError('Positive rows must be paired')
    diag=((a@w)*b).sum(1)
    q1,q2=product_moments(a,b,w,off)
    return diag.mean()-q1-.5*diag.square().mean()-.5*q2-.5*ridge*w.square().sum()

def dense_objective(a,b,w,off=True,ridge=0.):
    t=a@w@b.T; d=t.diagonal()
    q=t[~torch.eye(len(a),dtype=torch.bool)] if off else t.reshape(-1)
    return d.mean()-q.mean()-.5*d.square().mean()-.5*q.square().mean()-.5*ridge*w.square().sum()

def dense_quadratic(a,b,ridge=0.):
    # Small-case test oracle only; not the production HVP implementation.
    a,b=np.asarray(a),np.asarray(b); n=len(a)
    phi=np.einsum('ia,jb->ijab',a,b).reshape(n,n,-1)
    p=phi[np.arange(n),np.arange(n)]; q=phi[~np.eye(n,dtype=bool)]
    delta=p.mean(0)-q.mean(0)
    h=p.T@p/n+q.T@q/len(q)+ridge*np.eye(p.shape[1])
    return delta,h

def solve_box(delta,h):
    def fun(w): return .5*w@h@w-delta@w,h@w-delta
    r=minimize(fun,np.zeros(len(delta)),jac=True,bounds=[(-1.,1.)]*len(delta),
        method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-11,'maxiter':10000,'maxls':100})
    w=r.x; g=delta-h@w
    return w,dict(success=bool(r.success),message=str(r.message),iterations=int(r.nit),
        objective=float(delta@w-.5*w@h@w),fw_gap=max(0.,float(np.abs(g).sum()-g@w)))

def run_checks():
    rng=np.random.default_rng(20261016); rows=[]
    def check(name,value,bound,note=''):
        row=dict(name=name,value=float(value),required_max=float(bound),
                 passed=bool(np.isfinite(value) and value<=bound),note=note)
        rows.append(row)
        if not row['passed']: raise AssertionError(row)
    valerr=graderr=0.
    for n,r,s in [(2,2,3),(5,3,2),(11,4,5),(31,7,6)]:
        za=torch.tensor(rng.normal(size=(n,r)),dtype=D,requires_grad=True)
        zb=torch.tensor(rng.normal(size=(n,s)),dtype=D,requires_grad=True)
        w=torch.tensor(rng.uniform(-1,1,(r,s)),dtype=D,requires_grad=True)
        a,b=za.softmax(1),zb.softmax(1)
        check(f'global_bound_n{n}',float((a@w@b.T).abs().max().detach()),1.)
        for off in [False,True]:
            f=objective(a,b,w,off,.01); d=dense_objective(a,b,w,off,.01)
            valerr=max(valerr,abs(float((f-d).detach())))
            gf=torch.autograd.grad(f,(za,zb,w),retain_graph=True)
            gd=torch.autograd.grad(d,(za,zb,w),retain_graph=True)
            graderr=max(graderr,max(float((x-y).abs().max()) for x,y in zip(gf,gd)))
    check('dense_equals_moment_value',valerr,1e-12)
    check('dense_equals_moment_gradients',graderr,1e-11)
    a=torch.tensor(rng.dirichlet(np.ones(4),7),dtype=D)
    b=torch.tensor(rng.dirichlet(np.ones(3),9),dtype=D)
    w=torch.tensor(rng.uniform(-1,1,(4,3)),dtype=D)
    x,y=product_moments(a,b,w); t=a@w@b.T
    check('rectangular_independent_product',max(abs(float(x-t.mean())),abs(float(y-t.square().mean()))),1e-12)
    n=5
    a=torch.tensor(rng.dirichlet(np.ones(3),n),dtype=D)
    b=torch.tensor(rng.dirichlet(np.ones(3),n),dtype=D)
    w=torch.tensor(rng.uniform(-1,1,(3,3)),dtype=D)
    t=(a@w@b.T).numpy(); d=np.diag(t)
    perms=[p for p in itertools.permutations(range(n)) if all(p[i]!=i for i in range(n))]
    vals=[]
    for p in perms:
        q=t[np.arange(n),p]
        vals.append(d.mean()-.5*np.mean(d*d)-q.mean()-.5*np.mean(q*q))
    check('mean_44_derangements_equals_U',abs(np.mean(vals)-float(objective(a,b,w))),1e-12)
    check('sampling_variance_exists',0. if np.var(vals)>1e-10 else 1.,0.,
          f'Conditional permutation variance={np.var(vals):.8g}; not a natural-data result')
    tt=np.tanh(t); dd=np.diag(tt); qq=tt[~np.eye(n,dtype=bool)]
    f_tanh=dd.mean()-qq.mean()-.5*np.mean(dd*dd)-.5*np.mean(qq*qq)
    diff=abs(float(objective(a,b,w))-f_tanh)
    check('post_tanh_invalidates_moment_shortcut',0. if diff>1e-7 else 1.,0.,f'Counterexample difference={diff:.8g}')
    delta,h=dense_quadratic(a.numpy(),b.numpy(),.01)
    check('fixed_features_PSD_hessian',max(0.,-np.linalg.eigvalsh(h).min()),1e-12)
    ws,info=solve_box(delta,h)
    check('inner_solver_gap',info['fw_gap'],1e-7,info['message'])
    w0=rng.uniform(-.8,.8,len(delta)); g0=delta-h@w0
    f0=delta@w0-.5*w0@h@w0; fs=delta@ws-.5*ws@h@ws
    check('FW_certificate_bounds_objective_gap',max(0.,fs-f0-(np.abs(g0).sum()-g0@w0)),1e-10)
    p=np.array([[.30,.10],[.05,.55]])
    q=p.sum(1,keepdims=True)*p.sum(0,keepdims=True)
    eta=(p-q)/(p+q); s=.5*np.sum((p-q)**2/(p+q))
    j=np.sum((p-q)*eta)-.5*np.sum((p+q)*eta**2)
    check('hard_partition_oracle_S_equals_J',abs(s-j),1e-14)
    trial=rng.uniform(-1,1,p.shape)
    j=np.sum((p-q)*trial)-.5*np.sum((p+q)*trial**2)
    gap=.5*np.sum((p+q)*(trial-eta)**2)
    check('original_exact_gap_unchanged',abs(s-j-gap),1e-14)
    a1=torch.ones((n,1),dtype=D); w1=torch.zeros((1,1),dtype=D)
    check('K1_can_be_perfectly_solved_but_uninformative',abs(float(objective(a1,a1,w1))),1e-14)
    # Ordinary permutation matches the diagonal-INCLUDED target of current REG.
    allvals=[]
    for perm in itertools.permutations(range(n)):
        neg=t[np.arange(n),perm]
        allvals.append(d.mean()-.5*np.mean(d*d)-neg.mean()-.5*np.mean(neg*neg))
    check('mean_120_permutations_equals_V',
          abs(np.mean(allvals)-float(objective(a,b,w,off=False))),1e-12)
    # Row-normalized admissible pairs: subtract the same excluded set.
    allowed=np.ones((n,n),dtype=bool); np.fill_diagonal(allowed,False)
    allowed[0,1]=False; allowed[1,3]=False
    alpha=1.0/(n*allowed.sum(1)); aa,bb,ww=a.numpy(),b.numpy(),w.numpy()
    aq=(alpha[:,None]*aa).sum(0)
    ag=aa.T@(alpha[:,None]*aa); bg=bb.T@bb
    ef=float(aq@ww@bb.sum(0))-np.sum(alpha[:,None]*t*(~allowed))
    es=float(np.sum(ww*(ag@ww@bg)))-np.sum(alpha[:,None]*t*t*(~allowed))
    check('row_masked_first_and_second_moments',
          max(abs(ef-np.sum(alpha[:,None]*t*allowed)),
              abs(es-np.sum(alpha[:,None]*t*t*allowed))),1e-12)
    delta0,h0=dense_quadratic(a.numpy(),b.numpy(),0.)
    ws0,info0=solve_box(delta0,h0)
    check('unregularized_inner_solver_gap',info0['fw_gap'],1e-7,info0['message'])
    return dict(status='passed',n_checks=len(rows),checks=rows,inner_solver=info,
                scope='CPU algebra and derivatives only; no natural-data result or GPU timing.',seed=20261016,dtype='float64')

if __name__=='__main__':
    result=run_checks(); dest=Path(__file__).with_name('MATH_CHECK_RESULTS.json')
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'n_checks':result['n_checks'],'output':str(dest)},indent=2))
