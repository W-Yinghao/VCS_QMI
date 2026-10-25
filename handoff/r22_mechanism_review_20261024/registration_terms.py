"""CAL pose-gradient decomposition at fixed shared-cosine checkpoints."""
import argparse
import math
import time
import types
from mechanism_common import BASE,HERE,read,write,sha,verify,geometry,check_pass,output,done
L=geometry();T,R,A,torch=L.T,L.R,L.A,L.torch
NAMES=('P1','negative_P2_half','negative_Q1','negative_Q2_half','POS_QUAD','J')

def terms(u,v):
    u,v=T.normalized(u,v);p=(u*v).sum(-1);n=u.shape[-2]
    d=dict(P1=p.mean(-1),negative_P2_half=-.5*p.square().mean(-1),
           negative_Q1=-(u.mean(-2)*v.mean(-2)).sum(-1),
           negative_Q2_half=-.5*((u.transpose(-1,-2)@u/n)*(v.transpose(-1,-2)@v/n)).sum((-1,-2)))
    d['POS_QUAD']=d['P1']+d['negative_P2_half'];d['J']=sum(d[k] for k in NAMES[:4]);return d

def install(energy):
    inner=R.PE.inner(energy)
    def j(self,v):return terms(self.u[None],v)[self.mechanism_term]
    inner.j_critic=types.MethodType(j,inner);inner.mechanism_term='J';return inner

def error(p,truth):
    q=p.detach().cpu().tolist();return A.Q.corner_error_full(math.degrees(q[0]/A.Q.RHO),q[1:],truth[0],truth[1:])

def update(energy,inner,p0,term):
    # Actual host Adam hyperparameters and projection, copied initial pose and
    # fresh optimizer state, exactly as the beginning of each local refinement.
    p=p0.detach().clone().requires_grad_();opt=torch.optim.Adam([p],lr=A.Q.P.ADAM_LR)
    inner.mechanism_term=term;value=energy(p);g=torch.autograd.grad(value,p)[0]
    assert torch.isfinite(value) and torch.isfinite(g).all()
    p.grad=-g;opt.step()
    with torch.no_grad():
        p[0].clamp_(-A.Q.UMAX,A.Q.UMAX);n=p[1:].norm()
        if n>A.Q.TMAX:p[1:].mul_(A.Q.TMAX/n)
    return p.detach(),dict(value=float(value.detach()),gradient=g.detach().cpu().tolist(),gradient_norm=float(g.norm()),updated_pose=p.detach().cpu().tolist())

def check():
    f=verify();torch.set_num_threads(2);torch.manual_seed(24)
    u=torch.randn(2,7,5,dtype=torch.float64,requires_grad=True);v=torch.randn_like(u,requires_grad=True)
    z=terms(u,v);j=T.exact_j(*T.normalized(u,v));assert torch.allclose(z['J'],j,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(z['J'].sum(),(u,v),retain_graph=True);b=torch.autograd.grad(j.sum(),(u,v),retain_graph=True)
    assert all(torch.allclose(x,y,atol=1e-12,rtol=1e-12) for x,y in zip(a,b))
    assert torch.autograd.gradcheck(lambda x,y:torch.stack(list(terms(x,y).values())),(u,v))
    write(HERE/'checks/registration.json',dict(status='PASS',freeze_sha256=f,checks=['all-term values and gradients match complete exact J','float64 finite differences all terms','actual host Adam lr and pose projection reused']))

def run(ds,arm,seed):
    verify();check_pass('registration');torch.set_num_threads(4);started=time.time()
    name=f'reg_terms_{ds}_{arm}_{seed}';out=output(name)
    s=read(HERE/'SOURCE_LOCK.json')['geometry'][f'{ds}_{arm}_{seed}']
    assert sha(s['checkpoint'])==s['sha256'];record=read(s['completion']);assert record['status']=='COMPLETE'
    u,guard,H,fit,_=L.context(ds,seed,'cuda',images=False,gate_scope=True)
    t=T.GeometryTrainer(L.config(seed),{},fit,'cuda',arm);R.CP.load(t,s['checkpoint'],record['identity'])
    assert t.n_steps==30000
    # Deterministic IDs selected before outcomes; first four CAL windows, two
    # fixed perturbations. Small local diagnostic, never DEV/test grid rerun.
    cores=sorted(guard.included('CAL'))[:4];rows=[]
    with R.CP.evaluation_state(t):
        session=R.make_session(u,t,guard,H,'CAL',cores,'cuda')
        for core in cores:
            for i in (0,1):
                energy,_=session.energy(core,i);inner=install(energy);truth=session.perturbations[core][i]
                p0=torch.tensor([math.radians(truth[0])*A.Q.RHO,*truth[1:]],device='cuda')
                for radius in (1.,4.,16.):
                    for axis in range(3):
                        for sign in (-1,1):
                            p=p0.clone();p[axis]+=sign*radius
                            if abs(float(p[0]))>A.Q.UMAX or float(p[1:].norm())>A.Q.TMAX:continue
                            before=error(p,truth);r=dict(core=core,scene=guard.unit_of(core) if H else core,perturbation=i,radius=radius,axis=axis,sign=sign,error_before=before,terms={})
                            for k in NAMES:
                                after,d=update(energy,inner,p,k);d['error_after']=error(after,truth);d['error_delta']=d['error_after']-before;r['terms'][k]=d
                            # Identity must hold at every real CAL pose too.
                            assert math.isclose(r['terms']['J']['value'],sum(r['terms'][k]['value'] for k in NAMES[:4]),abs_tol=2e-6,rel_tol=2e-5)
                            gj=torch.tensor(r['terms']['J']['gradient']);gs=sum(torch.tensor(r['terms'][k]['gradient']) for k in NAMES[:4])
                            assert torch.allclose(gj,gs,atol=2e-5,rtol=1e-3)
                            R.append(out/'POSE_TERMS.jsonl',r);rows.append(r)
                # Old jobs did not retain their coarse top-k coordinates. Make
                # one CAL-only coarse pass and reevaluate only its top3, no local
                # search or full exact grid. Disclose this provenance explicitly.
                rerank={}
                for k in ('P1','J'):
                    inner.mechanism_term=k;ths,ts=A.Q.grid_points();tt=torch.tensor(ts,dtype=torch.float32,device='cuda')
                    with torch.no_grad():vals=R.np.concatenate([energy.grid_values(float(th),tt).float().cpu().numpy() for th in ths])
                    assert R.np.isfinite(vals).all();ids=R.np.argsort(vals)[::-1][:A.Q.P.N_STARTS];candidates=[]
                    for rank,index in enumerate(ids):
                        th=float(ths[index//len(ts)]);tr=ts[index%len(ts)]
                        candidates.append(dict(coarse_rank=rank,theta=th,translation=tr.tolist(),coarse_value=float(vals[index]),exact_value=energy.value(th,tr)))
                    rerank[k]=dict(candidates=candidates,exact_order=sorted(range(len(candidates)),key=lambda n:candidates[n]['exact_value'],reverse=True))
                R.append(out/'TOPK_RERANK.jsonl',dict(core=core,perturbation=i,readouts=rerank))
                print(name,core,i,'complete',flush=True)
    assert {r['role'] for r in guard.reads}<={'CAL'}
    write(out/'INPUT_TRACE.json',guard.reads)
    done(out,started,ds=ds,arm=arm,seed=seed,source=s,rows=len(rows),roles_read=sorted({r['role'] for r in guard.reads}),
         environment=R.environment(),pose_terms_sha256=sha(out/'POSE_TERMS.jsonl'),rerank_sha256=sha(out/'TOPK_RERANK.jsonl'),
         interpretation='CAL local one-step Adam from fresh state; isolated terms can change step direction, not a convergence or generalization proof. New small CAL coarse pass because historical candidates were not retained.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);p.add_argument('--ds');p.add_argument('--arm');p.add_argument('--seed',type=int,default=1001);a=p.parse_args()
    if a.mode=='check':check()
    else:run(a.ds,a.arm,a.seed)
