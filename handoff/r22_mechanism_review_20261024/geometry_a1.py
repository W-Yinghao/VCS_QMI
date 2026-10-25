"""Shared-cosine resampling adaptation with matched, already completed A0 controls."""
import argparse
import copy
import math
import time
from mechanism_common import BASE,HERE,read,write,sha,verify,geometry,load,check_pass,output,done
L=geometry();T,R,A,torch=L.T,L.R,L.A,L.torch
G=load('r21_geometry_adapter',BASE/'ssl_pilot_v10_r22c/v10/app_round_20261021/package/v10_round/geometry_fit.py')
ARMS=('vcs_exact','bce','nce')
DATASET={'bf':'BF_SHG','epfl':'EPFL_RGB_NIR'}

def check():
    f=verify();torch.set_num_threads(2)
    for ds in ('bf','epfl'):
        _,guard,H,fit,_=L.context(ds,1001,'cpu',images=False,gate_scope=True)
        sequences=[]
        for arm in ARMS:
            t,s=L.restore(ds,arm,1001,fit,{},'cpu');rows=[]
            for _ in range(5):
                hb=t.sampler.draw();rng=R.np.random.get_state();trng=torch.get_rng_state().clone()
                plan=G.plan_batch(hb,dataset=DATASET[ds],seed=1001)
                assert sum(x['requested'] for x in plan)==8
                assert plan==G.plan_batch(hb,dataset=DATASET[ds],seed=1001)
                assert torch.equal(trng,torch.get_rng_state()) and R.np.array_equal(rng[1],R.np.random.get_state()[1])
                assert all(not x['active'] or G.support_valid(hb.y0[x['row']],hb.x0[x['row']],160,x['theta_deg'],x['translation_yx_px']) for x in plan)
                rows.append((L.batch_sha(hb),plan))
            sequences.append(rows)
        assert all(x==sequences[0] for x in sequences)
    # Identity pullback must reproduce a crop, including original rot/flip.
    mov=torch.arange(512*512,dtype=torch.float32).reshape(1,512,512)/1000
    for k in range(4):
        for flip in (False,True):
            ref=torch.rot90(mov[:,100:260,200:360],k,(-2,-1))
            if flip:ref=torch.flip(ref,[-1])
            z=G.pullback_crop(A.Q,mov,y0=100,x0=200,crop=160,theta_deg=0.,translation_yx_px=[0.,0.],rot90=k,flip=flip)
            assert torch.allclose(z,ref,atol=1e-4,rtol=1e-6)
    write(HERE/'checks/a1.json',dict(status='PASS',freeze_sha256=f,checks=['six real source optimizers restored at step3000','all targets have identical five future batches and geometry plans','geometry stream does not consume training RNG','support constraints and identity crop/rotation/flip']))

def step(t,adapter):
    for g in t.opt.param_groups:g['lr']=L.lr_at(t.n_steps+1)
    b,meta=adapter.draw();start=time.perf_counter();row=t.step(batch=b);torch.cuda.synchronize()
    row.update(batch_sha256=L.batch_sha(b[0]),lr=L.lr_at(t.n_steps),geometry=meta,synchronized_step_seconds=time.perf_counter()-start)
    assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm_preclip'])
    return row

def gate(ds):
    verify();check_pass('a1');torch.set_num_threads(4);start=time.time();out=output(f'a1_gate_{ds}')
    u,guard,H,fit,imgs=L.context(ds,1001,'cuda',gate_scope=True);results=[]
    for arm in ARMS:
        t,s=L.restore(ds,arm,1001,fit,imgs,'cuda');factory=G.NativeMovingFactory(A,guard,DATASET[ds],'cuda');adapter=G.GeometryMatchedBatches(t,factory,dataset=DATASET[ds])
        rows=[step(t,adapter) for _ in range(3)]
        assert sum(x['geometry']['active'] for x in rows)>0
        # NativeMovingFactory itself verifies first real warp on every FIT core
        # against the pre-existing guarded registration input construction.
        ident={'gate':ds,'arm':arm,'freeze_sha256':sha(HERE/'FREEZE.json')};path=out/f'{arm}.pt';R.CP.save(t,path,ident)
        state=A.state_hash(*t.modules());opt=copy.deepcopy(t.opt.state_dict());t2=T.GeometryTrainer(L.config(1001),imgs,fit,'cuda',arm);R.CP.load(t2,path,ident)
        assert A.state_hash(*t2.modules())==state and L.tree_equal(opt,t2.opt.state_dict())
        with R.CP.evaluation_state(t2):
            core=sorted(guard.included('CAL'))[0];session=R.make_session(u,t2,guard,H,'CAL',[core],'cuda');energy,_=session.energy(core,0)
            import types
            inner=R.PE.inner(energy);inner.j_critic=types.MethodType(T.cosine_energy,inner);inner.geometry_readout='COS'
            truth=session.perturbations[core][0];p=torch.tensor([math.radians(truth[0])*A.Q.RHO,*truth[1:]],device='cuda',requires_grad=True)
            value=energy(p);grad=torch.autograd.grad(value,p)[0];assert torch.isfinite(value) and torch.isfinite(grad).all()
        results.append(dict(arm=arm,steps=rows,source=s,roundtrip=True,cal_gradient=True))
        del t,t2,factory,adapter,energy,session;torch.cuda.empty_cache()
    assert all([r['batch_sha256'] for r in x['steps']]==[r['batch_sha256'] for r in results[0]['steps']] for x in results)
    assert all([r['geometry'] for r in x['steps']]==[r['geometry'] for r in results[0]['steps']] for x in results)
    assert {r['role'] for r in guard.reads}<={'FIT','CAL'};write(out/'INPUT_TRACE.json',guard.reads)
    done(out,start,ds=ds,gate=True,results=results,input_trace_sha256=sha(out/'INPUT_TRACE.json'),roles_read=sorted({r['role'] for r in guard.reads}),environment=R.environment())

def run(ds,arm):
    verify();check_pass('a1')
    for d in ('bf','epfl'):
        g=read(HERE/'runs'/f'a1_gate_{d}'/'COMPLETE.json');assert g['status']=='COMPLETE' and g['freeze_sha256']==sha(HERE/'FREEZE.json')
    torch.set_num_threads(4);start=time.time();out=output(f'a1_{ds}_{arm}_1001')
    baseline=read(HERE/'SOURCE_LOCK.json')['geometry'][f'{ds}_{arm}_1001'];assert sha(baseline['completion'])==baseline['completion_sha256']
    u,guard,H,fit,imgs=L.context(ds,1001,'cuda');t,source=L.restore(ds,arm,1001,fit,imgs,'cuda')
    baseline_done=read(baseline['completion']);assert baseline_done['source']==source
    adapter=G.GeometryMatchedBatches(t,G.NativeMovingFactory(A,guard,DATASET[ds],'cuda'),dataset=DATASET[ds]);curves={};active=0;requested=0
    ident=dict(ds=ds,arm=arm,seed=1001,adaptation='A1_from_step3001',freeze_sha256=sha(HERE/'FREEZE.json'))
    # Compare to the already stored baseline input stream at every update.
    baseline_rows=iter(R.rows(PathLike(baseline['completion']).parent/'TRAIN.jsonl'))
    for stop in (10000,20000,30000):
        while t.n_steps<stop:
            r=step(t,adapter);ref=next(baseline_rows)
            assert r['step']==ref['step'] and r['batch_sha256']==ref['batch_sha256'] and r['lr']==ref['lr']
            active+=r['geometry']['active'];requested+=r['geometry']['requested']
            R.append(out/'TRAIN.jsonl',r)
            if t.n_steps%250==0:print(ds,arm,'A1',t.n_steps,'active',active,flush=True)
        R.CP.save(t,out/'checkpoints'/f'step_{stop}.pt',ident)
        p=out/f'cal_{stop}';p.mkdir(exist_ok=True);curves[str(stop)]=T.evaluate(t,u,guard,H,'CAL',p,('COS',))
    assert next(baseline_rows,None) is None
    p=out/'dev_30000';p.mkdir(exist_ok=True);dev=T.evaluate(t,u,guard,H,'DEV',p,('COS',))
    write(out/'INPUT_TRACE.json',guard.reads);assert {r['role'] for r in guard.reads}<={'FIT','CAL','DEV'}
    done(out,start,ds=ds,arm=arm,seed=1001,identity=ident,source=source,baseline=baseline,dev=dev,cal=curves,active=active,requested=requested,
         final_steps=30000,train_sha256=sha(out/'TRAIN.jsonl'),dev_sha256=sha(p/'DEV.jsonl'),environment=R.environment(),
         interpretation='A1 applied only to steps3001..30000; shared unadapted3000 prefix. Reuses fixed-budget A0, no extra repeat. Same objective/LR/sampler/COS endpoint; cross-run evaluation noise remains. Pilot1seed per dataset/target, not confirmation.')

from pathlib import Path as PathLike
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--ds');p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.mode=='check':check()
    elif a.mode=='gate':gate(a.ds)
    else:run(a.ds,a.arm)
