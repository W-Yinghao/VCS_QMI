"""Fixed-budget continuation; inherited models/objectives, no outcome-based stopping."""
import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import types

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE/'handoff/r22_geometry_train_20261023'))
import train as T
R, A, torch = T.R, T.A, T.torch


def read(p): return json.loads(Path(p).read_text())


def verify():
    for p, h in read(HERE/'FREEZE.json')['files'].items():
        assert R.sha(BASE/p) == h, p
    return R.sha(HERE/'FREEZE.json')


def lr_at(next_update):
    assert 1 <= next_update <= 30000
    return 3e-4 if next_update <= 15000 else (3e-5 if next_update <= 24000 else 3e-6)


def batch_sha(hb):
    h = hashlib.sha256(json.dumps([hb.step, hb.cores]).encode())
    for x in (hb.y0, hb.x0, hb.k, hb.flip, hb.yx, hb.perm):
        h.update(x.tobytes())
    return h.hexdigest()


def config(seed):
    return A.TrainConfig(method='MLP_perm1', seed=seed, encoder_lr=3e-4, head_lr=3e-4,
                         n_sc=16, n_pos=128, crop=160, grad_spot_every=100)


def source(ds, arm, seed):
    s = read(HERE/'SOURCE_LOCK.json')[f'{ds}_{arm}_{seed}']
    for kind in ('checkpoint', 'metadata', 'completion', 'source_train'):
        assert R.sha(s[kind]) == s[kind+'_sha256'], kind
    assert read(s['completion'])['status'] == 'COMPLETE'
    return s


def context(ds, seed, device, images=True):
    u = R.load_unit(BASE/'handoff/r22_c3_codex_20261023/units'/f'r22_pmi3_{ds}_MLPp1_s{seed}.json')
    guard, H = R.make_guard(u)
    fit = A.fit_cores(guard, 1., seed) if H is None else H.fit_windows(guard)
    imgs = {}
    if images:
        if H: guard.verify_data_lock({'FIT'})
        else:
            for row in read(u['data_lock']['path'])['files']:
                if row['role'] == 'FIT': assert R.sha(row['path']) == row['sha256']
        imgs = (A.load_learned_inputs if H is None else H.load_learned_inputs)(
            guard, fit, {'FIT'}, 'R22 fixed-budget continuation FIT', device)
    return u, guard, H, fit, imgs


def restore(ds, arm, seed, fit, imgs, device):
    s = source(ds, arm, seed)
    t = T.GeometryTrainer(config(seed), imgs, fit, device, arm)
    R.CP.load(t, s['checkpoint'], s['identity'])
    assert t.n_steps == t.sampler.step == 3000
    assert all(g['lr'] == 3e-4 for g in t.opt.param_groups)
    assert all(float(v['step']) == 3000 for v in t.opt.state.values())
    return t, s


def step(t):
    lr = lr_at(t.n_steps+1)
    for g in t.opt.param_groups: g['lr'] = lr
    b = t.draw()
    digest = batch_sha(b[0])
    start = time.perf_counter()
    row = t.step(batch=b)
    if t.device.type == 'cuda': torch.cuda.synchronize()
    row.update(lr=lr, batch_sha256=digest, synchronized_step_seconds=time.perf_counter()-start)
    assert row['step'] == t.n_steps and math.isfinite(row['loss']) and math.isfinite(row['grad_norm_preclip'])
    return row


def tree_equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a == b


def cpu_check():
    freeze = verify(); torch.set_num_threads(2)
    assert [lr_at(k) for k in (3001,15000,15001,24000,24001,30000)] == [3e-4,3e-4,3e-5,3e-5,3e-6,3e-6]
    checks = []
    for ds in ('bf','epfl'):
        for seed in (1001,1002,1003):
            _,_,_,fit,_ = context(ds,seed,'cpu',False)
            signatures = []
            for arm in T.ARMS:
                t,s = restore(ds,arm,seed,fit,{},'cpu')
                signatures.append([batch_sha(t.sampler.draw()) for _ in range(3)])
            assert all(x == signatures[0] for x in signatures)
            checks.append(dict(ds=ds,seed=seed,all4_sampler_steps3001_3003_equal=signatures[0]))
    # A real encoder/optimizer continuation on deterministic synthetic crops,
    # including saved Adam moments, local sampler and global torch RNG.
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
        cfg = A.TrainConfig(method='MLP_perm1',seed=123,n_sc=2,n_pos=7,crop=40,grad_spot_every=0)
        torch.manual_seed(17)
        imgs = {'synthetic':(torch.randn(3,A.P.SIZE,A.P.SIZE),torch.randn(1,A.P.SIZE,A.P.SIZE))}
        for arm in T.ARMS:
            t=T.GeometryTrainer(cfg,imgs,['synthetic'],'cpu',arm)
            step(t); p=Path(tmp)/(arm+'.pt'); ident={'arm':arm,'check':'CPU'}
            R.CP.save(t,p,ident); r1=step(t)
            expected_model=A.state_hash(*t.modules()); expected_opt=t.opt.state_dict()
            restored=T.GeometryTrainer(cfg,imgs,['synthetic'],'cpu',arm)
            R.CP.load(restored,p,ident); r2=step(restored)
            assert r1['batch_sha256']==r2['batch_sha256'] and r1['loss']==r2['loss']
            assert A.state_hash(*restored.modules())==expected_model and tree_equal(expected_opt,restored.opt.state_dict())
    R.write(HERE/'CPU_CHECK.json',dict(status='PASS',freeze_sha256=freeze,checks=checks,
        synthetic_full_state_continuation='all4 objectives bitwise equal',lr_boundaries='PASS'))
    print('LONG_CPU PASS',flush=True)


def gpu_gate(ds):
    freeze=verify();assert read(HERE/'CPU_CHECK.json')['freeze_sha256']==freeze
    assert torch.cuda.is_available();torch.set_num_threads(4);started=time.time()
    out=HERE/'gates'/ds;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists()
    u,guard,H,fit,imgs=context(ds,1001,'cuda');checks=[]
    for arm in T.ARMS:
        t,s=restore(ds,arm,1001,fit,imgs,'cuda')
        rows=[step(t) for _ in range(3)]
        # Saving and reloading full state must reproduce the exact stored values;
        # no nondeterministic GPU training re-execution equality is assumed.
        path=out/(arm+'.pt');ident={'gate':ds,'arm':arm,'freeze_sha256':freeze}
        R.CP.save(t,path,ident)
        model=A.state_hash(*t.modules());opt=t.opt.state_dict()
        rng=R.CP.rng_state(t); cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
        t2=T.GeometryTrainer(config(1001),imgs,fit,'cuda',arm);R.CP.load(t2,path,ident)
        assert A.state_hash(*t2.modules())==model and tree_equal(opt,t2.opt.state_dict())
        assert tree_equal(rng,R.CP.rng_state(t2)) and torch.equal(cpu_rng,torch.get_rng_state())
        assert tree_equal(gpu_rng,torch.cuda.get_rng_state_all())
        with R.CP.evaluation_state(t2):
            core=sorted(guard.included('CAL'))[0]
            session=R.make_session(u,t2,guard,H,'CAL',[core],'cuda')
            energy,_=session.energy(core,0);inner=R.PE.inner(energy)
            inner.j_critic=types.MethodType(T.cosine_energy,inner);inner.geometry_readout='COS'
            truth=session.perturbations[core][0]
            p=torch.tensor([math.radians(truth[0])*A.Q.RHO,*truth[1:]],device='cuda',requires_grad=True)
            value=energy(p);grad=torch.autograd.grad(value,p)[0]
            assert torch.isfinite(value) and torch.isfinite(grad).all()
        checks.append(dict(arm=arm,source=s,steps=rows,optimizer_and_rng_roundtrip=True,pose_gradient_finite=True))
        del t,t2,energy,session
    assert all([r['batch_sha256'] for r in c['steps']]==[r['batch_sha256'] for r in checks[0]['steps']] for c in checks)
    assert {r['role'] for r in guard.reads}<={'FIT','CAL'}
    R.write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=freeze,ds=ds,gate=True,seed=1001,
        checks=checks,seconds=time.time()-started,environment=R.environment(),job=os.environ['SLURM_JOB_ID'],
        roles_read=sorted({r['role'] for r in guard.reads})))
    print('LONG_GATE PASS',ds,flush=True)


def run(ds,arm,seed):
    freeze=verify();cpu=read(HERE/'CPU_CHECK.json');assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze
    for d in ('bf','epfl'):
        g=read(HERE/'gates'/d/'COMPLETE.json');assert g['status']=='COMPLETE' and g['freeze_sha256']==freeze
    assert torch.cuda.is_available();torch.set_num_threads(4);started=time.time()
    name=f'{ds}_{arm}_{seed}';out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    ident=dict(ds=ds,arm=arm,seed=seed,freeze_sha256=freeze)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COMPLETE.json').exists():assert read(out/'COMPLETE.json')['identity']==ident;return
        assert not list(out.glob('ATTEMPT_*.json')), 'Review incomplete attempt before retry; no silent overwrite.'
        job=os.environ['SLURM_JOB_ID'];attempt=out/f'ATTEMPT_{job}.json'
        R.write(attempt,dict(status='RUNNING',identity=ident,start=started,environment=R.environment()))
        u,guard,H,fit,imgs=context(ds,seed,'cuda');t,s=restore(ds,arm,seed,fit,imgs,'cuda')
        short=out/'dev_3000';short.mkdir(exist_ok=True)
        dev_short=T.evaluate(t,u,guard,H,'DEV',short,('COS',))
        cal={}
        torch.cuda.reset_peak_memory_stats()
        for stop in (10000,20000,30000):
            with (out/'TRAIN.jsonl').open('a') as trace:
                while t.n_steps<stop:
                    row=step(t);trace.write(json.dumps(R.plain(row),allow_nan=False)+'\n')
                    if t.n_steps%250==0:trace.flush();print(name,'step',t.n_steps,flush=True)
            R.CP.save(t,out/'checkpoints'/f'step_{stop}.pt',ident)
            d=out/f'cal_{stop}';d.mkdir(exist_ok=True)
            cal[stop]=T.evaluate(t,u,guard,H,'CAL',d,('COS',))
        long=out/'dev_30000';long.mkdir(exist_ok=True)
        dev_long=T.evaluate(t,u,guard,H,'DEV',long,('COS',))
        rows=R.rows(out/'TRAIN.jsonl');assert len(rows)==27000 and [r['step'] for r in rows]==list(range(3001,30001))
        assert all(r['lr']==lr_at(r['step']) for r in rows)
        assert t.n_steps==30000 and {r['role'] for r in guard.reads}<={'FIT','CAL','DEV'}
        for o in (short,long):assert len(R.rows(o/'DEV.jsonl'))==10*len(guard.included('DEV'))
        result=dict(status='COMPLETE',identity=ident,freeze_sha256=freeze,ds=ds,arm=arm,seed=seed,source=s,
            job=job,seconds=time.time()-started,environment=R.environment(),gpu=torch.cuda.get_device_name(),
            final_steps=t.n_steps,new_updates=len(rows),dev_short=dev_short,dev_long=dev_long,cal=cal,
            train_sha256=R.sha(out/'TRAIN.jsonl'),dev_short_sha256=R.sha(short/'DEV.jsonl'),dev_long_sha256=R.sha(long/'DEV.jsonl'),
            final_checkpoint_sha256=R.sha(out/'checkpoints/step_30000.pt'),
            synchronized_training_seconds=sum(r['synchronized_step_seconds'] for r in rows),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),roles_read=sorted({r['role'] for r in guard.reads}),
            note='Fixed budget developmental evidence; inherited prefix and LR decay disclosed; no checkpoint selection')
        R.write(out/'COMPLETE.json',result)
        R.write(attempt,dict(status='COMPLETE',identity=ident,seconds=time.time()-started,job=job))
        print('LONG_COMPLETE',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--ds',choices=['bf','epfl'])
    p.add_argument('--arm',choices=T.ARMS);p.add_argument('--seed',type=int,default=1001);a=p.parse_args()
    if a.mode=='check':cpu_check()
    elif a.mode=='gate':gpu_gate(a.ds)
    else:run(a.ds,a.arm,a.seed)
