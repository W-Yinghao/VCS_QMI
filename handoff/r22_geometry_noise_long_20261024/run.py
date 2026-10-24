"""Frozen continuation of both existing correspondence-noise series to 30000 updates."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
import types

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = load('inherited_clean_long', BASE/'handoff/r22_geometry_long_v2_20261024/run.py')
N = load('inherited_correspondence_noise', BASE/'handoff/r22_geometry_noise_20261023/train.py')
R, A, torch = L.R, L.A, L.torch
ARMS = N.ARMS
RATES = (.2, .4)


def read(p): return json.loads(Path(p).read_text())


def verify():
    for p, h in read(HERE/'FREEZE.json')['files'].items():
        assert R.sha(BASE/p) == h, p
    return R.sha(HERE/'FREEZE.json')


def source(ds, arm, seed, rate):
    s = read(HERE/'SOURCE_LOCK.json')[f'{ds}_{arm}_{seed}_n{round(rate*100)}']
    for k in ('checkpoint', 'metadata', 'completion', 'source_train'):
        assert R.sha(s[k]) == s[k+'_sha256'], k
    c = read(s['completion'])
    assert c['status'] == 'COMPLETE' and c['noise'] == rate
    assert c['train_rows_sha256'] == s['source_train_sha256']
    return s


def trainer(cfg, imgs, fit, device, arm, rate):
    # Original class snapshots NOISE in __init__; no change to its step or corruption.
    N.NOISE = rate
    return N.GeometryTrainer(cfg, imgs, fit, device, arm)


def restore(ds, arm, seed, rate, fit, imgs, device):
    s = source(ds, arm, seed, rate)
    t = trainer(L.config(seed), imgs, fit, device, arm, rate)
    R.CP.load(t, s['checkpoint'], s['identity'])
    assert t.n_steps == t.sampler.step == 3000 and t.noise == rate
    assert all(g['lr'] == 3e-4 for g in t.opt.param_groups)
    assert all(float(v['step']) == 3000 for v in t.opt.state.values())
    return t, s


def step(t):
    row = L.step(t)
    noise = row['correspondence_noise']
    assert noise['rate_requested'] == t.noise
    expected = max(2, round(t.noise*t.cfg.n_pos))/t.cfg.n_pos
    assert abs(noise['rate_realized']-expected) < 1e-7
    return row


def cpu_check():
    freeze = verify(); torch.set_num_threads(2)
    assert [L.lr_at(k) for k in (3001,15000,15001,24000,24001,30000)] == [3e-4,3e-4,3e-5,3e-5,3e-6,3e-6]
    checks = []
    for ds in ('bf', 'epfl'):
        for seed in (1001,1002,1003):
            _,guard,_,fit,_ = L.context(ds,seed,'cpu',False,gate_scope=True)
            assert {r['role'] for r in guard.reads} <= {'CAL'}
            clean,_ = L.restore(ds,'vcs_exact',seed,fit,{},'cpu')
            common = [L.batch_sha(clean.sampler.draw()) for _ in range(5)]
            prefix_batches = None
            for rate in RATES:
                prefix_assignments = None
                for arm in ARMS:
                    t,s = restore(ds,arm,seed,rate,fit,{},'cpu')
                    assert [L.batch_sha(t.sampler.draw()) for _ in range(5)] == common
                    rows = R.rows(s['source_train'])
                    assert [r['step'] for r in rows] == list(range(1,3001))
                    assert all(math.isfinite(r['loss']) for r in rows)
                    assert all(r['correspondence_noise']['rate_requested']==rate for r in rows)
                    batches = [r['correspondence_noise']['batch_sha256'] for r in rows]
                    assignments = [r['correspondence_noise']['assignment_sha256'] for r in rows]
                    if prefix_batches is None: prefix_batches = batches
                    else: assert batches == prefix_batches
                    if prefix_assignments is None: prefix_assignments = assignments
                    else: assert assignments == prefix_assignments
                checks.append(dict(ds=ds,seed=seed,rate=rate,prefix_3000_inputs_and_corruption_matched=True,
                                   restored_draws_match_clean3001_3005=common))
    # Verify marginal preservation at continuation steps, without changing inherited implementation.
    v = torch.arange(2*128*3,dtype=torch.float64).reshape(2,128,3).requires_grad_()
    for rate in RATES:
        for at in (3001,15001,24001,30000):
            altered,assignment=N.corrupt_pairs(v,1001,at,rate)
            assert torch.equal(assignment.sort(-1).values,torch.arange(128).repeat(2,1))
            assert ((assignment!=torch.arange(128)[None]).sum(1)==round(rate*128)).all()
            assert torch.equal(altered.sort(1).values,v.sort(1).values)
            assert torch.equal(N.corrupt_pairs(v,1001,at,rate)[1],assignment)
            assert torch.equal(torch.autograd.grad(altered.sum(),v)[0],torch.ones_like(v))
    # Full-state synthetic optimizer/RNG replay for each arm and both rates.
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
        cfg=A.TrainConfig(method='MLP_perm1',seed=123,n_sc=2,n_pos=7,crop=40,grad_spot_every=0)
        torch.manual_seed(17)
        imgs={'synthetic':(torch.randn(3,A.P.SIZE,A.P.SIZE),torch.randn(1,A.P.SIZE,A.P.SIZE))}
        for rate in RATES:
            for arm in ARMS:
                t=trainer(cfg,imgs,['synthetic'],'cpu',arm,rate);step(t)
                p=Path(tmp)/(arm+str(rate)+'.pt');ident={'check':'CPU','arm':arm,'rate':rate}
                R.CP.save(t,p,ident);r1=step(t)
                expected=A.state_hash(*t.modules());opt=t.opt.state_dict()
                restored=trainer(cfg,imgs,['synthetic'],'cpu',arm,rate);R.CP.load(restored,p,ident);r2=step(restored)
                assert r1['loss']==r2['loss'] and r1['batch_sha256']==r2['batch_sha256']
                assert r1['correspondence_noise']==r2['correspondence_noise']
                assert A.state_hash(*restored.modules())==expected and L.tree_equal(opt,restored.opt.state_dict())
    R.write(HERE/'CPU_CHECK.json',dict(status='PASS',freeze_sha256=freeze,checks=checks,
        full_state_replay='all4 objectives x2 noise rates bitwise equal',
        continuation_corruption='same inherited function; fixed count, bijection, isolated seed, marginal preservation PASS',
        lr_boundaries='PASS',source_checkpoints=48,roles='FIT/CAL metadata; no DEV evaluation or TEST'))
    print('NOISE_LONG_CPU PASS',flush=True)


def gpu_gate(ds):
    freeze=verify();assert read(HERE/'CPU_CHECK.json')['freeze_sha256']==freeze
    assert torch.cuda.is_available();torch.set_num_threads(4);started=time.time()
    out=HERE/'gates'/ds;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists()
    u,guard,H,fit,imgs=L.context(ds,1001,'cuda',gate_scope=True);checks=[]
    for rate in RATES:
        for arm in ARMS:
            t,s=restore(ds,arm,1001,rate,fit,imgs,'cuda');rr=[step(t) for _ in range(3)]
            path=out/f'{arm}_n{round(rate*100)}.pt';ident={'gate':ds,'arm':arm,'rate':rate,'freeze_sha256':freeze}
            R.CP.save(t,path,ident)
            model=A.state_hash(*t.modules());opt=t.opt.state_dict();rng=R.CP.rng_state(t)
            cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
            t2=trainer(L.config(1001),imgs,fit,'cuda',arm,rate);R.CP.load(t2,path,ident)
            assert A.state_hash(*t2.modules())==model and L.tree_equal(opt,t2.opt.state_dict())
            assert L.tree_equal(rng,R.CP.rng_state(t2)) and torch.equal(cpu_rng,torch.get_rng_state())
            assert L.tree_equal(gpu_rng,torch.cuda.get_rng_state_all())
            with R.CP.evaluation_state(t2):
                core=sorted(guard.included('CAL'))[0];session=R.make_session(u,t2,guard,H,'CAL',[core],'cuda')
                energy,_=session.energy(core,0);inner=R.PE.inner(energy)
                inner.j_critic=types.MethodType(L.T.cosine_energy,inner);inner.geometry_readout='COS'
                truth=session.perturbations[core][0]
                pos=torch.tensor([math.radians(truth[0])*A.Q.RHO,*truth[1:]],device='cuda',requires_grad=True)
                val=energy(pos);grad=torch.autograd.grad(val,pos)[0]
                assert torch.isfinite(val) and torch.isfinite(grad).all()
            checks.append(dict(arm=arm,noise=rate,source=s,steps=rr,optimizer_and_rng_roundtrip=True,pose_gradient_finite=True))
            del t,t2,energy,session
    assert all([r['batch_sha256'] for r in c['steps']]==[r['batch_sha256'] for r in checks[0]['steps']] for c in checks)
    for rate in RATES:
        cc=[c for c in checks if c['noise']==rate]
        assert all([r['correspondence_noise'] for r in c['steps']]==[r['correspondence_noise'] for r in cc[0]['steps']] for c in cc)
    R.write(out/'INPUT_TRACE.json',guard.reads)
    assert {r['role'] for r in guard.reads}<={'FIT','CAL'}
    R.write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=freeze,ds=ds,gate=True,seed=1001,
        checks=checks,seconds=time.time()-started,environment=R.environment(),job=os.environ['SLURM_JOB_ID'],
        input_trace_sha256=R.sha(out/'INPUT_TRACE.json'),roles_read=sorted({r['role'] for r in guard.reads})))
    print('NOISE_LONG_GATE PASS',ds,flush=True)


def run(ds,arm,seed,rate):
    freeze=verify();cpu=read(HERE/'CPU_CHECK.json');assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze
    for d in ('bf','epfl'):
        g=read(HERE/'gates'/d/'COMPLETE.json');assert g['status']=='COMPLETE' and g['freeze_sha256']==freeze
    assert torch.cuda.is_available();torch.set_num_threads(4);started=time.time()
    name=f'{ds}_{arm}_{seed}_n{round(rate*100)}';out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    ident=dict(ds=ds,arm=arm,seed=seed,noise=rate,freeze_sha256=freeze)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COMPLETE.json').exists():assert read(out/'COMPLETE.json')['identity']==ident;return
        assert not list(out.glob('ATTEMPT_*.json')), 'Review incomplete attempt before retry; no silent overwrite.'
        job=os.environ['SLURM_JOB_ID'];attempt=out/f'ATTEMPT_{job}.json'
        R.write(attempt,dict(status='RUNNING',identity=ident,start=started,environment=R.environment()))
        u,guard,H,fit,imgs=L.context(ds,seed,'cuda');t,s=restore(ds,arm,seed,rate,fit,imgs,'cuda')
        short=out/'dev_3000';short.mkdir(exist_ok=True)
        dev_short=L.T.evaluate(t,u,guard,H,'DEV',short,('COS',));cal={}
        torch.cuda.reset_peak_memory_stats()
        for stop in (10000,20000,30000):
            with (out/'TRAIN.jsonl').open('a') as trace:
                while t.n_steps<stop:
                    row=step(t);trace.write(json.dumps(R.plain(row),allow_nan=False)+'\n')
                    if t.n_steps%250==0:trace.flush();print(name,'step',t.n_steps,flush=True)
            R.CP.save(t,out/'checkpoints'/f'step_{stop}.pt',ident)
            d=out/f'cal_{stop}';d.mkdir(exist_ok=True)
            cal[stop]=L.T.evaluate(t,u,guard,H,'CAL',d,('COS',))
        long=out/'dev_30000';long.mkdir(exist_ok=True)
        dev_long=L.T.evaluate(t,u,guard,H,'DEV',long,('COS',))
        rr=R.rows(out/'TRAIN.jsonl');assert len(rr)==27000 and [r['step'] for r in rr]==list(range(3001,30001))
        assert all(r['lr']==L.lr_at(r['step']) for r in rr)
        assert t.n_steps==30000 and {r['role'] for r in guard.reads}<={'FIT','CAL','DEV'}
        for o in (short,long):assert len(R.rows(o/'DEV.jsonl'))==10*len(guard.included('DEV'))
        R.write(out/'INPUT_TRACE.json',guard.reads)
        result=dict(status='COMPLETE',identity=ident,freeze_sha256=freeze,ds=ds,arm=arm,seed=seed,noise=rate,source=s,
            job=job,seconds=time.time()-started,environment=R.environment(),gpu=torch.cuda.get_device_name(),
            final_steps=t.n_steps,new_updates=len(rr),dev_short=dev_short,dev_long=dev_long,cal=cal,
            train_sha256=R.sha(out/'TRAIN.jsonl'),dev_short_sha256=R.sha(short/'DEV.jsonl'),dev_long_sha256=R.sha(long/'DEV.jsonl'),
            final_checkpoint_sha256=R.sha(out/'checkpoints/step_30000.pt'),input_trace_sha256=R.sha(out/'INPUT_TRACE.json'),
            synchronized_training_seconds=sum(r['synchronized_step_seconds'] for r in rr),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),roles_read=sorted({r['role'] for r in guard.reads}),
            note='Fixed continuation of original corrupted P; same empirical marginal Q. Controlled developmental mechanism, not real-world noise evidence.')
        R.write(out/'COMPLETE.json',result)
        R.write(attempt,dict(status='COMPLETE',identity=ident,seconds=time.time()-started,job=job))
        print('NOISE_LONG_COMPLETE',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--ds',choices=['bf','epfl'])
    p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=1001);p.add_argument('--noise',type=float,choices=RATES)
    a=p.parse_args()
    if a.mode=='check':cpu_check()
    elif a.mode=='gate':gpu_gate(a.ds)
    else:run(a.ds,a.arm,a.seed,a.noise)
