"""Two missing seeds, original A1 update function, fixed 3000 -> 30000 budget."""
import argparse
import copy
import math
import os
from pathlib import Path
import time
import types
from common import (HERE, OUT, ARMS, SEEDS, DATASET, read, write, sha, verify,
                    units, source_check, original, storage, cpu_pass, gate_pass)

O = original()
L, G = O.L, O.G
T, R, A, torch = L.T, L.R, L.A, L.torch

def environment(key):
    now = R.environment(); ref = units()[key]['environment']
    fields = ('host', 'python', 'torch', 'numpy', 'cuda_runtime', 'cudnn', 'torch_threads', 'gpu_name', 'precision')
    for field in fields: assert now[field] == ref[field], (key, field, now[field], ref[field])
    return now

def start(out, mode):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'COMPLETE.json').exists() and not list(out.glob('ATTEMPT_*.json')), 'Never overwrite an attempt'
    write(out/f'ATTEMPT_{os.environ["SLURM_JOB_ID"]}.json', dict(status='STARTED', mode=mode, unix=time.time()))

def cpu():
    f = verify(); started = time.time(); torch.set_num_threads(2); traces = []; checks = {}
    assert [L.lr_at(x) for x in (3001,15000,15001,24000,24001,30000)] == [3e-4,3e-4,3e-5,3e-5,3e-6,3e-6]
    for ds in DATASET:
        for seed in SEEDS:
            _, guard, _, fit, _ = L.context(ds, seed, 'cpu', images=False, gate_scope=True)
            sequences = []
            for arm in ARMS:
                key = f'{ds}_{arm}_{seed}'; row = source_check(key)
                t, s = L.restore(ds, arm, seed, fit, {}, 'cpu')
                assert s == read(row['a0_completion'])['source']
                ref = iter(R.rows(Path(row['a0_completion']).parent/'TRAIN.jsonl')); sequence = []
                for _ in range(5):
                    hb = t.sampler.draw(); np_rng = copy.deepcopy(R.np.random.get_state()); rng = torch.get_rng_state().clone()
                    plan = G.plan_batch(hb, dataset=DATASET[ds], seed=seed)
                    assert plan == G.plan_batch(hb, dataset=DATASET[ds], seed=seed)
                    assert torch.equal(rng, torch.get_rng_state()) and L.tree_equal(np_rng[0], R.np.random.get_state()[0])
                    now = R.np.random.get_state(); assert R.np.array_equal(np_rng[1], now[1]) and np_rng[2:] == now[2:]
                    assert sum(x['requested'] for x in plan) == 8
                    assert all(not x['active'] or G.support_valid(hb.y0[x['row']], hb.x0[x['row']],160,x['theta_deg'],x['translation_yx_px']) for x in plan)
                    b = next(ref); assert L.batch_sha(hb) == b['batch_sha256'] and hb.step == b['step']
                    sequence.append(dict(step=hb.step, batch_sha256=L.batch_sha(hb), plan=plan))
                sequences.append(sequence); checks[key] = dict(source=s, first_five=sequence)
                del t
            assert all(x == sequences[0] for x in sequences)
            assert {x['role'] for x in guard.reads} <= {'CAL'}; traces.extend(guard.reads)
    mov = torch.arange(512*512, dtype=torch.float32).reshape(1,512,512)/1000
    for k in range(4):
        for flip in (False, True):
            ref = torch.rot90(mov[:,100:260,200:360],k,(-2,-1))
            if flip: ref = torch.flip(ref,[-1])
            got = G.pullback_crop(A.Q,mov,y0=100,x0=200,crop=160,theta_deg=0.,translation_yx_px=[0.,0.],rot90=k,flip=flip)
            assert torch.allclose(got,ref,atol=1e-4,rtol=1e-6)
    write(HERE/'checks/CPU_TRACE.json', traces)
    write(HERE/'checks/CPU.json', dict(status='PASS',freeze_sha256=f,units=checks,seconds=time.time()-started,
        job=os.environ['SLURM_JOB_ID'],trace_sha256=sha(HERE/'checks/CPU_TRACE.json'),
        checks=['12 actual restored prefixes and original A0 batch streams', 'same five geometry plans across three objectives per dataset/seed',
                'private geometry RNG and support', 'identity pullback and original rotation/flip', 'fixed LR boundaries']))
    print('A1_COMPLETION_CPU PASS', flush=True)

def gate(group):
    f = verify(); cpu_pass(); storage(); torch.set_num_threads(4); started = time.time()
    out = OUT/'gates'/group; start(out, 'gate'); checks = []; traces = []
    keys = sorted(k for k, v in units().items() if v['gate_group'] == group); assert keys
    for key in keys:
        rec = source_check(key); env = environment(key); ds, arm, seed = rec['ds'], rec['arm'], rec['seed']
        u, guard, H, fit, imgs = L.context(ds, seed, 'cuda', gate_scope=True)
        t, source = L.restore(ds, arm, seed, fit, imgs, 'cuda'); before = A.state_hash(*t.modules())
        adapter = G.GeometryMatchedBatches(t,G.NativeMovingFactory(A,guard,DATASET[ds],'cuda'),dataset=DATASET[ds])
        rows = [O.step(t,adapter) for _ in range(3)]; assert sum(x['geometry']['active'] for x in rows) > 0
        ref = R.rows(Path(rec['a0_completion']).parent/'TRAIN.jsonl')[:3]
        assert [(x['step'],x['batch_sha256'],x['lr']) for x in rows] == [(x['step'],x['batch_sha256'],x['lr']) for x in ref]
        ident = dict(key=key,mode='gate',freeze_sha256=f); path = out/(key+'.pt'); R.CP.save(t,path,ident)
        model = A.state_hash(*t.modules()); assert model != before
        opt = copy.deepcopy(t.opt.state_dict()); rng = R.CP.rng_state(t)
        t2 = T.GeometryTrainer(L.config(seed),imgs,fit,'cuda',arm); R.CP.load(t2,path,ident)
        assert A.state_hash(*t2.modules()) == model and L.tree_equal(opt,t2.opt.state_dict()) and L.tree_equal(rng,R.CP.rng_state(t2))
        with R.CP.evaluation_state(t2):
            core = sorted(guard.included('CAL'))[0]; session = R.make_session(u,t2,guard,H,'CAL',[core],'cuda'); energy,_ = session.energy(core,0)
            inner = R.PE.inner(energy); inner.j_critic = types.MethodType(T.cosine_energy,inner); inner.geometry_readout = 'COS'
            truth = session.perturbations[core][0]; p = torch.tensor([math.radians(truth[0])*A.Q.RHO,*truth[1:]],device='cuda',requires_grad=True)
            value = energy(p); grad = torch.autograd.grad(value,p)[0]; assert torch.isfinite(value) and torch.isfinite(grad).all()
        assert {r['role'] for r in guard.reads} <= {'FIT','CAL'}; traces.extend(guard.reads)
        checks.append(dict(key=key,source=source,steps=rows,model_changed=True,optimizer_rng_roundtrip=True,
            environment=env,environment_matched=True,baseline_batches_matched=True,cal_gradient_finite=True,checkpoint_sha256=sha(path)))
        del t,t2,adapter,energy,session,inner,imgs,p,value,grad; torch.cuda.empty_cache()
    write(out/'INPUT_TRACE.json',traces)
    write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=f,group=group,checks=checks,seconds=time.time()-started,
        cpu_sha256=sha(HERE/'checks/CPU.json'),trace_sha256=sha(out/'INPUT_TRACE.json'),roles_read=sorted({x['role'] for x in traces}),job=os.environ['SLURM_JOB_ID']))
    print('A1_COMPLETION_GATE PASS',group,flush=True)

def run(key):
    f = verify(); cpu_pass(); storage()
    for group in sorted({v['gate_group'] for v in units().values()}): gate_pass(group)
    torch.set_num_threads(4); started = time.time(); rec = source_check(key); env = environment(key)
    ds, arm, seed = rec['ds'], rec['arm'], rec['seed']; out = OUT/'runs'/key; start(out, 'full')
    u, guard, H, fit, imgs = L.context(ds,seed,'cuda'); t, source = L.restore(ds,arm,seed,fit,imgs,'cuda')
    assert read(rec['a0_completion'])['source'] == source
    adapter = G.GeometryMatchedBatches(t,G.NativeMovingFactory(A,guard,DATASET[ds],'cuda'),dataset=DATASET[ds])
    ident = dict(ds=ds,arm=arm,seed=seed,adaptation='A1_from_step3001',freeze_sha256=f)
    baseline = iter(R.rows(Path(rec['a0_completion']).parent/'TRAIN.jsonl')); curves = {}; active = requested = 0
    for stop in (10000,20000,30000):
        while t.n_steps < stop:
            row = O.step(t,adapter); ref = next(baseline)
            assert (row['step'],row['batch_sha256'],row['lr']) == (ref['step'],ref['batch_sha256'],ref['lr'])
            active += row['geometry']['active']; requested += row['geometry']['requested']; R.append(out/'TRAIN.jsonl',row)
            if t.n_steps % 250 == 0: print(key,'A1',t.n_steps,'active',active,flush=True)
        R.CP.save(t,out/'checkpoints'/f'step_{stop}.pt',ident)
        p = out/f'cal_{stop}'; p.mkdir(exist_ok=True); curves[str(stop)] = T.evaluate(t,u,guard,H,'CAL',p,('COS',))
    assert next(baseline,None) is None
    p = out/'dev_30000'; p.mkdir(exist_ok=True); dev = T.evaluate(t,u,guard,H,'DEV',p,('COS',))
    rows = R.rows(out/'TRAIN.jsonl'); assert len(rows) == 27000 and [r['step'] for r in rows] == list(range(3001,30001))
    assert all(x['lr'] == L.lr_at(x['step']) for x in rows)
    assert {x['role'] for x in guard.reads} <= {'FIT','CAL','DEV'}
    assert len(R.rows(p/'DEV.jsonl')) == 10*len(guard.included('DEV'))
    write(out/'INPUT_TRACE.json',guard.reads)
    write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=f,key=key,ds=ds,arm=arm,seed=seed,identity=ident,
        source=source,baseline=rec,environment=env,environment_matched=True,final_steps=30000,new_updates=27000,dev=dev,cal=curves,
        active=active,requested=requested,train_sha256=sha(out/'TRAIN.jsonl'),dev_sha256=sha(p/'DEV.jsonl'),
        trace_sha256=sha(out/'INPUT_TRACE.json'),checkpoint_sha256=sha(out/'checkpoints/step_30000.pt'),
        seconds=time.time()-started,job=os.environ['SLURM_JOB_ID'],roles_read=sorted({x['role'] for x in guard.reads}),
        interpretation='Two additional matched seeds. Original shared3000 prefix, fixed27000 continuation, original A1 update/geometry/COS. Same A0 node/GPU/driver/software/precision; cross-run nondeterministic evaluation remains. Historical1001 reported separately.'))
    print('A1_COMPLETION_FULL PASS',key,flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode',choices=['check','gate','run']); p.add_argument('--group'); p.add_argument('--key'); a=p.parse_args()
    if a.mode == 'check': cpu()
    elif a.mode == 'gate': gate(a.group)
    else: run(a.key)
