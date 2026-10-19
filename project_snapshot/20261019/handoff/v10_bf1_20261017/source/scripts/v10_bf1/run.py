"""Frozen BF1 full FIT -> checkpoints -> CAL -> selection -> final-seed DEV."""
from __future__ import annotations
import argparse
import contextlib
import dataclasses
import gc
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from common import (ROOT,REPO,FAMILIES,METHODS,Allocation,BudgetExceeded,append,candidates,digest,
                    freeze,identity,lines,lock,metrics,read,require_identity,selection_key,sha,
                    utc,verify_complete,write)
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
import torch.nn.functional as F
from v10 import adapter as A


def plain(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def runtime_check(device):
    assert os.environ.get('SLURM_JOB_ID'), 'Scientific execution requires Slurm'
    freeze()
    if A.code_hash()!='e9b4239450c4e2eb' or A.host_code_hash()!='6c6f094684fbc50a':
        raise RuntimeError('Accepted core/host identity changed')
    env=A.environment(device)
    p=read(ROOT/'PROTOCOL.json')['precision']
    if device=='cuda':
        if env.get('gpu_name')!='NVIDIA L40S':raise RuntimeError('BF1 budget is defined on L40S')
        for key in ('torch','cuda_runtime','cudnn','torch_threads'):
            if env[key]!=p['environment'][key]:raise RuntimeError('Environment mismatch: '+key)
        if A.precision_settings(env['precision'])!=p['settings']:
            raise RuntimeError('Precision settings differ from accepted T02')
        if env['precision']['env']!=p['env']:
            raise RuntimeError('Precision environment variables differ')
    return env


def check_data(roles):
    d=read(ROOT/'DATA_LOCK.json')
    if sha(d['r6_manifest']['path'])!=d['r6_manifest']['sha256']:
        raise RuntimeError('R6 role manifest changed')
    for r in d['files']:
        if r['role'] in roles and sha(r['path'])!=r['sha256']:
            raise RuntimeError('BF input changed: '+r['core'])


def config(ident):
    return A.TrainConfig(method=ident['method'],seed=ident['seed'],encoder_lr=ident['lr'],head_lr=ident['lr'],
        K=16,n_sc=16,n_pos=128,crop=160,m1=A.M1Config(max_iterations=ident['cap'] or 30))


def new_trainer(ident,guard,device,imgs=None):
    fit=A.fit_cores(guard,1.0,ident['seed'])
    assert len(fit)==32
    if imgs is None:imgs=A.load_learned_inputs(guard,fit,{'FIT'},'BF1 complete FIT training',device)
    return A.Trainer(config(ident),imgs,fit,device)


def save_checkpoint(t,path,ident):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'format':'BF1-resumable-pair-v1','identity':ident,'config':dataclasses.asdict(t.cfg),
        'step':t.n_steps,'modules':[m.state_dict() for m in t.modules()],
        'optimizer':t.opt.state_dict(),'sampler_state':t.sampler.g.bit_generator.state,
        'sampler_step':t.sampler.step,'extra_rng':t.g_extra.bit_generator.state,
        'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if t.device.type=='cuda' else [],
        'model_state_hash':A.state_hash(*t.modules())}
    tmp=path.with_name('.'+path.name+'.tmp')
    torch.save(payload,tmp);os.replace(tmp,path)
    meta={'identity':ident,'step':t.n_steps,'sha256':sha(path),'model_state_hash':payload['model_state_hash']}
    write(path.with_suffix('.json'),meta)
    return meta


def load_checkpoint(t,path,ident,restore_rng=True):
    path=Path(path);meta=read(path.with_suffix('.json'))
    if meta['identity']!=ident or sha(path)!=meta['sha256']:
        raise RuntimeError('Checkpoint identity or bytes mismatch: '+str(path))
    st=torch.load(path,map_location=t.device,weights_only=True)
    if st['identity']!=ident or st['config']!=dataclasses.asdict(t.cfg):
        raise RuntimeError('Checkpoint configuration mismatch')
    for m,s in zip(t.modules(),st['modules'],strict=True):m.load_state_dict(s,strict=True)
    t.opt.load_state_dict(st['optimizer']);t.n_steps=st['step'];t.sampler.step=st['sampler_step']
    t.sampler.g.bit_generator.state=st['sampler_state'];t.g_extra.bit_generator.state=st['extra_rng']
    if restore_rng:
        torch.set_rng_state(st['torch_rng'].cpu())
        if t.device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in st['cuda_rng']])
    if t.n_steps!=t.sampler.step or A.state_hash(*t.modules())!=st['model_state_hash']:
        raise RuntimeError('Restored checkpoint state mismatch')
    return meta


@contextlib.contextmanager
def evaluation_state(t):
    modules=t.modules();modes=[m.training for m in modules]
    flags=[[p.requires_grad for p in m.parameters()] for m in modules]
    rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all() if t.device.type=='cuda' else []
    before=A.state_hash(*modules)
    A.freeze_for_test(*modules)
    try:yield
    finally:
        if A.state_hash(*modules)!=before:raise RuntimeError('Evaluation altered a model or W')
        for m,mode,fs in zip(modules,modes,flags,strict=True):
            m.train(mode)
            for p,flag in zip(m.parameters(),fs,strict=True):p.requires_grad_(flag)
        torch.set_rng_state(rng)
        if t.device.type=='cuda':torch.cuda.set_rng_state_all(crng)


def evaluate(t,guard,role,cores,pert_indices,path,ident,checkpoint,allocation):
    """Complete, resumable records; the geometry truth is used only in this evaluator."""
    path=Path(path)
    expected={'unit':ident,'checkpoint_sha256':checkpoint['sha256'],'role':role,
              'cores':cores,'perturbations':list(pert_indices),'searcher':'unchanged R6-REG-P2'}
    idpath=path.with_suffix('.identity.json')
    if idpath.exists() and read(idpath)!=expected:raise RuntimeError('Evaluation reuse mismatch')
    write(idpath,expected)
    rows=lines(path);by_key={}
    for r in rows:
        key=(r['core'],r['perturbation'])
        if key in by_key or r['core'] not in cores or r['perturbation'] not in pert_indices:
            raise RuntimeError('Duplicate/unexpected registration')
        if r['checkpoint_sha256']!=checkpoint['sha256']:raise RuntimeError('Mixed checkpoints in evaluation')
        by_key[key]=r
    perts={}
    for c in cores:
        guard.log_rule(c,{role},'BF1 '+role+' frozen perturbation rule')
        perts[c]=A.Q.perturbations_p2(c)
    inputs=A.GuardedInputs(guard,{role},'BF1 '+role+' checkpoint evaluation',perts,t.device)
    with evaluation_state(t):
        for c in cores:
            ref=A.reference_descriptors(t.eU,inputs,c)
            for i in pert_indices:
                if (c,i) in by_key:continue
                allocation.check()
                rng=np.random.default_rng(A.Q.seed_of(f'R6-REG-P2|pos|{c}|{i}'))
                mov=inputs.moving(c,i)[1]
                if t.mf:
                    estimator='perm1' if t.cfg.method=='MF_perm1' else 'moments'
                    energy=A.MFRigidEnergy(ref,mov,rng,eV=t.eV,head=t.head,estimator=estimator)
                else:
                    energy=A.Q.RigidEnergy(torch,F,'critic',ref,mov,rng,eV=t.eV,critic=t.critic)
                r=plain(A.registration_record(energy,t.device,perts[c][i]))
                # Native optimiser may map a NaN grid to -inf; never accept that as a valid scientific row.
                for k in ('err_full','F_hat','F_true','theta_hat','seconds'):
                    if not math.isfinite(r[k]):raise RuntimeError('Non-finite registration: '+k)
                r.update(core=c,perturbation=i,role=role,checkpoint_sha256=checkpoint['sha256'])
                append(path,r);by_key[(c,i)]=r
                del energy
            # Keep only this core's image tensors resident during registration.
            inputs.ref.pop(c,None)
            for i in pert_indices:inputs.mov.pop((c,i),None)
    if len(by_key)!=len(cores)*len(pert_indices):raise RuntimeError('Registration set incomplete')
    result={'identity':expected,'status':'COMPLETE','metrics':metrics(list(by_key.values())),
            'rows_sha256':sha(path),'checkpoint_sha256':checkpoint['sha256']}
    write(path.with_suffix('.summary.json'),result)
    return result


def load_eval(path,ident,checkpoint,role,cores):
    p=Path(path)
    if not p.with_suffix('.summary.json').exists():return None
    r=read(p.with_suffix('.summary.json'))
    expected={'unit':ident,'checkpoint_sha256':checkpoint['sha256'],'role':role,
              'cores':cores,'perturbations':list(range(10)),'searcher':'unchanged R6-REG-P2'}
    if r['identity']!=expected or sha(p)!=r['rows_sha256']:
        raise RuntimeError('Stale evaluation summary')
    rows=lines(p)
    if len({(x['core'],x['perturbation']) for x in rows})!=len(cores)*10 or metrics(rows)!=r['metrics']:
        raise RuntimeError('Invalid registration coverage/metrics')
    return r


def run_unit(ident,directory,allocation,env):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with lock(directory/'RUN.lock'):
        old=verify_complete(directory,ident)
        if old:return old
        allocation.check()
        require_identity(directory,ident)
        write(directory/f"ENV_{os.environ['SLURM_JOB_ID']}.json",env)
        guard=A.DataGuard();check_data({'FIT','CAL'})
        cal=guard.included('CAL');assert len(cal)==8
        t=new_trainer(ident,guard,'cuda')
        write(directory/'CONFIG.json',dataclasses.asdict(t.cfg))
        checkpoints=[]
        existing=sorted((directory/'checkpoints').glob('step_*.json')) if (directory/'checkpoints').exists() else []
        latest_step=0
        if existing:
            latest=max(existing,key=lambda p:read(p)['step'])
            last=load_checkpoint(t,latest.with_suffix('.pt'),ident)
            latest_step=t.n_steps
            if latest_step not in range(250,3001,250):raise RuntimeError('Unexpected resume checkpoint')
            # All earlier evaluated checkpoints must be present, not silently skipped.
            for step in range(250,latest_step+1,250):
                cp=directory/'checkpoints'/f'step_{step:04d}.pt'
                meta=read(cp.with_suffix('.json'))
                if meta['identity']!=ident or sha(cp)!=meta['sha256']:raise RuntimeError('Prior checkpoint changed')
                evpath=directory/'CAL'/f'step_{step:04d}.jsonl'
                ev=load_eval(evpath,ident,meta,'CAL',cal)
                if ev is None:
                    load_checkpoint(t,cp,ident)
                    ev=evaluate(t,guard,'CAL',cal,range(10),evpath,ident,meta,allocation)
                checkpoints.append({'step':step,'metrics':ev['metrics'],'checkpoint':str(cp.relative_to(directory)),
                                    'checkpoint_sha256':meta['sha256']})
            load_checkpoint(t,latest.with_suffix('.pt'),ident)
        try:
            while t.n_steps<3000:
                allocation.check()
                rec=plain(t.step())
                if not all(math.isfinite(rec[k]) for k in ('loss','J','grad_norm_preclip')):
                    raise RuntimeError('Non-finite training state')
                rec.update(job_id=os.environ['SLURM_JOB_ID'],utc=utc())
                append(directory/'TRAIN_STEPS.jsonl',rec)
                if t.n_steps%250==0:
                    cp=directory/'checkpoints'/f'step_{t.n_steps:04d}.pt'
                    meta=save_checkpoint(t,cp,ident)
                    ev=evaluate(t,guard,'CAL',cal,range(10),directory/'CAL'/f'step_{t.n_steps:04d}.jsonl',ident,meta,allocation)
                    checkpoints.append({'step':t.n_steps,'metrics':ev['metrics'],'checkpoint':str(cp.relative_to(directory)),
                                        'checkpoint_sha256':meta['sha256']})
                    allocation.record('CHECKPOINT',unit=str(directory.name),step=t.n_steps)
                    print(f"CHECKPOINT {ident['family']} {ident['phase']} seed={ident['seed']} step={t.n_steps} CAL={ev['metrics']}",flush=True)
            assert [c['step'] for c in checkpoints]==list(range(250,3001,250))
            chosen=min(checkpoints,key=selection_key)
            extra={}
            if ident['phase']=='FINAL_DEV':
                check_data({'DEV'});dev=guard.included('DEV');assert len(dev)==32
                cp=directory/chosen['checkpoint'];meta=load_checkpoint(t,cp,ident)
                ev=evaluate(t,guard,'DEV',dev,range(10),directory/'DEV.jsonl',ident,meta,allocation)
                records=lines(directory/'DEV.jsonl')
                excluded_positions={c.split('_',1)[1] for c in guard.included('FIT')+cal}
                sensitive=[r for r in records if r['core'].split('_',1)[1] not in excluded_positions]
                assert len({r['core'] for r in sensitive})==24
                extra={'DEV32':ev['metrics'],'DEV24':metrics(sensitive),
                       'DEV24_cores':sorted({r['core'] for r in sensitive}),
                       'per_core':{c:metrics([r for r in records if r['core']==c]) for c in dev}}
            write(directory/'DATA_READS.json',guard.reads)
            artifacts={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*')
                       if p.is_file() and p.name not in ('RUN.lock','COMPLETE.json') and not p.name.startswith('.')}
            result={'status':'COMPLETE','identity':ident,'steps':3000,'checkpoints':checkpoints,'selected':chosen,
                    'completed_utc':utc(),'job_id':os.environ['SLURM_JOB_ID'],'artifacts':artifacts,**extra}
            write(directory/'COMPLETE.json',result)
            return result
        finally:
            write(directory/f"READS_{os.environ['SLURM_JOB_ID']}.json",guard.reads)
            del t;gc.collect();torch.cuda.empty_cache()


def require_gpu_gate():
    gate=read(ROOT/'GPU_GATE.json')
    if gate['status']!='PASS' or gate['freeze_sha256']!=sha(ROOT/'FROZEN.json'):
        raise RuntimeError('BF1 GPU gate has not passed for this frozen driver')


def run_family(family,allocation,env):
    require_gpu_gate()
    fd=ROOT/'results'/'CAL'/family
    with lock(fd/'FAMILY.lock'):
        if (fd/'SELECTION.json').exists():
            checked_selection(family)
            return  # preserve the dependency digest of already started final seeds
        all_candidates=candidates(family);completed=[];budget_hit=False
        for cd in all_candidates:
            ident=identity(family,1017,cd['lr'],cd['cap'],'CAL')
            ud=fd/f"candidate_{cd['order']:02d}"
            try:
                result=run_unit(ident,ud,allocation,env)
            except BudgetExceeded:
                budget_hit=True
                write(fd/'BUDGET_EXHAUSTED.json',{'utc':utc(),'candidate':cd,'previous_seconds':allocation.previous,
                    'current_seconds':allocation.elapsed(),'partial_candidate_excluded_from_fixed_steps_selection':True})
                break
            completed.append({'candidate':cd,'unit':str(ud.relative_to(ROOT)),'complete_sha256':sha(ud/'COMPLETE.json'),
                              'choice':{**result['selected'],'order':cd['order']}})
        if not completed:raise RuntimeError('No complete full-length CAL candidate; family selection is BLOCKED')
        if not budget_hit and len(completed)!=len(all_candidates):raise RuntimeError('Missing CAL candidates')
        best=min(completed,key=lambda r:selection_key(r['choice']))
        sel={'status':'SELECTED','family':family,'freeze_sha256':sha(ROOT/'FROZEN.json'),
             'completed_candidates':completed,'planned_candidates':len(all_candidates),'budget_exhausted':budget_hit,
             'selected':best,'utc':utc(),'selection_scope':'full 3000-step candidates with all 12 complete CAL checkpoints'}
        write(fd/'SELECTION.json',sel)
        print('FAMILY_SELECTED',family,best['candidate'],flush=True)


def checked_selection(family):
    p=ROOT/'results'/'CAL'/family/'SELECTION.json';s=read(p)
    if s['status']!='SELECTED' or s['family']!=family or s['freeze_sha256']!=sha(ROOT/'FROZEN.json'):
        raise RuntimeError('Family selection identity mismatch')
    verified=[]
    for c in s['completed_candidates']:
        cd=c['candidate'];ud=ROOT/c['unit']
        if sha(ud/'COMPLETE.json')!=c['complete_sha256']:raise RuntimeError('Selection dependency changed')
        r=verify_complete(ud,identity(family,1017,cd['lr'],cd['cap'],'CAL'))
        if r is None or {**r['selected'],'order':cd['order']}!=c['choice']:
            raise RuntimeError('Selection does not match candidate result')
        verified.append(c)
    if min(verified,key=lambda x:selection_key(x['choice']))!=s['selected']:
        raise RuntimeError('CAL lexicographic choice mismatch')
    return s,sha(p)


def run_final(family,seed,allocation,env):
    require_gpu_gate();s,sel_sha=checked_selection(family);cd=s['selected']['candidate']
    ident=identity(family,seed,cd['lr'],cd['cap'],'FINAL_DEV',dependency=sel_sha)
    return run_unit(ident,ROOT/'results'/'FINAL_DEV'/family/f'seed_{seed}',allocation,env)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('cal','final','all'))
    p.add_argument('family',choices=FAMILIES);p.add_argument('--seed',type=int,choices=(1001,1002,1003))
    args=p.parse_args()
    if args.phase=='final' and args.seed is None:p.error('final needs --seed')
    allocation=Allocation(args.family,'CAL' if args.phase in ('cal','all') else 'FINAL_DEV')
    def terminated(signum,frame):raise InterruptedError(f'Slurm signal {signum}; resume from last verified checkpoint')
    signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGUSR1,terminated)
    status='FAIL'
    try:
        env=runtime_check('cuda')
        if args.phase in ('cal','all'):
            def budget_alarm(signum,frame):raise BudgetExceeded('Allocated CAL time cap alarm')
            signal.signal(signal.SIGALRM,budget_alarm)
            remaining=8*3600-30-allocation.previous-allocation.elapsed()
            if remaining>0:signal.setitimer(signal.ITIMER_REAL,remaining)
            try:run_family(args.family,allocation,env)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            if args.phase=='all':
                allocation.record('END',status='COMPLETE')
                allocation=Allocation(args.family,'FINAL_DEV',start=time.time())
                for seed in (1001,1002,1003):run_final(args.family,seed,allocation,env)
        else:run_final(args.family,args.seed,allocation,env)
        status='COMPLETE';return 0
    except Exception as e:
        write(ROOT/'failures'/f"{os.environ.get('SLURM_JOB_ID')}.json",{'utc':utc(),'phase':args.phase,
              'family':args.family,'seed':args.seed,'exception':repr(e),'traceback':traceback.format_exc()})
        raise
    finally:allocation.record('END',status=status)


if __name__=='__main__':raise SystemExit(main())
