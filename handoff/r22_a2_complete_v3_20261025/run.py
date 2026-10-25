"""Run original A2 fit unchanged. Gate-only interception is restored before return."""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import math
import os
import subprocess
import sys
import time
from common import BASE,HERE,REPO,SRC,OLD,OUT,ARMS,SEEDS,read,write,sha,verify,source,reference,storage,cpu_pass,gate_pass

def lrs(epoch): return [2e-4*.1**(epoch//30), 1e-3*.1**(epoch//30)]
def environment(R):
    precision, env=R.gpu_setup();ref,_=reference(901,'VCS_SHARP')
    assert precision==ref['precision_effective'], ('precision',precision)
    for k in R.c1.PAIRED_KEYS: assert env[k]==ref['environment'][k], ('environment',k,env[k])
    return precision,env
def matching(R,seed,arm,ctx,inp,student,check_init=True):
    oldarm=arm.removesuffix('_SHARP');ref,manifest=reference(seed,oldarm);_,pilot=reference(901,arm)
    assert asdict(ctx['cfg'])==manifest['cfg']==pilot['cfg']
    assert student.manifest['objective']==pilot['objective'] and student.manifest['loss']==pilot['loss']
    assert {k:v for k,v in student.manifest.items() if k not in ('objective','loss','ess_U')}=={k:v for k,v in manifest.items() if k not in ('objective','loss','ess_U')}
    # ESS is a reported fp64 reduction, not a training input. Raw weights/order remain exact.
    assert abs(student.manifest['ess_U']-manifest['ess_U'])<=8*math.ulp(manifest['ess_U']), 'ESS exceeds eight fp64 ULPs'
    assert inp['sha256']==ref['inputs_sha256']
    checks=R.c1.pool_checks(student,inp['rows'],inp['weights'],seed,ctx['cfg'],{**R.c1.PROTOCOL_POOL,'updates':45320})
    assert checks['ok'] and checks['data_order_sha256']==ref['data_order_sha256']
    checks['historical_ess_U']=manifest['ess_U'];checks['current_ess_U']=student.manifest['ess_U']
    checks['ess_absolute_delta']=abs(student.manifest['ess_U']-manifest['ess_U'])
    init=R.c1.initial_model_sha256(ctx['cfg'],seed) if check_init else None
    if check_init: assert init==ref['initial_model_sha256'], ('initial model',seed)
    return checks,init

def check(R,finish):
    import pytest
    import torch
    import v9.n2.train as train
    from v9.n2.train import WeightedStudent
    from v9.n2.teacher import HostConfig
    from losses_sharp import OBJECTIVES_SHARP
    git_ref=read(HERE/'CPU_GIT_REFERENCE.json')
    for rel,h in git_ref['sha256'].items():
        blob=subprocess.check_output(['git','-C',str(REPO),'show',git_ref['commit']+':'+rel])
        assert hashlib.sha256(blob).hexdigest()==h
    (HERE/'checks').mkdir(parents=True,exist_ok=True)
    # Existing tests include formula/gradient, A/hinge identity and stateful resume.
    rc=pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(HERE/'checks/pytest'),
        str(SRC/'test_r22.py'),str(SRC/'test_r22_a2.py')])
    assert rc==0, ('original CPU suite',rc)
    checked={}
    for seed in SEEDS:
        for arm in ARMS:
            inp=R.weight_inputs(seed,R.WEIGHT_FILE[arm]);_,m=reference(seed,arm.removesuffix('_SHARP'))
            cfg=HostConfig(**m['cfg'])
            student=WeightedStudent(cfg,inp['rows'],inp['weights'],None,None,None,HERE/'checks/inputs'/f'{seed}_{arm}',seed,
                weight_lock=inp['lock'],formal=True,loss_scale='sum',objective=OBJECTIVES_SHARP[arm]())
            checks,_=matching(R,seed,arm,{'cfg':cfg},inp,student,check_init=False);checked[f'{seed}_{arm}']=checks
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.rand(4,4)*.1+.45)
        def forward(self,ims,caps,lens): return self.w
    class Student(WeightedStudent):
        def validate(self,host): return {'rsum':float(host.w.detach().sum())}
    oldbuild,oldstep=train.build_host,torch.optim.Adam.step;schedules={}
    try:
        train.build_host=lambda cfg:Toy()
        for arm in ARMS:
            applied=[]
            def step(opt,*a,**kw):
                applied.append([g['lr'] for g in opt.param_groups]);return oldstep(opt,*a,**kw)
            torch.optim.Adam.step=step
            cfg=replace(cfg,batch_size=4);rows=[{'pair_id':str(i),'role':'U'} for i in range(4)]
            st=Student(cfg,rows,{str(i):.5 for i in range(4)},lambda r:(torch.zeros(4,1),torch.zeros(4,1),[1]*4),None,None,
                HERE/'checks'/f'toy_{arm}',902,objective=OBJECTIVES_SHARP[arm](),loss_scale='sum')
            st.fit(40,log_every=1);log=read(st.out/'train_log.json')
            assert applied==[lrs(e) for e in range(40)]
            assert all(math.isfinite(e['mean_loss']) and e['steps_in_epoch']==1 for e in log['epochs'])
            ck=torch.load(st.out/'last.pt',map_location='cpu',weights_only=False)
            assert [g['lr'] for g in ck['opt']['param_groups']]==lrs(39) and ck['epoch']==39
            schedules[arm]=applied
    finally: train.build_host,torch.optim.Adam.step=oldbuild,oldstep
    R.c1.require_clean_trace(finish.forbidden_hits)
    return dict(status='PASS',input_checks=checked,actual_adam_lr=schedules,original_cpu_suite_exit=rc,
        cpu_initial_fingerprint_note='Original CPU initial tensor hash is checked on required RTX execution host, before GPU updates; source and all six input order identities checked here.')

def gate(R,finish):
    import torch
    import v9.n2.train as train
    cpu_pass();precision,env=environment(R);ctx=R.c1.load_context();rows=[]
    for seed in SEEDS:
        first=None
        for arm in ARMS:
            out=OUT/'gates'/f's{seed}__{arm}';assert not out.exists(),'No hidden gate restart'
            inp=R.weight_inputs(seed,R.WEIGHT_FILE[arm]);batches=[]
            def collate(items):
                if len(batches)<8:batches.append(hashlib.sha256('\n'.join(r['pair_id'] for r in items).encode()).hexdigest())
                return ctx['collate'](items)
            small={**ctx,'dev_images':ctx['dev_images'][:16],'dev_captions':ctx['dev_captions'][:80]}
            st=R.make_student(small,inp,seed,arm,out,collate=collate,trace_hits=finish.forbidden_hits)
            checks,init=matching(R,seed,arm,ctx,inp,st)
            oi,os_=train.iterate_batches,torch.optim.Adam.step;calls=[0];applied=[]
            def limited(*a,**kw):
                e=calls[0];calls[0]+=1;return itertools.islice(oi(*a,**kw),8 if e==0 else 1)
            def step(opt,*a,**kw):
                applied.append([g['lr'] for g in opt.param_groups]);return os_(opt,*a,**kw)
            try:
                train.iterate_batches,torch.optim.Adam.step=limited,step;st.fit(40,log_every=1)
            finally:train.iterate_batches,torch.optim.Adam.step=oi,os_
            log=read(out/'train_log.json');counts=[8]+[1]*39
            assert calls[0]==40 and [e['steps_in_epoch'] for e in log['epochs']]==counts
            assert applied==[lrs(e) for e,n in enumerate(counts) for _ in range(n)]
            assert all(math.isfinite(e['mean_loss']) and math.isfinite(e['dev']['rsum']) for e in log['epochs'])
            assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm_pre_clip']) for x in log['steps'])
            if first is None:first=batches
            assert len(batches)==8 and batches==first
            ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            assert ck['epoch']==39 and [g['lr'] for g in ck['opt']['param_groups']]==lrs(39)
            train.seed_everything(seed);initial=train.build_host(ctx['cfg']).state_dict()
            changed=any(not torch.equal(ck['model'][k],v) for k,v in initial.items())
            b=float(ck['objective']['critic.b']);assert changed and math.isfinite(b) and b!=-10
            assert all(torch.isfinite(v).all() for v in ck['model'].values() if v.is_floating_point())
            assert R.c1.initial_model_sha256(ctx['cfg'],seed)==init
            rows.append(dict(seed=seed,arm=arm,epochs=40,updates=47,model_changed=changed,b_changed=b!=-10,b_last=b,
                initial_model_sha256=init,pool_checks=checks,first8_batch_sha256=batches,actual_adam_lr=applied,
                log_sha256=sha(out/'train_log.json'),checkpoint_sha256=sha(out/'last.pt')))
            del ck,initial,st
            print('GATE_CELL_PASS',seed,arm,flush=True)
    return dict(status='COMPLETE',rows=rows,precision=precision,environment=env,cpu_sha256=sha(HERE/'checks/CPU.json'),
        scope='Six real SGR cells x40 sparse epochs (47 updates/cell,282 total), technical only; no task efficacy requirement.')

def run(R,finish,seed,arm):
    import torch
    g=gate_pass();precision,env=environment(R)
    assert seed in SEEDS and arm in ARMS
    out=OUT/'runs'/f's{seed}__{arm}';assert not out.exists() and not (OLD/'runs'/out.name).exists(),'No duplicate or silent resume'
    ctx=R.c1.load_context();inp=R.weight_inputs(seed,R.WEIGHT_FILE[arm])
    probe=R.make_student(ctx,inp,seed,arm,OUT/'preflight'/out.name,trace_hits=finish.forbidden_hits)
    checks,init=matching(R,seed,arm,ctx,inp,probe);del probe
    original_context,original_setup=R.c1.load_context,R.gpu_setup
    try:
        R.c1.load_context=lambda:ctx;R.gpu_setup=lambda:(precision,env)
        status=R.run(OUT,seed,arm,trace_hits=finish.forbidden_hits)
    finally:R.c1.load_context,R.gpu_setup=original_context,original_setup
    assert status=='COMPLETE_VERIFIED'
    r=read(out/'unit_result.json');assert r['initial_model_sha256']==init and r['data_order_sha256']==checks['data_order_sha256']
    log=read(out/'train_log.json');assert [e['lr'] for e in log['epochs']]==[lrs(e)[0] for e in range(40)]
    ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==39 and [x['lr'] for x in ck['opt']['param_groups']]==lrs(39)
    assert all(torch.isfinite(v).all() for v in ck['model'].values() if v.is_floating_point())
    assert math.isfinite(float(ck['objective']['critic.b']))
    return dict(status='COMPLETE',seed=seed,arm=arm,unit_result_sha256=sha(out/'unit_result.json'),
        gate_sha256=sha(OUT/'gates/COMPLETE.json'),initial_model_sha256=init,data_order_sha256=checks['data_order_sha256'])

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'),'All numerical work requires Slurm'
    f=verify();guard=storage();R=source();finish=R.trace_open.install('c1' if a.mode!='check' else 'c0');started=time.time()
    dest=HERE/'checks/CPU.json' if a.mode=='check' else OUT/'gates/COMPLETE.json' if a.mode=='gate' else OUT/'runs'/f's{a.seed}__{a.arm}'/'COMPLETE.json'
    trace=HERE/'checks/CPU_TRACE.json' if a.mode=='check' else OUT/'gates/OPEN_TRACE.json' if a.mode=='gate' else dest.parent/f"OPEN_TRACE_{os.environ['SLURM_JOB_ID']}.json"
    assert not dest.exists() and not trace.exists(),'Completion/trace may not be overwritten'
    try:
        result=check(R,finish) if a.mode=='check' else gate(R,finish) if a.mode=='gate' else run(R,finish,a.seed,a.arm)
        R.c1.require_clean_trace(finish.forbidden_hits)
    except BaseException as exc:
        finish(trace,{'error':repr(exc),'successful':False})
        write(dest.parent/f"FAILED_{a.mode}_{os.environ['SLURM_JOB_ID']}.json",dict(status='FAILED',error=repr(exc),freeze_sha256=f,seconds=time.time()-started))
        raise
    tr=finish(trace,{'successful':True});assert tr['status']=='PASS'
    assert verify()==f
    result.update(freeze_sha256=f,job=os.environ['SLURM_JOB_ID'],seconds=time.time()-started,trace_sha256=sha(trace),storage_guard=guard)
    write(dest,result);print(a.mode.upper()+'_PASS',flush=True)

if __name__=='__main__':main()
