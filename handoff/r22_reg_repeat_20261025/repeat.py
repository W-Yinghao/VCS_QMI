"""Same-code balanced-order repeatability diagnostic for four pose-gradient arms."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from common import BASE,HERE,read,write,sha,verify

spec=importlib.util.spec_from_file_location('frozen_gradient_factor',BASE/'handoff/r22_reg_gradfactor_paired_20261025/gradfactor.py')
F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
V=F.V;torch=F.torch;METHODS=tuple(F.ARMS);REPEATS=4


def order(problem_index,repeat_index):
    offset=(problem_index+repeat_index)%len(METHODS)
    return METHODS[offset:]+METHODS[:offset]


def check():
    F.check();report=read(HERE/'CPU_CHECK.json')
    for i in range(20):
        schedule=[order(i,r) for r in range(REPEATS)]
        assert all(set(row)==set(METHODS) for row in schedule)
        assert all({row[k] for row in schedule}==set(METHODS) for k in range(4))
    class Toy:
        trajectory_objective='P1'
        def __call__(self,p):
            a=-((p-torch.tensor([4.,-3.,8.],dtype=p.dtype))**2).sum()/100
            b=-.02*(p**2).sum()/100;c=.001*p.sum();d=-.001*(p**2).sum()/100
            self.trajectory_terms=dict(zip(V.TERMS,(a,b,c,d,a+b,a+b+c+d)))
            return self.trajectory_terms[self.trajectory_objective]
    toy=Toy();initial=torch.tensor([1.,10.,-4.],dtype=torch.float64);reference={}
    for rep in range(4):
        for method in order(0,rep):
            p=F.path(toy,toy,initial,method,[0.,0.,0.],'bf')
            if method in reference:assert p==reference[method]
            reference[method]=p
    report['repeat_checks']=['Latin rotations balance every arm at every rank within each problem',
        'unchanged frozen four-arm path implementation','fresh Adam per path','four CPU repeats bitwise identical at all recorded values/state under balanced order']
    write(HERE/'CPU_CHECK.json',report);print('REPEAT_CPU PASS',flush=True)


def run(ds,arm,seed,gate=False):
    assert os.environ.get('SLURM_JOB_ID');freeze=verify();cpu=read(HERE/'CPU_CHECK.json')
    assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze and len(cpu['repeat_checks'])==4
    if not gate:assert 'full' in read(HERE/'ADMISSION.json')['groups']
    torch.set_num_threads(4);assert torch.cuda.is_available()
    key=f'{ds}_{arm}_{seed}';prior=read(HERE/'BASELINE_LOCK.json')[key]
    previous=read(prior['completion']);assert sha(prior['completion'])==prior['completion_sha256']
    assert sha(prior['trajectories'])==prior['trajectories_sha256']
    env=V.R.environment()
    for k in ('host','gpu_name','torch','numpy','cuda_runtime','cudnn','precision'):
        assert env[k]==previous['environment'][k],('frozen environment',k)
    name=f'gate_{ds}' if gate else key;out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(out.glob('ATTEMPT*')) and not (out/'COMPLETE.json').exists()
        start=time.time();job=os.environ['SLURM_JOB_ID']
        write(out/f'ATTEMPT_{job}.json',dict(status='STARTED',job=job,freeze_sha256=freeze,unix=start))
        try:
            oldrows={(r['core'],r['perturbation']):r for r in map(json.loads,open(prior['trajectories']).read().splitlines())}
            u,guard,H,fit,cores=V.context(ds,seed,'cuda');trainer,src=V.restored(ds,arm,seed,fit,'cuda')
            active=cores[:1] if gate else cores;indices=(0,) if gate else (0,1);problems=[]
            with V.R.CP.evaluation_state(trainer):
                session=V.R.make_session(u,trainer,guard,H,'CAL',active,'cuda')
                for core in active:
                    for i in indices:
                        old=oldrows[(core,i)];energy,_=session.energy(core,i);inner=V.install(energy);truth=session.perturbations[core][i]
                        assert list(truth)==old['truth'] and inner.u.shape[0]==old['n_pos']==4096
                        assert hashlib.sha256(inner.x.detach().cpu().numpy().tobytes()).hexdigest()==old['position_sha256']
                        q=dict(core=core,scene=old['scene'],perturbation=i,truth=old['truth'],start_sha256=old['start_sha256'],position_sha256=old['position_sha256'],repeats=[])
                        for rep in range(REPEATS):
                            schedule=order(len(problems),rep);record=dict(repeat=rep,order=list(schedule),arms={})
                            rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
                            for position,method in enumerate(schedule):
                                paths=[]
                                for k in range(3):
                                    native=old['objectives']['P1']['paths'][k]
                                    initial=torch.tensor(native['initial_pose'],dtype=torch.float32,device='cuda')
                                    got=F.path(energy,inner,initial,method,truth,ds)
                                    F.assert_initial(got['trajectory'][0],native['trajectory'][0]);paths.append(got)
                                chosen=max(range(3),key=lambda k:paths[k]['final_cosine'])
                                record['arms'][method]=dict(position=position,paths=paths,selected_by_common_cosine=chosen,
                                    selected_error_native=paths[chosen]['final_error_native'],mean_three_start_error_native=sum(x['final_error_native'] for x in paths)/3)
                            assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
                            assert list(record['arms'])==list(schedule)
                            q['repeats'].append(record)
                        assert all({r['order'][k] for r in q['repeats']}==set(METHODS) for k in range(4))
                        for k in range(3):
                            ref=q['repeats'][0]['arms']['P1_NATIVE']['paths'][k]['trajectory'][0]
                            for rep in q['repeats']:
                                for method in METHODS:F.assert_initial(rep['arms'][method]['paths'][k]['trajectory'][0],ref)
                        V.R.append(out/'TRAJECTORIES.jsonl',q);problems.append((core,i));print(name,core,i,'FOUR_REPEATS_COMPLETE',flush=True)
            assert len(problems)==len(active)*len(indices)
            assert {r['role'] for r in guard.reads}=={'CAL'} and sha(src['checkpoint'])==src['sha256']
            write(out/'INPUT_TRACE.json',guard.reads)
            write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=freeze,ds=ds,arm=arm,seed=seed,gate=gate,job=job,seconds=time.time()-start,
                source=src,baseline=prior,environment=env,roles_read=['CAL'],source_unchanged=True,problems=problems,n_problems=len(problems),
                starts=3,adam_steps=100,repeats=REPEATS,arms=list(METHODS),balanced_order=True,torch_rng_unchanged=True,
                n_paths=len(problems)*REPEATS*4*3,trajectory_sha256=sha(out/'TRAJECTORIES.jsonl'),input_trace_sha256=sha(out/'INPUT_TRACE.json')))
        except BaseException as exc:
            write(out/f'FAILURE_{job}.json',dict(status='FAILED',seconds=time.time()-start,error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--ds',choices=['bf','epfl']);p.add_argument('--arm',default='vcs_exact');p.add_argument('--seed',type=int,default=1001);a=p.parse_args()
    if a.mode=='check':check()
    else:run(a.ds,a.arm,a.seed,a.mode=='gate')
