"""Fixed-CAL pose gradient direction/norm interventions; not a new VCS loss."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import time
import traceback
from common import BASE, HERE, read, write, sha, verify

spec=importlib.util.spec_from_file_location('frozen_trajectory_host',BASE/'handoff/r22_reg_trajectory_v2_20261025/trajectory.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
torch=V.torch
ARMS={'P1_NATIVE':('P1','P1'),'POS_NATIVE':('POS_QUAD','POS_QUAD'),
      'P1_DIR_POS_NORM':('P1','POS_QUAD'),'POS_DIR_P1_NORM':('POS_QUAD','P1')}
CROSSED=('P1_DIR_POS_NORM','POS_DIR_P1_NORM')
EPS=1e-12

def compose(gd,gn):
    nd,nn=gd.norm(),gn.norm()
    assert torch.isfinite(nd) and torch.isfinite(nn)
    if nd<=EPS:
        assert nn<=EPS,'Undefined direction with nonzero target norm'
        return torch.zeros_like(gd),0.
    scale=nn/nd
    g=gd*scale
    assert torch.allclose(g.norm(),nn,rtol=2e-5,atol=1e-9)
    return g,float(scale)

def path(energy,inner,initial,arm,truth,ds):
    direction,magnitude=ARMS[arm];inner.trajectory_objective='P1'
    p=initial.detach().clone().requires_grad_();opt=torch.optim.Adam([p],lr=.5)
    rows=[];scales=[]
    for step in range(101):
        opt.zero_grad();energy(p);z=inner.trajectory_terms
        grads={k:torch.autograd.grad(z[k].sum(),p,retain_graph=True)[0] for k in ('P1','POS_QUAD')}
        g,scale=compose(grads[direction],grads[magnitude]);scales.append(scale)
        assert torch.isfinite(g).all()
        if step in V.RECORD:
            fullgrads={k:torch.autograd.grad(z[k].sum(),p,retain_graph=True)[0] for k in V.TERMS}
            assert all(torch.isfinite(v).all() for v in fullgrads.values())
            assert torch.allclose(fullgrads['J'],sum(fullgrads[k] for k in V.TERMS[:4]),atol=2e-5,rtol=1e-3)
            assert all(torch.isfinite(v).all() for v in z.values())
            state=opt.state.get(p,{});err=V.pose_error(p,truth)
            rows.append(dict(step=step,pose=p.detach().cpu().tolist(),error_full=err,error_native=err/4 if ds=='epfl' else err,
                values={k:float(z[k].detach().sum()) for k in V.TERMS},gradients={k:v.detach().cpu().tolist() for k,v in fullgrads.items()},
                applied_ascent_gradient=g.detach().cpu().tolist(),gradient_norms={k:float(v.norm()) for k,v in grads.items()},scale=scale,
                adam_step=int(state['step']) if state else 0,
                adam_exp_avg=state['exp_avg'].detach().cpu().tolist() if state else [0.]*3,
                adam_exp_avg_sq=state['exp_avg_sq'].detach().cpu().tolist() if state else [0.]*3))
            if step==50:
                other=p.detach().clone().requires_grad_();o=torch.optim.Adam([other],lr=.5)
                o.load_state_dict(copy.deepcopy(opt.state_dict()));assert V.L.tree_equal(opt.state_dict(),o.state_dict())
        if step==100:break
        p.grad=-g.detach().clone();opt.step();V.project(p)
        assert torch.isfinite(p).all() and abs(float(p[0]))<=V.A.Q.UMAX+1e-4 and float(p[1:].norm())<=V.A.Q.TMAX+1e-4
    return dict(arm=arm,direction=direction,norm_source=magnitude,initial_pose=initial.detach().cpu().tolist(),trajectory=rows,
        final_pose=rows[-1]['pose'],final_error_native=rows[-1]['error_native'],final_cosine=rows[-1]['values']['P1'],
        min_scale=min(scales),max_scale=max(scales))

def assert_initial(got,old):
    assert got['pose']==old['pose']
    for k in V.TERMS:
        assert math.isclose(got['values'][k],old['values'][k],abs_tol=1e-6,rel_tol=1e-5),('initial value',k)
        assert all(math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-5) for a,b in zip(got['gradients'][k],old['gradients'][k])),('initial gradient',k)

def history_difference(got,old):
    errors=[]
    assert [x['step'] for x in got['trajectory']]==[x['step'] for x in old['trajectory']]==list(V.RECORD)
    for a,b in zip(got['trajectory'],old['trajectory']):
        diff=max(abs(x-y) for x,y in zip(a['pose'],b['pose']));err=abs(a['error_native']-b['error_native'])
        errors.append(dict(step=a['step'],max_pose_error=diff,native_error_difference=err))
    return errors

def check():
    assert os.environ.get('SLURM_JOB_ID')
    # Inherited exact-term finite differences, literal100-step equivalence,
    # state50 roundtrip,18 source restores and CAL scope checks.
    V.cpu_check();cpu=read(HERE/'CPU_CHECK.json');torch.set_num_threads(2)
    for gd,gn in [(torch.tensor([3.,4.,0.],dtype=torch.float64),torch.tensor([1.,0.,0.],dtype=torch.float64)),
                  (torch.zeros(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64))]:
        g,_=compose(gd,gn);assert torch.allclose(g.norm(),gn.norm())
    try:compose(torch.zeros(3),torch.ones(3))
    except AssertionError:pass
    else:raise AssertionError('Undefined direction did not fail closed')
    class Toy:
        trajectory_objective='P1'
        def __call__(self,p):
            a=-((p-torch.tensor([4.,-3.,8.],dtype=p.dtype))**2).sum()/100
            b=-.02*(p**2).sum()/100;c=.001*p.sum();d=-.001*(p**2).sum()/100
            self.trajectory_terms=dict(zip(V.TERMS,(a,b,c,d,a+b,a+b+c+d)))
            return self.trajectory_terms[self.trajectory_objective]
    toy=Toy();initial=torch.tensor([V.A.Q.UMAX,50.,10.],dtype=torch.float64)
    for arm,objective in [('P1_NATIVE','P1'),('POS_NATIVE','POS_QUAD')]:
        original=V.run_path(toy,toy,initial,objective,[0.,0.,0.],'bf')
        got=path(toy,toy,initial,arm,[0.,0.,0.],'bf')
        assert got['final_pose']==original['final_pose'],'Manual native gradient must exactly match host CPU update'
    for arm in CROSSED:
        got=path(toy,toy,initial,arm,[0.,0.,0.],'bf');assert len(got['trajectory'])==7
    cpu['gradient_factor_checks']=['direction/norm formula','zero-direction fail closed','both native manual gradients bitwise match original100step CPU trajectory','both crossed100step updates and real state50 roundtrip']
    cpu['status']='PASS';write(HERE/'CPU_CHECK.json',cpu);print('GRADFACTOR_CPU PASS',flush=True)

def run(ds,arm,seed,gate=False):
    assert os.environ.get('SLURM_JOB_ID');freeze=verify();cpu=read(HERE/'CPU_CHECK.json')
    assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze and cpu['gradient_factor_checks']
    if not gate:assert 'full' in read(HERE/'ADMISSION.json')['groups']
    torch.set_num_threads(4);assert torch.cuda.is_available()
    key=f'{ds}_{arm}_{seed}';prior=read(HERE/'BASELINE_LOCK.json')[key]
    previous=read(prior['completion']);assert sha(prior['completion'])==prior['completion_sha256']
    env=V.R.environment();oldenv=previous['environment']
    for k in ('host','gpu_name','torch','numpy','cuda_runtime','cudnn'):
        assert env[k]==oldenv[k],('baseline environment',k)
    assert env['precision']==oldenv['precision'],'Exact recorded precision and driver required'
    name=f'gate_{ds}' if gate else key;out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(out.glob('ATTEMPT*')) and not (out/'COMPLETE.json').exists()
        start=time.time();job=os.environ['SLURM_JOB_ID'];write(out/f'ATTEMPT_{job}.json',dict(status='STARTED',job=job,freeze_sha256=freeze,unix=start))
        try:
            assert sha(prior['trajectories'])==prior['trajectories_sha256']
            oldrows=[json.loads(s) for s in open(prior['trajectories']).read().splitlines()]
            oldrows={(r['core'],r['perturbation']):r for r in oldrows}
            u,guard,H,fit,cores=V.context(ds,seed,'cuda');trainer,src=V.restored(ds,arm,seed,fit,'cuda')
            active=cores[:1] if gate else cores;indices=(0,) if gate else (0,1);problems=[];replay=[]
            with V.R.CP.evaluation_state(trainer):
                session=V.R.make_session(u,trainer,guard,H,'CAL',active,'cuda')
                for core in active:
                    for i in indices:
                        old=oldrows[(core,i)];energy,_=session.energy(core,i);inner=V.install(energy)
                        truth=session.perturbations[core][i]
                        assert list(truth)==old['truth'] and inner.u.shape[0]==old['n_pos']==4096
                        assert hashlib.sha256(inner.x.detach().cpu().numpy().tobytes()).hexdigest()==old['position_sha256']
                        result=dict(core=core,scene=old['scene'],perturbation=i,truth=old['truth'],start_sha256=old['start_sha256'],position_sha256=old['position_sha256'],arms={})
                        for method in tuple(ARMS):
                            paths=[]
                            for k in range(3):
                                natural=old['objectives']['P1']['paths'][k]
                                initial=torch.tensor(natural['initial_pose'],dtype=torch.float32,device='cuda')
                                got=path(energy,inner,initial,method,truth,ds)
                                assert_initial(got['trajectory'][0],natural['trajectory'][0])
                                if method in ('P1_NATIVE','POS_NATIVE'):
                                    ref='P1' if method=='P1_NATIVE' else 'POS_QUAD'
                                    replay.append(dict(core=core,perturbation=i,start=k,arm=method,errors=history_difference(got,old['objectives'][ref]['paths'][k])))
                                if method!='P1_NATIVE':assert_initial(got['trajectory'][0],result['arms']['P1_NATIVE']['paths'][k]['trajectory'][0])
                                paths.append(got)
                            chosen=max(range(3),key=lambda k:paths[k]['final_cosine'])
                            result['arms'][method]=dict(paths=paths,selected_by_common_cosine=chosen,selected_error_native=paths[chosen]['final_error_native'],mean_three_start_error_native=sum(p['final_error_native'] for p in paths)/3)
                        V.R.append(out/'TRAJECTORIES.jsonl',result);problems.append((core,i));print(name,core,i,'COMPLETE',flush=True)
            assert {r['role'] for r in guard.reads}=={'CAL'} and sha(src['checkpoint'])==src['sha256']
            assert len(replay)==6*len(problems)
            write(out/'INPUT_TRACE.json',guard.reads);write(out/'HISTORICAL_DIFFERENCES.json',replay)
            write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=freeze,ds=ds,arm=arm,seed=seed,gate=gate,job=job,seconds=time.time()-start,
                source=src,baseline=prior,environment=env,roles_read=['CAL'],source_unchanged=True,problems=problems,n_problems=len(problems),
                starts=3,adam_steps=100,arms=list(ARMS),paired_reference_paths=len(replay),historical_reference_reuse=False,
                trajectory_sha256=sha(out/'TRAJECTORIES.jsonl'),input_trace_sha256=sha(out/'INPUT_TRACE.json'),historical_differences_sha256=sha(out/'HISTORICAL_DIFFERENCES.json')))
        except Exception as exc:
            write(out/f'FAILURE_{job}.json',dict(status='FAILED',seconds=time.time()-start,error=repr(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['check','gate','run']);a.add_argument('--ds',choices=['bf','epfl']);a.add_argument('--arm',default='vcs_exact');a.add_argument('--seed',type=int,default=1001);args=a.parse_args()
    if args.mode=='check':check()
    else:run(args.ds,args.arm,args.seed,args.mode=='gate')
