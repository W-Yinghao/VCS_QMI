"""DEV pose profiles at fixed historical checkpoints; all readouts share one encoding."""
import argparse
from collections import defaultdict
import fcntl
import math
import os
import statistics
import sys
import time
import types
from common import BASE, HERE, read, write, sha, verify, receipt

sys.path.insert(0, str(BASE/'ssl_pilot_v10_r22c/v10/r22_pmi'))
import run_unit as R
torch = R.torch
VARIANTS = ('J', 'PMI', 'PMI_C2', 'EPT', 'POS')
RADII = (.5, 1., 2., 4., 8., 16.)

def readouts(J, T):
    pmi = 2*torch.atanh(T.clamp(-1+1e-4, 1-1e-4))
    return dict(J=J, PMI=pmi.mean(), PMI_C2=pmi.clamp(-2, 2).mean(),
                EPT=T.mean(), POS=(T-.5*T.square()).mean())

def probe(energy, pose):
    inner = R.PE.inner(energy)
    assert 'j_critic' not in inner.__dict__
    captured = []
    def capture(self, dv):
        captured.append(dv)
        return type(self).j_critic(self, dv)
    inner.j_critic = types.MethodType(capture, inner)
    try:
        J = energy(pose)
    finally:
        del inner.j_critic
    assert len(captured) == 1
    T = R.PE.positive_scores(inner, captured[0])
    values = readouts(J, T)
    out = {}
    for name, value in values.items():
        gradient = torch.autograd.grad(value, pose, retain_graph=True)[0]
        assert bool(torch.isfinite(value)) and bool(torch.isfinite(gradient).all())
        out[name] = {'value': float(value.detach()), 'gradient': gradient.detach().cpu().tolist(),
                     'gradient_norm': float(gradient.norm())}
    out['score'] = {'abs_T_ge_099': float((T.abs() >= .99).float().mean()),
                    'abs_T_ge_09999': float((T.abs() >= .9999).float().mean()),
                    'mean_T': float(T.mean().detach())}
    return out

def check():
    # Check capture against the direct host critic on both actual energy classes.
    from v10.supplement.learned import MLPRigidEnergy
    torch.manual_seed(20261023)
    for cls in (R.A.Q.RigidEnergy, MLPRigidEnergy):
        e=cls.__new__(cls); e.torch=torch; e.kind='critic'
        e.u=torch.randn(9,16,dtype=torch.float64)
        e.kw={'critic':torch.nn.Sequential(torch.nn.Linear(32,5),torch.nn.Tanh(),torch.nn.Linear(5,1)).double()}
        e.perm=torch.randperm(9); e.permutations=e.perm[None]
        W=torch.randn(3,9*16,dtype=torch.float64)
        def batch(self, th, tr):
            p=torch.cat((th*R.A.Q.RHO,tr[0]))
            return self.j_critic((p@W).reshape(1,9,16))
        e.batch=types.MethodType(batch,e)
        p=torch.tensor([.13,-.27,.31],dtype=torch.float64,requires_grad=True)
        direct=e(p); direct_grad=torch.autograd.grad(direct,p)[0]
        observed=probe(e,p)
        assert observed['J']['value'] == float(direct)
        assert torch.equal(torch.tensor(observed['J']['gradient'],dtype=torch.float64),direct_grad)
        for axis in range(3):
            plus=p.detach().clone();minus=plus.clone();plus[axis]+=1e-5;minus[axis]-=1e-5
            a=probe(e,plus.requires_grad_());b=probe(e,minus.requires_grad_())
            for v in VARIANTS:
                fd=(a[v]['value']-b[v]['value'])/2e-5
                assert math.isclose(fd,observed[v]['gradient'][axis],rel_tol=2e-5,abs_tol=1e-7),(cls.__name__,v,fd)
        assert 'j_critic' not in e.__dict__
    write(HERE/'REG_CHECK.json',dict(status='PASS',freeze_sha256=sha(HERE/'FREEZE.json'),
          checks=['both host energy classes: captured J value and gradient identical',
                  'all five readout pose gradients match finite differences', 'original method restored']))
    print('REG_CHECK PASS',flush=True)

def run(ds,seed,model_key):
    assert read(HERE/'REG_CHECK.json')['freeze_sha256']==sha(HERE/'FREEZE.json')
    assert torch.cuda.is_available()
    out=HERE/'runs'/f'reg_{ds}_{model_key}_{seed}';out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COMPLETE.json').exists():
            assert read(out/'COMPLETE.json')['freeze_sha256']==sha(HERE/'FREEZE.json')
            return
        write(out/'RUNNING.json',receipt(started,'RUNNING'))
        summaries={}; sources={}
        for key in (model_key,):
            source=BASE/'handoff/r22_c3_codex_20261023/units'/f'r22_pmi3_{ds}_{key}_s{seed}.json'
            u=R.load_unit(source); guard,H=R.make_guard(u)
            trainer,model=R.build_trainer(u,guard,H,'cuda')
            cores=sorted(guard.included('DEV'))
            path=out/f'{key}.jsonl';records=[];historical=R.stored_index(u,'DEV')
            # Do not mix partial attempts or hardware: retain the old file then start the complete profile again.
            if path.exists():path.rename(out/f'{key}.previous_{os.environ["SLURM_JOB_ID"]}.jsonl')
            with R.CP.evaluation_state(trainer):
                session=R.make_session(u,trainer,guard,H,'DEV',cores,'cuda')
                for core in cores:
                    for perturbation in range(10):
                        energy,_=session.energy(core,perturbation)
                        truth=session.perturbations[core][perturbation]
                        p0=torch.tensor([math.radians(truth[0])*R.A.Q.RHO,*truth[1:]],device='cuda',dtype=torch.float32)
                        at_truth=probe(energy,p0.clone().requires_grad_())
                        # Direct J value sanity check on every problem, without changing the energy.
                        direct=energy.value(truth[0],truth[1:])
                        assert math.isclose(direct,at_truth['J']['value'],abs_tol=2e-6,rel_tol=2e-5)
                        curve=[]
                        for radius in RADII:
                            for axis in range(3):
                                for sign in (-1,1):
                                    delta=torch.zeros_like(p0);delta[axis]=sign*radius
                                    point=probe(energy,(p0+delta).detach().requires_grad_())
                                    for name in VARIANTS:
                                        row=point[name]
                                        inward=-sign*row['gradient'][axis]
                                        row.update(truth_advantage=at_truth[name]['value']-row['value'],
                                                   inward_derivative=inward,
                                                   inward_cosine=inward/row['gradient_norm'] if row['gradient_norm']>0 else None)
                                    curve.append(dict(radius=radius,axis=axis,sign=sign,readouts=point))
                        row=dict(core=core,scene=guard.unit_of(core) if H else core,perturbation=perturbation,
                                 truth=list(truth),historical_J_error=historical[core,perturbation]['err_full'],at_truth=at_truth,curve=curve)
                        R.append(path,row);records.append(row)
                        del energy
                    print(f'{ds} {key} {seed} {core} profiles complete',flush=True)
            assert set(r['role'] for r in guard.reads)<={'DEV'}
            # First average rays within a problem, then problems within a scene, then scenes.
            groups=defaultdict(lambda:defaultdict(list))
            for row in records:
                for radius in RADII:
                    for name in VARIANTS:
                        vals=[x['readouts'][name] for x in row['curve'] if x['radius']==radius]
                        groups[name,str(radius)][row['scene']].append({
                            'truth_preferred':statistics.mean(float(v['truth_advantage']>0) for v in vals),
                            'inward_gradient':statistics.mean(float(v['inward_derivative']>0) for v in vals),
                            'zero_gradient':statistics.mean(float(v['gradient_norm']==0) for v in vals),
                            'inward_cosine':statistics.mean(v['inward_cosine'] or 0. for v in vals)})
            summaries[key]={}
            for (name,radius),scenes in groups.items():
                fields=next(iter(scenes.values()))[0]
                summaries[key].setdefault(name,{})[radius]={f:statistics.mean(statistics.mean(r[f] for r in rows) for rows in scenes.values()) for f in fields}
            sources[key]=dict(model=model,checkpoint_sha256=u['checkpoint_sha256'],rows_sha256=sha(path),
                              problems=len(records),scenes=len({r['scene'] for r in records}),roles_read=['DEV'])
            del trainer;torch.cuda.empty_cache()
        write(out/'COMPLETE.json',receipt(started,'COMPLETE',gpu=torch.cuda.get_device_name(),
              ds=ds,seed=seed,summaries=summaries,sources=sources,environment=R.environment(),
              interpretation='DEV local axis profiles, not a registration convergence guarantee or an objective-only comparison'))
        print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);p.add_argument('--ds',choices=['bf','epfl']);p.add_argument('--seed',type=int);p.add_argument('--model',choices=['MLPp1','InfoNCE'])
    a=p.parse_args();verify()
    if a.mode=='check':check()
    else:run(a.ds,a.seed,a.model)
