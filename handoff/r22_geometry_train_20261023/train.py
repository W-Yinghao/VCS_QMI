"""Shared cosine geometry: exact/sample VCS, bounded-score BCE and cosine InfoNCE."""
import argparse
from collections import defaultdict
import dataclasses
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import types

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
sys.path.insert(0,str(BASE/'ssl_pilot_v10_r22c/v10/r22_pmi'))
import run_unit as R
torch=R.torch;F=R.F;A=R.A
ARMS=('vcs_exact','vcs_perm','bce','nce')

def freeze():
    assert os.environ.get('SLURM_JOB_ID')
    for rel,digest in json.loads((HERE/'FREEZE.json').read_text())['files'].items():
        if R.sha(BASE/rel)!=digest:raise RuntimeError('Frozen source changed: '+rel)
    return R.sha(HERE/'FREEZE.json')

def normalized(u,v):
    return F.normalize(u,dim=-1,eps=1e-8),F.normalize(v,dim=-1,eps=1e-8)

def exact_j(u,v):
    p=(u*v).sum(-1)
    mu=u.mean(-2);mv=v.mean(-2)
    cu=u.transpose(-1,-2)@u/u.shape[-2];cv=v.transpose(-1,-2)@v/v.shape[-2]
    return (p-.5*p.square()).mean(-1)-(mu*mv).sum(-1)-.5*(cu*cv).sum((-2,-1))

def objective(u,v,arm,perm):
    u,v=normalized(u,v)
    if arm=='vcs_exact':
        j=exact_j(u,v).mean();return -j,j
    p=(u*v).sum(-1)
    if arm=='vcs_perm':
        q=(u*torch.stack([v[i,perm[i]] for i in range(len(v))])).sum(-1)
        j=(p-.5*p.square()).mean()-(q+.5*q.square()).mean();return -j,j
    scores=u@v.transpose(-1,-2)
    if arm=='bce':
        # Twice balanced BCE, common bounded score T=cos; Q is the full within-scene product including diagonal.
        loss=-torch.log(((1+p)/2).clamp(1e-6,1-1e-6)).mean()-torch.log(((1-scores)/2).clamp(1e-6,1-1e-6)).mean()-2*math.log(2)
    elif arm=='nce':
        labels=torch.arange(u.shape[1],device=u.device).expand(u.shape[:2]).reshape(-1)
        loss=.5*(F.cross_entropy((scores/.07).reshape(-1,u.shape[1]),labels)+F.cross_entropy((scores.transpose(1,2)/.07).reshape(-1,u.shape[1]),labels))
    else:raise ValueError(arm)
    return loss,None

class NoFreeCritic(torch.nn.Module):
    """Compatibility module only. Evaluation reads normalized descriptors directly."""
    def forward(self,x):
        a,b=x.chunk(2,-1);a,b=normalized(a,b)
        return torch.atanh((a*b).sum(-1,keepdim=True).clamp(-1+1e-6,1-1e-6))

class GeometryTrainer(A.Trainer):
    def __init__(self,cfg,imgs,fit,device,arm):
        super().__init__(cfg,imgs,fit,device)
        self.arm=arm;self.critic=NoFreeCritic().to(self.device)
        self.clip_params=list(self.eU.parameters())+list(self.eV.parameters())
        self.opt=torch.optim.AdamW(self.clip_params,lr=cfg.encoder_lr,weight_decay=cfg.weight_decay)
        self.optimizers=[self.opt];self.spot_groups={'encoders':self.clip_params}
    def forward(self,hb,bu,bv,mark=None):
        u,v=A.batch_descriptors(self.eU,self.eV,bu,bv,hb.yx)
        perm=torch.as_tensor(hb.perm,device=self.device)
        loss,j=objective(u,v,self.arm,perm)
        return loss,dict(J=j,du=u,dv=v,perm=perm,info={})

def cosine_energy(self,dv):
    u,v=normalized(self.u[None],dv)
    if self.geometry_readout=='COS':return (u*v).sum(-1).mean(-1)
    if self.geometry_readout=='J_EXACT':return exact_j(u.expand_as(v),v)
    raise ValueError(self.geometry_readout)

def check():
    torch.manual_seed(20261023)
    u=torch.randn(2,7,5,dtype=torch.float64,requires_grad=True)
    v=torch.randn(2,7,5,dtype=torch.float64,requires_grad=True)
    un,vn=normalized(u,v);scores=un@vn.transpose(-1,-2);p=scores.diagonal(dim1=-2,dim2=-1)
    dense=(p-.5*p.square()).mean(-1)-(scores+.5*scores.square()).mean((-2,-1))
    moments=exact_j(un,vn)
    assert torch.allclose(dense,moments,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(dense.sum(),(u,v),retain_graph=True);gb=torch.autograd.grad(moments.sum(),(u,v),retain_graph=True)
    assert all(torch.allclose(a,b,atol=1e-12,rtol=1e-12) for a,b in zip(ga,gb))
    # Enumerating cyclic permutations gives exactly the same empirical product reference.
    js=[]
    for shift in range(7):
        perm=torch.arange(7).roll(shift).repeat(2,1)
        js.append(objective(u,v,'vcs_perm',perm)[1])
    assert torch.allclose(torch.stack(js).mean(),moments.mean(),atol=1e-12)
    for arm in ARMS:
        loss,_=objective(u,v,arm,torch.arange(7).repeat(2,1))
        grads=torch.autograd.grad(loss,(u,v),retain_graph=True)
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
    # Same encoders, draws and update code; only the objective changes.
    trainers=[GeometryTrainer(A.TrainConfig(method='MLP_perm1',seed=1001,n_sc=2,n_pos=7,crop=40),{},['dummy'],'cpu',arm) for arm in ARMS]
    assert len({A.state_hash(t.eU,t.eV) for t in trainers})==1
    assert all(len(list(t.critic.parameters()))==0 for t in trainers)
    R.write(HERE/'CPU_CHECK.json',dict(status='PASS',freeze_sha256=R.sha(HERE/'FREEZE.json'),
            checks=['exact moments equal dense J values and encoder gradients','ordinary-permutation mean equals empirical product',
                    'all matched objective gradients finite','all arms have identical encoders and no free critic']))
    print('GEOMETRY_CPU_CHECK PASS',flush=True)

def evaluate(t,u,guard,H,role,out,variants):
    path=out/f'{role}.jsonl';expected=[(c,i) for c in sorted(guard.included(role)) for i in range(10)]
    existing=R.rows(path);done={(r['core'],r['perturbation']):r for r in existing}
    assert len(done)==len(existing) and all(k in expected and set(r['E'])==set(variants) for k,r in done.items())
    with R.CP.evaluation_state(t):
        session=R.make_session(u,t,guard,H,role,sorted(guard.included(role)),'cuda')
        for core,i in expected:
            if (core,i) in done:continue
            energy,_=session.energy(core,i);inner=R.PE.inner(energy)
            inner.j_critic=types.MethodType(cosine_energy,inner)
            truth=session.perturbations[core][i];values={}
            for variant in variants:
                inner.geometry_readout=variant
                values[variant]=R.plain(A.registration_record(energy,'cuda',truth))
                assert all(math.isfinite(values[variant][k]) for k in ('err_full','F_true','F_hat'))
            row=dict(core=core,scene=guard.unit_of(core) if H else core,perturbation=i,role=role,E=values)
            R.append(path,row);done[core,i]=row;del energy
    summary={}
    for variant in variants:
        scenes=defaultdict(list)
        for row in done.values():scenes[row['scene']].append(row['E'][variant]['err_full'])
        summary[variant]=dict(success42=statistics.mean(sum(v<42 for v in values)/len(values) for values in scenes.values()),
                             success9=statistics.mean(sum(v<9 for v in values)/len(values) for values in scenes.values()),
                             median_err=statistics.median(v for values in scenes.values() for v in values),n=len(done))
    R.write(out/f'{role}_SUMMARY.json',summary)
    return summary

def save(t,path,identity):
    R.CP.save(t,path,identity)

def run(ds,arm,seed,smoke=False):
    gate=json.loads((HERE/'CPU_CHECK.json').read_text());assert gate['status']=='PASS' and gate['freeze_sha256']==R.sha(HERE/'FREEZE.json')
    assert torch.cuda.is_available();started=time.time();torch.set_num_threads(4)
    name=f'{ds}_{arm}_{seed}'+('_smoke' if smoke else '')
    out=HERE/('smoke' if smoke else 'runs')/name;out.mkdir(parents=True,exist_ok=True)
    identity=dict(ds=ds,arm=arm,seed=seed,freeze_sha256=R.sha(HERE/'FREEZE.json'),smoke=smoke)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COMPLETE.json').exists():
            assert json.loads((out/'COMPLETE.json').read_text())['identity']==identity;return
        job=os.environ['SLURM_JOB_ID'];R.write(out/f'ATTEMPT_{job}.json',dict(identity=identity,status='RUNNING',start=time.time(),gpu=torch.cuda.get_device_name()))
        source=BASE/'handoff/r22_c3_codex_20261023/units'/f'r22_pmi3_{ds}_MLPp1_s{seed}.json'
        u=R.load_unit(source);guard,H=R.make_guard(u)
        # Verify FIT bytes too: historical readout jobs only needed CAL/DEV.
        data_lock=json.loads(Path(u['data_lock']['path']).read_text())
        if H:guard.verify_data_lock({'FIT'})
        else:
            for row in data_lock['files']:
                if row['role']=='FIT':assert R.sha(row['path'])==row['sha256']
        fit=A.fit_cores(guard,1.,seed) if H is None else H.fit_windows(guard)
        images=(A.load_learned_inputs if H is None else H.load_learned_inputs)(guard,fit,{'FIT'},'R22 geometry full FIT training','cuda')
        cfg=A.TrainConfig(method='MLP_perm1',seed=seed,encoder_lr=3e-4,head_lr=3e-4,n_sc=16,n_pos=128,crop=160,grad_spot_every=100)
        t=GeometryTrainer(cfg,images,fit,'cuda',arm)
        initial=A.state_hash(t.eU,t.eV);last=out/'last.pt'
        if last.exists():R.CP.load(t,last,identity)
        # Short real-data gate also tests both readouts' pose gradients without full CAL search.
        if smoke:
            rows=[t.step() for _ in range(3)]
            with R.CP.evaluation_state(t):
                core=sorted(guard.included('CAL'))[0];session=R.make_session(u,t,guard,H,'CAL',[core],'cuda')
                energy,_=session.energy(core,0);inner=R.PE.inner(energy);inner.j_critic=types.MethodType(cosine_energy,inner)
                truth=session.perturbations[core][0]
                for variant in ('COS','J_EXACT'):
                    inner.geometry_readout=variant
                    p=torch.tensor([math.radians(truth[0])*A.Q.RHO,*truth[1:]],device='cuda',dtype=torch.float32,requires_grad=True)
                    value=energy(p);g=torch.autograd.grad(value,p)[0]
                    assert torch.isfinite(value) and torch.isfinite(g).all()
            R.write(out/'COMPLETE.json',dict(status='COMPLETE',identity=identity,freeze_sha256=identity['freeze_sha256'],seconds=time.time()-started,
                    initial_encoder_sha256=initial,steps=rows,roles_read=sorted({r['role'] for r in guard.reads}),gpu=torch.cuda.get_device_name()))
            print('SMOKE PASS',name,flush=True);return
        # Every arm keeps the same three CAL candidates. Selection precedes all DEV evaluation.
        cal={}
        for stop in (1000,2000,3000):
            checkpoint=out/'checkpoints'/f'step_{stop}.pt';checkpoint.parent.mkdir(exist_ok=True)
            if t.n_steps<stop:
                while t.n_steps<stop:
                    row=t.step();assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm_preclip'])
                    R.append(out/'TRAIN.jsonl',row)
                    if t.n_steps%250==0:save(t,last,identity);print(name,'step',t.n_steps,flush=True)
                save(t,checkpoint,identity)
            if t.n_steps==stop and not checkpoint.exists():save(t,checkpoint,identity)
            assert checkpoint.exists()
            cal_path=out/f'cal_{stop}'/'CAL_SUMMARY.json'
            if not cal_path.exists():
                assert t.n_steps==stop
                (out/f'cal_{stop}').mkdir(exist_ok=True)
                evaluate(t,u,guard,H,'CAL',out/f'cal_{stop}',('COS',))
            cal[stop]=json.loads(cal_path.read_text())['COS']
        selected=max(cal,key=lambda k:(cal[k]['success42'],cal[k]['success9'],-cal[k]['median_err'],-k))
        selection=dict(step=selected,cal=cal,rule='CAL scene/core macro success42, success9, lower median error, earliest step',
                       checkpoint_sha256=R.sha(out/'checkpoints'/f'step_{selected}.pt'))
        R.write(out/'SELECTION_BEFORE_DEV.json',selection)
        R.CP.load(t,out/'checkpoints'/f'step_{selected}.pt',identity)
        devdir=out/'dev';devdir.mkdir(exist_ok=True)
        dev=evaluate(t,u,guard,H,'DEV',devdir,('COS','J_EXACT'))
        assert {r['role'] for r in guard.reads}<={'FIT','CAL','DEV'}
        result=dict(status='COMPLETE',identity=identity,freeze_sha256=identity['freeze_sha256'],seconds=time.time()-started,
                    initial_encoder_sha256=initial,selection=selection,dev=dev,job=job,gpu=torch.cuda.get_device_name(),
                    environment=R.environment(),roles_read=sorted({r['role'] for r in guard.reads}),
                    train_rows_sha256=R.sha(out/'TRAIN.jsonl'),dev_rows_sha256=R.sha(devdir/'DEV.jsonl'),
                    interpretation='PILOT: new shared-geometry function class, fixed optimization recipe, no independent TEST evidence')
        R.write(out/'COMPLETE.json',result)
        R.write(out/f'ATTEMPT_{job}.json',dict(status='COMPLETE',seconds=time.time()-started,gpu=torch.cuda.get_device_name(),identity=identity))
        print('COMPLETE',name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run','smoke']);p.add_argument('--ds',choices=['bf','epfl']);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=1001)
    a=p.parse_args();freeze()
    if a.mode=='check':check()
    else:run(a.ds,a.arm,a.seed,a.mode=='smoke')
