"""FIT-only paired cost validation, including encoder/backward/optimizer and blocked controls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / 'handoff/r22_dense_geometry_20261023'))
import train as T
from torch.utils.checkpoint import checkpoint
torch, A, R = T.torch, T.A, T.R
ARMS = ('vcs_exact', 'vcs_dense', 'vcs_perm', 'bce', 'bce_block', 'nce', 'nce_block')
CHUNK = 256


def blocked(u, v, arm, chunk=CHUNK):
    u, v = T.normalized(u, v)
    b, n, _ = u.shape
    if arm == 'bce_block':
        p = (u * v).sum(-1)
        loss = -torch.log(((1+p)/2).clamp(1e-6,1-1e-6)).mean() - 2*math.log(2)
        def block(a, z):
            return -torch.log(((1-a@z.transpose(-1,-2))/2).clamp(1e-6,1-1e-6)).sum()/(b*n*n)
        for start in range(0,n,chunk):
            loss = loss + checkpoint(block, u[:,start:start+chunk], v, use_reentrant=False)
        return loss
    if arm == 'nce_block':
        loss = u.new_zeros(())
        for x,y in ((u,v),(v,u)):
            for start in range(0,n,chunk):
                labels = torch.arange(start,min(start+chunk,n),device=u.device).expand(b,-1).reshape(-1)
                def block(a,z,labels):
                    return T.F.cross_entropy(((a@z.transpose(-1,-2))/.07).reshape(-1,n),labels,reduction='sum')/(2*b*n)
                loss = loss + checkpoint(block,x[:,start:start+chunk],y,labels,use_reentrant=False)
        return loss
    raise ValueError(arm)


def loss(u,v,arm,perm):
    if arm.endswith('_block'):
        return blocked(u,v,arm)
    if arm == 'vcs_dense':
        u,v = T.normalized(u,v); s = u@v.transpose(-1,-2); p = s.diagonal(dim1=-2,dim2=-1)
        return -(p-.5*p.square()).mean()+(s+.5*s.square()).mean()
    return T.objective(u,v,arm,perm)[0]


def flatten(grads):
    return torch.cat([x.detach().flatten() for x in grads])


def cpu_check():
    torch.set_num_threads(2); torch.manual_seed(20261024)
    checks=[]
    for n in (7,17):
        u=torch.randn(2,n,5,dtype=torch.float64,requires_grad=True)
        v=torch.randn(2,n,5,dtype=torch.float64,requires_grad=True)
        perm=torch.arange(n).repeat(2,1)
        for ref,alt in (('bce','bce_block'),('nce','nce_block')):
            a=loss(u,v,ref,perm); ga=flatten(torch.autograd.grad(a,(u,v)))
            z=blocked(u,v,alt,chunk=3); gz=flatten(torch.autograd.grad(z,(u,v)))
            assert torch.allclose(a,z,atol=1e-12,rtol=1e-12)
            assert torch.allclose(ga,gz,atol=1e-12,rtol=1e-10)
            checks.append(dict(n=n,reference=ref,blocked=alt,value_error=float(abs(a-z).detach()),
                               relative_gradient_error=float((ga-gz).norm()/ga.norm())))
    R.write(HERE/'CPU_CHECK.json',dict(status='PASS',freeze_sha256=R.sha(HERE/'FREEZE.json'),checks=checks))
    print('CPU_CHECK PASS',flush=True)


def verify_freeze():
    h=R.sha(HERE/'FREEZE.json')
    for rel,digest in json.loads((HERE/'FREEZE.json').read_text())['files'].items():
        assert R.sha(BASE/rel)==digest,rel
    cpu=json.loads((HERE/'CPU_CHECK.json').read_text())
    assert cpu['status']=='PASS' and cpu['freeze_sha256']==h
    return h


def gpu_run(ds,seed,gate=False):
    started=time.time(); torch.set_num_threads(4)
    freeze=verify_freeze(); T.freeze(); assert torch.cuda.is_available()
    out=HERE/('gates' if gate else 'runs')/f'{ds}_{seed}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        assert json.loads((out/'COMPLETE.json').read_text())['freeze_sha256']==freeze
        return
    job=os.environ['SLURM_JOB_ID']
    R.write(out/f'ATTEMPT_{job}.json',dict(status='RUNNING',start=time.time(),job=job))
    source=BASE/'handoff/r22_geometry_train_20261023/runs'/f'{ds}_vcs_exact_{seed}'
    complete=json.loads((source/'COMPLETE.json').read_text()); assert complete['status']=='COMPLETE'
    model=source/'checkpoints'/f"step_{complete['selection']['step']}.pt"
    assert R.sha(model)==complete['selection']['checkpoint_sha256']
    pinned=json.loads((HERE/'SOURCE_LOCK.json').read_text())[f'{ds}_{seed}']
    assert R.sha(model)==pinned['checkpoint_sha256'] and R.sha(source/'COMPLETE.json')==pinned['completion_sha256']
    unit=R.load_unit(BASE/'handoff/r22_c3_codex_20261023/units'/f'r22_pmi3_{ds}_MLPp1_s{seed}.json')
    guard,H=R.make_guard(unit)
    lock=json.loads(Path(unit['data_lock']['path']).read_text())
    if H: guard.verify_data_lock({'FIT'})
    else:
        for row in lock['files']:
            if row['role']=='FIT': assert R.sha(row['path'])==row['sha256']
    fit=A.fit_cores(guard,1.,seed) if H is None else H.fit_windows(guard)
    imgs=(A.load_learned_inputs if H is None else H.load_learned_inputs)(guard,fit,{'FIT'},'R22 paired full-step cost','cuda')
    cfg=A.TrainConfig(method='MLP_perm1',seed=seed,encoder_lr=3e-4,head_lr=3e-4,n_sc=16,n_pos=128,crop=160,grad_spot_every=100)
    trainer=T.GeometryTrainer(cfg,imgs,fit,'cuda','vcs_exact'); R.CP.load(trainer,model,complete['identity'])
    assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) for net in (trainer.eU,trainer.eV) for m in net.modules())
    for group in trainer.opt.param_groups: group['lr']=0.0
    initial=A.state_hash(trainer.eU,trainer.eV)
    parameters=list(trainer.eU.parameters())+list(trainer.eV.parameters())
    records=[]; equivalence=[]
    for batch_idx in range(1 if gate else 3):
        hb,bu,bv=trainer.draw()
        rng=R.np.random.default_rng(A.Q.seed_of(f'R22|full-cost|{ds}|{seed}|{batch_idx}'))
        yx=rng.uniform(trainer.sampler.inner,159-trainer.sampler.inner,(16,2048,2))
        for n in (128,512,2048):
            coords=yx[:,:n]
            perm=torch.as_tensor(R.np.stack([rng.permutation(n) for _ in range(16)]),device='cuda')
            def encoder_value_grad(arm):
                u,v=A.batch_descriptors(trainer.eU,trainer.eV,bu,bv,coords)
                value=loss(u,v,arm,perm)
                return float(value.detach()),flatten(torch.autograd.grad(value,parameters))
            for ref,alt in (('vcs_exact','vcs_dense'),('bce','bce_block'),('nce','nce_block')):
                value,gr=encoder_value_grad(ref); other,ga=encoder_value_grad(alt)
                rel=float((gr-ga).norm()/gr.norm().clamp_min(1e-12))
                assert abs(value-other)<=2e-5+2e-4*abs(value) and rel<2e-4, (ds,n,ref,rel)
                equivalence.append(dict(batch=batch_idx,n=n,reference=ref,alternative=alt,value_error=abs(value-other),encoder_gradient_relative_l2=rel))
                del gr,ga
            order=list(ARMS); shift=(seed+batch_idx+n)%len(order); order=order[shift:]+order[:shift]
            costs={}
            for arm in order:
                def step():
                    trainer.opt.zero_grad(set_to_none=True)
                    u,v=A.batch_descriptors(trainer.eU,trainer.eV,bu,bv,coords)
                    value=loss(u,v,arm,perm); value.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, trainer.cfg.clip_norm)
                    trainer.opt.step()
                    return value.detach()
                for _ in range(1 if gate else 3):
                    value=step(); assert torch.isfinite(value); del value
                trainer.opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
                base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                times=[]
                for _ in range(2 if gate else 10):
                    torch.cuda.synchronize(); ts=time.perf_counter(); value=step(); torch.cuda.synchronize()
                    times.append(time.perf_counter()-ts); assert torch.isfinite(value); del value
                costs[arm]=dict(median_seconds=statistics.median(times),all_seconds=times,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),baseline_bytes=base,
                    peak_increment_bytes=torch.cuda.max_memory_allocated()-base)
                trainer.opt.zero_grad(set_to_none=True)
            records.append(dict(batch=batch_idx,n_pos=n,order=order,costs=costs,
                batch_sha256=hashlib.sha256(repr(hb.cores).encode()+coords.tobytes()+hb.y0.tobytes()+hb.x0.tobytes()+hb.k.tobytes()+hb.flip.tobytes()).hexdigest()))
            R.write(out/'PROGRESS.json',dict(completed=len(records),expected=3 if gate else 9,utc=R.environment()['utc']))
            print('COST',ds,seed,batch_idx,n,flush=True)
    assert A.state_hash(trainer.eU,trainer.eV)==initial
    assert {r['role'] for r in guard.reads}<={'FIT','CAL','DEV'}
    result=dict(status='COMPLETE',freeze_sha256=freeze,seconds=time.time()-started,ds=ds,seed=seed,gate=gate,
        job=job,rows=records,equivalence=equivalence,environment=R.environment(),gpu=torch.cuda.get_device_name(),
        source_checkpoint_sha256=R.sha(model),source_completion_sha256=R.sha(source/'COMPLETE.json'),
        numerical_roles=['FIT'],integrity_roles=sorted({r['role'] for r in guard.reads}),weights_unchanged=True,summaries={},
        scope='Fixed real FIT crops: zero_grad, encoder forward, sampling descriptors, objective, encoder backward, gradient clip and zero-lr AdamW step. Image loading/crop draw excluded. No parameter learning or pose evaluation.',
        limitations=['Zero-lr fixed-weight steps measure computation, not convergence or task quality.',
        'Checkpointed row blocks preserve full BCE/NCE objectives; fixed block256 is not claimed globally optimal.',
        'Current checkpoints are VCS-trained; all objectives share them. Six jobs are paired device measurements, not independent task tests.'])
    R.write(out/'COMPLETE.json',result)
    R.write(out/f'ATTEMPT_{job}.json',dict(status='COMPLETE',job=job,seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--gate',action='store_true')
    p.add_argument('--ds',choices=('bf','epfl'));p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.cpu_check: cpu_check()
    else:
        assert args.ds and args.seed
        gpu_run(args.ds,args.seed,args.gate)
