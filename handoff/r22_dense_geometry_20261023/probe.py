"""FIT-only same-device descriptor objective costs and conditional permutation gradient errors."""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
import train as T

R=T.R;A=T.A;torch=T.torch;HERE=T.HERE;BASE=T.BASE

def loss_for(u,v,arm,perms):
    if arm=='vcs_dense':
        a,b=T.normalized(u,v);scores=a@b.transpose(-1,-2)
        p=scores.diagonal(dim1=-2,dim2=-1)
        return -(p-.5*p.square()).mean()+(scores+.5*scores.square()).mean()
    if arm=='vcs_perm8':
        return torch.stack([T.objective(u,v,'vcs_perm',p)[0] for p in perms]).mean()
    return T.objective(u,v,arm,perms[0])[0]

def value_grad(u,v,arm,perms):
    value=loss_for(u,v,arm,perms)
    grads=torch.autograd.grad(value,(u,v))
    return value.detach(),tuple(g.detach() for g in grads)

def flat(grads):
    return torch.cat([g.flatten() for g in grads])

def bench(u,v,arm,perms):
    for _ in range(3):
        value,grads=value_grad(u,v,arm,perms)
        assert torch.isfinite(value) and all(torch.isfinite(g).all() for g in grads)
        del value,grads
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
    elapsed=[]
    for _ in range(10):
        torch.cuda.synchronize();start=time.perf_counter()
        value,grads=value_grad(u,v,arm,perms)
        torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
        del value,grads
    return dict(median_seconds=statistics.median(elapsed),all_seconds=elapsed,
                peak_increment_bytes=torch.cuda.max_memory_allocated()-baseline,
                scope='objective value and gradient w.r.t. fixed descriptors; excludes encoder, sampling and permutation creation')

def run(ds,seed):
    started=time.time();torch.set_num_threads(4)
    T.freeze();assert torch.cuda.is_available()
    gate=json.loads((HERE/'CPU_CHECK.json').read_text());assert gate['status']=='PASS' and gate['freeze_sha256']==R.sha(HERE/'FREEZE.json')
    source=BASE/'handoff/r22_geometry_train_20261023/runs'/f'{ds}_vcs_exact_{seed}'
    clean=json.loads((source/'COMPLETE.json').read_text());assert clean['status']=='COMPLETE'
    checkpoint=source/'checkpoints'/f"step_{clean['selection']['step']}.pt"
    assert R.sha(checkpoint)==clean['selection']['checkpoint_sha256']
    unit=R.load_unit(BASE/'handoff/r22_c3_codex_20261023/units'/f'r22_pmi3_{ds}_MLPp1_s{seed}.json')
    guard,H=R.make_guard(unit)
    data_lock=json.loads(Path(unit['data_lock']['path']).read_text())
    if H:guard.verify_data_lock({'FIT'})
    else:
        for row in data_lock['files']:
            if row['role']=='FIT':assert R.sha(row['path'])==row['sha256']
    fit=A.fit_cores(guard,1.,seed) if H is None else H.fit_windows(guard)
    images=(A.load_learned_inputs if H is None else H.load_learned_inputs)(guard,fit,{'FIT'},'R22 dense reference FIT benchmark','cuda')
    cfg=A.TrainConfig(method='MLP_perm1',seed=seed,encoder_lr=3e-4,head_lr=3e-4,n_sc=16,n_pos=128,crop=160,grad_spot_every=100)
    trainer=T.GeometryTrainer(cfg,images,fit,'cuda','vcs_exact')
    R.CP.load(trainer,checkpoint,clean['identity'])
    # One fixed real FIT crop batch. A separate RNG gives nested positions across density levels.
    hb,bu,bv=trainer.draw()
    rng=R.np.random.default_rng(A.Q.seed_of(f'R22|dense-probe|{ds}|{seed}'))
    yx=rng.uniform(trainer.sampler.inner,160-trainer.sampler.inner-1,(16,2048,2))
    with torch.no_grad():
        uu,vv=A.batch_descriptors(trainer.eU,trainer.eV,bu,bv,yx)
    rows=[]
    for n in (128,512,2048):
        u=uu[:,:n].detach().clone().requires_grad_();v=vv[:,:n].detach().clone().requires_grad_()
        rg=R.np.random.default_rng(A.Q.seed_of(f'R22|dense-probe-perms|{ds}|{seed}|{n}'))
        def draws(k):
            return [torch.as_tensor(R.np.stack([rg.permutation(n) for _ in range(16)]),device='cuda') for _ in range(k)]
        perms=draws(8)
        exact,gexact=value_grad(u,v,'vcs_exact',perms)
        dense,gdense=value_grad(u,v,'vcs_dense',perms)
        assert torch.allclose(exact,dense,atol=2e-5,rtol=2e-4)
        gx=flat(gexact);gd=flat(gdense)
        relative=float(torch.linalg.vector_norm(gx-gd)/torch.linalg.vector_norm(gx).clamp_min(1e-12))
        assert relative<2e-4
        record=dict(n_pos=n,exact_dense_gradient_relative_l2=relative,costs={},permutation={})
        # Rotate order deterministically to limit order/thermal bias; timings remain descriptive.
        arms=['vcs_exact','vcs_dense','vcs_perm','vcs_perm8','bce','nce']
        shift=seed%len(arms);arms=arms[shift:]+arms[:shift]
        for arm in arms:record['costs'][arm]=bench(u,v,arm,perms)
        norm2=gx.square().sum().clamp_min(1e-24)
        for arm,k in (('vcs_perm',1),('vcs_perm8',8)):
            mean=torch.zeros_like(gx);errors=[];values=[]
            for i in range(64):
                val,grads=value_grad(u,v,arm,draws(k));g=flat(grads)
                mean.add_((g-mean)/(i+1))
                errors.append(float((g-gx).square().sum()/norm2));values.append(float(val))
            record['permutation'][arm]=dict(draws=64,mean_relative_squared_error=statistics.mean(errors),
                mean_gradient_relative_squared_error=float((mean-gx).square().sum()/norm2),
                mean_objective=statistics.mean(values),exact_objective=float(exact),
                scope='conditional descriptor-gradient Monte Carlo error; excludes scene/crop sampling and encoder-gradient uncertainty')
        rows.append(record)
        print('PROBE',ds,seed,n,flush=True)
    assert {r['role'] for r in guard.reads}<={'FIT','CAL','DEV'}
    out=HERE/'probes'/f'{ds}_{seed}';out.mkdir(parents=True,exist_ok=True)
    R.write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=R.sha(HERE/'FREEZE.json'),ds=ds,seed=seed,summaries={},
            seconds=time.time()-started,rows=rows,gpu=torch.cuda.get_device_name(),environment=R.environment(),job=os.environ['SLURM_JOB_ID'],
            source_checkpoint_sha256=R.sha(checkpoint),source_completion_sha256=R.sha(source/'COMPLETE.json'),
            numerical_roles=['FIT'],integrity_roles=sorted({r['role'] for r in guard.reads}),
            batch_sha256=hashlib.sha256(repr(hb.cores).encode()+yx.tobytes()+hb.y0.tobytes()+hb.x0.tobytes()).hexdigest(),
            limitations=['One fixed batch per dataset and trained seed; all methods timed on exactly the same descriptors/device.',
                        'Full dense baseline has quadratic materialized storage; blocked/checkpointed kernels could reduce memory at a different time cost.',
                        'Descriptor timing is not end-to-end training speedup.']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ds',choices=['bf','epfl'],required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();run(a.ds,a.seed)
