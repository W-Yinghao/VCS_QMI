"""Frozen selected KD checkpoints: FIT saturation and local auxiliary/anchor gradients."""
import argparse
import fcntl
import math
import os
import sys
import time
from common import BASE,HERE,read,write,sha,verify,receipt
CODE=BASE/'outputs_apps/r22/code_adc5de49'
sys.path[:0]=[str(CODE),str(CODE/'engine'),str(CODE/'engine/shared')]
import torch
import torch.nn.functional as F
import numpy as np

def objectives(p,qs,qt,tau):
    tp,tqs,tqt=[torch.tanh(x/tau) for x in (p,qs,qt)]
    vcs=-2*(tp-.5*tp.square()).mean()+(tqs+.5*tqs.square()).mean()+(tqt+.5*tqt.square()).mean()
    # Per-direction BCE is twice balanced BCE, matching VCS score derivatives at f=0.
    bce=2*F.softplus(-2*p/tau).mean()+F.softplus(2*qs/tau).mean()+F.softplus(2*qt/tau).mean()-4*math.log(2)
    return vcs,bce

def stats(x):
    x=x.detach().float().flatten()
    # Quantiles of at most 8192 uniformly spaced entries; fractions and means use all entries.
    sample=x[torch.linspace(0,len(x)-1,min(len(x),8192),device=x.device).long()]
    return dict(n=x.numel(),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),
                quantiles=torch.quantile(sample,torch.tensor([.01,.1,.5,.9,.99],device=x.device)).cpu().tolist())

def grad_compare(aux,anchor):
    a,b=aux.detach(),anchor.detach();an,bn=float(a.norm()),float(b.norm())
    return dict(aux_norm=an,anchor_norm=bn,aux_over_anchor=an/bn if bn else None,
                cosine=float((a*b).sum())/(an*bn) if an and bn else None)

def check():
    p=torch.tensor([-.3,.0,.4],dtype=torch.float64,requires_grad=True)
    qs=torch.tensor([[-.7,.05],[.0,.8]],dtype=torch.float64,requires_grad=True)
    qt=torch.tensor([[-.2,.4],[.1,.9]],dtype=torch.float64,requires_grad=True)
    for tau in (.07,.2,.5):
        a,b=objectives(p,qs,qt,tau)
        ga=torch.autograd.grad(a,(p,qs,qt),retain_graph=True)
        gb=torch.autograd.grad(b,(p,qs,qt),retain_graph=True)
        for x,u,v in zip((p,qs,qt),ga,gb):
            assert torch.allclose(u,v*(1-torch.tanh(x/tau).square()),atol=1e-12,rtol=1e-8)
        assert torch.autograd.gradcheck(lambda x,y,z:objectives(x,y,z,tau)[0],(p,qs,qt))
    write(HERE/'KD_CHECK.json',dict(status='PASS',freeze_sha256=sha(HERE/'FREEZE.json'),
          checks=['VCS/BCE score gradient factor is exactly 1-T^2','two-direction J matches finite-difference gradients']))
    print('KD_CHECK PASS',flush=True)

def run(seed):
    assert torch.cuda.is_available()
    assert read(HERE/'KD_CHECK.json')['freeze_sha256']==sha(HERE/'FREEZE.json')
    import guard
    from kd.runner import _teacher,_loader,_device,environment
    from kd.data import read_manifest,RoleDataset,canonical_hash
    from kd.models import create_model,feature_dim
    from kd.losses import DistillationLoss
    from kd.crd_adapter import OfficialCRD
    torch.set_num_threads(4)
    out=HERE/'runs'/f'kd_{seed}';out.mkdir(parents=True,exist_ok=True);started=time.time()
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COMPLETE.json').exists():
            assert read(out/'COMPLETE.json')['freeze_sha256']==sha(HERE/'FREEZE.json')
            return
        write(out/'RUNNING.json',receipt(started,'RUNNING'))
        results={}
        for arm,lam in [('vcscrd','0.8'),('dkd_vcscrd','3')]:
            cfgpath=CODE/'configs/kd'/f'{arm}_l{lam}_s{seed}.json';cfg=read(cfgpath)
            guard.install(guard.kd_roots(cfg))
            folder=BASE/'outputs_apps/r22/kd/runs'/f'{arm}_l{lam}_s{seed}'
            record=read(folder/'run.json')
            assert record['status']=='complete' and record['epochs_completed']==240
            assert not record['test_loaded_during_train'] and sha(folder/'best.pt')==record['best_checkpoint_sha256']
            ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            assert ckpt['config_sha256']==record['config_sha256']==canonical_hash(cfg)
            manifest=read_manifest(cfg['data']['manifest']);device=_device(cfg)
            fit=RoleDataset(cfg['data']['root'],manifest,'fit',augment=True,seed=seed)
            student=create_model(cfg['student'],manifest['num_classes'],seed);sdim=feature_dim(student)
            teacher,tdim,_=_teacher(cfg,manifest,device)
            losses=DistillationLoss(sdim,tdim,cfg['loss'])
            losses.crd=OfficialCRD(sdim,tdim,fit.ids,fit.labels[fit.ids].tolist(),cfg['loss']['crd'],seed)
            student.load_state_dict(ckpt['student']);losses.load_state_dict(ckpt['losses'])
            student.to(device).train();losses.to(device).eval()
            # Training-mode BN uses batch statistics but does not write any checkpoint. Restore buffers after each probe.
            bn_saved={n:b.detach().clone() for n,b in student.named_buffers()}
            memory=losses.crd.loss.contrast
            bank_before=[x.detach().clone() for x in (memory.memory_v1,memory.memory_v2)]
            rows=[]
            # Fixed diagnostic augmentation/order epoch shared across both recipes, separate from selected epoch.
            for batch,(images,labels,ids) in enumerate(_loader(fit,cfg,train=True,epoch=0)):
                if batch==8:break
                images,labels=images.to(device),labels.to(device)
                logits_s,feat_s=student(images)
                with torch.no_grad():logits_t,feat_t=teacher(images)
                anchor,_=losses(logits_s,logits_t,feat_s,feat_t,labels,ckpt['epoch'])
                ga=torch.autograd.grad(anchor,feat_s,retain_graph=True)[0]
                fs,ft=losses.crd.loss.embed_s(feat_s),losses.crd.loss.embed_t(feat_t)
                positive=(fs*ft).sum(-1)
                positives=np.asarray([losses.crd.lookup[int(i)] for i in ids],dtype=np.int64)
                neg=[];all_ids=np.arange(losses.crd.n_data)
                for i,pos in zip(ids,positives):
                    rng=np.random.default_rng(seed+int(i)*9176)
                    pool=all_ids[all_ids!=pos]
                    neg.append(rng.choice(pool,size=losses.crd.k,replace=losses.crd.k>len(pool)))
                negatives=torch.tensor(np.stack(neg),device=device)
                mt=memory.memory_v2[negatives].detach();ms=memory.memory_v1[negatives].detach()
                qs=torch.bmm(mt,fs[:,:,None]).squeeze(-1);qt=torch.bmm(ms,ft[:,:,None]).squeeze(-1)
                # Only current-positive VCS and matched BCE are compared here; neither memory is updated.
                row=dict(batch=batch,ids=ids.tolist(),cos_positive=stats(positive),cos_qs=stats(qs),cos_qt=stats(qt),
                         teacher_correct=int((logits_t.argmax(1)==labels).sum()),student_correct=int((logits_s.argmax(1)==labels).sum()),temperature={})
                target=(logits_t.argmax(1)==labels)&(logits_s.argmax(1)!=labels)
                row['teacher_correct_student_wrong']=int(target.sum())
                for tau in (.07,.2,.5):
                    lv,lb=objectives(positive,qs,qt,tau)
                    gv=torch.autograd.grad(cfg['loss']['crd_weight']*lv,feat_s,retain_graph=True)[0]
                    gb=torch.autograd.grad(cfg['loss']['crd_weight']*lb,feat_s,retain_graph=True)[0]
                    lp=-2*(torch.tanh(positive/tau)-.5*torch.tanh(positive/tau).square()).mean()
                    gp=torch.autograd.grad(cfg['loss']['crd_weight']*lp,feat_s,retain_graph=True)[0]
                    measures={}
                    for name,scores in [('P',positive),('Qs',qs),('Qt',qt)]:
                        t=torch.tanh(scores/tau);factor=1-t.square()
                        measures[name]=dict(abs_T_ge_099=float((t.abs()>=.99).float().mean()),
                                            exact_abs_T_1=float((t.abs()==1).float().mean()),
                                            gradient_factor=stats(factor),factor_lt_1e_3=float((factor<1e-3).float().mean()))
                    row['temperature'][str(tau)]=dict(vcs_loss=float(lv.detach()),bce_loss=float(lb.detach()),scores=measures,
                        vcs=grad_compare(gv,ga),matched_bce=grad_compare(gb,ga),positive=grad_compare(gp,ga),negative=grad_compare(gv-gp,ga),
                        teacher_correct_student_wrong=grad_compare(gv[target],ga[target]) if target.any() else None)
                    assert torch.isfinite(gv).all() and torch.isfinite(gb).all()
                # Exact implementation cross-check once, on disposable in-memory buffers. Restore immediately.
                if batch==0:
                    old=losses.crd.tau;losses.crd.tau=.07
                    reference=losses.crd.vcs(feat_s,feat_t,torch.tensor(positives,device=device),negatives)
                    actual=objectives(positive,qs,qt,.07)[0]
                    assert torch.allclose(actual.detach(),reference.detach(),atol=1e-6,rtol=1e-5)
                    with torch.no_grad():
                        memory.memory_v1.copy_(bank_before[0]);memory.memory_v2.copy_(bank_before[1])
                    losses.crd.tau=old
                rows.append(row)
                with torch.no_grad():
                    for n,b in student.named_buffers():b.copy_(bn_saved[n])
                print(arm,seed,'batch',batch,'diagnosed',flush=True)
                del mt,ms,qs,qt,positive,fs,ft,logits_s,feat_s
            assert all(torch.equal(a,b) for a,b in zip(bank_before,(memory.memory_v1,memory.memory_v2)))
            assert all(torch.equal(b,bn_saved[n]) for n,b in student.named_buffers())
            assert all(p.grad is None for p in teacher.parameters())
            assert sha(folder/'best.pt')==record['best_checkpoint_sha256']
            results[arm]=dict(lambda_selected=float(lam),checkpoint_sha256=record['best_checkpoint_sha256'],
                             selected_epoch=ckpt['epoch'],batches=rows)
            del student,teacher,losses,memory,bank_before,ckpt;torch.cuda.empty_cache()
        write(out/'COMPLETE.json',receipt(started,'COMPLETE',seed=seed,gpu=torch.cuda.get_device_name(),
              environment=environment(),results=results,guard=guard.status(),test_read=False,
              interpretation='FIT diagnostic at CAL-selected checkpoints; counterfactual temperature gradients do not establish retraining gains or explain early training'))
        print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);p.add_argument('--seed',type=int)
    a=p.parse_args();verify()
    if a.mode=='check':check()
    else:run(a.seed)
