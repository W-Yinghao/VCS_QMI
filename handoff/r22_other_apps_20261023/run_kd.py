import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
from common import BASE,HERE,read,write,sha,verify
sys.path.insert(0,str(HERE/'engine'))
import guard
import torch
from kd import runner as K
from kd import decision as D
from kd.losses import DistillationLoss

def check():
    torch.set_num_threads(4);torch.manual_seed(27182)
    s=torch.randn(9,11,dtype=torch.float64,requires_grad=True)
    t=torch.randn_like(s,requires_grad=True)
    u,v=D.coordinates(s),D.coordinates(t.detach());score=u@v.T
    mask=~torch.eye(9,dtype=torch.bool);p=score.diagonal()
    dense=(p-.5*p.square()).mean()-(score[mask]+.5*score[mask].square()).mean()
    j=D.quadratic_u(u,v)
    assert torch.allclose(dense,j,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(dense,s,retain_graph=True)[0];gb=torch.autograd.grad(j,s,retain_graph=True)[0]
    assert torch.allclose(ga,gb,atol=1e-12,rtol=1e-12)
    for method in ('cosine','vcs','bce'):
        loss,_=D.decision_loss(s,t,method)
        grads=torch.autograd.grad(loss,(s,t),allow_unused=True,retain_graph=True)
        assert torch.isfinite(loss) and torch.isfinite(grads[0]).all() and grads[1] is None
        shifted,_=D.decision_loss(s+torch.randn(9,1,dtype=torch.float64)*10,t+torch.randn(9,1,dtype=torch.float64)*10,method)
        assert torch.allclose(loss,shifted,atol=1e-11,rtol=1e-11)
        perm=torch.randperm(11)
        permuted,_=D.decision_loss(s[:,perm],t[:,perm],method)
        assert torch.allclose(loss,permuted,atol=1e-12,rtol=1e-12)
    path=BASE/'ssl_pilot_app_r22/apps/r22/engine/kd/losses.py'
    spec=importlib.util.spec_from_file_location('original_kd_losses',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    labels=torch.arange(9)%11
    for anchor in ('none','dkd'):
        cfg=dict(anchor=anchor,feature='none',ce_weight=1.,dkd_beta=6.,decision={'method':'none','weight':0.})
        a=DistillationLoss(11,11,cfg);b=old.DistillationLoss(11,11,cfg)
        la,_=a(s,t,s,t,labels,10);lb,_=b(s,t,s,t,labels,10)
        assert torch.equal(la,lb)
        assert torch.equal(torch.autograd.grad(la,s,retain_graph=True)[0],torch.autograd.grad(lb,s,retain_graph=True)[0])
        assert not list(a.parameters())
    write(HERE/'KD_CPU_CHECK.json',dict(status='PASS',freeze_sha256=sha(HERE/'FREEZE.json'),checks=['exact U moments equal dense values and gradients',
          'teacher detached, finite gradients, logit translation invariance, common class permutation invariance',
          'CE/DKD anchors bitwise identical to original when auxiliary disabled; no trainable head']))
    print('KD_DECISION_CPU PASS',flush=True)

def gate():
    torch.set_num_threads(4);assert torch.cuda.is_available()
    cfg=K.read_config(HERE/'configs/kd_pilot_ce_none_l0_s27182.json');guard.install(guard.kd_roots(cfg))
    manifest=K.read_manifest(cfg['data']['manifest']);device=K._device(cfg)
    fit=K.RoleDataset(cfg['data']['root'],manifest,'fit',augment=True,seed=cfg['seed'])
    teacher,tdim,_=K._teacher(cfg,manifest,device);teacher_sha=K.state_digest(teacher)
    rows=[]
    for anchor in ('ce','dkd'):
        for method in ('none','cosine','vcs','bce'):
            name=f'kd_pilot_{anchor}_{method}_l{0 if method=="none" else 10}_s27182'
            c=K.read_config(HERE/'configs'/f'{name}.json')
            student=K.create_model(c['student'],manifest['num_classes'],c['seed']).to(device)
            initial=K.state_digest(student);losses=DistillationLoss(K.feature_dim(student),tdim,c['loss']).to(device)
            optimizer=torch.optim.SGD(student.parameters(),lr=c['optimizer']['lr'],momentum=.9,weight_decay=c['optimizer']['weight_decay'])
            trace=hashlib.sha256();values=[];student.train()
            for i,(images,labels,ids) in enumerate(K._loader(fit,c,True,0)):
                if i==4:break
                trace.update(ids.numpy().tobytes());trace.update(images.numpy().tobytes())
                images,labels=images.to(device),labels.to(device)
                ls,fs=student(images)
                with torch.no_grad():lt,ft=teacher(images)
                total,parts=losses(ls,lt,fs,ft,labels,0);optimizer.zero_grad();total.backward()
                assert torch.isfinite(total) and all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
                assert all(p.grad is None for p in teacher.parameters())
                optimizer.step();values.append(float(total))
            assert len(values)==4
            rows.append(dict(anchor=anchor,method=method,initial=initial,input_trace=trace.hexdigest(),losses=values))
            print('KD_GATE',anchor,method,'PASS',flush=True)
    assert len({r['initial'] for r in rows})==1 and len({r['input_trace'] for r in rows})==1
    assert K.state_digest(teacher)==teacher_sha
    write(HERE/'gates/kd/COMPLETE.json',dict(status='COMPLETE',freeze_sha256=sha(HERE/'FREEZE.json'),ds='kd_decision',seed=27182,summaries={},
        job=os.environ['SLURM_JOB_ID'],rows=rows,gpu=torch.cuda.get_device_name(),guard=guard.status()))

def run(name):
    cfg_path=HERE/'configs'/f'{name}.json';cfg=K.read_config(cfg_path)
    out=HERE/'runs'/name
    if (out/'COMPLETE.json').exists():
        assert read(out/'COMPLETE.json')['freeze_sha256']==sha(HERE/'FREEZE.json');return
    assert read(HERE/'gates/kd/COMPLETE.json')['status']=='COMPLETE'
    guard.install(guard.kd_roots(cfg));start=time.time()
    r=K.train(cfg_path,out)
    assert r['status']=='complete' and r['teacher_frozen_verified'] and not r['test_loaded_during_train']
    # Teacher correctness/margins are evaluated only on CAL, alongside the student's stored predictions.
    manifest=K.read_manifest(cfg['data']['manifest']);teacher,_,_=K._teacher(cfg,manifest,K._device(cfg))
    teacher_metrics,rows=K.infer(teacher,K.RoleDataset(cfg['data']['root'],manifest,'cal'),cfg,K._device(cfg))
    K.write_predictions(out/'teacher_cal_predictions.csv',rows)
    history=read(out/'history.json');assert len(history)==cfg['training']['epochs']
    order=hashlib.sha256(''.join(x['sample_order_sha256'] for x in history).encode()).hexdigest()
    write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=sha(HERE/'FREEZE.json'),ds='kd_decision',seed=cfg['seed'],summaries={},
         name=name,job=os.environ['SLURM_JOB_ID'],seconds=time.time()-start,gpu=torch.cuda.get_device_name(),config_sha256=sha(cfg_path),
         metrics=read(out/'best_cal_metrics.json'),teacher_metrics=teacher_metrics,student_initial_sha256=r['student_initial_state_sha256'],
         sample_order_sha256=order,history_sha256=sha(out/'history.json'),predictions_sha256=sha(out/'best_cal_predictions.csv'),
         teacher_predictions_sha256=sha(out/'teacher_cal_predictions.csv'),best_checkpoint_sha256=sha(out/'best.pt'),
         anchor=cfg['loss']['anchor'],decision=cfg['loss']['decision'],epochs=cfg['training']['epochs'],guard=guard.status(),label='PILOT/DEVELOPMENT; CAL only'))
    print('KD_DECISION COMPLETE',name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--name');a=p.parse_args();verify()
    if a.mode=='check':check()
    else:
        c=read(HERE/'KD_CPU_CHECK.json');assert c['status']=='PASS' and c['freeze_sha256']==sha(HERE/'FREEZE.json')
        if a.mode=='gate':gate()
        else:run(a.name)
