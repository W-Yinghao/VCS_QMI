"""Fixed-final-checkpoint FIT gradient and disposable SGD diagnostics."""
import argparse
import copy
import hashlib
import sys
import time
from mechanism_common import BASE,HERE,read,write,sha,verify,check_pass,output,done
sys.path.insert(0,str(BASE/'handoff/r22_other_apps_20261023/engine'))
import guard
import torch
from kd import runner as K,decision as D
from kd.losses import DistillationLoss

def terms(s,t):
    u,v=D.coordinates(s),D.coordinates(t.detach());p=(u*v).sum(1)
    cosine=-p.mean();positive=-(p-.5*p.square()).mean();full=-D.quadratic_u(u,v)
    return dict(cosine=cosine,positive_quadratic=positive,full_vcs=full,reference=full-positive)

def margin(logits,labels):
    other=logits.clone();other.scatter_(1,labels[:,None],-torch.inf)
    return logits.gather(1,labels[:,None])[:,0]-other.max(1).values

def check():
    f=verify();torch.set_num_threads(2);torch.manual_seed(24)
    s=torch.randn(7,11,dtype=torch.float64,requires_grad=True);t=torch.randn_like(s,requires_grad=True)
    z=terms(s,t);v,_=D.decision_loss(s,t,'vcs');c,_=D.decision_loss(s,t,'cosine')
    assert torch.allclose(z['full_vcs'],v,atol=1e-12,rtol=1e-12)
    for x,y in ((z['full_vcs'],v),(z['cosine'],c)):
        gx,gt=torch.autograd.grad(x,(s,t),allow_unused=True,retain_graph=True);gy=torch.autograd.grad(y,s,retain_graph=True)[0]
        assert gt is None and torch.allclose(gx,gy,atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(lambda x:torch.stack(list(terms(x,t).values())),(s,))
    write(HERE/'checks/kd.json',dict(status='PASS',freeze_sha256=f,checks=['complete VCS equals production decision_loss value/gradient','cosine gradient equals production loss (constant offset removed)','all terms finite difference, teacher detached']))

def run(seed,anchor,device_name="cuda",max_batches=8,technical=""):
    verify();check_pass('kd');torch.set_num_threads(4);started=time.time();out=output(f'{technical}kd_terms_{anchor}_{seed}')
    sources=read(HERE/'SOURCE_LOCK.json')['kd'];results={};input_hashes=[]
    for source_method in ('cosine','vcs'):
        src=sources[f'{anchor}_{source_method}_{seed}'];p=BASE/src['directory'];cfg=read(p/'resolved_config.json');rec=read(p/'run.json')
        assert rec['status']=='complete' and rec['epochs_completed']==240 and not rec['test_loaded_during_train']
        assert sha(p/'last.pt')==src['sha256']
        guard.install(guard.kd_roots(cfg));manifest=K.read_manifest(cfg['data']['manifest']);device=torch.device(device_name)
        fit=K.RoleDataset(cfg['data']['root'],manifest,'fit',augment=True,seed=seed)
        ckpt=torch.load(p/'last.pt',map_location='cpu',weights_only=True);assert ckpt['epoch']==239
        assert ckpt['config_sha256']==K.canonical_hash(cfg)
        teacher,tdim,_=K._teacher(cfg,manifest,device);teacher_hash=K.state_digest(teacher)
        model=K.create_model(cfg['student'],manifest['num_classes'],seed)
        student_dim=K.feature_dim(model)  # CPU probe before moving model to the target device
        model=model.to(device);model.load_state_dict(ckpt['student'])
        lc=copy.deepcopy(cfg['loss']);lc.pop('decision',None);losses=DistillationLoss(student_dim,tdim,lc).to(device)
        assert not list(losses.parameters())
        lr=ckpt['optimizer']['param_groups'][0]['lr'];weight=cfg['loss']['decision']['weight'];rows=[];hashes=[]
        for batch,(images,labels,ids) in enumerate(K._loader(fit,cfg,True,0)):
            if batch==max_batches:break
            h=hashlib.sha256(ids.numpy().tobytes()+images.numpy().tobytes()).hexdigest();hashes.append(h)
            images,labels=images.to(device),labels.to(device)
            with torch.no_grad():lt,ft=teacher(images)
            model.load_state_dict(ckpt['student']);model.train();ls,fs=model(images)
            task,_=losses(ls,lt,fs,ft,labels,239);ga=torch.autograd.grad(task,ls,retain_graph=True)[0]
            z=terms(ls,lt);m=margin(ls,labels);gm=torch.autograd.grad(m.sum(),ls,retain_graph=True)[0]
            subsets={'all':torch.ones_like(labels,dtype=torch.bool),'teacher_correct_student_wrong':(lt.argmax(1)==labels)&(ls.argmax(1)!=labels),
                     'teacher_correct_student_margin_le1':(lt.argmax(1)==labels)&(m<=1)}
            row=dict(batch=batch,batch_sha256=h,positive_cosine=float((D.coordinates(ls)*D.coordinates(lt)).sum(1).mean().detach()),terms={})
            for name,val in z.items():
                g=torch.autograd.grad(val,ls,retain_graph=True)[0];norm=float(g.norm());an=float(ga.norm());info={}
                for label,mask in subsets.items():
                    info[label]=dict(n=int(mask.sum()),logit_margin_direction=float((-gm[mask]*g[mask]).sum()) if mask.any() else None,
                        anchor_cosine=float((g[mask]*ga[mask]).sum()/(g[mask].norm()*ga[mask].norm()).clamp_min(1e-30)) if mask.any() else None)
                row['terms'][name]=dict(value=float(val.detach()),gradient_norm=norm,weighted_over_anchor=weight*norm/an if an else None,subsets=info)
            # Matched disposable updates: clone model+momentum every time;
            # compare three equal-weight auxiliary objectives and anchor-only.
            # BN fixed at checkpoint for pre/post measurement, training mode for
            # actual update; an identical batch is used. No persistent writes.
            model.load_state_dict(ckpt['student']);model.eval()
            with torch.no_grad():before=margin(model(images)[0],labels)
            updates={}
            for name in ('anchor_only','cosine','positive_quadratic','full_vcs'):
                model.load_state_dict(ckpt['student']);model.train()
                opt=torch.optim.SGD([{'params':list(model.parameters()),'weight_decay':cfg['optimizer']['weight_decay']}],lr=cfg['optimizer']['lr'],momentum=cfg['optimizer'].get('momentum',.9),nesterov=False)
                opt.load_state_dict(copy.deepcopy(ckpt['optimizer']));assert opt.param_groups[0]['lr']==lr
                opt.zero_grad(set_to_none=True);s,f=model(images);a,_=losses(s,lt,f,ft,labels,239)
                total=a if name=='anchor_only' else a+weight*terms(s,lt)[name]
                total.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
                # Restore source BN buffers to isolate parameter update effects.
                with torch.no_grad():
                    for n,b in model.named_buffers():b.copy_(ckpt['student'][n].to(device))
                model.eval()
                with torch.no_grad():delta=margin(model(images)[0],labels)-before
                updates[name]={label:dict(n=int(mask.sum()),mean_margin_delta=float(delta[mask].mean()) if mask.any() else None,improved_count=int((delta[mask]>0).sum())) for label,mask in subsets.items()}
            row['disposable_update']=updates;rows.append(row)
            print(anchor,source_method,seed,batch,'PASS',flush=True)
        input_hashes.append(hashes)
        assert K.state_digest(teacher)==teacher_hash and all(p.grad is None for p in teacher.parameters())
        assert sha(p/'last.pt')==src['sha256']
        results[source_method]=dict(source=src,fixed_epoch=240,aux_weight=weight,optimizer_lr=lr,batches=rows)
        del model,teacher,losses,ckpt;torch.cuda.empty_cache()
    assert input_hashes[0]==input_hashes[1]
    status=guard.status();assert not status.get('forbidden_hits',[]),status
    done(out,started,seed=seed,anchor=anchor,technical=technical,max_batches=max_batches,execution_device=device_name,results=results,guard=status,environment=K.environment(),test_read=False,
         interpretation='Fixed FIT eight-batch local diagnostics at epoch240; equal lambda within each checkpoint. Copied SGD includes historical momentum; anchor-only matched update controls its contribution. No retraining/generalization claim; positive-only is a mechanism control, not VCS-QMI.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','cpu_path','gate','run']);p.add_argument('--seed',type=int);p.add_argument('--anchor',choices=['ce','dkd']);a=p.parse_args()
    if a.mode=='check':check()
    elif a.mode=='cpu_path':
        for anchor in ('ce','dkd'):
            for seed in (31415,16180,57721):run(seed,anchor,'cpu',1,'cpu_path_')
    else:
        start=time.time()
        try:run(a.seed,a.anchor,'cuda',1 if a.mode=='gate' else 8,'gate_' if a.mode=='gate' else '')
        except Exception as exc:
            import os,traceback
            write(HERE/('FAILURE_'+os.environ['SLURM_JOB_ID']+'.json'),dict(status='FAILED',seconds=time.time()-start,error=repr(exc),traceback=traceback.format_exc(),mode=a.mode,anchor=a.anchor,seed=a.seed))
            raise
