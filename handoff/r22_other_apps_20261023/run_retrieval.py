import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from common import BASE,HERE,read,write,sha,verify

UPSTREAM=BASE/'ssl_pilot_v9_r22a2_dev/v9_r22/vcsloss'
sys.path.insert(0,str(UPSTREAM))
import r22_student as R
import torch
import numpy as np
from losses_nce_w import WeightedInfoNCEObjective

METHODS=('vcs','bce','constant')
def protected(w):
    w=np.asarray(w,dtype=np.float64).clip(1e-6,1-1e-6)
    z=np.log(w)-np.log1p(-w)
    lo,hi=-60.,60.
    for _ in range(90):
        mid=(lo+hi)/2;p=.1+.8/(1+np.exp(-(z+mid)))
        if p.mean()<.5:lo=mid
        else:hi=mid
    shift=(lo+hi)/2
    return .1+.8/(1+np.exp(-(z+shift))),shift

def prepare_and_check():
    reports={}
    for seed in (901,902,903):
        methods={};original={}
        for method,wname in (('vcs','VCS_Weight'),('bce','BCE_Weight')):
            inp=R.weight_inputs(seed,wname)
            ids=sorted(inp['weights']);w=np.array([inp['weights'][i] for i in ids])
            new,shift=protected(w)
            assert abs(new.mean()-.5)<1e-12 and new.min()>=.1 and new.max()<=.9
            order=np.argsort(w,kind='stable');assert np.all(np.diff(new[order])>=-1e-15)
            out=HERE/'weights'/f'{method}_{seed}.json'
            write(out,dict(weights=dict(zip(ids,map(float,new))),source=inp['paths'],source_sha256=inp['sha256'],
                           transform='0.1+0.8*sigmoid(logit(clip(w,1e-6,1-1e-6))+shift)',shift=shift,target_mean=.5))
            methods[method]=dict(path=str(out),sha256=sha(out),mean=float(new.mean()),minimum=float(new.min()),maximum=float(new.max()),
                 ess=float(new.sum()**2/np.square(new).sum()),original_mean=float(w.mean()))
            original[method]=w
        assert len(original['vcs'])==len(original['bce'])==130500
        out=HERE/'weights'/f'constant_{seed}.json';write(out,dict(weights={i:.5 for i in ids},target_mean=.5))
        methods['constant']=dict(path=str(out),sha256=sha(out),mean=.5,minimum=.5,maximum=.5,ess=len(ids))
        # Fixed positive-slope scalar critics use the same underlying score; audit the remaining numeric ordering difference.
        va=read(methods['vcs']['path'])['weights'];vb=read(methods['bce']['path'])['weights']
        delta=np.array([va[i]-vb[i] for i in ids])
        reports[str(seed)]=dict(methods=methods,shared_full_pool_normalizer=.55,protected_max_abs_difference=float(np.abs(delta).max()),
             protected_mean_abs_difference=float(np.abs(delta).mean()),
             equal_stable_order=bool(np.array_equal(np.argsort(original['vcs'],kind='stable'),np.argsort(original['bce'],kind='stable'))),
             interpretation='Both scalar critics are monotone in the same teacher score; this comparison primarily tests task-weight calibration, not a new ranking signal.')
    # Real objective identity and weighting/normalizer contracts, without oracle labels.
    torch.manual_seed(901);x=torch.randn(7,7,dtype=torch.float64,requires_grad=True);v=torch.linspace(.1,1.,7,dtype=torch.float64)
    obj=WeightedInfoNCEObjective('VCS_Weight');loss,_=obj(x,v,.55,'sum')
    z=x/.05;ell=.5*(-torch.log_softmax(z,1).diagonal()-torch.log_softmax(z,0).diagonal())
    expected=(v*ell/.55).sum()
    assert torch.equal(loss,expected) or torch.allclose(loss,expected,atol=1e-12,rtol=1e-12)
    assert torch.allclose(torch.autograd.grad(loss,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0],atol=1e-12,rtol=1e-12)
    write(HERE/'RETRIEVAL_CPU_CHECK.json',dict(status='PASS',freeze_sha256=sha(HERE/'FREEZE.json'),weights=reports,
         checks=['original N2F observed/weight hashes verified','all U means .5 and bounds [.1,.9]; anchors remain1 in host',
                 'transform monotone, no oracle labels used','symmetric weighted NCE value and score gradient equal explicit formula']))
    print('RETRIEVAL_CPU PASS',flush=True)

class Objective(WeightedInfoNCEObjective):
    def __init__(self,method):
        super().__init__('VCS_Weight' if method!='bce' else 'BCE_Weight');self.method=method
        self.name='nce_protected_mass_'+method
    def config(self):
        r=super().config();r.update(name=self.name,weights=f'{self.method}: fixed meanU=.5, floor=.1, ceiling=.9; A=1',
            normaliser='(14500+130500*.5)/145000=.55; fixed across all3 arms',interpretation='Task weights after monotone calibration, not clean posterior probabilities')
        return r

def inputs(seed,method):
    cpu=read(HERE/'RETRIEVAL_CPU_CHECK.json');item=cpu['weights'][str(seed)]['methods'][method]
    assert sha(item['path'])==item['sha256']
    original=R.weight_inputs(seed,'VCS_Weight' if method!='bce' else 'BCE_Weight')
    weights=read(item['path'])['weights'];assert set(weights)==set(original['weights'])
    return dict(rows=original['rows'],weights=weights,lock={'sha256':item['sha256'],'record':{'method':'protected_mass_'+method}},
                paths=original['paths'],sha256=original['sha256'],transformed=item)

def train_one(seed,method,smoke,finish):
    start=time.time();precision,env=R.gpu_setup();ctx=R.c1.load_context();spec=R.c1.protocol();cfg=ctx['cfg']
    inp=inputs(seed,method)
    name=f'retrieval_{method}_s{seed}';out=HERE/('smoke_retrieval' if smoke else 'runs')/name
    if (out/'COMPLETE.json').exists():return read(out/'COMPLETE.json')
    from v9.n2.train import WeightedStudent
    class Student(WeightedStudent):
        def validate(self,host):
            result=super().validate(host);R.c1.require_clean_trace(finish.forbidden_hits);return result
    batches=[]
    def collate(rows):
        if len(batches)<8:batches.append(hashlib.sha256('\n'.join(r['pair_id'] for r in rows).encode()).hexdigest())
        return ctx['collate'](rows)
    student=Student(cfg,inp['rows'],inp['weights'],collate,ctx['dev_images'][:16] if smoke else ctx['dev_images'],
        ctx['dev_captions'][:80] if smoke else ctx['dev_captions'],out,seed,spec['lr_mult'],device='cuda',weight_lock=inp['lock'],
        formal=True,loss_scale=spec['loss_scale'],stop_epoch_steps=8 if smoke else None,objective=Objective(method))
    checks=R.c1.pool_checks(student,inp['rows'],inp['weights'],seed,cfg,{**R.c1.PROTOCOL_POOL,'updates':spec['updates']})
    assert checks['ok'] and abs(student.mean_v_full-.55)<1e-12
    init=R.c1.initial_model_sha256(cfg,seed)
    R.c1.attempt_log(out,dict(event='start',environment=env,resumes_from_last_pt=(out/'last.pt').exists()))
    student.fit(1 if smoke else cfg.num_epochs,log_every=1 if smoke else 20)
    log=read(out/'train_log.json')
    if smoke:
        assert log['epochs'][0]['steps_in_epoch']==8 and all(math.isfinite(r['loss']) for r in log['steps'])
        post={'ok':True,'scope':'8 FIT updates plus first16 DEV images/80 captions; technical smoke only'}
    else:
        post,_=R.c1.log_checks(out,cfg.num_epochs,R.c1.PROTOCOL_POOL['steps_per_epoch']);assert post['ok']
    R.c1.require_clean_trace(finish.forbidden_hits)
    result=dict(status='COMPLETE',freeze_sha256=sha(HERE/'FREEZE.json'),ds='retrieval_protected',seed=seed,summaries={},name=name,
        method=method,seconds=time.time()-start,job=os.environ['SLURM_JOB_ID'],environment=env,gpu=env['gpu'],precision=precision,
        transformed_weight=inp['transformed'],pool_checks=checks,log_checks=post,initial_model_sha256=init,
        sample_order_sha256=checks['data_order_sha256'],first_batch_sha256=batches,endpoints=R.endpoints(log),
        train_log_sha256=sha(out/'train_log.json'),best_checkpoint_sha256=sha(out/'best.pt'),label='PILOT/DEVELOPMENT; DEV selection, no TEST',smoke=smoke)
    R.c1.attempt_log(out,dict(event='end',status='COMPLETE',elapsed_s=result['seconds']))
    # The caller writes completion only after final read-audit passes.
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--method',choices=METHODS);p.add_argument('--seed',type=int,default=901)
    a=p.parse_args();verify()
    finish=R.trace_open.install('c0' if a.mode=='check' else 'c1')
    if a.mode=='check':
        prepare_and_check();trace=finish(HERE/'records/retrieval_cpu_trace.json');assert trace['status']=='PASS';return
    cpu=read(HERE/'RETRIEVAL_CPU_CHECK.json');assert cpu['status']=='PASS' and cpu['freeze_sha256']==sha(HERE/'FREEZE.json')
    if a.mode=='gate':
        rows=[train_one(901,m,True,finish) for m in METHODS]
        assert len({r['initial_model_sha256'] for r in rows})==1
        assert len({r['sample_order_sha256'] for r in rows})==1
        assert all(r['first_batch_sha256']==rows[0]['first_batch_sha256'] for r in rows)
        trace=finish(HERE/'records/retrieval_gpu_trace.json');assert trace['status']=='PASS'
        write(HERE/'gates/retrieval/COMPLETE.json',dict(status='COMPLETE',freeze_sha256=sha(HERE/'FREEZE.json'),ds='retrieval_protected',seed=901,
             summaries={},job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),rows=rows,trace_sha256=sha(HERE/'records/retrieval_gpu_trace.json')))
        print('RETRIEVAL_GATE PASS',flush=True)
    else:
        assert read(HERE/'gates/retrieval/COMPLETE.json')['status']=='COMPLETE'
        r=train_one(a.seed,a.method,False,finish);out=HERE/'runs'/r['name']
        trace=finish(out/f'OPEN_TRACE_{os.environ["SLURM_JOB_ID"]}.json');assert trace['status']=='PASS'
        write(out/'COMPLETE.json',r);print('RETRIEVAL COMPLETE',r['name'],flush=True)

if __name__=='__main__':main()
