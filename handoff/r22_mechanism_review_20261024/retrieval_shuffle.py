"""One frozen permutation of protected VCS weights within U for each seed."""
import argparse
import hashlib
import os
import sys
import time
from mechanism_common import BASE,HERE,read,write,sha,verify,load,check_pass
sys.path.insert(0,str(BASE/'handoff/r22_other_apps_20261023'))
O=load('protected_retrieval',BASE/'handoff/r22_other_apps_20261023/run_retrieval.py')
np,torch=O.np,O.torch
OLD=O.HERE
ORIGINAL_INPUTS=O.inputs

def permutation(seed,n):
    # Private stream, not training RNG; no search over permutations.
    return np.random.default_rng(202610240000+seed).permutation(n)

def check():
    f=verify();finish=O.R.trace_open.install('c0');reports={}
    for seed in (901,902,903):
        original=ORIGINAL_INPUTS(seed,'vcs');ids=sorted(original['weights']);w=np.array([original['weights'][k] for k in ids])
        perm=permutation(seed,len(ids));v=w[perm]
        assert len(ids)==130500 and np.array_equal(np.sort(v),np.sort(w)) and not np.array_equal(v,w)
        assert abs(v.mean()-.5)<1e-12 and abs(v.sum()**2/(v@v)-w.sum()**2/(w@w))<1e-7
        assert np.array_equal(perm,permutation(seed,len(ids)))
        path=HERE/'shuffle_weights'/f's{seed}.json';write(path,dict(weights=dict(zip(ids,map(float,v)))))
        reports[str(seed)]=dict(path=str(path),sha256=sha(path),source_sha256=original['transformed']['sha256'],
            n=len(ids),mean=float(v.mean()),min=float(v.min()),max=float(v.max()),ess=float(v.sum()**2/(v@v)),
            permutation_sha256=hashlib.sha256(perm.astype('<i8').tobytes()).hexdigest(),unchanged_pair_count=int((perm==np.arange(len(ids))).sum()))
    # Exact multiset preservation is the hypothesis contrast; objective unchanged.
    x=torch.randn(8,8,dtype=torch.float64,requires_grad=True);v=torch.linspace(.1,.9,8,dtype=torch.float64)
    obj=O.Objective('vcs');z=x/.05;expected=(v*(-.5*(torch.log_softmax(z,1).diagonal()+torch.log_softmax(z,0).diagonal()))/.55).sum()
    actual=obj(x,v,.55,'sum')[0];assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    assert torch.allclose(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0],atol=1e-12,rtol=1e-12)
    trace=finish(HERE/'checks/retrieval_read_trace.json');assert trace['status']=='PASS'
    write(HERE/'checks/retrieval.json',dict(status='PASS',freeze_sha256=f,weights=reports,trace_sha256=sha(HERE/'checks/retrieval_read_trace.json')))

def inputs(seed,method):
    current=O.HERE
    try:
        O.HERE=OLD;old=ORIGINAL_INPUTS(seed,'vcs')
    finally:O.HERE=current
    if method=='real_gate':return old
    item=read(HERE/'checks/retrieval.json')['weights'][str(seed)]
    assert sha(item['path'])==item['sha256'] and item['source_sha256']==old['transformed']['sha256']
    weights=read(item['path'])['weights'];assert set(weights)==set(old['weights'])
    return {**old,'weights':weights,'lock':{'sha256':item['sha256'],'record':{'method':'fixed_U_shuffle'}},'transformed':item}

class Objective(O.Objective):
    def __init__(self,method):
        super().__init__('vcs');self.method=method;self.name='nce_protected_'+method
    def config(self):
        r=super().config();r.update(weights='same protected VCS U multiset, one fixed within-U permutation; anchors=1',interpretation='Allocation mechanism control, not clean posterior inference');return r

def run(seed,gate):
    verify();check_pass('retrieval');started=time.time();finish=O.R.trace_open.install('c1')
    O.HERE=HERE;O.inputs=inputs;O.Objective=Objective
    if gate:
        rows=[O.train_one(901,m,True,finish) for m in ('real_gate','shuffle')]
        assert len({r['initial_model_sha256'] for r in rows})==len({r['sample_order_sha256'] for r in rows})==1
        assert rows[0]['first_batch_sha256']==rows[1]['first_batch_sha256']
        trace=finish(HERE/'checks/retrieval_gpu_trace.json');assert trace['status']=='PASS'
        write(HERE/'gates/retrieval/COMPLETE.json',dict(status='COMPLETE',freeze_sha256=sha(HERE/'FREEZE.json'),seconds=time.time()-started,job=os.environ['SLURM_JOB_ID'],rows=rows,trace_sha256=sha(HERE/'checks/retrieval_gpu_trace.json')))
    else:
        g=read(HERE/'gates/retrieval/COMPLETE.json');assert g['status']=='COMPLETE' and g['freeze_sha256']==sha(HERE/'FREEZE.json')
        p=HERE/'runs'/f'retrieval_shuffle_s{seed}'
        assert not (p/'COMPLETE.json').exists() and not (p/'train_log.json').exists(),'Do not duplicate or silently restart'
        r=O.train_one(seed,'shuffle',False,finish)
        trace=finish(p/f'OPEN_TRACE_{os.environ["SLURM_JOB_ID"]}.json');assert trace['status']=='PASS'
        write(p/'COMPLETE.json',r)
    print('SHUFFLE',seed,'GATE' if gate else 'FULL','PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--seed',type=int,default=901);a=p.parse_args()
    if a.mode=='check':check()
    else:run(a.seed,a.mode=='gate')
