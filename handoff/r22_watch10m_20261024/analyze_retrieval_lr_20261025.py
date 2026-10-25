"""Read-only audit of six LR runs with six frozen historical controls. No TEST."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import watch10m as w

P = w.BASE/'handoff/r22_retrieval_lr_20261025'
SEEDS = (901,902,903)
METRICS = ('best_rsum','last_rsum','mean_last5_rsum','mean_last10_rsum')
RECALLS = ('i2t_R1','i2t_R5','i2t_R10','t2i_R1','t2i_R5','t2i_R10')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def close(a,b): assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)

def gate_audit():
    freeze=w.read(P/'FREEZE.json');fh=sha(P/'FREEZE.json')
    for path,h in freeze['files'].items():assert sha(w.BASE/path)==h,path
    cpu=w.read(P/'checks/CPU.json');g=w.read(P/'gates/COMPLETE.json')
    assert cpu['status']=='PASS' and cpu['freeze_sha256']==fh
    assert g['status']=='COMPLETE' and g['freeze_sha256']==fh and g['cpu_sha256']==sha(P/'checks/CPU.json')
    for name,r in [('CPU',cpu),('GPU',g)]:
        trace=P/'checks'/f'{name}_TRACE.json'
        assert sha(trace)==r['trace_sha256'] and w.read(trace)['status']=='PASS' and not w.read(trace)['forbidden_hits']
    assert {(r['method'],r['period']) for r in g['rows']}=={(m,p) for m in ('vcs','constant') for p in (10,30)}
    logs={}
    for r in g['rows']:
        d=P/'gates'/f"{r['method']}_period{r['period']}";log=w.read(d/'train_log.json')
        assert sha(d/'train_log.json')==r['log_sha256'] and sha(d/'last.pt')==r['last_checkpoint_sha256']
        lr=[2e-4*.1**(e//r['period']) for e in range(40)];counts=[8]+[1]*39
        assert [x['steps_in_epoch'] for x in log['epochs']]==counts
        assert [x['lr'] for x in log['epochs']]==lr
        assert r['actual_adam_lr']==[[l] for l,n in zip(lr,counts) for _ in range(n)]
        assert r['updates']==47 and r['initial_fingerprint_rechecked'] and r['pool_checks']['ok']
        logs[(r['method'],r['period'])]=log
    # Independent equivalence check before the first differing LR. Exact
    # equality is expected with fixed RNG/precision and the unmodified fit.
    prefix={}
    for method in ('vcs','constant'):
        a,b=logs[(method,30)],logs[(method,10)]
        for x,y in zip(a['epochs'][:10],b['epochs'][:10]):
            assert x['mean_loss']==y['mean_loss'] and x['dev']==y['dev'],(method,x['epoch'])
        prefix[method]=dict(first10_epoch_losses_and_DEV_identical=True)
    return dict(status='PASS',freeze_sha256=fh,frozen_files=len(freeze['files']),
        gate_job=g['job'],gate_sha256=sha(P/'gates/COMPLETE.json'),gate_seconds=g['seconds'],
        prefix_equivalence=prefix,environment=g['environment'],precision=g['precision'],
        scope='Technical only; sparse gate has no task-performance interpretation.')

def unit(directory,period):
    r=w.read(directory/'COMPLETE.json');m=w.read(directory/'run_manifest.json')
    log=w.read(directory/'train_log.json');rows=log['epochs'];ep=r['endpoints']
    assert r['status']=='COMPLETE' and not r['smoke'] and r['pool_checks']['ok'] and r['log_checks']['ok']
    assert sha(directory/'train_log.json')==r['train_log_sha256']
    assert sha(directory/'best.pt')==r['best_checkpoint_sha256']
    assert [x['epoch'] for x in rows]==list(range(40)) and all(x['steps_in_epoch']==1133 for x in rows)
    assert m['cfg']['lr_update']==period and m['cfg']['num_epochs']==40
    for x in rows:
        close(x['lr'],2e-4*.1**(x['epoch']//period));close(x['dev']['rsum'],sum(x['dev'][k] for k in RECALLS))
        assert math.isfinite(x['mean_loss'])
    scores=[x['dev']['rsum'] for x in rows]
    metrics=dict(best_rsum=max(scores),last_rsum=scores[-1],mean_last5_rsum=mean(scores[-5:]),mean_last10_rsum=mean(scores[-10:]))
    for k in ('best_rsum','last_rsum','mean_last5_rsum'):close(ep[k],metrics[k])
    assert ep['best_epoch']==scores.index(max(scores))==w.read(directory/'done.json')['best']['epoch']
    trace=w.read(directory/f"OPEN_TRACE_{r['job']}.json");assert trace['status']=='PASS' and not trace['forbidden_hits']
    attempts=[json.loads(x) for x in (directory/'attempts.jsonl').read_text().splitlines()]
    starts=[x for x in attempts if x['event']=='start'];ends=[x for x in attempts if x['event']=='end']
    assert len(starts)==len(ends)==1 and not starts[0]['resumes_from_last_pt'] and ends[0]['status']=='COMPLETE'
    return r,m,dict(seed=r['seed'],method=r['method'],period=period,metrics=metrics,best_epoch_one_based=ep['best_epoch']+1,
        curve_rsum=scores,curve_loss=[x['mean_loss'] for x in rows],lr=[x['lr'] for x in rows],
        receipt_gpu_hours=r['seconds']/3600,completion_sha256=sha(directory/'COMPLETE.json'),environment=r['environment'],attempts=attempts)

def main(check_gate):
    if check_gate:
        r=gate_audit();w.write(P/'GATE_INDEPENDENT_VERIFICATION.json',r);print(json.dumps(r));return
    missing=[f'{m}_{s}' for s in SEEDS for m in ('vcs','constant') if not (P/'runs'/f'retrieval_{m}_s{s}'/'COMPLETE.json').exists()]
    if missing:print(json.dumps(dict(status='WAITING_COMPLETE_SIX',missing=missing)));return
    gate=gate_audit();locks=w.read(P/'SOURCE_LOCK.json')['controls'];units={}
    for seed in SEEDS:
        for method in ('vcs','constant'):
            lock=locks[f'{method}_{seed}'];old=Path(lock['directory'])
            for name,h in lock['files'].items():assert sha(old/name)==h
            ref,rm,u=unit(old,30);units[f'{method}_30_{seed}']=u
            r,m,u=unit(P/'runs'/f'retrieval_{method}_s{seed}',10);units[f'{method}_10_{seed}']=u
            assert r['freeze_sha256']==gate['freeze_sha256'] and r['technical_gate_sha256']==gate['gate_sha256']
            assert m=={**rm,'cfg':{**rm['cfg'],'lr_update':10}}
            for key in ('initial_model_sha256','sample_order_sha256','first_batch_sha256','precision','transformed_weight'):
                assert r[key]==ref[key],(seed,method,key)
            for key in ('gpu','driver','torch','cuda','cudnn','python','numpy'):
                assert r['environment'][key]==ref['environment'][key],key
            u['host_equal_to_control']=r['environment']['host']==ref['environment']['host']
            submitted=[x for x in (w.POOL/'SUBMITTED.tsv').read_text().splitlines() if f'\tr22_lrs10_{method}_s{seed}\t' in x]
            assert len(submitted)==1 and submitted[0].split('\t')[-1]==r['job']
    contrasts={}
    definitions={'vcs_schedule_gain': [('vcs',10,1),('vcs',30,-1)],
        'constant_schedule_gain':[('constant',10,1),('constant',30,-1)],
        'weight_schedule_interaction':[('vcs',10,1),('vcs',30,-1),('constant',10,-1),('constant',30,1)],
        'vcs_minus_constant_period10':[('vcs',10,1),('constant',10,-1)],
        'vcs_minus_constant_period30':[('vcs',30,1),('constant',30,-1)]}
    for name,terms in definitions.items():
        contrasts[name]={}
        for metric in METRICS:
            values=[sum(sign*units[f'{m}_{p}_{s}']['metrics'][metric] for m,p,sign in terms) for s in SEEDS]
            contrasts[name][metric]=dict(seeds=list(SEEDS),values=values,mean=mean(values),sd=stdev(values),positive=sum(v>0 for v in values))
    decision={name:(c['best_rsum']['mean']>=3 and c['best_rsum']['positive']>=2) for name,c in contrasts.items()}
    result=dict(status='COMPLETE_AUDIT_PENDING_INTERPRETATION',utc=w.utc(),freeze_sha256=gate['freeze_sha256'],
        analyzer_sha256=sha(Path(__file__)),series='r22_retrieval_lr_20261025:other_retrieval_lr_full',
        units=units,contrasts=contrasts,registered_bestDEV_signal=decision,gate=gate,
        means={f'{m}_{p}':{k:mean(units[f'{m}_{p}_{s}']['metrics'][k] for s in SEEDS) for k in METRICS} for m in ('vcs','constant') for p in (10,30)},
        cost=dict(new_full_receipt_gpu_hours=sum(u['receipt_gpu_hours'] for u in units.values() if u['period']==10),
            reused_control_receipt_gpu_hours=sum(u['receipt_gpu_hours'] for u in units.values() if u['period']==30),
            shared_gate_seconds_once=gate['gate_seconds'],scope='CPU, scheduler startup, previous failed studies separate; no cross-node speed claim.'),
        limitations=['Repeated development DEV; schedule selected after prior curves; three seeds including pilot901, not confirmation.',
            'lr_update10 changes both decay timing and number, not an isolated single-milestone intervention.',
            'Interaction differs from absolute VCS gain, and this lacks a new matched BCE arm.',
            'Report best, final and last5 separately. Same GPU/driver/software required; per-unit hosts disclosed.'])
    w.write(w.HERE/'ANALYSIS_RETRIEVAL_LR_ALL6_20261025.json',result)
    with (w.HERE/'RETRIEVAL_LR_ALL12_CURVES_20261025.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','method','period','epoch_one_based','lr','dev_rsum','fit_loss']);writer.writeheader()
        for u in units.values():
            for i,(score,loss,lr) in enumerate(zip(u['curve_rsum'],u['curve_loss'],u['lr'])):
                writer.writerow(dict(seed=u['seed'],method=u['method'],period=u['period'],epoch_one_based=i+1,lr=lr,dev_rsum=score,fit_loss=loss))
    print(json.dumps({k:result[k] for k in ('status','means','contrasts','registered_bestDEV_signal','cost')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-gate',action='store_true');a=p.parse_args();main(a.check_gate)
