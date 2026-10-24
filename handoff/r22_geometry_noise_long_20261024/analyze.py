"""Audit a complete long-training dataset series; never select on partial runs."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ARMS=('vcs_exact','vcs_perm','bce','nce')


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]


def metrics(rr):
    scenes=defaultdict(list)
    for r in rr:scenes[r['scene']].append(r['E']['COS']['err_full'])
    errors=[v for vs in scenes.values() for v in vs]
    assert all(math.isfinite(x) for x in errors)
    sorted_errors=sorted(errors)
    return dict(success9=statistics.mean(sum(x<9 for x in v)/len(v) for v in scenes.values()),
                success42=statistics.mean(sum(x<42 for x in v)/len(v) for v in scenes.values()),
                median_err=statistics.median(errors),p95_err=sorted_errors[math.ceil(.95*len(errors))-1],
                failed100=sum(x>=100 for x in errors),n=len(errors),scenes=len(scenes))


def delta(a,b):
    d=[x-y for x,y in zip(a,b)]
    return dict(differences=d,mean=statistics.mean(d),positive=sum(x>0 for x in d))


def analyze(ds, rate):
    paths={f'{arm}_{seed}':HERE/'runs'/f'{ds}_{arm}_{seed}_n{round(rate*100)}' for arm in ARMS for seed in (1001,1002,1003)}
    missing=[k for k,p in paths.items() if not (p/'COMPLETE.json').exists()]
    if missing:return dict(status='WAITING_FOR_COMPLETE_SERIES',dataset=ds,missing=missing)
    freeze=sha(HERE/'FREEZE.json')
    for p,h in read(HERE/'FREEZE.json')['files'].items():assert sha(BASE/p)==h,p
    source=read(HERE/'SOURCE_LOCK.json');units={};samplers={};inputs={};corruption={};cost=[]
    for key,p in paths.items():
        r=read(p/'COMPLETE.json');arm=r['arm'];seed=r['seed']
        assert r['status']=='COMPLETE' and r['ds']==ds and key==f'{arm}_{seed}'
        assert r['freeze_sha256']==freeze and r['final_steps']==30000 and r['new_updates']==27000 and r['noise']==rate
        assert set(r['roles_read'])<={'FIT','CAL','DEV'}
        s=source[f'{ds}_{arm}_{seed}_n{round(rate*100)}'];assert s==r['source']
        for k in ('checkpoint','metadata','completion','source_train'):assert sha(s[k])==s[k+'_sha256']
        assert sha(p/'TRAIN.jsonl')==r['train_sha256']
        tr=rows(p/'TRAIN.jsonl');assert len(tr)==27000
        assert [x['step'] for x in tr]==list(range(3001,30001))
        assert all(x['lr']==(3e-4 if x['step']<=15000 else 3e-5 if x['step']<=24000 else 3e-6) for x in tr)
        assert all(math.isfinite(x[k]) for x in tr for k in ('loss','grad_norm_preclip','synchronized_step_seconds'))
        sig=[x['batch_sha256'] for x in tr]
        if seed in samplers:assert sig==samplers[seed]
        else:samplers[seed]=sig
        assert all(x['correspondence_noise']['rate_requested']==rate and
                   x['correspondence_noise']['rate_realized']==round(rate*128)/128 for x in tr)
        noise_sig=[x['correspondence_noise'] for x in tr]
        if seed in corruption:assert corruption[seed]==noise_sig
        else:corruption[seed]=noise_sig
        assert sha(p/'INPUT_TRACE.json')==r['input_trace_sha256']
        assert {x['role'] for x in read(p/'INPUT_TRACE.json')}==set(r['roles_read'])
        clean=BASE/'handoff/r22_geometry_long_v2_20261024/runs'/f'{ds}_{arm}_{seed}'
        assert (clean/'COMPLETE.json').exists(), 'Wait for complete fixed-budget clean comparator'
        cr=read(clean/'COMPLETE.json');assert cr['status']=='COMPLETE' and cr['final_steps']==30000
        assert sha(clean/'TRAIN.jsonl')==cr['train_sha256']
        assert sig==[x['batch_sha256'] for x in rows(clean/'TRAIN.jsonl')]
        unit={'arm':arm,'seed':seed,'noise':rate,'job':r['job'],'environment':r['environment']}
        for budget,label in ((3000,'short'),(30000,'long')):
            dev=p/f'dev_{budget}'/'DEV.jsonl';assert sha(dev)==r[f'dev_{label}_sha256']
            rr=rows(dev);assert len({(x['core'],x['perturbation']) for x in rr})==len(rr)
            assert all(x['role']=='DEV' and set(x['E'])=={'COS'} for x in rr)
            identity=[(x['core'],x['perturbation'],x['E']['COS']['theta'],x['E']['COS']['t']) for x in rr]
            if seed in inputs:assert identity==inputs[seed]
            else:inputs[seed]=identity
            for core in {x['core'] for x in rr}:assert {x['perturbation'] for x in rr if x['core']==core}==set(range(10))
            unit[label]=metrics(rr)
            for k in ('success9','success42','median_err'):assert abs(unit[label][k]-r[f'dev_{label}']['COS'][k])<1e-12
        unit['cal']={}
        for step in (10000,20000,30000):
            ck=p/'checkpoints'/f'step_{step}.pt';meta=read(ck.with_suffix('.json'))
            assert meta['identity']==r['identity'] and meta['step']==step and meta['sha256']==sha(ck)
            if step==30000:assert sha(ck)==r['final_checkpoint_sha256']
            rr=rows(p/f'cal_{step}'/'CAL.jsonl');assert all(x['role']=='CAL' for x in rr)
            unit['cal'][step]=metrics(rr)
        unit['fit_curve']={stop:dict(loss=statistics.mean(x['loss'] for x in tr if stop-1000<x['step']<=stop),
            grad=statistics.mean(x['grad_norm_preclip'] for x in tr if stop-1000<x['step']<=stop)) for stop in (4000,10000,20000,30000)}
        attempts=[read(x) for x in p.glob('ATTEMPT_*.json')];assert len(attempts)==1 and attempts[0]['status']=='COMPLETE'
        cost.append(dict(job=r['job'],seconds=r['seconds'],inherited_prefix_seconds=read(s['completion'])['seconds'],compute_seconds=r['synchronized_training_seconds'],
                         peak_allocated_bytes=r['peak_allocated_bytes'],environment=r['environment']))
        units[key]=unit
    clean_report=BASE/'handoff/r22_watch10m_20261024'/f'ANALYSIS_LONG_{ds.upper()}_20261024.json'
    assert clean_report.exists(), 'Complete clean comparator requires its full audit before final interpretation'
    clean_review=read(clean_report);assert clean_review['status']=='AUDIT_PASS'
    assert clean_review['freeze_sha256']==sha(BASE/'handoff/r22_geometry_long_v2_20261024/FREEZE.json')
    clean_units=clean_review['units']
    noise_penalty={arm:{k:delta([units[f'{arm}_{seed}']['long'][k] for seed in (1001,1002,1003)],
        [clean_units[f'{arm}_{seed}']['long'][k] for seed in (1001,1002,1003)])
        for k in ('success9','success42','median_err')} for arm in ARMS}
    interactions={arm:{k:delta(
        [units[f'vcs_exact_{seed}']['long'][k]-units[f'{arm}_{seed}']['long'][k] for seed in (1001,1002,1003)],
        [clean_units[f'vcs_exact_{seed}']['long'][k]-clean_units[f'{arm}_{seed}']['long'][k] for seed in (1001,1002,1003)])
        for k in ('success9','success42','median_err')} for arm in ('vcs_perm','bce','nce')}
    effects={} 
    for arm in ('vcs_perm','bce','nce'):
        effects[arm]={k:delta([units[f'vcs_exact_{s}']['long'][k] for s in (1001,1002,1003)],
                             [units[f'{arm}_{s}']['long'][k] for s in (1001,1002,1003)]) for k in ('success9','success42','median_err')}
    gains={arm:{k:delta([units[f'{arm}_{s}']['long'][k] for s in (1001,1002,1003)],
                         [units[f'{arm}_{s}']['short'][k] for s in (1001,1002,1003)]) for k in ('success9','success42','median_err')} for arm in ARMS}
    signal=all(effects[a]['success9']['mean']>=.01 and effects[a]['success9']['positive']>=2 and effects[a]['success42']['mean']>=-.0025 for a in ('bce','nce'))
    result=dict(status='AUDIT_PASS',utc=datetime.now(timezone.utc).isoformat(),dataset=ds,noise=rate,freeze_sha256=freeze,
        units=units,exact_minus=effects,long_minus_short=gains,noise_minus_clean=noise_penalty,noise_interaction=interactions,clean_audit_sha256=sha(clean_report),registered_practical_signal=signal,
        cost=dict(receipt_gpu_hours=sum(x['seconds'] for x in cost)/3600,inherited_prefix_gpu_hours=sum(x['inherited_prefix_seconds'] for x in cost)/3600,records=cost,
          note='New continuation/evaluation receipts only; inherited3000-step costs andGPUgates separate. Compute timer excludes crop assembly; total job receipts include it. No cross-hardware speedup claim.'),
        limitations=['Development data reused; no independent TEST evidence','Same-seed batches, fixed budgets and original encoders/objectives; new jobs constrained to L40S; inherited prefixes and reused clean controls can differ in hardware/driver',
          'Long-short changes duration and LR schedule together; samejob fixed endpoints','All seeds/cores retained; perturbations are not independent replicates; no automatic scientific stopping from reporting threshold'])
    path=BASE/'handoff/r22_watch10m_20261024'/f'ANALYSIS_NOISE_LONG_N{round(rate*100)}_{ds.upper()}_20261024.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(status='AUDIT_PASS',path=str(path),registered_practical_signal=signal,exact_minus=effects,long_minus_short=gains)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ds',choices=['bf','epfl'],required=True);p.add_argument('--noise',type=float,choices=[.2,.4],required=True);a=p.parse_args()
    print(json.dumps(analyze(a.ds,a.noise),ensure_ascii=False))
