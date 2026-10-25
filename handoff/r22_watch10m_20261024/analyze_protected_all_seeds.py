"""Complete protected-retrieval review; read FIT/DEV artifacts, never TEST."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import watch10m as w

BASE=w.BASE/'handoff/r22_other_apps_20261023'
SEEDS=(901,902,903)
METHODS=('vcs','bce','constant')
RECALLS=('i2t_R1','i2t_R5','i2t_R10','t2i_R1','t2i_R5','t2i_R10')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def close(x,y):
    assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-8),(x,y)

def main():
    freeze=sha(BASE/'FREEZE.json'); frozen=w.read(BASE/'FREEZE.json')
    for p,h in frozen['files'].items():assert sha(w.BASE/p)==h,p
    cpu=w.read(BASE/'RETRIEVAL_CPU_CHECK.json')
    assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze
    units={};curves=[];weight_audits={};ledger=(w.POOL/'SUBMITTED.tsv').read_text().splitlines()
    for seed in SEEDS:
        receipts={};manifests={};weights={}
        for method in METHODS:
            key=f'{method}_{seed}';p=BASE/'runs'/f'retrieval_{method}_s{seed}'
            r=w.read(p/'COMPLETE.json');m=w.read(p/'run_manifest.json');receipts[method]=r;manifests[method]=m
            assert r['status']=='COMPLETE' and r['freeze_sha256']==freeze and not r['smoke']
            assert r['pool_checks']['ok'] and r['log_checks']['ok']
            assert sha(p/'train_log.json')==r['train_log_sha256']
            assert sha(p/'best.pt')==r['best_checkpoint_sha256']
            trace=w.read(p/f"OPEN_TRACE_{r['job']}.json")
            assert trace['status']=='PASS' and not trace['forbidden_hits']
            rows=w.read(p/'train_log.json')['epochs'];assert [x['epoch'] for x in rows]==list(range(40))
            assert all(x['steps_in_epoch']==1133 for x in rows)
            for x in rows:
                assert math.isfinite(x['mean_loss'])
                close(x['lr'],2e-4 if x['epoch']<30 else 2e-5)
                close(x['dev']['rsum'],sum(x['dev'][k] for k in RECALLS))
                curves.append(dict(seed=seed,method=method,epoch_one_based=x['epoch']+1,dev_rsum=x['dev']['rsum'],fit_loss=x['mean_loss'],lr=x['lr']))
            vals=[x['dev']['rsum'] for x in rows];best=vals.index(max(vals));ep=r['endpoints']
            assert ep['epochs']==40 and ep['best_epoch']==best
            close(ep['best_rsum'],max(vals));close(ep['best_rsum'],w.read(p/'done.json')['best']['rsum'])
            close(ep['last_rsum'],vals[-1]);close(ep['mean_last5_rsum'],mean(vals[-5:]))
            for k in RECALLS:close(ep[k],rows[best]['dev'][k])
            attempts=[json.loads(x) for x in (p/'attempts.jsonl').read_text().splitlines()]
            starts=[x for x in attempts if x['event']=='start'];ends=[x for x in attempts if x['event']=='end']
            assert len(starts)==len(ends)==1 and not starts[0]['resumes_from_last_pt']
            assert ends[0]['status']=='COMPLETE';close(ends[0]['elapsed_s'],r['seconds'])
            submitted=[x for x in ledger if f'\tr22_retrieval_{method}_s{seed}\t' in x]
            assert len(submitted)==1 and submitted[0].split('\t')[-1]==r['job']
            assert m['cfg']['num_epochs']==40 and m['cfg']['lr_update']==30 and m['objective']['tau']==.05
            assert (m['n_pool'],m['n_A'],m['n_U'],m['mean_v_full_pool'])==(145000,14500,130500,.55)
            meta=r['transformed_weight'];assert sha(meta['path'])==meta['sha256']==m['weight_lock_sha256']
            weights[method]=w.read(meta['path']);ww=list(weights[method]['weights'].values())
            assert len(ww)==130500 and .1<=min(ww)<=max(ww)<=.9
            close(mean(ww),.5);close(sum(ww)**2/sum(x*x for x in ww),meta['ess'])
            if method!='constant':
                for n,source in weights[method]['source'].items():assert sha(source)==weights[method]['source_sha256'][n]
            units[key]=dict(seed=seed,method=method,role='selected pilot' if seed==901 else 'development replication',
                endpoints={**ep,'mean_last10_rsum':mean(vals[-10:])},curve_rsum=vals,curve_loss=[x['mean_loss'] for x in rows],
                best_epoch_one_based=best+1,environment=r['environment'],precision=r['precision'],
                receipt_gpu_hours=r['seconds']/3600,attempts=attempts,job=r['job'],completion_sha256=sha(p/'COMPLETE.json'),
                best_checkpoint_sha256_verified=True,trace_sha256=sha(p/f"OPEN_TRACE_{r['job']}.json"),weights=meta)
        for attr in ['initial_model_sha256','sample_order_sha256','first_batch_sha256','precision']:
            assert len({json.dumps(r[attr],sort_keys=True) for r in receipts.values()})==1,(seed,attr)
        assert len({json.dumps(m['cfg'],sort_keys=True) for m in manifests.values()})==1
        for attr in ['torch','cuda','cudnn','python','numpy']:
            assert len({r['environment'][attr] for r in receipts.values()})==1
        assert len({tuple(sorted(x['weights'])) for x in weights.values()})==1
        assert weights['vcs']['source_sha256']['observed']==weights['bce']['source_sha256']['observed']
        ids=sorted(weights['vcs']['weights']);a=weights['vcs']['weights'];b=weights['bce']['weights']
        diffs=[abs(a[i]-b[i]) for i in ids]
        weight_audits[str(seed)]=dict(mean_abs_difference=mean(diffs),max_abs_difference=max(diffs),
            equal_stable_order=sorted(ids,key=a.get)==sorted(ids,key=b.get),
            gpu_models={k:r['gpu'] for k,r in receipts.items()},drivers={k:r['environment']['driver'] for k,r in receipts.items()})
    metrics=('best_rsum','last_rsum','mean_last5_rsum','mean_last10_rsum')
    means={m:{k:mean(units[f'{m}_{s}']['endpoints'][k] for s in SEEDS) for k in metrics} for m in METHODS}
    pairs={}
    for ctrl in ('bce','constant'):
        pairs[ctrl]={}
        for k in metrics:
            diff=[units[f'vcs_{s}']['endpoints'][k]-units[f'{ctrl}_{s}']['endpoints'][k] for s in SEEDS]
            pairs[ctrl][k]=dict(seeds=list(SEEDS),differences=diff,mean=mean(diff),sd=stdev(diff),positive=sum(x>0 for x in diff),
                replication_902_903_mean=mean(diff[1:]),replication_positive=sum(x>0 for x in diff[1:]))
    rule=w.read(w.BASE/'handoff/r22_retrieval_complete_20261024/PROTOCOL_ADDENDUM.json')['original_decision']
    result=dict(status='AUDIT_PASS',utc=w.utc(),series='r22_retrieval_complete_20261024:other_retrieval_complete',
        freeze_sha256=freeze,frozen_files_verified=len(frozen['files']),units=units,means=means,pairs_vcs_minus_control=pairs,
        weight_audits=weight_audits,original_pilot_decision_preserved=rule,
        cost=dict(new_six_gpu_hours=sum(u['receipt_gpu_hours'] for u in units.values() if u['seed']!=901),
            pilot_three_gpu_hours=sum(u['receipt_gpu_hours'] for u in units.values() if u['seed']==901),
            all_nine_gpu_hours=sum(u['receipt_gpu_hours'] for u in units.values()),full_attempts=9,failed_or_resumed_full_attempts=0,
            excludes='Prior teacher/weight fitting, shared gates, scheduler startup; not full historical project cost.'),
        limitations=['Selected pilot901 separate from development replication902/903; neither is TEST.',
            'Seed901 BCE uses L40S versus RTX for VCS/constant; replicas902/903 use same GPU model, drivers disclosed.',
            'Best DEV is primary; late endpoints do not replace it. Three seeds and reused DEV are not independent confirmation.',
            'Near-identical scalar weight ranking tests calibration shape, not distinct identification of noisy pairs.'],
        decision='Complete already-frozen shuffled-U experiments; no blind floor/mean/critic grid or longer unchanged schedule.')
    out=w.HERE/'ANALYSIS_PROTECTED_ALL_SEEDS_20261024.json';w.write(out,result)
    with (w.HERE/'PROTECTED_ALL_SEEDS_CURVES_20261024.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    print(json.dumps({k:result[k] for k in ['status','means','pairs_vcs_minus_control','weight_audits','cost']},ensure_ascii=False))

if __name__=='__main__':main()
