"""Descriptive full-run readout: paired training seeds, no test metrics or pseudo-replication."""
import csv
import statistics
from pathlib import Path
from common import ROOT, read, write


def collapse(log):
    epochs = {x['epoch']: x for x in log['epochs']}
    # Epoch records keep the original validation dictionary under 'dev'.
    def rsum(e):
        for k in ('dev', 'validation', 'val'):
            if k in e:
                return e[k]['rsum']
        return e.get('dev_rsum')
    base = rsum(epochs.get(4, {}))
    bad = []; details = []
    for epoch, e in epochs.items():
        if epoch < 5 or base is None: continue
        ss = [s for s in log['steps'] if s['epoch'] == epoch]
        hi = max((s['step'] for s in ss), default=-1)
        ss = [s for s in ss if s['step'] > hi-200]
        locked = sum(s['u_risk_unnormalised'] >= .39 for s in ss)/len(ss) if ss else None
        r = rsum(e)
        flag = r is not None and r < .5*base and locked is not None and locked >= .9
        if flag: bad.append(epoch)
        details.append({'epoch': epoch, 'rsum': r, 'locked_share': locked, 'n_logged_steps': len(ss), 'flag': flag})
    return {'severe_collapse': any(e+1 in bad for e in bad), 'flag_epochs': bad, 'windows': details}


def summarize():
    rows = []; errors = []
    for group in read(ROOT/'GROUPS.json'):
        for u in group['units']:
            p = ROOT/'results'/group['id']/f"{u['method']}_{u['scale']}"
            row = {'group': group['id'], 'seed': group['seed'], **u, 'status': 'PENDING'}
            if (p/'full_result.json').exists():
                r = read(p/'full_result.json'); log = read(p/'train_log.json')
                c = collapse(log)
                row.update(status='COMPLETE', best_dev_rsum=r['done']['best']['rsum'], best_epoch=r['done']['best']['epoch'],
                           severe_collapse=c['severe_collapse'], n_pool=r['n_pool'], gpu=r['hardware']['gpu'])
                write(p/'collapse_readout.json', c)
            elif (p/'progress.jsonl').exists():
                row['status'] = 'INCOMPLETE'
            rows.append(row)
        ep=ROOT/'results'/group['id']/'FAILED.json'
        if ep.exists(): errors.append(read(ep))
    complete = {(r['seed'],r['method'],r['scale']):r for r in rows if r['status']=='COMPLETE'}
    def contrast(a,b):
        vals=[]
        for seed in (901,902,903):
            x,y=complete.get((seed,*a)),complete.get((seed,*b))
            if x is not None and y is not None: vals.append({'seed':seed,'delta':x['best_dev_rsum']-y['best_dev_rsum']})
        ds=[x['delta'] for x in vals]
        return {'pairs':vals,'n':len(vals),'mean':statistics.mean(ds) if ds else None,'positive':sum(d>0 for d in ds),
                'range':[min(ds),max(ds)] if ds else None}
    contrasts={}
    for method in ('VCS_Weight','BCE_Weight','RankMass','Unweighted_A'):
        contrasts[f'{method}:SUM-MEAN']=contrast((method,'sum'),(method,'mean'))
    for scale in ('mean','sum'):
        for other in ('BCE_Weight','RankMass','Unweighted_A'):
            contrasts[f'{scale}:VCS-{other}']=contrast(('VCS_Weight',scale),(other,scale))
        for method in ('BCE_Weight','RankMass'):
            contrasts[f'{scale}:{method}-Unweighted_A']=contrast((method,scale),('Unweighted_A',scale))
    contrasts['sum:OracleDrop-OracleMask']=contrast(('OracleDrop','sum'),('OracleMask','sum'))
    contrasts['sum:VCS-RuLSIF_Weight']=contrast(('VCS_Weight','sum'),('RuLSIF_Weight','sum'))
    contrasts['sum:RuLSIF_Weight-Unweighted_A']=contrast(('RuLSIF_Weight','sum'),('Unweighted_A','sum'))
    main=[r for r in rows if not r['method'].startswith('Oracle')]
    decision={'status':'WAIT_FULL_MATRIX','main_complete':sum(r['status']=='COMPLETE' for r in main),'main_planned':27}
    def passes(key,margin,a,b,scale):
        c=contrasts[key]
        if c['n']!=3: return False
        ca=sum(complete[(s,a,scale)]['severe_collapse'] for s in (901,902,903))
        cb=sum(complete[(s,b,scale)]['severe_collapse'] for s in (901,902,903))
        return c['mean']>=margin and c['positive']>=2 and ca<=cb
    if decision['main_complete']==27:
        vcs=all(passes(f'sum:VCS-{other}',3,'VCS_Weight',other,'sum') for other in ('BCE_Weight','RankMass','RuLSIF_Weight'))
        generic={scale:any(passes(f'{scale}:VCS-Unweighted_A' if method=='VCS_Weight' else f'{scale}:{method}-Unweighted_A',
                                 5,method,'Unweighted_A',scale) for method in (('VCS_Weight','BCE_Weight','RankMass','RuLSIF_Weight') if scale=='sum' else ('VCS_Weight','BCE_Weight','RankMass')))
                 for scale in ('mean','sum')}
        decision.update(status='FULL_DEV_RESOURCE_DECISION',vcs_specific_confirmation_worth_testing=vcs,
                        generic_weighting_gate=generic,stop_current_route=not any(generic.values()),
                        warning='Resource gates, not significance; DEV is exploratory. No TEST read.')
    report={'units':rows,'contrasts':contrasts,'decision':decision,'failures':errors}
    write(ROOT/'READOUT.json', report)
    with (ROOT/'FULL_UNITS.csv').open('w',newline='') as f:
        cols=['group','seed','method','scale','status','best_dev_rsum','best_epoch','severe_collapse','n_pool','gpu']
        w=csv.DictWriter(f,cols); w.writeheader(); w.writerows(rows)
    return report


if __name__=='__main__':
    import json
    print(json.dumps(summarize()['decision'],ensure_ascii=False))
