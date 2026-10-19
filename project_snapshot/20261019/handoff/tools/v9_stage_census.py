"""Snapshot only existing DEV logs and manifests, never TEST or tensor checkpoints."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[2]
SRC=Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k')
OUT=ROOT/'handoff/evidence/v9_stage_20261017'


def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=[]; sources={}; collapse_rows=[]
    for p in sorted(SRC.glob('rho*/s*/unit_*')):
        if '_diag_' in p.name:continue
        f=p/'progress.jsonl'
        if not f.exists():continue
        raw=f.read_bytes();es=[]
        for line in raw.splitlines():
            try:e=json.loads(line)
            except json.JSONDecodeError:continue
            if 'dev_rsum' in e:es.append(e)
        if not es:continue
        rel=str(p.relative_to(SRC));sources[str(f)]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
        dst=OUT/'dev_logs'/p.parent.parent.name/p.parent.name/p.name;dst.mkdir(parents=True,exist_ok=True)
        (dst/'progress.jsonl').write_bytes(raw)
        m=json.loads((p/'run_manifest.json').read_text()); (dst/'run_manifest.json').write_text(json.dumps(m,indent=1))
        best=max(es,key=lambda e:e['dev_rsum']);e4=next((e for e in es if e['epoch']==4),None)
        row={'unit':rel,'formal':m['formal'],'done':(p/'done.json').exists(),'epochs_observed':len(es),'last_epoch':es[-1]['epoch'],
             'best_epoch':best['epoch'],'best_dev_rsum':best['dev_rsum'],'last_dev_rsum':es[-1]['dev_rsum'],
             'epoch4_dev_rsum':e4['dev_rsum'] if e4 else None}
        row['dev_below_half_epoch4_ever']= bool(e4 and any(e['epoch']>=5 and e['dev_rsum']<.5*e4['dev_rsum'] for e in es))
        if row['dev_below_half_epoch4_ever']:collapse_rows.append(rel)
        rows.append(row)
    from collections import Counter
    counts=Counter((r['unit'].split('/')[0], 'formal' if r['formal'] else 'tune','complete' if r['done'] else 'incomplete') for r in rows)
    snap={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':{'/'.join(k):v for k,v in counts.items()},
          'units':rows,'dev_below_half_epoch4_ever':collapse_rows,'metric_note':'Trajectory flag only; not the old pre-registered step-level collapse definition.',
          'sources':sources,'test_read':False}
    (OUT/'CENSUS.json').write_text(json.dumps(snap,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'UNITS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'utc':snap['utc'],'counts':snap['counts'],'dev_below_half_epoch4_ever':collapse_rows},indent=2))


if __name__=='__main__':main()
