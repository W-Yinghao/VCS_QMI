"""Idempotent one-shot technical admission into the existing unique GPU pool."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
sys.path.insert(0,str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def verify():
    freeze=sha(HERE/'FREEZE.json')
    for p,h in read(HERE/'FREEZE.json')['files'].items(): assert sha(BASE/p)==h,p
    cpu=read(HERE/'CPU_CHECK.json')
    assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze
    assert sha(HERE/'CPU_CHECK.json')==read(HERE/'ADMISSION.json')['cpu_check_sha256']
    jobs=read(HERE/'GATE_JOBS.json')
    missing=[j['name'] for j in jobs if not Path(j['completion']).exists()]
    if missing:return dict(status='WAITING_FOR_BOTH_GPU_GATES',missing=missing)
    gates=[];sources=read(HERE/'SOURCE_LOCK.json');arms={'vcs_exact','vcs_perm','bce','nce'}
    for j in jobs:
        r=read(j['completion']);ds=r['ds']
        assert r['status']=='COMPLETE' and r['gate'] and r['seed']==1001
        assert r['freeze_sha256']==j['freeze_sha256']==freeze
        assert set(r['roles_read'])<={'FIT','CAL'} and 'FIT' in r['roles_read']
        trace=Path(j['completion']).parent/'INPUT_TRACE.json'
        assert sha(trace)==r['input_trace_sha256']
        assert {x['role'] for x in read(trace)}==set(r['roles_read'])
        assert all(x['role'] in ('FIT','CAL') for x in read(trace))
        checks=r['checks'];assert len(checks)==4 and {c['arm'] for c in checks}==arms
        hashes=[]
        for c in checks:
            assert c['optimizer_and_rng_roundtrip'] and c['pose_gradient_finite']
            assert [x['step'] for x in c['steps']]==[3001,3002,3003]
            assert all(x['lr']==3e-4 and math.isfinite(x['loss']) and math.isfinite(x['grad_norm_preclip']) for x in c['steps'])
            hashes.append([x['batch_sha256'] for x in c['steps']])
            s=sources[f"{ds}_{c['arm']}_1001"];assert c['source']==s
            for k in ('checkpoint','metadata','completion','source_train'):assert sha(s[k])==s[k+'_sha256']
        assert all(h==hashes[0] for h in hashes)
        gates.append(dict(ds=ds,job=r['job'],path=j['completion'],sha256=sha(j['completion'])))
    assert {x['ds'] for x in gates}=={'bf','epfl'}
    full=read(HERE/'FULL_JOBS_PENDING_GATES.json')
    assert len(full)==24 and len({j['name'] for j in full})==24
    assert all(j['freeze_sha256']==freeze for j in full)
    return dict(status='GATES_VERIFIED_READY',gates=gates,freeze_sha256=freeze,
                full_manifest_sha256=sha(HERE/'FULL_JOBS_PENDING_GATES.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    with submission_lock():
        result=verify()
        if result['status']=='GATES_VERIFIED_READY' and not a.check_only:
            path=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json'
            old=read(path);manifest=str(HERE/'FULL_JOBS_PENDING_GATES.json')
            if manifest not in old:
                names={j['name'] for f in old for j in read(f)}
                assert not names&{j['name'] for j in read(manifest)}
                old.append(manifest);tmp=path.with_suffix('.long_admit')
                tmp.write_text(json.dumps(old,indent=2)+'\n');tmp.replace(path)
            result.update(status='FULL24_ADMITTED',utc=datetime.now(timezone.utc).isoformat())
            (HERE/'FULL_ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
