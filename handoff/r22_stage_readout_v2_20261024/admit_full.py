"""One-shot stage-policy technical gate audit/admission, no background worker."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def verify():
    freeze=sha(HERE/'FREEZE.json')
    for rel,h in read(HERE/'FREEZE.json')['files'].items(): assert sha(BASE/rel)==h,rel
    cpu=read(HERE/'CPU_CHECK.json')
    assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze
    assert sha(HERE/'CPU_CHECK.json')==read(HERE/'ADMISSION.json')['cpu_check_sha256']
    jobs=read(HERE/'GATE_JOBS.json')
    missing=[j['name'] for j in jobs if not Path(j['completion']).exists()]
    if missing:return dict(status='WAITING_FOR_BOTH_GPU_GATES',missing=missing)
    arms={'vcs_exact','vcs_perm','bce','nce'}
    policies={a+'__'+b for a in ('COS','J_EXACT') for b in ('COS','J_EXACT')}
    source=read(HERE/'SOURCE_LOCK.json'); gates=[]
    for j in jobs:
        path=Path(j['completion']);r=read(path);ds=r['ds']
        assert r['status']=='COMPLETE' and r['gate'] and r['weights_unchanged']
        assert r['freeze_sha256']==j['freeze_sha256']==freeze and r['seed']==1001 and r['numerical_roles']==['CAL']
        assert sha(path.parent/'ROWS.jsonl')==r['rows_sha256'] and sha(path.parent/'INPUT_TRACE.json')==r['input_trace_sha256']
        rows=[json.loads(x) for x in (path.parent/'ROWS.jsonl').read_text().splitlines()]
        assert len(rows)==8 and set(r['summaries'])==arms and len(r['technical_checks'])==8
        signatures={}
        for arm in arms:
            rr=[x for x in rows if x['arm']==arm];assert len(rr)==2
            signatures[arm]=[(x['core'],x['perturbation'],x['policies']['COS__COS']['theta'],x['policies']['COS__COS']['t']) for x in rr]
            for row in rr:
                assert row['role']=='CAL' and row['perturbation']==0 and set(row['policies'])==policies
                assert set(row['order'])==policies and len(row['order'])==4
                assert len({v['n_grid'] for v in row['policies'].values()})==1
                for v in row['policies'].values():
                    assert all(math.isfinite(v[k]) for k in ('err_full','F_true','F_hat','synchronized_wall_seconds'))
                    assert v['grid_calls']==21 and v['local_calls']==303
            assert set(r['summaries'][arm])==policies
        assert all(signatures[a]==signatures['vcs_exact'] for a in arms)
        for check in r['technical_checks']:
            assert len(check['checks'])==4
            assert {x['coarse']+'__'+x['local'] for x in check['checks']}==policies
            assert all(math.isfinite(x[k]) for x in check['checks'] for k in ('value_error','gradient_max_error','grid_max_error'))
        assert len(r['models'])==4
        for model in r['models']:
            expected=source[f"{ds}_{model['arm']}_1001"]
            assert model['unchanged'] and model['source']==expected
            for kind in ('completion','checkpoint','cal_rows'):assert sha(expected[kind])==expected[kind+'_sha256']
        attempts=list(path.parent.glob('ATTEMPT_*.json'));assert len(attempts)==1 and read(attempts[0])['status']=='COMPLETE'
        gates.append(dict(ds=ds,job=r['job'],sha256=sha(path),path=str(path)))
    assert {g['ds'] for g in gates}=={'bf','epfl'}
    full=read(HERE/'FULL_JOBS_PENDING_GATES.json');assert len(full)==24 and len({j['name'] for j in full})==24
    return dict(status='GATES_VERIFIED_READY',gates=gates,freeze_sha256=freeze,full_manifest_sha256=sha(HERE/'FULL_JOBS_PENDING_GATES.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    with submission_lock():
        result=verify()
        if result['status']=='GATES_VERIFIED_READY' and not a.check_only:
            path=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json';old=read(path);manifest=str(HERE/'FULL_JOBS_PENDING_GATES.json')
            if manifest not in old:
                current={j['name'] for f in old for j in read(f)}
                assert not current&{j['name'] for j in read(manifest)}
                old.append(manifest);tmp=path.with_suffix('.stage_admit');tmp.write_text(json.dumps(old,indent=2)+'\n');tmp.replace(path)
            result.update(status='FULL24_ADMITTED',utc=datetime.now(timezone.utc).isoformat(),admission_script_sha256=sha(Path(__file__)))
            (HERE/'FULL_ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
