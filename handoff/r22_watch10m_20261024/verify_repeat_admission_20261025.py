"""Independent read-only audit of repeat diagnostic gates and pool admission."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
from analyze_reg_trajectory_20261025 import read,sha,write,finite,norm,equal
from analyze_gradpair_20261025 import METHODS,FACTORS,STEPS

BASE=Path(__file__).resolve().parents[2]
PACK=BASE/'handoff/r22_reg_repeat_20261025'


def main():
    f=read(PACK/'FREEZE.json');fh=sha(PACK/'FREEZE.json')
    for rel,h in f['files'].items():assert sha(BASE/rel)==h,rel
    for name in ('CPU_CHECK.json','STATIC_CHECK.json'):
        d=read(PACK/name);assert d['status']=='PASS' and d['freeze_sha256']==fh
    assert len(read(PACK/'CPU_CHECK.json')['repeat_checks'])==4
    admission=read(PACK/'ADMISSION.json')
    if 'full' not in admission['groups']:
        print(json.dumps(dict(status='WAITING_FOR_BOTH_GPU_GATES',groups=admission['groups'])));return
    assert admission['counts']==dict(gates=2,full=18)
    sources=read(BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json')
    alljobs=[j for p in sources for j in read(p)]
    assert len(alljobs)==len({j['name'] for j in alljobs})
    for group,n in [('gates',2),('full',18)]:
        p=PACK/f'JOBS_{group}.json';assert sources.count(str(p))==1
        jobs=read(p);assert len(jobs)==n and all(j['freeze_sha256']==fh for j in jobs)
    locks=read(PACK/'BASELINE_LOCK.json');source=read(PACK/'SOURCE_LOCK.json');gates=[]
    for ds in ('bf','epfl'):
        p=PACK/'runs'/f'gate_{ds}';r=read(p/'COMPLETE.json');key=f'{ds}_vcs_exact_1001'
        assert sha(p/'COMPLETE.json')==admission['gpu_gate_hashes'][ds]
        assert r['freeze_sha256']==fh and r['status']=='COMPLETE' and r['gate']
        assert r['n_problems']==1 and r['n_paths']==48 and r['repeats']==4 and r['adam_steps']==100
        assert r['starts']==3 and r['arms']==list(METHODS)
        assert r['balanced_order'] and r['torch_rng_unchanged'] and r['source_unchanged'] and r['roles_read']==['CAL']
        assert r['source']==source[key] and r['baseline']==locks[key]
        assert sha(r['source']['checkpoint'])==r['source']['sha256']
        previous=read(locks[key]['completion'])
        for k in ('host','gpu_name','torch','numpy','cuda_runtime','cudnn','precision'):
            assert r['environment'][k]==previous['environment'][k]
        for name,k in [('TRAJECTORIES.jsonl','trajectory_sha256'),('INPUT_TRACE.json','input_trace_sha256')]:assert sha(p/name)==r[k]
        assert {q['role'] for q in read(p/'INPUT_TRACE.json')}=={'CAL'}
        assert len(list(p.glob('ATTEMPT_*')))==1 and not list(p.glob('FAILURE_*'))
        rows=list(map(json.loads,(p/'TRAJECTORIES.jsonl').read_text().splitlines()));assert len(rows)==1
        q=rows[0];finite(q);assert len(q['repeats'])==4
        initial=[a['initial_pose'] for a in q['repeats'][0]['arms']['P1_NATIVE']['paths']]
        for rep in q['repeats']:
            i=rep['repeat'];assert rep['order']==list(METHODS[i:]+METHODS[:i])
            assert list(rep['arms'])==rep['order']
            for position,m in enumerate(rep['order']):
                a=rep['arms'][m];assert a['position']==position and len(a['paths'])==3
                assert [x['initial_pose'] for x in a['paths']]==initial
                assert a['selected_by_common_cosine']==max(range(3),key=lambda k:a['paths'][k]['final_cosine'])
                for path in a['paths']:
                    assert path['arm']==m and (path['direction'],path['norm_source'])==FACTORS[m]
                    tr=path['trajectory'];assert [x['step'] for x in tr]==list(STEPS)==[x['adam_step'] for x in tr]
                    assert path['final_pose']==tr[-1]['pose'] and equal(path['final_error_native'],tr[-1]['error_native'])
                    for x in tr:
                        target=x['gradient_norms'][path['norm_source']];direction=x['gradient_norms'][path['direction']]
                        assert math.isclose(norm(x['applied_ascent_gradient']),target,abs_tol=1e-8,rel_tol=5e-5)
                        if direction>1e-12:assert math.isclose(x['scale'],target/direction,abs_tol=1e-8,rel_tol=2e-5)
        assert all({rep['order'][i] for rep in q['repeats']}==set(METHODS) for i in range(4))
        gates.append(dict(dataset=ds,job=r['job'],seconds=r['seconds'],paths=48,actual_updates=4800))
    state=read(BASE/'handoff/r22_task_diagnostics_20261023/POOL_STATE.json')['submitted']
    report=dict(status='FULL_ADMISSION_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=fh,
        frozen_files_verified=len(f['files']),cpu_job='1063729',gates=gates,shared_gate_seconds=sum(g['seconds'] for g in gates),
        full_count=18,unique_manifests_and_job_names=True,balanced_order_verified=True,source_and_initial_environment_verified=True,
        actual_applied_norm_and_scale_verified=True,cal_only=True,no_training=True,
        full_submitted={j['name']:state[j['name']] for j in read(PACK/'JOBS_full.json') if j['name'] in state})
    write(PACK/'ADMISSION_VERIFICATION.json',report)
    p=BASE/'handoff/r22_next_round_20261025/STATUS.json';d=read(p);d.setdefault('REG_REPEAT',{}).update(report);write(p,d)
    print(json.dumps({k:v for k,v in report.items() if k!='full_submitted'}));print('submitted',len(report['full_submitted']))


if __name__=='__main__':main()
