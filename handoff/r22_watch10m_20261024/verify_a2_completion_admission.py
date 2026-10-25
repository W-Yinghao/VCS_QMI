"""Read-only independent metadata/hash audit; never trains, submits or opens TEST."""
import hashlib
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
H=BASE/'handoff/r22_a2_complete_v3_20261025'
OUT=Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r22_a2_complete_v3_20261025')
ARMS=('VCS_SHARP','VCSW_SHARP','BCE_SHARP')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def audit():
    freeze=read(H/'FREEZE.json');f=sha(H/'FREEZE.json')
    for p,h in freeze['files'].items():assert sha(p)==h,p
    result=dict(status='WAITING_CPU',freeze_sha256=f,frozen_files=len(freeze['files']))
    paths=read(BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json')
    names=[j['name'] for p in paths for j in read(p)];assert len(names)==len(set(names))
    assert len(paths)==len(set(paths))
    for old in ('r22_a2_complete_20261025','r22_a2_complete_v2_20261025'):
        assert not any('/'+old+'/' in p for p in paths)
    if not (H/'checks/CPU.json').exists():return result
    c=read(H/'checks/CPU.json');assert c['status']=='PASS' and c['freeze_sha256']==f
    assert c['trace_sha256']==sha(H/'checks/CPU_TRACE.json') and read(H/'checks/CPU_TRACE.json')['status']=='PASS'
    expect=[[2e-4*.1**(e//30),1e-3*.1**(e//30)] for e in range(40)]
    assert set(c['actual_adam_lr'])==set(ARMS)
    for lr in c['actual_adam_lr'].values():assert lr==expect
    assert set(c['input_checks'])=={f'{s}_{a}' for s in (902,903) for a in ARMS}
    for x in c['input_checks'].values():
        assert x['ok'] and x['expected_updates']==45320 and x['steps_per_epoch']==[1133]
        assert x['ess_absolute_delta']<=8*math.ulp(x['historical_ess_U'])
    result.update(status='CPU_PASS_WAITING_GPU_GATE',cpu_job=c['job'],cpu_seconds=c['seconds'])
    if not (OUT/'gates/COMPLETE.json').exists():return result
    g=read(OUT/'gates/COMPLETE.json');assert g['status']=='COMPLETE' and g['freeze_sha256']==f
    assert g['cpu_sha256']==sha(H/'checks/CPU.json') and g['trace_sha256']==sha(OUT/'gates/OPEN_TRACE.json')
    assert read(OUT/'gates/OPEN_TRACE.json')['status']=='PASS'
    assert {(x['seed'],x['arm']) for x in g['rows']}=={(s,a) for s in (902,903) for a in ARMS}
    assert len(g['rows'])==6
    for x in g['rows']:
        p=OUT/'gates'/f"s{x['seed']}__{x['arm']}"
        assert x['epochs']==40 and x['updates']==47 and x['model_changed'] and x['b_changed']
        assert sha(p/'train_log.json')==x['log_sha256'] and sha(p/'last.pt')==x['checkpoint_sha256']
        log=read(p/'train_log.json');counts=[8]+[1]*39
        assert [e['epoch'] for e in log['epochs']]==list(range(40))
        assert [e['steps_in_epoch'] for e in log['epochs']]==counts
        assert x['actual_adam_lr']==[v for v,n in zip(expect,counts) for _ in range(n)]
        assert len(x['first8_batch_sha256'])==8
        for e in log['epochs']:assert math.isfinite(e['mean_loss']) and math.isfinite(e['dev']['rsum'])
        ref=read(BASE/'outputs_v9_r22/vcsloss/runs'/f"s{x['seed']}__{x['arm'].removesuffix('_SHARP')}"/'unit_result.json')
        assert x['initial_model_sha256']==ref['initial_model_sha256'] and x['pool_checks']['data_order_sha256']==ref['data_order_sha256']
    for s in (902,903):
        bs=[x['first8_batch_sha256'] for x in g['rows'] if x['seed']==s];assert bs[0]==bs[1]==bs[2]
    spec=read(H/'JOBS_SPEC.json')
    for group in ('gates','full'):
        p=H/f'JOBS_{group}.json';assert paths.count(str(p))==1
        assert read(p)==[{**j,'freeze_sha256':f} for j in spec[group]]
    result.update(status='FULL_ADMISSION_VERIFIED',gate_job=g['job'],gate_seconds=g['seconds'],gate_updates=282,
        unique_full_jobs=6,unique_gate_manifest=True,unique_full_manifest=True,all_pool_names_unique=True)
    return result

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
