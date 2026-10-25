"""Independent, read-only, stdlib verification; no training, submission or TEST."""
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PKG = BASE/'handoff/r22_a1_complete_20261025'
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/r22_a1_complete_20261025')

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def main():
    f=sha(PKG/'FREEZE.json');frozen=read(PKG/'FREEZE.json')['files']
    for p,h in frozen.items():assert sha(BASE/p)==h,p
    source=read(PKG/'SOURCE_LOCK.json');units=source['units']
    expected={f'{d}_{a}_{s}' for d in ('bf','epfl') for a in ('vcs_exact','bce','nce') for s in (1002,1003)}
    assert set(units)==expected and len(source['gate_groups'])==5
    spec=read(PKG/'JOBS_SPEC.json');assert len(spec['gates'])==5 and len(spec['full'])==12
    c=read(PKG/'checks/CPU.json');assert c['status']=='PASS' and c['freeze_sha256']==f and set(c['units'])==expected
    trace=PKG/'checks/CPU_TRACE.json';assert sha(trace)==c['trace_sha256'] and {r['role'] for r in read(trace)}<={'CAL'}
    for key,u in units.items():
        for p,h in u['files'].items():assert sha(p)==h,p
        a0=read(u['a0_completion']);assert a0['status']=='COMPLETE' and a0['final_steps']==30000 and a0['new_updates']==27000
        rows=c['units'][key]['first_five'];assert [x['step'] for x in rows]==list(range(3001,3006))
        with (Path(u['a0_completion']).parent/'TRAIN.jsonl').open() as stream:
            old=[json.loads(next(stream)) for _ in range(5)]
        assert [x['batch_sha256'] for x in rows]==[x['batch_sha256'] for x in old]
        assert c['units'][key]['source']==a0['source']
        job=next(j for j in spec['full'] if j['name']=='r22_a1c_'+key)
        assert '--nodelist='+u['environment']['host'] in job['args']
        assert '--partition='+u['environment']['slurm_partition'] in job['args']
    for ds in ('bf','epfl'):
        for seed in (1002,1003):
            rows=[c['units'][f'{ds}_{a}_{seed}']['first_five'] for a in ('vcs_exact','bce','nce')]
            assert rows[0]==rows[1]==rows[2]
    paths=read(BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json')
    assert paths.count(str(PKG/'JOBS_gates.json'))==1
    names=[j['name'] for p in paths for j in read(p)];assert len(names)==len(set(names))
    assert read(PKG/'JOBS_gates.json')==[{**j,'freeze_sha256':f} for j in spec['gates']]
    complete=[];waiting=[];gate_seconds=0
    for group in sorted(source['gate_groups']):
        p=OUT/'gates'/group/'COMPLETE.json'
        if not p.exists():waiting.append(group);continue
        g=read(p);assert g['status']=='COMPLETE' and g['freeze_sha256']==f and g['cpu_sha256']==sha(PKG/'checks/CPU.json')
        assert {x['key'] for x in g['checks']}=={k for k,u in units.items() if u['gate_group']==group}
        trace=p.parent/'INPUT_TRACE.json';assert sha(trace)==g['trace_sha256'] and {x['role'] for x in read(trace)}<={'FIT','CAL'}
        for x in g['checks']:
            key=x['key'];ref=units[key]['environment'];env=x['environment']
            for field in ('host','python','torch','numpy','cuda_runtime','cudnn','torch_threads','gpu_name','precision'):
                assert env[field]==ref[field],(key,field)
            assert all(x[k] for k in ('model_changed','optimizer_rng_roundtrip','environment_matched','baseline_batches_matched','cal_gradient_finite'))
            assert sha(p.parent/(key+'.pt'))==x['checkpoint_sha256']
            rows=x['steps'];assert [r['step'] for r in rows]==[3001,3002,3003]
            for r,cr in zip(rows,c['units'][key]['first_five']):
                assert r['batch_sha256']==cr['batch_sha256'] and r['geometry']['scenes']==cr['plan']
                assert r['lr']==3e-4 and math.isfinite(r['loss']) and math.isfinite(r['grad_norm_preclip'])
            assert sum(r['geometry']['active'] for r in rows)>0
        complete.append(group);gate_seconds+=g['seconds']
    full=not waiting and str(PKG/'JOBS_full.json') in paths
    if full:
        assert paths.count(str(PKG/'JOBS_full.json'))==1
        assert read(PKG/'JOBS_full.json')==[{**j,'freeze_sha256':f} for j in spec['full']]
    else:assert str(PKG/'JOBS_full.json') not in paths
    result=dict(status='FULL_ADMISSION_VERIFIED' if full else 'CPU_PASS_WAITING_GPU_GATES' if waiting else 'GATES_PASS_WAITING_ADMISSION',
        freeze_sha256=f,frozen_files=len(frozen),cpu_job=c['job'],cpu_seconds=c['seconds'],gates_complete=complete,gates_waiting=waiting,
        completed_gate_seconds=gate_seconds,full_admitted=full,new_full_jobs=12,unique_pool_names=True)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
