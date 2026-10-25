import sys
from mechanism_common import HERE,BASE,read,write,sha
sys.path.insert(0,str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock

def main():
    with submission_lock():
        freeze=sha(HERE/'FREEZE.json')
        for p,h in read(HERE/'FREEZE.json')['files'].items():assert sha(BASE/p)==h,p
        c=read(HERE/'checks/kd.json');assert c['status']=='PASS' and c['freeze_sha256']==freeze
        checks={}
        for anchor in ('ce','dkd'):
            for seed in (31415,16180,57721):
                p=HERE/'runs'/f'cpu_path_kd_terms_{anchor}_{seed}'/'COMPLETE.json';r=read(p)
                assert r['status']=='COMPLETE' and r['freeze_sha256']==freeze and r['max_batches']==1 and r['execution_device']=='cpu'
                assert not r['test_read'] and all(len(x['batches'])==1 for x in r['results'].values());checks[str(p)]=sha(p)
        ready=['gates'];gates={}
        for anchor in ('ce','dkd'):
            p=HERE/'runs'/f'gate_kd_terms_{anchor}_31415'/'COMPLETE.json'
            if p.exists():
                r=read(p);assert r['status']=='COMPLETE' and r['freeze_sha256']==freeze and r['max_batches']==1 and r['execution_device']=='cuda' and not r['test_read']
                assert all(len(x['batches'])==1 for x in r['results'].values());gates[anchor]=sha(p)
        if len(gates)==2:ready.append('full')
        pool=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json';paths=read(pool)
        old=BASE/'handoff/r22_mechanism_review_20261024';original=old/'JOBS_diagnostics.json';retained=old/'JOBS_registration_retained.json'
        jobs=read(original);keep=[x for x in jobs if x['group']!='other_mechanism_kd'];failed=[x for x in jobs if x['group']=='other_mechanism_kd']
        assert len(keep)==6 and len(failed)==6
        if str(original) in paths:
            assert all(not __import__('pathlib').Path(j['completion']).exists() for j in failed)
            write(retained,keep);paths=[str(retained) if p==str(original) else p for p in paths]
            write(HERE/'RETIREMENT.json',dict(reason='Six initialization failures; replaced by independently frozen repair with CPU and GPU gates',original_manifest_sha256=sha(original),retained_manifest_sha256=sha(retained),failed_jobs=failed,original_files_preserved=True,cost_record=str(old/'FAILURE_KD_20261024.json')))
        names={j['name'] for path in paths for j in read(__import__('pathlib').Path(path))}
        spec=read(HERE/'JOBS_SPEC.json')
        for group in ready:
            jobs=[{**j,'freeze_sha256':freeze} for j in spec[group]];p=HERE/f'JOBS_{group}.json'
            if p.exists():assert read(p)==jobs
            else:write(p,jobs)
            if str(p) not in paths:
                assert not names & {j['name'] for j in jobs}
                paths.insert(0,str(p));names.update(j['name'] for j in jobs)
        write(pool,paths)
        write(HERE/'ADMISSION.json',dict(status='ADMITTED_AVAILABLE_GROUPS',freeze_sha256=freeze,cpu_checks=checks,gates=gates,groups=ready,counts={g:len(spec[g]) for g in ready}))
        print('ADMITTED',ready,flush=True)

if __name__=='__main__':main()
