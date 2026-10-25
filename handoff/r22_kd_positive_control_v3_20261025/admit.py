"""Admit verified gates, then six controls to the existing pool, idempotently."""
import sys
from common import HERE,BASE,read,write,sha,verify
sys.path.insert(0,str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock
def main():
    with submission_lock():
        f=verify();c=read(HERE/'CPU_CHECK.json');assert c['status']=='PASS' and c['freeze_sha256']==f and len(c['rows'])==6
        static = read(HERE/'STATIC_CHECK.json'); assert static['status'] == 'PASS' and static['freeze_sha256'] == f
        groups=['gates'];gate_hashes={}
        for seed in (31415,16180):
            p=HERE/'gates'/f'{seed}.json'
            if p.exists():
                g=read(p);assert g['status']=='COMPLETE' and g['freeze_sha256']==f and g['technical']
                for anchor in ('ce','dkd'):
                    r=read(HERE/'runs'/f'gate_{anchor}_{seed}'/'COMPLETE.json')
                    assert r['status']=='COMPLETE' and r['freeze_sha256']==f and r['epochs']==1 and r['technical'] and not r['test_read']
                gate_hashes[str(seed)]=sha(p)
        if len(gate_hashes)==2:groups.append('full')
        pool=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json';paths=read(pool);spec=read(HERE/'JOBS_SPEC.json')
        names={j['name'] for path in paths for j in read(path)}
        for group in groups:
            jobs=[{**j,'freeze_sha256':f} for j in spec[group]];path=HERE/f'JOBS_{group}.json'
            if path.exists():assert read(path)==jobs
            else:write(path,jobs)
            if str(path) not in paths:
                assert not names & {j['name'] for j in jobs};paths.append(str(path));names.update(j['name'] for j in jobs)
        write(pool,paths);write(HERE/'ADMISSION.json',dict(status='ADMITTED_AVAILABLE_GROUPS',freeze_sha256=f,groups=groups,gpu_gate_hashes=gate_hashes,cpu_sha256=sha(HERE/'CPU_CHECK.json'),counts={g:len(spec[g]) for g in groups}))
        print('ADMITTED',groups,flush=True)
if __name__=='__main__':main()
