"""Idempotent gated admission to the sole existing R22 pool."""
import sys
from common import BASE,HERE,OUT,read,write,sha,verify,cpu_pass,gate_pass,storage
sys.path.insert(0,str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock

def main():
    with submission_lock():
        f=verify();cpu_pass();guard=storage();groups=['gates']
        gp=OUT/'gates/COMPLETE.json'
        if gp.exists():gate_pass();groups.append('full')
        pool=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json';paths=read(pool)
        alljobs=[j for p in paths for j in read(p)];names=[j['name'] for j in alljobs]
        assert len(names)==len(set(names)),'Existing pool names must be unique'
        spec=read(HERE/'JOBS_SPEC.json')
        for group in groups:
            jobs=[{**j,'freeze_sha256':f} for j in spec[group]];p=HERE/f'JOBS_{group}.json'
            if p.exists():assert read(p)==jobs
            else:write(p,jobs)
            if str(p) not in paths:
                assert not set(names).intersection(j['name'] for j in jobs)
                if group=='gates':paths.insert(0,str(p))
                else:paths.append(str(p))
                names.extend(j['name'] for j in jobs)
        write(pool,paths)
        result=dict(status='ADMITTED_AVAILABLE_GROUPS',freeze_sha256=f,groups=groups,counts={g:len(spec[g]) for g in groups},
            cpu_sha256=sha(HERE/'checks/CPU.json'),gate_sha256=sha(gp) if gp.exists() else None,storage=guard)
        write(HERE/'ADMISSION.json',result);print(result)

if __name__=='__main__':main()
