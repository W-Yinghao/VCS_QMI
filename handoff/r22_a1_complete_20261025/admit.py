"""Idempotent CPU -> five environment gates -> twelve full jobs, original pool."""
import sys
from common import BASE,HERE,OUT,read,write,sha,verify,units,cpu_pass,gate_pass,storage
sys.path.insert(0,str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock

def main():
    with submission_lock():
        f=verify();cpu_pass();guard=storage();groups=['gates']
        gates=sorted({r['gate_group'] for r in units().values()})
        if all((OUT/'gates'/g/'COMPLETE.json').exists() for g in gates):
            for g in gates:gate_pass(g)
            groups.append('full')
        pool=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json';paths=read(pool)
        names=[r['name'] for p in paths for r in read(p)];assert len(names)==len(set(names))
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
        write(HERE/'ADMISSION.json',dict(status='ADMITTED_AVAILABLE_GROUPS',freeze_sha256=f,groups=groups,
            counts={g:len(spec[g]) for g in groups},cpu_sha256=sha(HERE/'checks/CPU.json'),storage=guard))
        print('A1_COMPLETION_ADMITTED',groups,flush=True)

if __name__=='__main__':main()
