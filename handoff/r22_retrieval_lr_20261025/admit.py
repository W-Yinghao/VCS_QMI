"""Original pool only; CPU admits one gate, verified gate admits six full jobs."""
import sys
from lr_common import BASE,HERE,read,write,sha,verify,cpu_pass
sys.path.insert(0,str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock

def main():
    with submission_lock():
        f=verify();cpu_pass();cpu={'freeze_sha256':f,'sha256':sha(HERE/'checks/CPU.json')}
        p=HERE/'CPU_ADMISSION.json'
        if p.exists():assert read(p)==cpu
        else:write(p,cpu)
        groups=['gates'];gp=HERE/'gates/COMPLETE.json'
        if gp.exists():
            g=read(gp);assert g['status']=='COMPLETE' and g['freeze_sha256']==f and g['cpu_sha256']==cpu['sha256']
            assert g['trace_sha256']==sha(HERE/'checks/GPU_TRACE.json')
            assert read(HERE/'checks/GPU_TRACE.json')['status']=='PASS'
            assert {(r['method'],r['period']) for r in g['rows']}=={(m,p) for m in ('vcs','constant') for p in (30,10)}
            for r in g['rows']:
                assert r['epochs']==40 and r['updates']==47 and r['pool_checks']['ok'] and r['initial_fingerprint_rechecked']
                q=HERE/'gates'/f"{r['method']}_period{r['period']}"
                assert sha(q/'train_log.json')==r['log_sha256'] and sha(q/'last.pt')==r['last_checkpoint_sha256']
            groups.append('full')
        spec=read(HERE/'JOBS_SPEC.json');pool=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json';paths=read(pool)
        names={j['name'] for p in paths for j in read(p)}
        for group in groups:
            jobs=[{**j,'freeze_sha256':f} for j in spec[group]];p=HERE/f'JOBS_{group}.json'
            if p.exists():assert read(p)==jobs
            else:write(p,jobs)
            if str(p) not in paths:
                assert not names.intersection(j['name'] for j in jobs)
                if group=='gates':paths.insert(0,str(p))
                else:paths.append(str(p))
                names.update(j['name'] for j in jobs)
        write(pool,paths)
        result=dict(status='ADMITTED_AVAILABLE_GROUPS',freeze_sha256=f,groups=groups,
            counts={k:len(spec[k]) for k in groups},gate_sha256=sha(gp) if gp.exists() else None)
        write(HERE/'ADMISSION.json',result);print(result)

if __name__=='__main__':main()
