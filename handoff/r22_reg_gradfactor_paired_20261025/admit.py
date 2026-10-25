"""Admit fixed checks and crossed-gradient jobs through the existing pool only."""
import sys
from common import BASE,HERE,read,write,sha,verify
sys.path.insert(0,str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock

def main():
    with submission_lock():
        f=verify();c=read(HERE/'CPU_CHECK.json');s=read(HERE/'STATIC_CHECK.json')
        assert c['status']==s['status']=='PASS' and c['freeze_sha256']==s['freeze_sha256']==f
        assert len(c['sources'])==18 and len(c['gradient_factor_checks'])==4
        groups=['gates'];gates={}
        for ds in ('bf','epfl'):
            p=HERE/'runs'/f'gate_{ds}'/'COMPLETE.json'
            if p.exists():
                d=read(p);assert d['status']=='COMPLETE' and d['freeze_sha256']==f and d['gate']
                assert d['paired_reference_paths']==6 and d['historical_reference_reuse'] is False and d['n_problems']==1 and d['adam_steps']==100 and d['starts']==3
                assert d['arms']==['P1_NATIVE','POS_NATIVE','P1_DIR_POS_NORM','POS_DIR_P1_NORM']
                assert d['roles_read']==['CAL'] and d['source_unchanged']
                for name,key in [('TRAJECTORIES.jsonl','trajectory_sha256'),('INPUT_TRACE.json','input_trace_sha256'),('HISTORICAL_DIFFERENCES.json','historical_differences_sha256')]:assert sha(p.parent/name)==d[key]
                gates[ds]=sha(p)
        if len(gates)==2:groups.append('full')
        pool=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json';sources=read(pool);spec=read(HERE/'JOBS_SPEC.json')
        names={j['name'] for p in sources for j in read(p)}
        for group in groups:
            jobs=[{**j,'freeze_sha256':f} for j in spec[group]];p=HERE/f'JOBS_{group}.json'
            if p.exists():assert read(p)==jobs
            else:write(p,jobs)
            if str(p) not in sources:
                assert not names & {j['name'] for j in jobs};sources.append(str(p));names.update(j['name'] for j in jobs)
        write(pool,sources);write(HERE/'ADMISSION.json',dict(status='ADMITTED_AVAILABLE_GROUPS',groups=groups,freeze_sha256=f,
            gpu_gate_hashes=gates,cpu_sha256=sha(HERE/'CPU_CHECK.json'),counts={g:len(spec[g]) for g in groups}))
        print('GRADFACTOR_PAIRED_ADMITTED',groups,flush=True)
if __name__=='__main__':main()
