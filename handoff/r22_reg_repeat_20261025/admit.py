import sys
from common import BASE,HERE,read,write,sha,verify
sys.path.insert(0,str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock


def main():
    with submission_lock():
        freeze=verify();c=read(HERE/'CPU_CHECK.json');s=read(HERE/'STATIC_CHECK.json')
        assert c['status']==s['status']=='PASS' and c['freeze_sha256']==s['freeze_sha256']==freeze
        assert len(c['sources'])==18 and len(c['repeat_checks'])==4
        groups=['gates'];gates={}
        for ds in ('bf','epfl'):
            p=HERE/'runs'/f'gate_{ds}'/'COMPLETE.json'
            if p.exists():
                d=read(p);assert d['status']=='COMPLETE' and d['freeze_sha256']==freeze and d['gate']
                assert d['n_problems']==1 and d['n_paths']==48 and d['repeats']==4 and d['starts']==3 and d['adam_steps']==100
                assert d['balanced_order'] and d['torch_rng_unchanged'] and d['source_unchanged'] and d['roles_read']==['CAL']
                assert d['arms']==['P1_NATIVE','POS_NATIVE','P1_DIR_POS_NORM','POS_DIR_P1_NORM']
                for name,key in [('TRAJECTORIES.jsonl','trajectory_sha256'),('INPUT_TRACE.json','input_trace_sha256')]:assert sha(p.parent/name)==d[key]
                gates[ds]=sha(p)
        if len(gates)==2:groups.append('full')
        p=BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json';sources=read(p);spec=read(HERE/'JOBS_SPEC.json')
        names={j['name'] for s in sources for j in read(s)}
        for group in groups:
            jobs=[{**j,'freeze_sha256':freeze} for j in spec[group]];manifest=HERE/f'JOBS_{group}.json'
            if manifest.exists():assert read(manifest)==jobs
            else:write(manifest,jobs)
            if str(manifest) not in sources:
                assert not names & {j['name'] for j in jobs};sources.append(str(manifest));names.update(j['name'] for j in jobs)
        write(p,sources);write(HERE/'ADMISSION.json',dict(status='ADMITTED_AVAILABLE_GROUPS',groups=groups,freeze_sha256=freeze,gpu_gate_hashes=gates,counts={g:len(spec[g]) for g in groups}))
        print('REPEAT_ADMITTED',groups,flush=True)


if __name__=='__main__':main()
