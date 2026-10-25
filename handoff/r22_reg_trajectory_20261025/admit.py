"""Append gates, then diagnostics to the existing pool under its original lock."""
import sys
from common import BASE, HERE, read, write, sha, verify
sys.path.insert(0, str(BASE/'handoff/r22_submission_coord_20261024'))
from coord import submission_lock


def main():
    with submission_lock():
        f = verify(); cpu = read(HERE/'CPU_CHECK.json')
        assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == f and len(cpu['sources']) == 18
        groups, gates = ['gates'], {}
        for ds in ('bf', 'epfl'):
            p = HERE/'runs'/f'gate_{ds}'/'COMPLETE.json'
            if p.exists():
                d = read(p)
                assert d['status'] == 'COMPLETE' and d['freeze_sha256'] == f and d['gate'] and d['ds'] == ds
                assert d['n_problems'] == 1 and d['adam_steps'] == 100 and d['starts'] == 3
                assert d['objectives'] == ['P1', 'POS_QUAD', 'J'] and d['roles_read'] == ['CAL'] and d['source_unchanged']
                assert sha(p.parent/'TRAJECTORIES.jsonl') == d['trajectory_sha256']
                assert sha(p.parent/'INPUT_TRACE.json') == d['input_trace_sha256']
                gates[ds] = sha(p)
        if len(gates) == 2: groups.append('full')
        spec = read(HERE/'JOBS_SPEC.json')
        pool = BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json'
        sources = read(pool); names = {j['name'] for p in sources for j in read(p)}
        for group in groups:
            jobs = [{**j, 'freeze_sha256': f} for j in spec[group]]; p = HERE/f'JOBS_{group}.json'
            if p.exists(): assert read(p) == jobs
            else: write(p, jobs)
            if str(p) not in sources:
                assert not names & {j['name'] for j in jobs}
                sources.append(str(p)); names.update(j['name'] for j in jobs)
        write(pool, sources)
        write(HERE/'ADMISSION.json', dict(status='ADMITTED_AVAILABLE_GROUPS', freeze_sha256=f,
              groups=groups, gates=gates, cpu_sha256=sha(HERE/'CPU_CHECK.json'), counts={g: len(spec[g]) for g in groups}))
        print('TRAJECTORY_ADMITTED', groups, flush=True)


if __name__ == '__main__': main()
