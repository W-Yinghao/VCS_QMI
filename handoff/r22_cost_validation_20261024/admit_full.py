"""One-shot verification/admission of six frozen cost probes. Never starts a controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
POOL = BASE / 'handoff/r22_task_diagnostics_20261023'
sys.path.insert(0, str(BASE / 'handoff/r22_submission_coord_20261024'))
from coord import submission_lock


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify():
    freeze_hash = sha(HERE / 'FREEZE.json')
    for rel, digest in read(HERE / 'FREEZE.json')['files'].items():
        assert sha(BASE / rel) == digest, rel
    cpu = read(HERE / 'CPU_CHECK.json')
    assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == freeze_hash
    assert sha(HERE / 'CPU_CHECK.json') == read(HERE / 'ADMISSION.json')['cpu_check_sha256']
    source = read(HERE / 'SOURCE_LOCK.json')
    for r in source.values():
        assert sha(r['checkpoint']) == r['checkpoint_sha256']
        assert sha(r['completion']) == r['completion_sha256']
    gate_jobs = read(HERE / 'GATE_JOBS.json')
    missing = [r['name'] for r in gate_jobs if not Path(r['completion']).exists()]
    if missing:
        return dict(status='WAITING_FOR_GPU_GATES', missing=missing, full_probes_admitted=False)
    gates = []
    arms = {'vcs_exact', 'vcs_dense', 'vcs_perm', 'bce', 'bce_block', 'nce', 'nce_block'}
    pairs = {('vcs_exact','vcs_dense'), ('bce','bce_block'), ('nce','nce_block')}
    for ds, job in zip(('bf','epfl'), gate_jobs):
        r = read(job['completion'])
        assert r['status'] == 'COMPLETE' and r['gate'] is True and r['weights_unchanged'] is True
        assert r['ds'] == ds and r['seed'] == 1001
        assert r['freeze_sha256'] == job['freeze_sha256'] == freeze_hash
        assert r['numerical_roles'] == ['FIT'] and set(r['integrity_roles']) <= {'FIT','CAL','DEV'}
        assert r['source_checkpoint_sha256'] == source[f'{ds}_1001']['checkpoint_sha256']
        assert r['source_completion_sha256'] == source[f'{ds}_1001']['completion_sha256']
        assert len(r['rows']) == 3 and {(x['batch'],x['n_pos']) for x in r['rows']} == {(0,n) for n in (128,512,2048)}
        for row in r['rows']:
            assert set(row['costs']) == arms and len(row['order']) == 7 and set(row['order']) == arms
            for cost in row['costs'].values():
                assert len(cost['all_seconds']) == 2 and all(math.isfinite(t) and t > 0 for t in cost['all_seconds'])
                assert cost['peak_allocated_bytes'] >= cost['baseline_bytes'] > 0
        assert len(r['equivalence']) == 9
        assert {(x['batch'],x['n'],x['reference'],x['alternative']) for x in r['equivalence']} == {
            (0,n,a,b) for n in (128,512,2048) for a,b in pairs}
        assert all(math.isfinite(x['encoder_gradient_relative_l2']) and x['encoder_gradient_relative_l2'] < 2e-4
                   and math.isfinite(x['value_error']) for x in r['equivalence'])
        # Value tolerance was checked against the reference value inside the frozen GPU gate.
        gates.append(dict(job=r['job'], path=job['completion'], sha256=sha(job['completion']),
                          max_gradient_relative_l2=max(x['encoder_gradient_relative_l2'] for x in r['equivalence'])))
    manifest = HERE / 'FULL_JOBS_PENDING_GATES.json'
    jobs = read(manifest)
    assert len(jobs) == 6 and len({r['name'] for r in jobs}) == 6
    assert all(r['freeze_sha256'] == freeze_hash and r['group'] == 'other_cost' for r in jobs)
    return dict(status='GATES_VERIFIED_READY', freeze_sha256=freeze_hash, gates=gates,
                manifest=str(manifest), manifest_sha256=sha(manifest), full_probes_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    with submission_lock():
        result = verify()
        if result['status'] == 'GATES_VERIFIED_READY' and not args.check_only:
            path = POOL / 'POOL_SOURCES.json'
            sources = read(path)
            if result['manifest'] not in sources:
                current = [j for source in sources for j in read(source)]
                proposed = read(result['manifest'])
                assert not ({j['name'] for j in current} & {j['name'] for j in proposed})
                sources.append(result['manifest'])
                tmp = path.with_name(path.name + '.cost_admission')
                tmp.write_text(json.dumps(sources, indent=2) + '\n'); tmp.replace(path)
            result.update(status='FULL_PROBES_ADMITTED', full_probes_admitted=True,
                          utc=datetime.now(timezone.utc).isoformat(), admission_script_sha256=sha(Path(__file__)))
            (HERE / 'FULL_ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))
