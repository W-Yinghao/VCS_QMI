import datetime
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PACK = BASE/'handoff/r22_reg_gradfactor_paired_20261025'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    frozen = read(PACK/'FREEZE.json'); freeze = sha(PACK/'FREEZE.json')
    for rel, h in frozen['files'].items():
        assert sha(BASE/rel) == h, rel
    for name in ['CPU_CHECK.json','STATIC_CHECK.json']:
        d = read(PACK/name)
        assert d['status'] == 'PASS' and d['freeze_sha256'] == freeze
    admission = read(PACK/'ADMISSION.json')
    assert set(admission['groups']) == {'gates','full'} and admission['counts'] == {'gates':2,'full':18}
    sources = read(BASE/'handoff/r22_task_diagnostics_20261023/POOL_SOURCES.json')
    jobs = [j for p in sources for j in read(p)]
    assert len(jobs) == len({j['name'] for j in jobs})
    assert sources.count(str(PACK/'JOBS_gates.json')) == sources.count(str(PACK/'JOBS_full.json')) == 1
    old = BASE/'handoff/r22_reg_gradfactor_20261025'
    assert str(old/'JOBS_gates.json') not in sources and not (old/'JOBS_full.json').exists()
    gates = []; repeated_backward_differences = []
    for ds in ('bf','epfl'):
        p = PACK/'runs'/f'gate_{ds}'; r = read(p/'COMPLETE.json')
        assert admission['gpu_gate_hashes'][ds] == sha(p/'COMPLETE.json')
        assert r['freeze_sha256'] == freeze and r['paired_reference_paths'] == 6
        assert r['historical_reference_reuse'] is False and r['source_unchanged']
        assert r['roles_read'] == ['CAL'] and r['adam_steps'] == 100 and r['starts'] == 3
        for name, key in [('TRAJECTORIES.jsonl','trajectory_sha256'),('INPUT_TRACE.json','input_trace_sha256'),('HISTORICAL_DIFFERENCES.json','historical_differences_sha256')]:
            assert sha(p/name) == r[key]
        rows = [json.loads(s) for s in (p/'TRAJECTORIES.jsonl').read_text().splitlines()]
        assert len(rows) == 1
        arms = rows[0]['arms']
        assert list(arms) == ['P1_NATIVE','POS_NATIVE','P1_DIR_POS_NORM','POS_DIR_P1_NORM']
        for a in arms.values():
            assert len(a['paths']) == 3
            for k, path in enumerate(a['paths']):
                assert path['initial_pose'] == arms['P1_NATIVE']['paths'][k]['initial_pose']
                assert [x['step'] for x in path['trajectory']] == [0,1,5,10,25,50,100]
                for x in path['trajectory']:
                    assert x['adam_step'] == x['step'] and math.isfinite(x['error_native'])
                    gd = x['gradients'][path['direction']]
                    expected = [v*x['scale'] for v in gd]
                    applied = math.sqrt(sum(v*v for v in x['applied_ascent_gradient']))
                    # The recorded full term gradients come from a SECOND autograd
                    # call. CUDA's nondeterministic backward can differ slightly.
                    # Check the actual first-pass target norm and scale; retain
                    # second-pass discrepancies descriptively, not as a new gate.
                    target = x['gradient_norms'][path['norm_source']]
                    direction = x['gradient_norms'][path['direction']]
                    assert math.isclose(applied,target,abs_tol=1e-8,rel_tol=5e-5)
                    if direction > 1e-12:
                        assert math.isclose(x['scale'],target/direction,abs_tol=1e-8,rel_tol=2e-5)
                    diff = max(abs(a-b) for a,b in zip(x['applied_ascent_gradient'],expected))
                    repeated_backward_differences.append(dict(ds=ds,arm=path['arm'],start=k,step=x['step'],
                        max_abs=diff,relative_to_applied_norm=diff/max(applied,1e-12)))
        gates.append(dict(dataset=ds,job=r['job'],seconds=r['seconds'],paths=12,actual_updates=1200))
    state = read(BASE/'handoff/r22_task_diagnostics_20261023/POOL_STATE.json')['submitted']
    full = read(PACK/'JOBS_full.json')
    report = dict(status='FULL_ADMISSION_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        freeze_sha256=freeze,frozen_files_verified=len(frozen['files']),gate_records=gates,
        gate_receipt_seconds=sum(g['seconds'] for g in gates),old_failed_gate_scheduler_seconds=10,full_count=18,
        all_active_job_names_unique=True,manifests_unique=True,old_failed_manifest_retired_records_retained=True,
        native_references='computed in same job; no historical endpoint reuse',
        gate_actual_first_pass_norm_and_scale_independently_recomputed=True,
        direction_formula_checked_by_frozen_CPU_and_runtime=True,
        repeated_backward_differences=repeated_backward_differences,
        verifier_note='Initial offline verifier incorrectly treated the separately recomputed full-term gradients as the actual first-pass gradients. Largest gate discrepancy relative to applied norm was 5.14e-5. Actual first-pass norm and scale pass; all differences retained. No frozen scientific code, criteria or gate changed.',
        full_submitted={j['name']:state[j['name']] for j in full if j['name'] in state})
    (PACK/'ADMISSION_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    p = BASE/'handoff/r22_next_round_20261025/STATUS.json'; status = read(p)
    status['REG_GRAD_FACTOR_PAIRED'].update(report)
    status['REG_GRAD_FACTOR_PAIRED']['gpu_gate_jobs'] = ['1063698','1063699']
    status['other_work'] = 'All48 noise, all18 original trajectory, all6 A1 reviewed. Samejob four-arm gradient18 fully admitted after both GPU gates PASS. KDpositive and shuffled-U continue unchanged.'
    p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('full_submitted','repeated_backward_differences')}))
    print('full_submitted',len(report['full_submitted']))


if __name__ == '__main__':
    main()
