"""Freeze the R36 protocol after CPU test + GPU gate ALL_PASS: binds code, configs, package, R35 prep, audit and gate
hashes into ROOT/PROTOCOL_FROZEN.json (train_unit.check_frozen refuses formal units otherwise)."""
import glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, PREP, PACKAGE, sha256, write_json, read_json, utc

NOT_BOUND = ['summarize.py', 'freeze.py', 'gate.py', 'make_configs.py']


def main():
    cpu = read_json(ROOT / 'CPU_TEST.json'); gate = read_json(ROOT / 'GPU_GATE.json')
    assert cpu['ALL_PASS'] and gate['ALL_PASS'], 'CPU test and GPU gate must both pass'
    units = read_json(ROOT / 'configs' / 'UNITS.json'); assert len(units) == 27
    code = {Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'r36' / '*.py')))}
    for f, h in gate['code_sha256'].items():
        if f not in NOT_BOUND:
            assert code.get(f) == h, f'code changed since the gate: {f}'
    write_json(ROOT / 'PROTOCOL_FROZEN.json', dict(
        line='R36 SUPERVISION/SELECTION (VCS_R36_Supervision_Selection_20261102)', frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
        package=dict(dir=str(PACKAGE), matrix_sha256=sha256(PACKAGE / 'configs' / 'MATRIX.csv'), plan_sha256=sha256(PACKAGE / 'docs' / 'SERVER_PLAN_CN.md'),
                     protocol_sha256=sha256(PACKAGE / 'configs' / 'PROTOCOL.json'), manifest_sha256=sha256(PACKAGE / 'MANIFEST_SHA256.json'),
                     evidence_commit='892638d66615fbc4495d728d96b538b2ef43f5fa'),
        units=units, config_sha256={u: sha256(ROOT / 'configs' / f'{u}.json') for u in units},
        resolved_matrix_sha256=sha256(ROOT / 'RESOLVED_MATRIX.csv'), code_sha256=code, code_not_bound=NOT_BOUND,
        r35_prep_manifest_sha256=sha256(PREP / 'PREP_MANIFEST.json'), cpu_test=dict(job=cpu['job'], sha256=sha256(ROOT / 'CPU_TEST.json')),
        gpu_gate=dict(job=gate['job'], gpu=gate['gpu'], sha256=sha256(ROOT / 'GPU_GATE.json')),
        audit_results_sha256=sha256(ROOT / 'AUDIT_RESULTS.json') if (ROOT / 'AUDIT_RESULTS.json').exists() else None,
        primary_endpoint='fixed final step (M/T/N 8000; V phase-2 step 4000 = total 12000, secondary phase-1 8000); curves only for interpretation',
        not_submitted='importance-sampling alternative (plan section 8) is not part of this matrix', test_read=False))
    print('PROTOCOL_FROZEN written')


if __name__ == '__main__':
    main()
