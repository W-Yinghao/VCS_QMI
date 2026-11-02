"""Freeze the R35 protocol after CPU test + GPU gate ALL_PASS: binds code, configs, prep manifest, gate and
package hashes into ROOT/PROTOCOL_FROZEN.json (train_unit.check_frozen refuses formal units otherwise)."""
import glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, PREP, PACKAGE, sha256, write_json, read_json, utc

NOT_BOUND = ['summarize.py', 'readouts6.py', 'delivery.py', 'freeze.py', 'gate.py', 'make_configs.py']


def main():
    cpu = read_json(ROOT / 'CPU_TEST.json'); gate = read_json(ROOT / 'GPU_GATE.json')
    assert cpu['ALL_PASS'] and gate['ALL_PASS'], 'CPU test and GPU gate must both pass'
    units = read_json(ROOT / 'configs' / 'UNITS.json')
    code = {Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'r35' / '*.py')))}
    for f, h in gate['code_sha256'].items():
        if f not in NOT_BOUND:
            assert code.get(f) == h, f'code changed since the gate: {f}'
    write_json(ROOT / 'PROTOCOL_FROZEN.json', dict(
        line='R35 EXPANDED', frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
        package=dict(dir=str(PACKAGE), matrix_sha256=sha256(PACKAGE / 'configs' / 'MATRIX.csv'), plan_sha256=sha256(PACKAGE / 'docs' / 'SERVER_PLAN_CN.md'),
                     protocol_sha256=sha256(PACKAGE / 'configs' / 'PROTOCOL.json'), manifest_sha256=sha256(PACKAGE / 'MANIFEST_SHA256.json')),
        units=units, config_sha256={u: sha256(ROOT / 'configs' / f'{u}.json') for u in units},
        matrix_resolved_sha256=sha256(ROOT / 'MATRIX_RESOLVED.csv'), code_sha256=code, code_not_bound=NOT_BOUND,
        prep_manifest_sha256=sha256(PREP / 'PREP_MANIFEST.json'), cpu_test=dict(job=cpu['job'], sha256=sha256(ROOT / 'CPU_TEST.json')),
        gpu_gate=dict(job=gate['job'], gpu=gate['gpu'], sha256=sha256(ROOT / 'GPU_GATE.json')),
        primary_endpoint='fixed final step (A/B/C 8000, D 4000); curves only for interpretation', test_read=False))
    print('PROTOCOL_FROZEN written')


if __name__ == '__main__':
    main()
