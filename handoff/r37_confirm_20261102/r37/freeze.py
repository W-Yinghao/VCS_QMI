"""Freeze the R37 protocol after CPU test + GPU gate ALL_PASS: binds code, configs, package, parent caches/prep and gate
hashes into ROOT/PROTOCOL_FROZEN.json (train_unit.check_frozen refuses formal units otherwise)."""
import glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, PACKAGE, CONDS, R35_PREP, R37_PREP, sha256, write_json, read_json, utc

NOT_BOUND = ['summarize.py', 'freeze.py', 'gate.py', 'make_configs.py', 'build_cache.py', 'prep.py']


def main():
    cpu = read_json(ROOT / 'CPU_TEST.json')
    assert cpu['ALL_PASS'], 'CPU test must pass'
    # standalone GPU gate waived by the user (2026-11-02 「好，取消gate，直接提交正式实验」); each formal unit runs a fail-fast step-0 native check
    units = read_json(ROOT / 'configs' / 'UNITS.json'); assert len(units) == 18
    code = {Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'r37' / '*.py')))}
    mans = [str(Path(c['cache_dir']) / 'CACHE_MANIFEST.json') for c in CONDS.values()] + [str(R35_PREP / 'PREP_MANIFEST.json'), str(R37_PREP / 'PREP_MANIFEST.json')]
    write_json(ROOT / 'PROTOCOL_FROZEN.json', dict(
        line='R37 CONFIRM (VCS_R36_Review_ee659c3b NEXT_DECISIONS: A confirm 12 + B structure 6; C CC152K not run: missing same-series parent)',
        frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
        package=dict(dir=str(PACKAGE), matrix_sha256=sha256(PACKAGE / 'MATRIX_PROPOSED.csv'), next_decisions_sha256=sha256(PACKAGE / 'docs' / 'NEXT_DECISIONS_CN.md'),
                     manifest_sha256=sha256(PACKAGE / 'MANIFEST_SHA256.json'), evidence_commit='ee659c3b7437a69035dd66b7f8d6acf9a85730ae'),
        units=units, config_sha256={u: sha256(ROOT / 'configs' / f'{u}.json') for u in units},
        resolved_matrix_sha256=sha256(ROOT / 'RESOLVED_MATRIX.csv'), parent_binding_sha256=sha256(ROOT / 'PARENT_BINDING.json'),
        code_sha256=code, code_not_bound=NOT_BOUND, parent_cache_manifest_sha256={m: sha256(m) for m in mans},
        cpu_test=dict(job=cpu['job'], sha256=sha256(ROOT / 'CPU_TEST.json')), gpu_gate=dict(waived_by_user=True, utc=utc(), quote='好，取消gate，直接提交正式实验，R33可以让位', replacement='fail-fast step-0 native-H rank check inside every formal unit (train_unit.py; eval_step0.json step0_native_check)', cancelled_gate_job='1082030 (never started)'),
        primary_endpoint='fixed step 8000 (curves 0/250/500/1000/2000/4000/8000 for interpretation only; no per-method stop/extend/best-step)',
        test_read=False))
    print('PROTOCOL_FROZEN written')


if __name__ == '__main__':
    main()
