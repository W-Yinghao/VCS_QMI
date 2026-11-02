"""Freeze the R37C protocol after CPU_TEST ALL_PASS (standalone GPU gate waived as for R37; each unit runs the fail-fast
step-0 native check): binds code, configs, package matrix, parent binding, cache/prep manifests into
ROOT/PROTOCOL_FROZEN.json (train_unit.check_frozen refuses formal units otherwise)."""
import glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, PACKAGE, CONDS, PREP, sha256, write_json, read_json, utc

NOT_BOUND = ['summarize.py', 'freeze.py', 'make_configs.py', 'build_cache.py', 'prep.py']


def main():
    cpu = read_json(ROOT / 'CPU_TEST.json')
    assert cpu['ALL_PASS'], 'CPU test must pass'
    units = read_json(ROOT / 'configs' / 'UNITS.json'); assert len(units) == 6
    code = {Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'r37c' / '*.py')))}
    mans = [str(Path(CONDS['cc']['cache_dir']) / 'CACHE_MANIFEST.json'), str(PREP / 'PREP_MANIFEST.json')]
    write_json(ROOT / 'PROTOCOL_FROZEN.json', dict(
        line='R37C CC152K (R37 package block C_real_noise: T_G/N_G x J/BCE/NCE x seed1001 head, 8000 steps); parent R31 H0_default cc_s1001 (user 2026-11-02); development experiment, no TEST read',
        frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
        package=dict(dir=str(PACKAGE), matrix_sha256=sha256(PACKAGE / 'MATRIX_PROPOSED.csv'), manifest_sha256=sha256(PACKAGE / 'MANIFEST_SHA256.json')),
        units=units, config_sha256={u: sha256(ROOT / 'configs' / f'{u}.json') for u in units},
        resolved_matrix_sha256=sha256(ROOT / 'RESOLVED_MATRIX.csv'), parent_binding_sha256=sha256(ROOT / 'PARENT_BINDING.json'),
        code_sha256=code, code_not_bound=NOT_BOUND, parent_cache_manifest_sha256={m: sha256(m) for m in mans},
        cpu_test=dict(job=cpu['job'], sha256=sha256(ROOT / 'CPU_TEST.json')),
        gpu_gate=dict(waived_by_user=True, utc=utc(), basis='R37 waiver (user 2026-11-02 「好，取消gate，直接提交正式实验」) carried to R37C per the supervising session: no standalone GPU gate; cache build requires exact native DEV reproduction; every unit asserts step-0 two-stage ranks == native ranks (fail-fast)'),
        primary_endpoint='fixed step 8000 (curves 0/250/500/1000/2000/4000/8000 for interpretation only; no per-method stop/extend/best-step)',
        test_read=False))
    print('PROTOCOL_FROZEN written')


if __name__ == '__main__':
    main()
