"""Freeze the R39 protocol after CPU_TEST_pre/post ALL_PASS, both COMPLETE preps, both JOINT preps (cache equivalence) and both
JOINT functional checks (standalone GPU gate waived by the user: every formal unit asserts the step-0 native-rank check, the R38
first/second-update graph checks, the R38-reference init/batch-chain identity, and for JOINT the step-0 student-vs-cache
equivalence and the update-1 student/frozen checks; COMPLETE asserts its row invariants on every batch). Binds both code trees,
the 21 configs, package files, R38 references, parent/teacher caches and preps, R39 prep manifests.
train_unit.check_frozen refuses formal units otherwise. Usage: python3 -I -B tools/freeze.py"""
import csv, glob, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r39c'))
from common import sha256, write_json, read_json, utc, PACKAGE, CONDS as CC, R39_PREP

FLICKR_MANS = ['/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n60/CACHE_MANIFEST.json',
               '/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json',
               '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1002/CACHE_MANIFEST.json',
               '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1003/CACHE_MANIFEST.json']


def main():
    pre, post = read_json(W / 'CPU_TEST_pre.json'), read_json(W / 'CPU_TEST_post.json')
    assert pre['ALL_PASS'] and post['ALL_PASS'], 'CPU tests must pass'
    smoke = {}
    for ds in ('f60', 'cc'):
        s = read_json(R39_PREP / ds / 'JOINT_SMOKE.json')
        assert s['step0_cache_equivalence']['passed'] and s['one_step_check']['passed'], f'JOINT functional check failed: {ds}'
        smoke[ds] = dict(sha256=sha256(R39_PREP / ds / 'JOINT_SMOKE.json'), seconds_per_update=s['seconds_per_update'], gpu=s['gpu'])
    units = read_json(W / 'configs' / 'UNITS.json'); assert len(units) == 21
    code = {t: {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r39f', 'r39c')}
    mans = list(FLICKR_MANS)
    for k in ('cc', 'cc_s1002', 'cc_s1003'):
        mans += [str(Path(CC[k]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(CC[k]['prep_dir']) / 'PREP_MANIFEST.json')]
    for ds in ('f60', 'cc'):
        mans += [str(R39_PREP / ds / 'COMPLETE_PREP_MANIFEST.json'), str(R39_PREP / ds / 'JOINT_PREP_MANIFEST.json')]
    pkg = read_json(PACKAGE / 'FILE_SHA256.json')
    pkg_ok = {f: sha256(PACKAGE / f) == h for f, h in (pkg.items() if isinstance(pkg, dict) else [])}
    write_json(W / 'PROTOCOL_FROZEN.json', dict(
        line='R39 COMPLETE x JOINT (package VCS_R38_Review_R39_Design_20261102: 18 non-oracle + 3 Flickr oracle, 8000 steps, cosine, N/shared G); development experiment, no TEST read',
        frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
        package=dict(dir=str(PACKAGE), matrix_sha256=sha256(PACKAGE / 'MATRIX.csv'), reference_matrix_sha256=sha256(PACKAGE / 'REFERENCE_MATRIX.csv'),
                     plan_sha256=sha256(PACKAGE / 'docs' / 'SERVER_PLAN_CN.md'), targets_sha256=sha256(PACKAGE / 'reference' / 'targets.py'), file_sha256_check=pkg_ok),
        units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units},
        resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX.csv'), reuse_references_sha256=sha256(W / 'REUSE_REFERENCES.json'),
        code_sha256_by_tree=code, code_not_bound=[], parent_cache_manifest_sha256={m: sha256(m) for m in mans},
        cpu_test=dict(pre=dict(job=pre['job'], sha256=sha256(W / 'CPU_TEST_pre.json')), post=dict(job=post['job'], sha256=sha256(W / 'CPU_TEST_post.json'))),
        joint_functional_check=smoke,
        gpu_gate=dict(waived_by_user=True, utc=utc(), basis='user waived the standalone GPU gate (R37/R38 practice, R39 task brief): CPU checks first, then in-unit fail-fast checks in every formal unit'),
        primary_endpoint='fixed step 8000 (curves 0/250/500/1000/2000/4000/8000; no per-method stop/extend/best-step)', test_read=False))
    print('PROTOCOL_FROZEN written', all(pkg_ok.values()) if pkg_ok else 'no package hash list')


if __name__ == '__main__':
    main()
