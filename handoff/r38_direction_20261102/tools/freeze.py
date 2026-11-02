"""Freeze the R38 protocol after CPU_TEST ALL_PASS and PARENT_ARTIFACT_CHECK ALL_PASS (standalone GPU gate waived, as R37/R37C:
every formal unit asserts the step-0 native-rank check, the first-update graph check and, where a reference exists, the R36/R37C
init/batch-chain identity). Binds both code trees, the 30 configs, package matrix/manifest, reuse references, parent caches/preps.
train_unit.check_frozen refuses formal units otherwise. Usage: python -I -B tools/freeze.py"""
import csv, glob, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r38c'))
from common import sha256, write_json, read_json, utc, PACKAGE, CONDS as CC

NOT_BOUND = ['build_cache.py', 'prep.py']


def main():
    cpu = read_json(W / 'CPU_TEST.json'); art = read_json(W / 'PARENT_ARTIFACT_CHECK.json')
    assert cpu['ALL_PASS'] and art['ALL_PASS'], 'CPU test and parent artifact check must pass'
    units = read_json(W / 'configs' / 'UNITS.json'); assert len(units) == 30
    code = {t: {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r38f', 'r38c')}
    mans = ['/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n60/CACHE_MANIFEST.json',
            '/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json']
    for k in ('cc', 'cc_s1002', 'cc_s1003'):
        mans += [str(Path(CC[k]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(CC[k]['prep_dir']) / 'PREP_MANIFEST.json')]
    write_json(W / 'PROTOCOL_FROZEN.json', dict(
        line='R38 directional readout x schedule (package VCS_R37_Review_R38_Plan_20261102, 30 critic-only fits, 8000 steps); development experiment, no TEST read',
        frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
        package=dict(dir=str(PACKAGE), matrix_sha256=sha256(PACKAGE / 'configs' / 'MATRIX_NEW.csv'), protocol_sha256=sha256(PACKAGE / 'configs' / 'PROTOCOL.json'),
                     manifest_sha256=sha256(PACKAGE / 'MANIFEST.json'), reference_sha256=sha256(PACKAGE / 'reference' / 'directional_reference.py')),
        units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units},
        resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX.csv'), parent_binding_sha256=sha256(W / 'PARENT_BINDING.json'),
        reuse_references_sha256=sha256(W / 'REUSE_REFERENCES.json'),
        code_sha256_by_tree=code, code_not_bound=NOT_BOUND, parent_cache_manifest_sha256={m: sha256(m) for m in mans},
        cpu_test=dict(job=cpu['job'], sha256=sha256(W / 'CPU_TEST.json')), parent_artifact_check=dict(job=art['job'], sha256=sha256(W / 'PARENT_ARTIFACT_CHECK.json')),
        gpu_gate=dict(waived_by_user=True, utc=utc(), basis='R37 waiver (user 2026-11-02 「好，取消gate，直接提交正式实验」) carried to R38 by the supervising session (package SERVER_PROMPT item 4): no standalone GPU gate; every unit asserts step-0 two-stage ranks == native ranks, the first-update graph check (directional routing / score identity / grad sum) and, for the s1001 parents, the R36/R37C init-state and batch-chain identity (fail-fast)'),
        primary_endpoint='fixed step 8000 (curves 0/250/500/1000/2000/4000/8000; no per-method stop/extend/best-step)', test_read=False))
    print('PROTOCOL_FROZEN written')


if __name__ == '__main__':
    main()
