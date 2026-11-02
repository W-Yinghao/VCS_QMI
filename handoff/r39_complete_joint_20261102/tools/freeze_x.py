"""Freeze the R39X extension protocol (PROTOCOL_FROZEN_R39X.json) after CPU_TEST_R39X ALL_PASS and the five R39X JOINT preps
(student/cache equivalence max |diff| <= the R39 in-unit tolerance). Standalone GPU gate waived as for R39 (every unit asserts the
step-0 native-rank check, step-0 student-vs-cache equivalence, update-1 student/frozen checks and the reference init/batch-chain
identity). The frozen R39 21-unit protocol (PROTOCOL_FROZEN.json) is not touched. Usage: python3 -I -B tools/freeze_x.py"""
import glob, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r39xc'))
from common import sha256, write_json, read_json, utc, CONDS as CC, R39_PREP

R39_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')
FLICKR_MANS = ['/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n00/CACHE_MANIFEST.json',
               '/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/locks/LOCKS.json',
               '/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json',
               '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/prep/PREP_MANIFEST.json',
               '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1002/CACHE_MANIFEST.json',
               '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1003/CACHE_MANIFEST.json']
PREP_KEYS = ('f00_s1001', 'f60_s1002', 'f60_s1003', 'cc_s1002', 'cc_s1003', 'cc_r37s')


def main():
    cpu = read_json(W / 'CPU_TEST_R39X.json'); assert cpu['ALL_PASS'], 'R39X CPU test must pass'
    units = read_json(W / 'configs' / 'UNITS_R39X.json'); assert len(units) == 21
    tol = read_json(W / 'configs' / f'{units[0]}.json')['joint_rule']['equivalence_tol_max_abs']
    eqs = {}
    for k in PREP_KEYS:
        m = read_json(R39_PREP / k / 'JOINT_PREP_MANIFEST.json')
        eqs[k] = dict(max_abs=m['equivalence_summary']['max_abs_over_all'], dev_native_from_student=m['equivalence']['dev_native_from_student_vectors'])
        assert eqs[k]['max_abs'] <= tol, (k, eqs[k])
    mans = list(FLICKR_MANS)
    for k in ('cc_s1002', 'cc_s1003', 'cc_r37s'):
        mans += [str(Path(CC[k]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(CC[k]['prep_dir']) / 'PREP_MANIFEST.json')]
    mans += [str(R39_PREP / k / 'JOINT_PREP_MANIFEST.json') for k in PREP_KEYS]
    code = {t: {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r39xf', 'r39xc')}
    write_json(W / 'PROTOCOL_FROZEN_R39X.json', dict(
        line='R39X JOINT confirmation extension (N route x joint_global x J/BCE/NCE on five additional parents; R39 JOINT recipe unchanged); development experiment, no TEST read',
        frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
        authorisation=dict(user_quote='把你觉得值得提交的都提交了吧', relayed_by='main session (cs-qmi)',
                           reason='confirm whether the Flickr60 JOINT gain and the CC152K JOINT loss of R39 hold on other parents / the R30-style CC152K recipe',
                           addition='main session (same authorisation; user: everything meaningful is worth running): Flickr 0% N x JOINT capability diagnostic with matching N-frozen-cosine cells (no R38 reference at 0%)'),
        parents=dict(f00_s1001='R30-H r30_f00_s1001_H0 (0%, capability diagnostic; 3 new N-frozen-cosine cells + 3 JOINT)', f60_s1002='R30-H r30v11_f60_s1002_HH', f60_s1003='R30-H r30v11_f60_s1003_HH', cc_s1002='R31 H0_default cc_s1002', cc_s1003='R31 H0_default cc_s1003',
                     cc_r37s='R37S R30-style r37s_cc_s1001_H0'),
        n_frozen_references=read_json(W / 'REUSE_REFERENCES_R39X.json')['references'],
        units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units},
        resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX_R39X.csv'), reuse_references_sha256=sha256(W / 'REUSE_REFERENCES_R39X.json'),
        r39_protocol_frozen_sha256=sha256(W / 'PROTOCOL_FROZEN.json'), r39_protocol_untouched=True,
        code_sha256_by_tree=code, code_not_bound=[], parent_cache_manifest_sha256={m: sha256(m) for m in mans},
        joint_prep_equivalence=eqs, cpu_test=dict(job=cpu['job'], sha256=sha256(W / 'CPU_TEST_R39X.json')),
        gpu_gate=dict(waived_by_user=True, utc=utc(), basis='same waiver as R39: CPU checks, then in-unit fail-fast checks in every formal unit'),
        primary_endpoint='fixed step 8000 (curves 0/250/500/1000/2000/4000/8000)', test_read=False))
    print('PROTOCOL_FROZEN_R39X written')


if __name__ == '__main__':
    main()
