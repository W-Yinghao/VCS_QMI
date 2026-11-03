"""Freeze the R40 protocol (PROTOCOL_FROZEN.json) after CPU_TEST.json ALL_PASS. Standalone GPU gate waived (user waiver carried over from
R37/R38/R39): every formal unit asserts step-0 native ranks, step-0 student-vs-cache equivalence (JOINT), the R38 first-update graph check,
the R40 one-step active-path rule, PRESENT p_d / masked-gradient checks and the reference init / student-init / batch-chain identity.
R39 / R39X / R38 files are only read. Usage: python -I -B tools/freeze.py"""
import glob, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r40c'))
from common import sha256, write_json, read_json, utc, R40_PACKAGE

R39 = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')
MANS = ['/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n60/CACHE_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/locks/LOCKS.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/cache/cc/CACHE_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/prep/PREP_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/prep/f60/JOINT_PREP_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/prep/cc/JOINT_PREP_MANIFEST.json']


def main():
    cpu = read_json(W / 'CPU_TEST.json'); assert cpu['ALL_PASS'], 'R40 CPU test must pass'
    assert read_json(W / 'DEDUPE.json')['ALL_UNIQUE']
    units = read_json(W / 'configs' / 'UNITS.json'); assert len(units) == 27
    code = {t: {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r40f', 'r40c')}
    pkg = read_json(R40_PACKAGE / 'MANIFEST_SHA256.json')
    write_json(W / 'PROTOCOL_FROZEN.json', dict(
        line='R40 update scope x incomplete-correspondence supervision (A: image_only/text_only; B: N_PRESENT_ONLY x {frozen, both_joint}; C: Flickr ORACLE_MISSING x both_joint); '
             'parents fixed to the R39 parents (F60 R30-H f60 s1001, CC31 R31 H0_default cc_s1001); R39 recipe otherwise unchanged; development experiment, no TEST read',
        frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
        authorisation=dict(user_quote='查看这个，准备并提交相关实验，R39X的实验还是值得做完的，这个优先级排到那批后面，比R33高，R33还是可以让位', relayed_by='main session',
                           priority='normal QOS nice 100: after R39X (nice 0), above R33 (nice 10000); formal units never runfill'),
        package=dict(path=str(R40_PACKAGE), manifest_sha256=sha256(R40_PACKAGE / 'MANIFEST_SHA256.json'), matrix_sha256=sha256(R40_PACKAGE / 'MATRIX.csv'),
                     reference_objectives_sha256=sha256(R40_PACKAGE / 'reference' / 'objectives.py'), files=len(pkg) if isinstance(pkg, (list, dict)) else None),
        units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units},
        resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX.csv'), parent_binding_sha256=sha256(W / 'PARENT_BINDING.json'), dedupe_sha256=sha256(W / 'DEDUPE.json'),
        reuse_references_sha256=sha256(W / 'REUSE_REFERENCES.json'), protocol_diff_vs_r39_sha256=sha256(W / 'PROTOCOL_DIFF_VS_R39.json'),
        r39_protocol_frozen_sha256=sha256(R39 / 'PROTOCOL_FROZEN.json'), r39_files_untouched=True,
        code_sha256_by_tree=code, code_not_bound=[], parent_cache_manifest_sha256={m: sha256(m) for m in MANS},
        cpu_test=dict(job=cpu['job'], sha256=sha256(W / 'CPU_TEST.json')),
        gpu_gate=dict(waived_by_user=True, utc=utc(), basis='user waiver (R37/R38/R39): CPU checks on real data, then in-unit fail-fast checks in every formal unit'),
        primary_endpoint='fixed step 8000 (curves 0/250/500/1000/2000/4000/8000)', test_read=False))
    print('PROTOCOL_FROZEN written')


if __name__ == '__main__':
    main()
