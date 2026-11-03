"""Freeze R41 part 1 (A/B/D/E -> PROTOCOL_FROZEN_R41.json, after CPU_TEST_R41_1 ALL_PASS) or part 2 (C -> PROTOCOL_FROZEN_R41C.json, after
the new-parent prep and CPU_TEST_R41_2 ALL_PASS). R40 freeze files and outputs are not touched; R39/R39X/R38 only read.
Usage: python -I -B tools/freeze_r41.py 1|2"""
import glob, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r41f'))
from common import sha256, write_json, read_json, utc, CONDS, JOINT_PARENTS

BASE_MANS = ['/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n60/CACHE_MANIFEST.json',
             '/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n00/CACHE_MANIFEST.json',
             '/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/locks/LOCKS.json',
             '/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json',
             '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/prep/PREP_MANIFEST.json',
             '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1002/CACHE_MANIFEST.json',
             '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1003/CACHE_MANIFEST.json']


def main():
    part = sys.argv[1]
    cpu = read_json(W / f'CPU_TEST_R41_{part}.json'); assert cpu['ALL_PASS']
    units = read_json(W / 'configs' / ('UNITS_R41.json' if part == '1' else 'UNITS_R41C.json'))
    conds = sorted({read_json(W / 'configs' / f'{u}.json')['cond'] for u in units})
    mans = list(BASE_MANS) if part == '1' else []
    for c in conds:
        if c in ('f40in', 'f80in'):
            mans += [str(Path(CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json')]
        mans.append(str(Path(JOINT_PARENTS[c]['prep_dir']) / 'JOINT_PREP_MANIFEST.json'))
    mans = sorted(set(mans))
    out = dict(line='R41 extension of R40 (A budget 16000, B training-seed replicate, C IN2R-native Flickr parents, D oracle x frozen/JOINT on s1002, E image_only on s1002/s1003); '
                    'R40 recipe and in-unit checks; development experiment, no TEST read', part=part, frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
               authorisation=dict(user_quote='你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我', relayed_by='main session',
                                  priority='normal QOS nice 50; RTX6000PRO/L40S/H100 excl. node51/53/60; 16000-update jobs double time'),
               units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX_R41.csv'),
               reuse_references_sha256=sha256(W / 'REUSE_REFERENCES_R41.json'), diff_vs_r40_sha256=sha256(W / 'R41_DIFF_VS_R40.patch'), dedupe_sha256=sha256(W / 'DEDUPE_R41.json'),
               r40_protocol_frozen_sha256=sha256(W / 'PROTOCOL_FROZEN.json'), code_sha256_by_tree={'r41f': {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / 'r41f' / '*.py')))}},
               code_not_bound=[], parent_cache_manifest_sha256={m: sha256(m) for m in mans}, cpu_test=dict(job=cpu['job'], sha256=sha256(W / f'CPU_TEST_R41_{part}.json')),
               gpu_gate=dict(waived_by_user=True, utc=utc(), basis='user waiver (R37-R40): CPU checks on real data + in-unit fail-fast checks'),
               primary_endpoint='fixed final step (16000 for block A with step 8000 also reported; 8000 otherwise)', test_read=False)
    write_json(W / ('PROTOCOL_FROZEN_R41.json' if part == '1' else 'PROTOCOL_FROZEN_R41C.json'), out)
    print('R41 frozen part', part, len(units))


if __name__ == '__main__':
    main()
