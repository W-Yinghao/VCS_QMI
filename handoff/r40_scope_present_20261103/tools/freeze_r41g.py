"""Freeze the R41-G amendment (PROTOCOL_FROZEN_R41G.json) after CPU_TEST_R41G ALL_PASS: r41g code, 9 configs, parent manifests, amendment reason +
user quote. R41 part 1/2 and R41-F freeze files are not modified. Usage: python -I -B tools/freeze_r41g.py"""
import glob, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r41g'))
from common import sha256, write_json, read_json, utc, JOINT_PARENTS
cpu = read_json(W / 'CPU_TEST_R41G.json'); assert cpu['ALL_PASS']
units = read_json(W / 'configs/UNITS_R41G.json')
mans = ['/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n00/CACHE_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/locks/LOCKS.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/prep/PREP_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1003/CACHE_MANIFEST.json',
        str(Path(JOINT_PARENTS['n00']['prep_dir']) / 'JOINT_PREP_MANIFEST.json'), str(Path(JOINT_PARENTS['n60s1003']['prep_dir']) / 'JOINT_PREP_MANIFEST.json')]
write_json(W / 'PROTOCOL_FROZEN_R41G.json', dict(
    line='R41-G amendment to R41: G1 Flickr 0% R30-H f00_s1001 N x both_joint, training seed 2002 (3); G2 ORACLE_MISSING x {frozen, both_joint} on Flickr60 R30-H s1003 (6); R41/R40 recipe',
    frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
    amendment=dict(name='R41-G', user_quote='你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我', relayed_by='main session',
                   reasons=dict(G1='replicate the 0% NCE-JOINT 551.1 outlier and give a second 0% J/BCE point', G2='second-parent replication of the ORACLE-JOINT gap alongside block D (s1002)')),
    units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX_R41G.csv'),
    diff_vs_r41f_sha256=sha256(W / 'R41G_DIFF_VS_R41F.patch'),
    other_freeze_files_untouched={f: sha256(W / f) for f in ('PROTOCOL_FROZEN_R41.json', 'PROTOCOL_FROZEN_R41C.json', 'PROTOCOL_FROZEN_R41F.json') if (W / f).exists()},
    code_sha256_by_tree={'r41g': {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / 'r41g' / '*.py')))}}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sha256(m) for m in mans}, cpu_test=dict(job=cpu['job'], sha256=sha256(W / 'CPU_TEST_R41G.json')),
    gpu_gate=dict(waived_by_user=True, utc=utc(), basis='user waiver; CPU checks + in-unit fail-fast checks'), primary_endpoint='fixed step 8000', test_read=False))
print('R41-G frozen')
