"""Freeze the R41-F amendment (PROTOCOL_FROZEN_R41F.json) after CPU_TEST_R41F ALL_PASS. R41 part 1/2 freeze files are not modified; this file
registers the amendment (reason + user quote) and binds the r41c code, the 3 configs and the CC parent manifests. Usage: python -I -B tools/freeze_r41f.py"""
import glob, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r41c'))
from common import sha256, write_json, read_json, utc
cpu = read_json(W / 'CPU_TEST_R41F.json'); assert cpu['ALL_PASS']
units = read_json(W / 'configs/UNITS_R41F.json')
c0 = read_json(W / 'configs' / f'{units[0]}.json')
mans = ['/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/cache/cc/CACHE_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/prep/PREP_MANIFEST.json',
        '/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/prep/cc/JOINT_PREP_MANIFEST.json']
write_json(W / 'PROTOCOL_FROZEN_R41F.json', dict(
    line='R41-F amendment to R41: CC152K CC31 (R31 H0_default cc_s1001) N x both_joint x J/BCE/NCE, R39 CC N-JOINT recipe with student lr 1e-6 -> 1e-7 (10x lower); one variant, not a grid',
    frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS', amendment=c0['amendment'], units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units},
    resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX_R41F.csv'), diff_vs_r40c_sha256=sha256(W / 'R41F_DIFF_VS_R40C.patch'),
    r41_part_files_untouched={f: sha256(W / f) for f in ('PROTOCOL_FROZEN_R41.json', 'PROTOCOL_FROZEN_R41C.json')},
    dedupe_note='cell key equals R39_CC152K_N_joint_global_<obj> except optimizer.lr_encoder (1e-6 vs 1e-5): not a duplicate (declared single step-size variant)',
    code_sha256_by_tree={'r41c': {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / 'r41c' / '*.py')))}}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sha256(m) for m in mans}, cpu_test=dict(job=cpu['job'], sha256=sha256(W / 'CPU_TEST_R41F.json')),
    gpu_gate=dict(waived_by_user=True, utc=utc(), basis='user waiver; CPU checks + in-unit fail-fast checks'), primary_endpoint='fixed step 8000', test_read=False))
print('R41-F frozen')
