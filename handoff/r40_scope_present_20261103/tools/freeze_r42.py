"""Freeze R42 (PROTOCOL_FROZEN_R42.json) after the seven new-parent builds and CPU_TEST_R42 ALL_PASS. R40/R41 freeze files untouched.
Usage: python -I -B tools/freeze_r42.py"""
import glob, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r42f'))
from common import sha256, write_json, read_json, utc, CONDS, JOINT_PARENTS
PART = sys.argv[1] if len(sys.argv) > 1 else '1'
cpu = read_json(W / ('CPU_TEST_R42.json' if PART == '1' else 'CPU_TEST_R42B.json')); assert cpu['ALL_PASS']
units = [u for u in read_json(W / 'configs/UNITS_R42.json') if ('F40_IN2R_H1003' in u) == (PART == '2')]; assert len(units) == (42 if PART == '1' else 6)
conds = sorted({read_json(W / 'configs' / f'{u}.json')['cond'] for u in units})
mans = ['/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/locks/LOCKS.json', '/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json']
for c in conds:
    mans += [str(Path(CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json'), str(Path(JOINT_PARENTS[c]['prep_dir']) / 'JOINT_PREP_MANIFEST.json')]
parents = {c: dict(h_unit=CONDS[c]['h_unit'], noise_file=CONDS[c]['noise_file'], noise_sha256=CONDS[c]['noise_sha256'], noise_percent=CONDS[c]['noise_percent'],
                   cache_native_dev_rsum=read_json(Path(CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json')['native_dev']['rsum'],
                   cache_gpu=read_json(Path(CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json')['env']['gpu']) for c in conds}
write_json(W / ('PROTOCOL_FROZEN_R42.json' if PART == '1' else 'PROTOCOL_FROZEN_R42B.json'), dict(part=PART,
    line='R42 noise dose-response of JOINT on IN2R-native Flickr parents (20/40/80%) x {N-frozen-shared-cosine, N-both_joint} + ORACLE_MISSING x {frozen, both_joint} on f80 s1001; R41 C recipe; development experiment, no TEST read',
    frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
    authorisation=dict(user_quote='你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我', relayed_by='main session',
                       reason='R41: JOINT-FROZEN +17-21 at 40% but -0.1 to -1.6 at 80% (IN2R-native), CC152K fails, seed variance about +-1.5, ORACLE-JOINT replicates: noise dose-response with replication and whether the 80% failure is a supervision-quality limit',
                       priority='normal QOS nice 50; frozen cells 2h, joint 8h; one oracle job at a time'),
    units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX_R42.csv'),
    diff_vs_r41g_sha256=sha256(W / 'R42_DIFF_VS_R41G.patch'), parents=parents,
    other_freeze_files_untouched={f: sha256(W / f) for f in ('PROTOCOL_FROZEN.json', 'PROTOCOL_FROZEN_R41.json', 'PROTOCOL_FROZEN_R41C.json', 'PROTOCOL_FROZEN_R41F.json', 'PROTOCOL_FROZEN_R41G.json')},
    code_sha256_by_tree={'r42f': {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / 'r42f' / '*.py')))}}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sha256(m) for m in sorted(set(mans))}, cpu_test=dict(job=cpu['job'], sha256=sha256(W / ('CPU_TEST_R42.json' if PART == '1' else 'CPU_TEST_R42B.json'))),
    gpu_gate=dict(waived_by_user=True, utc=utc(), basis='user waiver; CPU checks + in-unit fail-fast checks'), primary_endpoint='fixed step 8000', test_read=False))
print('R42 frozen part', PART, len(units))
