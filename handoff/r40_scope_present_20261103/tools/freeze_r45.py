"""Freeze R45 (PROTOCOL_FROZEN_R45.json, with parent_cache_manifest_sha256 read by check_frozen) after CPU_TEST_R45 ALL_PASS. Usage: python -I -B tools/freeze_r45.py"""
import glob, hashlib, json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
cpu = rj(W / 'CPU_TEST_R45.json'); assert cpu['ALL_PASS']
units = rj(W / 'configs/UNITS_R45.json'); assert len(units) == 6
out = W / 'PROTOCOL_FROZEN_R45.json'; assert not out.exists()
sys.path.insert(0, str(W / 'r45o')); import common as C
mans = []
for c in sorted({rj(W / 'configs' / f'{u}.json')['cond'] for u in units}):
    mans += [str(Path(C.CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(C.CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json'), str(Path(C.JOINT_PARENTS[c]['prep_dir']) / 'JOINT_PREP_MANIFEST.json')]
mans.append(str(C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json')))
out.write_text(json.dumps(dict(line='R45: COCO N x both_joint 16000-update budget (R41-A recipe) on R33 H0 c60 s1001 / c80 s1001; development experiment, no TEST read',
    frozen_utc=utc, status='FROZEN_BEFORE_FORMAL_UNITS',
    authorisation=dict(user_quote='你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我', relayed_by='main session', priority='normal QOS nice 50'),
    units=units, config_sha256={u: sh(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sh(W / 'RESOLVED_MATRIX_R45.csv'), diff_sha256={'R45_DIFF_VS_R43O.patch': sh(W / 'R45_DIFF_VS_R43O.patch')},
    other_freeze_files_untouched={Path(f).name: sh(f) for f in sorted(glob.glob(str(W / 'PROTOCOL_FROZEN*.json')))},
    code_sha256_by_tree={'r45o': {Path(f).name: sh(f) for f in sorted(glob.glob(str(W / 'r45o' / '*.py')))}}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sh(m) for m in sorted(set(mans))}, cpu_test=dict(job=cpu['job'], sha256=sh(W / 'CPU_TEST_R45.json')),
    gpu_gate=dict(waived_by_user=True, utc=utc, basis='user waiver; CPU checks + in-unit fail-fast checks'), primary_endpoint='fixed step 16000 (8000 also reported)', test_read=False), indent=1, ensure_ascii=False))
print('R45 frozen', len(units))
