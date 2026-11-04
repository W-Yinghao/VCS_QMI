"""Freeze R50 (PROTOCOL_FROZEN_R50.json with parent_cache_manifest_sha256 and code hashes, read by check_frozen) after CPU_TEST_R50 ALL_PASS and the three
GPU smokes ALL_PASS (GPU_SMOKE_R50.json; every (tree, cond, supervision, encoder) cell covered). The trees' check_frozen reads the shared ROOT/GPU_GATE.json
(R46, not modified); the R50 gate is enforced here and recorded in GPU_GATE_R50.json. Modelled on freeze_r48.py.
Usage: python -I -B tools/freeze_r50.py <smoke_job_f> <smoke_job_c> <smoke_job_oracle>"""
import glob, hashlib, json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
TREES = ('r50f', 'r50c')
PKG = Path('/home/infres/yinwang/CS_QMI/r50_layout_audit/VCS_Layout_Audit_R50_20261104')
cpu = rj(W / 'CPU_TEST_R50.json'); assert cpu['ALL_PASS']
sm = rj(W / 'GPU_SMOKE_R50.json'); smk = {j: sm[j] for j in sys.argv[1:4]}
assert len(smk) == 3 and all(v['ALL_PASS'] for v in smk.values()), {j: v['ALL_PASS'] for j, v in smk.items()}
assert 'NPY_DISABLE_CPU_FEATURES="AVX512F' in (W / 'sbatch/units_r50.sbatch').read_text(), 'amendment A2 env missing in units_r50.sbatch'
units = rj(W / 'configs/UNITS_R50.json'); assert len(units) == 24
cfg = {u: rj(W / 'configs' / f'{u}.json') for u in units}
smoked = {u for v in smk.values() for u in v['units']}
key = lambda c: (c['code_tree'], c['cond'], c['supervision'], c['adaptation'], c['oracle'])
assert {key(c) for c in cfg.values()} == {key(cfg[u]) for u in smoked if u in cfg}, 'a tree/cond/supervision/encoder cell is not covered by the smokes'
out = W / 'PROTOCOL_FROZEN_R50.json'; assert not out.exists()
gg = W / 'GPU_GATE_R50.json'; assert not gg.exists()
gg.write_text(json.dumps(dict(line='R50 GPU smoke / gate', ALL_PASS=True, smoke_jobs=list(smk), gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R50.json'), utc=utc), indent=1))
mans = []
for t in TREES:
    sys.path.insert(0, str(W / t)); sys.modules.pop('common', None)
    import common as C
    for c in sorted({cfg[u]['cond'] for u in units if cfg[u]['code_tree'] == t}):
        mans += [str(Path(C.CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(C.CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json'),
                 str(Path(C.JOINT_PARENTS[c]['prep_dir']) / 'JOINT_PREP_MANIFEST.json'), str(Path(C.CONDS[c]['cache_dir']) / f"NORMALIZATION_{C.CONDS[c]['lf_base']}.json")]
    if t == 'r50c':
        tm = C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json')
        mans.append(str(tm) if tm.exists() else str(C.TOKENS_FILE))
    sys.path.remove(str(W / t)); sys.modules.pop('common', None)
assert all(Path(m).exists() for m in mans), [m for m in mans if not Path(m).exists()]
out.write_text(json.dumps(dict(
    line='R50 (package VCS_Layout_Audit_R50_20261104, protocol FIT_FIXED_CENTER_V1): A 12 = PRESENT_ONLY x {frozen, both_joint} x J/BCE/NCE x {F60 R30-H s1001, '
         'CC152K R31-H0 s1001}; B 12 = ORACLE_MISSING x {frozen, both_joint} x J/BCE/NCE x {F60 R30-H s1001, F80 IN2R s1001} (oracle runner, isolated); 8000 updates, '
         'R47 G / R48 recipe of each parent; N references = R48 N-frozen / R47 G N-JOINT weights re-read under the same protocol (REUSE_RECEIPT_R50.json)',
    frozen_utc=utc, status='FROZEN_BEFORE_FORMAL_UNITS', protocol_id='FIT_FIXED_CENTER_V1',
    authorisation=dict(user_quote='查看这个包，准备相关实验，提交', relayed_by='main session', priority='normal QOS nice 50', test_read=False),
    package=dict(dir=str(PKG), sha256_prefix='07998b44', matrix_sha256=sh(PKG / 'MATRIX.csv'), plan_sha256=sh(PKG / 'docs/SERVER_PLAN_CN.md'), file_sha256_json_sha256=sh(PKG / 'FILE_SHA256.json')),
    units=units, config_sha256={u: sh(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sh(W / 'RESOLVED_MATRIX_R50.csv'),
    tool_sha256={p: sh(W / 'tools' / p) for p in ('make_configs_r50.py', 'r50_patch.py', 'r50_build.py', 'r50_smoke_check.py', 'freeze_r50.py', 'r50_reread.py', 'r50_replay_check.py', 'r50_replay_stop.py', 'summarize_r50.py', 'r50_reuse_receipt.py', 'r50_src/r50ext.py')},
    test_sha256={p: sh(W / 'tests' / p) for p in ('cpu_test_r50.py', 'r50_cpu_real.py', 'r50_e2e.py', 'train_oracle_e2e_r50.py')},
    registered_diffs={p: sh(W / p) for p in ('R50_DIFF_R50F.patch', 'R50_DIFF_R50C.patch')},
    other_freeze_files_untouched={Path(f).name: sh(f) for f in sorted(glob.glob(str(W / 'PROTOCOL_FROZEN*.json'))) if Path(f).name != out.name},
    code_sha256_by_tree={t: {Path(f).name: sh(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in TREES}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sh(m) for m in sorted(set(mans))}, cpu_test=dict(job=cpu['job'], sha256=sh(W / 'CPU_TEST_R50.json')),
    gpu_gate=dict(waived_by_user=False, smoke_jobs={j: dict(ALL_PASS=v['ALL_PASS'], units={u: x['passed'] for u, x in v['units'].items()}) for j, v in smk.items()},
                  gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R50.json'), gpu_gate_r50_file_sha256=sh(gg),
                  note='check_frozen reads the shared ROOT/GPU_GATE.json (R46, ALL_PASS, not modified); the R50 gate is enforced by freeze_r50.py and recorded in GPU_GATE_R50.json'),
    tolerances=cpu.get('tolerances'), amendments={f: sh(W / f) for f in ('R50_AMENDMENT_A1.json', 'R50_AMENDMENT_A2.json')},
    tie_policy=dict(env_NPY_DISABLE_CPU_FEATURES='AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR', in_unit_assert='r50ext.protocol_check', sbatch_sha256={f: sh(W / 'sbatch' / f) for f in ('units_r50.sbatch', 'submit_formal_r50.sh')}), primary_endpoint='fixed step 8000; DEV two-stage readout on the R50 FIT-fixed cache (protocol FIT_FIXED_CENTER_V1)',
    outputs=dict(runs='/home/infres/yinwang/CS_QMI/outputs_v9_r50/runs_r50', oracle_runs='/home/infres/yinwang/CS_QMI/outputs_v9_r50/oracle_runs_r50',
                 note='on home: /projects had 301 GB free (the 300 GB floor) when R50 was built'),
    test_read=False), indent=1, ensure_ascii=False))
print('R50 frozen', len(units))
