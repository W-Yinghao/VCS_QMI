"""Freeze R47 (PROTOCOL_FROZEN_R47.json with parent_cache_manifest_sha256, read by check_frozen) after CPU_TEST_R47 ALL_PASS and GPU_SMOKE_R47
(all three smoke jobs ALL_PASS). The trees' check_frozen reads ROOT/GPU_GATE.json (shared, written by the R46 freeze); it is NOT modified here.
The R47 gate is enforced by this script (refuses to freeze unless every listed smoke job is ALL_PASS) and recorded in GPU_GATE_R47.json.
Usage: python -I -B tools/freeze_r47.py <smoke_job_f> <smoke_job_o> <smoke_job_c>"""
import glob, hashlib, json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
cpu = rj(W / 'CPU_TEST_R47.json'); assert cpu['ALL_PASS']
sm = rj(W / 'GPU_SMOKE_R47.json'); smk = {j: sm[j] for j in sys.argv[1:4]}; assert len(smk) == 3 and all(v['ALL_PASS'] for v in smk.values()), {j: v['ALL_PASS'] for j, v in smk.items()}
units = rj(W / 'configs/UNITS_R47.json'); assert len(units) == 30
cfg = {u: rj(W / 'configs' / f'{u}.json') for u in units}
smoked = {u for v in smk.values() for u in v['units']}
assert {(c['code_tree'], c['cond']) for c in cfg.values()} == {(cfg[u]['code_tree'], cfg[u]['cond']) for u in smoked}, 'a tree/cond is not covered by the smokes'
out = W / 'PROTOCOL_FROZEN_R47.json'; assert not out.exists()
gg = W / 'GPU_GATE_R47.json'; assert not gg.exists()
gg.write_text(json.dumps(dict(line='R47 GPU smoke / gate', ALL_PASS=True, smoke_jobs=list(smk), gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R47.json'), utc=utc), indent=1))
mans = []
for t in ('r47f', 'r47o', 'r47c'):
    sys.path.insert(0, str(W / t)); sys.modules.pop('common', None)
    import common as C
    for c in sorted({cfg[u]['cond'] for u in units if cfg[u]['code_tree'] == t}):
        mans += [str(Path(C.CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(C.CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json'), str(Path(C.JOINT_PARENTS[c]['prep_dir']) / 'JOINT_PREP_MANIFEST.json')]
    if t in ('r47o', 'r47c'):
        mans.append(str(C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json')) if C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json').exists() else str(C.TOKENS_FILE))
    sys.path.remove(str(W / t)); sys.modules.pop('common', None)
assert all(Path(m).exists() for m in mans), [m for m in mans if not Path(m).exists()]
out.write_text(json.dumps(dict(line='R47: N-JOINT with layout-free FIT centring (G: fixed global parent means; S: own training-batch means) x J/BCE/NCE x {F60 R30-H s1001, F80 IN2R s1001, F00 R30-H s1001, C60 IN2R s1001, CC31}; primary DEV readout with the matching layout-free centring (parent cache rebuilt layout-free); development experiment, no TEST read',
    frozen_utc=utc, status='FROZEN_BEFORE_FORMAL_UNITS',
    authorisation=dict(basis='R44 audit (layout leak through the JOINT student native per-batch centring); main session go-ahead: once CPU check and GPU smokes pass, freeze PROTOCOL_FROZEN_R47.json and submit all 30 formal units (normal, nice 50, no TEST)',
                       standing_user_quote='你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我', relayed_by='main session', priority='normal QOS nice 50'),
    units=units, config_sha256={u: sh(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sh(W / 'RESOLVED_MATRIX_R47.csv'),
    diff_sha256={p: sh(W / p) for p in ('R47_DIFF_R47F.patch', 'R47_DIFF_R47O.patch', 'R47_DIFF_R47C.patch')},
    tool_sha256={p: sh(W / 'tools' / p) for p in ('r47_patch.py', 'r47_build_lf.py', 'make_configs_r47.py', 'r47_smoke_check.py')},
    other_freeze_files_untouched={Path(f).name: sh(f) for f in sorted(glob.glob(str(W / 'PROTOCOL_FROZEN*.json')))},
    code_sha256_by_tree={t: {Path(f).name: sh(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r47f', 'r47o', 'r47c')}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sh(m) for m in sorted(set(mans))}, cpu_test=dict(job=cpu['job'], sha256=sh(W / 'CPU_TEST_R47.json')),
    gpu_gate=dict(waived_by_user=False, smoke_jobs={j: dict(ALL_PASS=v['ALL_PASS'], units={u: x['passed'] for u, x in v['units'].items()}) for j, v in smk.items()},
                  gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R47.json'), gpu_gate_r47_file_sha256=sh(gg),
                  note='check_frozen reads the shared ROOT/GPU_GATE.json (R46, ALL_PASS, not modified); the R47 gate is enforced by freeze_r47.py and recorded in GPU_GATE_R47.json'),
    primary_endpoint='fixed step 8000; DEV two-stage readout with the variant layout-free centring (G: global means; S: fixed random 512-item batches, seed [4701, split, modality, peer]); native-layout readout secondary',
    test_read=False), indent=1, ensure_ascii=False))
print('R47 frozen', len(units))
