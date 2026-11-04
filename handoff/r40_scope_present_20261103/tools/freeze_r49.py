"""Freeze R49 (PROTOCOL_FROZEN_R49.json with parent_cache_manifest_sha256, read by check_frozen) after CPU_TEST_R49 ALL_PASS and GPU_SMOKE_R49
(all three smoke jobs ALL_PASS). Modelled on freeze_r47.py. The trees' check_frozen reads the shared ROOT/GPU_GATE.json (R46, not modified); the R49
gate is enforced here (refuses to freeze unless every listed smoke job is ALL_PASS) and recorded in GPU_GATE_R49.json.
Usage: python -I -B tools/freeze_r49.py <smoke_job_f> <smoke_job_o> <smoke_job_c>"""
import glob, hashlib, json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
TREES = ('r49f', 'r49o', 'r49c')
cpu = rj(W / 'CPU_TEST_R49.json'); assert cpu['ALL_PASS']
sm = rj(W / 'GPU_SMOKE_R49.json'); smk = {j: sm[j] for j in sys.argv[1:4]}
assert len(smk) == 3 and all(v['ALL_PASS'] for v in smk.values()), {j: v['ALL_PASS'] for j, v in smk.items()}
units = rj(W / 'configs/UNITS_R49.json'); assert len(units) == 57
cfg = {u: rj(W / 'configs' / f'{u}.json') for u in units}
smoked = {u for v in smk.values() for u in v['units']}
assert {(c['code_tree'], c['cond']) for c in cfg.values()} == {(cfg[u]['code_tree'], cfg[u]['cond']) for u in smoked}, 'a tree/cond is not covered by the smokes'
out = W / 'PROTOCOL_FROZEN_R49.json'; assert not out.exists()
gg = W / 'GPU_GATE_R49.json'; assert not gg.exists()
gg.write_text(json.dumps(dict(line='R49 GPU smoke / gate', ALL_PASS=True, smoke_jobs=list(smk), gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R49.json'), utc=utc), indent=1))
mans = []
for t in TREES:
    sys.path.insert(0, str(W / t)); sys.modules.pop('common', None)
    import common as C
    for c in sorted({cfg[u]['cond'] for u in units if cfg[u]['code_tree'] == t}):
        mans += [str(Path(C.CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(C.CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json')]
    if t in ('r49o', 'r49c'):
        tm = C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json')
        mans.append(str(tm) if tm.exists() else str(C.TOKENS_FILE))
    sys.path.remove(str(W / t)); sys.modules.pop('common', None)
assert all(Path(m).exists() for m in mans), [m for m in mans if not Path(m).exists()]
out.write_text(json.dumps(dict(
    line='R49: N-frozen (shared G head, cosine schedule, frozen parent encoders) trained and read out on R47-style layout-free G parent caches x J/BCE/NCE x 19 parents '
         '(Flickr60 s1002/3, Flickr20 s1001-3, Flickr40 s1001-3, Flickr80 s1002/3, COCO60 s1002/3, COCO80 s1001-3, COCO20 s1001, CC152K R31 s1002/3, R37S)',
    frozen_utc=utc, status='FROZEN_BEFORE_FORMAL_UNITS',
    authorisation=dict(user_quote='好，准备R49吧，smoke过了就提交，TEST也再读一次', basis='R44 audit: R34-R43 frozen-critic heads were trained on native per-batch-centred FIT features; '
                       'R49 extends the R48 layout-free N-frozen to the remaining parents', priority='normal QOS nice 50'),
    units=units, config_sha256={u: sh(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sh(W / 'RESOLVED_MATRIX_R49.csv'),
    tool_sha256={p: sh(W / 'tools' / p) for p in ('make_configs_r49.py', 'r49_smoke_check.py', 'freeze_r49.py')},
    other_freeze_files_untouched={Path(f).name: sh(f) for f in sorted(glob.glob(str(W / 'PROTOCOL_FROZEN*.json'))) if Path(f).name != out.name},
    code_sha256_by_tree={t: {Path(f).name: sh(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in TREES}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sh(m) for m in sorted(set(mans))}, cpu_test=dict(job=cpu['job'], sha256=sh(W / 'CPU_TEST_R49.json')),
    gpu_gate=dict(waived_by_user=False, smoke_jobs={j: dict(ALL_PASS=v['ALL_PASS'], units={u: x['passed'] for u, x in v['units'].items()}) for j, v in smk.items()},
                  gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R49.json'), gpu_gate_r49_file_sha256=sh(gg),
                  note='check_frozen reads the shared ROOT/GPU_GATE.json (R46, ALL_PASS, not modified); the R49 gate is enforced by freeze_r49.py and recorded in GPU_GATE_R49.json'),
    primary_endpoint='fixed step 8000; DEV two-stage readout with the layout-free G centring (same as R47 G / R48)',
    test_read=False), indent=1, ensure_ascii=False))
print('R49 frozen', len(units))
