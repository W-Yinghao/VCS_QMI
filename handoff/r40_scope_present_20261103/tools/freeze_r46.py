"""Freeze R46 (PROTOCOL_FROZEN_R46.json with parent_cache_manifest_sha256, read by check_frozen) after CPU_TEST_R46 ALL_PASS and GPU_SMOKE_R46 (both
smoke jobs ALL_PASS). Usage: python -I -B tools/freeze_r46.py <smoke_job_o> <smoke_job_f>"""
import glob, hashlib, json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
cpu = rj(W / 'CPU_TEST_R46.json'); assert cpu['ALL_PASS']
sm = rj(W / 'GPU_SMOKE_R46.json'); smk = {j: sm[j] for j in sys.argv[1:3]}; assert all(v['ALL_PASS'] for v in smk.values()), {j: v['ALL_PASS'] for j, v in smk.items()}
units = rj(W / 'configs/UNITS_R46.json'); assert len(units) == 30
out = W / 'PROTOCOL_FROZEN_R46.json'; assert not out.exists()
gg = W / 'GPU_GATE.json'   # read by the trees' check_frozen when the gate is not waived; only R46 freezes use it (R40-R45 freezes record waived_by_user)
assert not gg.exists()
gg.write_text(json.dumps(dict(line='R46 GPU smoke / gate', ALL_PASS=True, smoke_jobs=list(smk), gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R46.json'), utc=utc), indent=1))
mans = []
for t in ('r46f', 'r46o'):
    sys.path.insert(0, str(W / t)); sys.modules.pop('common', None)
    import common as C
    for c in sorted({rj(W / 'configs' / f'{u}.json')['cond'] for u in units if rj(W / 'configs' / f'{u}.json')['code_tree'] == t}):
        mans += [str(Path(C.CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(C.CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json'), str(Path(C.JOINT_PARENTS[c]['prep_dir']) / 'JOINT_PREP_MANIFEST.json')]
    if t == 'r46o':
        mans.append(str(C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json')))
    sys.path.remove(str(W / t)); sys.modules.pop('common', None)
mans += sorted(glob.glob('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/r46_parents/r2cand/R2CAND_MANIFEST_*.json'))
assert all(Path(m).exists() for m in mans)
out.write_text(json.dumps(dict(line='R46: A FIT-pool restriction on COCO 80% (29k / 58k images) x {N-frozen, N-JOINT}; B iterated N supervision round 2 (REBUILT / CONTINUED) on F60 / F80 / C80; development experiment, no TEST read',
    frozen_utc=utc, status='FROZEN_BEFORE_FORMAL_UNITS',
    authorisation=dict(user_quote='现在就开始设计R46吧，我同意你的设计，该提交的CPU，GPUgate smoke test现在就准备', relayed_by='main session', priority='normal QOS nice 50'),
    units=units, config_sha256={u: sh(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sh(W / 'RESOLVED_MATRIX_R46.csv'),
    dedup_sha256=sh(W / 'DEDUP_R46.json'), adaptation_sha256=sh(W / 'R46_ADAPTATION.json'), diff_sha256={p: sh(W / p) for p in ('R46_DIFF_R46F.patch', 'R46_DIFF_R46O.patch')},
    patch_tool_sha256=sh(W / 'tools' / 'r46_patch.py'), other_freeze_files_untouched={Path(f).name: sh(f) for f in sorted(glob.glob(str(W / 'PROTOCOL_FROZEN*.json')))},
    code_sha256_by_tree={t: {Path(f).name: sh(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r46f', 'r46o')}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sh(m) for m in sorted(set(mans))}, cpu_test=dict(job=cpu['job'], sha256=sh(W / 'CPU_TEST_R46.json')),
    gpu_gate=dict(waived_by_user=False, smoke_jobs={j: dict(ALL_PASS=v['ALL_PASS'], units={u: x['passed'] for u, x in v['units'].items()}) for j, v in smk.items()},
                  gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R46.json'), gpu_gate_file_sha256=sh(gg), basis='user asked for a CPU check AND a GPU smoke / gate before the formal units'),
    primary_endpoint='fixed step 8000 (B: 8000 round-2 updates)', test_read=False), indent=1, ensure_ascii=False))
print('R46 frozen', len(units))
