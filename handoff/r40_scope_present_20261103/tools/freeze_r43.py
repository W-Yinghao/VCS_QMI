"""Freeze R43 after the part's CPU check ALL_PASS: part B (TRUST+N, trees r43f / r43c) -> PROTOCOL_FROZEN_R43.json; part A (MS-COCO, tree r43o)
-> PROTOCOL_FROZEN_R43A.json. Earlier freeze files untouched. Usage: python -I -B tools/freeze_r43.py B|A"""
import glob, json, sys, hashlib, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
PART = sys.argv[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rj = lambda p: json.loads(Path(p).read_text())
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
cpu = rj(W / f'CPU_TEST_R43_{PART}.json'); assert cpu['ALL_PASS']
units = [u for u in rj(W / 'configs/UNITS_R43.json') if u.startswith(f'R43_{PART}_')]; assert len(units) == (24 if PART == 'B' else 42)
trees = ('r43f', 'r43c') if PART == 'B' else ('r43o',)
mans = []
for t in trees:
    sys.path.insert(0, str(W / t)); sys.modules.pop('common', None)
    import common as C
    for c in sorted({rj(W / 'configs' / f'{u}.json')['cond'] for u in units if rj(W / 'configs' / f'{u}.json')['code_tree'] == t}):
        mans += [str(Path(C.CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(C.CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json')]
        jp = C.JOINT_PARENTS[c]['prep_dir'] if hasattr(C, 'JOINT_PARENTS') and c in getattr(C, 'JOINT_PARENTS') else C.R39_PREP / C.DS_KEY
        mans.append(str(Path(jp) / 'JOINT_PREP_MANIFEST.json'))
    if PART == 'A':
        mans.append(str(C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json')))
    sys.path.remove(str(W / t))
patches = ('R43F_DIFF_VS_R42F.patch', 'R43C_DIFF_VS_R41C.patch') if PART == 'B' else ('R43O_DIFF.patch',)
out = W / ('PROTOCOL_FROZEN_R43.json' if PART == 'B' else 'PROTOCOL_FROZEN_R43A.json')
assert not out.exists()
out.write_text(json.dumps(dict(part=PART,
    line=('R43 B: deployable cleaner supervision under JOINT (R37 TRUST_NATIVE_NULL) x {frozen, both_joint} on the 80% IN2R-native Flickr parents and CC152K CC31' if PART == 'B'
          else 'R43 A: MS-COCO (second injected-noise dataset) IN2R-native R33 H0 parents x {N-frozen-shared-cosine, N-both_joint}') + '; R41 C / R42 recipe; development experiment, no TEST read',
    frozen_utc=utc, status='FROZEN_BEFORE_FORMAL_UNITS',
    authorisation=dict(user_quote='你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我', relayed_by='main session',
                       priority='normal QOS nice 50; frozen 2h, joint 8h (COCO memory/time from the in-build timing run, registered in R43_ADAPTATION.json)'),
    units=units, config_sha256={u: sh(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sh(W / 'RESOLVED_MATRIX_R43.csv'),
    diff_sha256={p: sh(W / p) for p in patches}, adaptation_sha256=sh(W / 'R43_ADAPTATION.json') if (W / 'R43_ADAPTATION.json').exists() else None,
    other_freeze_files_untouched={f: sh(W / f) for f in ('PROTOCOL_FROZEN.json', 'PROTOCOL_FROZEN_R41.json', 'PROTOCOL_FROZEN_R41C.json', 'PROTOCOL_FROZEN_R41F.json',
                                                          'PROTOCOL_FROZEN_R41G.json', 'PROTOCOL_FROZEN_R42.json', 'PROTOCOL_FROZEN_R42B.json') + (('PROTOCOL_FROZEN_R43.json',) if PART == 'A' else ())},
    code_sha256_by_tree={t: {Path(f).name: sh(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in trees}, code_not_bound=[],
    parent_cache_manifest_sha256={m: sh(m) for m in sorted(set(mans))}, cpu_test=dict(job=cpu['job'], sha256=sh(W / f'CPU_TEST_R43_{PART}.json')),
    gpu_gate=dict(waived_by_user=True, utc=utc, basis='user waiver; CPU checks + in-unit fail-fast checks (step-0 native reproduction, cache equivalence, R38 graph check, R40 one-step rule)'),
    primary_endpoint='fixed step 8000', test_read=False), indent=1, ensure_ascii=False))
print('R43 frozen part', PART, len(units))
