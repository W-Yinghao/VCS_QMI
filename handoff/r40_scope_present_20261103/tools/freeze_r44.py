"""Freeze R44 (PROTOCOL_FROZEN_R44.json + R44_PROTOCOL_DIFF.json) after CPU_TEST_R44 ALL_PASS. Earlier freeze files untouched. Usage: python -I -B tools/freeze_r44.py"""
import glob, hashlib, json, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
PKG = Path('/home/infres/yinwang/CS_QMI/r44_plan/VCS_R39X_R42_Review_R44_Plan_20261103')
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
cpu = rj(W / 'CPU_TEST_R44.json'); assert cpu['ALL_PASS']
units = rj(W / 'configs/UNITS_R44.json'); assert len(units) == 39
out = W / 'PROTOCOL_FROZEN_R44.json'; assert not out.exists()


def parent_manifests():
    """cache / prep / JOINT prep manifests of every R44 condition (key read by the frozen trainer's check_frozen)."""
    import sys
    mans = []
    for t in ('r44f', 'r44c'):
        sys.path.insert(0, str(W / t)); sys.modules.pop('common', None)
        import common as C
        for c in sorted({rj(W / 'configs' / f'{u}.json')['cond'] for u in units if rj(W / 'configs' / f'{u}.json')['code_tree'] == t}):
            mans += [str(Path(C.CONDS[c]['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(C.CONDS[c]['prep_dir']) / 'PREP_MANIFEST.json')]
            jp = C.JOINT_PARENTS[c]['prep_dir'] if c in getattr(C, 'JOINT_PARENTS', {}) else C.R39_PREP / C.DS_KEY
            mans.append(str(Path(jp) / 'JOINT_PREP_MANIFEST.json'))
        sys.path.remove(str(W / t)); sys.modules.pop('common', None)
    assert all(Path(m).exists() for m in mans), [m for m in mans if not Path(m).exists()]
    return {m: sh(m) for m in sorted(set(mans))}
diff = dict(utc=utc, vs='R39 / R39X N-JOINT recipe (R41-A 16k stretched cosine for block A; R40 N_PRESENT_ONLY for block C)',
    A='training seed 2002 (R41-B definition: build_heads(..., 2002), sampler [2002, 3]) on the R41-A 16000-update recipe from the original H + fresh head; readouts + 12000',
    B='routing ROUTE_PRESENT / ROUTE_RANDOM: head + null = grad of the full-N loss, student encoders = grad of the masked conditional loss (observed-present rows / independent '
      'Bernoulli(p_d) rows from default_rng([init_seed, 44, step])), fixed p_d = R40 presence_probs; two autograd.grad on one graph; per-peer single clip; single AdamW step',
    C='R40 N_PRESENT_ONLY x both_joint on F00 (R30-H f00_s1001) and F80 (R33 H0 f80_s1001); F60 / CC PRESENT_ALL reused from R40',
    unchanged='N lists / labels / Q, shared G + cosine (head 1e-4 -> 1e-5, encoder 1e-5 -> 1e-6), AdamW wd 1e-4, per-peer clip 2, 16 queries / direction, '
              'DEV two-stage frozen-H top-100 rerank, step-0 native reproduction, R38 graph check, R40 one-step rule, frozen H / buffers',
    in_unit_checks='ROUTING_CHECK.json at update 0 (all-ones p=1 = ALL encoder grad rel 1e-5; head grad = ALL head grad bitwise; encoder grad = conditional objective bitwise; '
                   'masked rows zero encoder / nonzero head score-gradient); every update: masked rows zero encoder score-gradient, one optimizer step',
    code=dict(r44f='r42f + tools/r44_patch.py (train_unit) + common R44 section (R44_DIFF_R44F.patch); oracle runner / oracle_truth not copied',
              r44c='r41c + tools/r44_patch.py (train_unit) + common R44 section (R44_DIFF_R44C.patch)'))
(W / 'R44_PROTOCOL_DIFF.json').write_text(json.dumps(diff, indent=1))
out.write_text(json.dumps(dict(line='R44 (package VCS_R39X_R42_Review_R44_Plan_20261103): A 16k seed-2002 confirmation, B head/encoder gradient routing, C PRESENT_ALL on F00/F80; development experiment, no TEST read',
    frozen_utc=utc, status='FROZEN_BEFORE_FORMAL_UNITS', package=dict(path=str(PKG), manifest_sha256=sh(PKG / 'MANIFEST.sha256'), matrix_sha256=sh(PKG / 'configs' / 'MATRIX.csv'),
                                                                       reuse_references_sha256=sh(PKG / 'configs' / 'REUSE_REFERENCES.csv')),
    authorisation=dict(user_quote='查看这个包，准备相关的实验，并提交，这个我也倾向是normal，优先级还是比R33高', relayed_by='main session', priority='normal QOS nice 50 (R33 nice 10000)'),
    units=units, config_sha256={u: sh(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sh(W / 'RESOLVED_MATRIX_R44.csv'),
    dedup_sha256=sh(W / 'DEDUP_R44.json'), parent_binding_sha256=sh(W / 'PARENT_BINDING_R44.json'), protocol_diff_sha256=sh(W / 'R44_PROTOCOL_DIFF.json'),
    diff_sha256={p: sh(W / p) for p in ('R44_DIFF_R44F.patch', 'R44_DIFF_R44C.patch')}, patch_tool_sha256=sh(W / 'tools' / 'r44_patch.py'),
    other_freeze_files_untouched={Path(f).name: sh(f) for f in sorted(glob.glob(str(W / 'PROTOCOL_FROZEN*.json'))) if Path(f).name != out.name},
    code_sha256_by_tree={t: {Path(f).name: sh(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r44f', 'r44c')}, code_not_bound=[],
    parent_cache_manifest_sha256=parent_manifests(), cpu_test=dict(job=cpu['job'], sha256=sh(W / 'CPU_TEST_R44.json')),
    gpu_gate=dict(waived_by_user=True, utc=utc, basis='user waiver; CPU checks + in-unit fail-fast checks'), primary_endpoint='fixed step 16000 (A) / 8000 (B, C)', test_read=False), indent=1, ensure_ascii=False))
print('R44 frozen', len(units))
