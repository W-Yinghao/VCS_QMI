"""R39X CPU checks (Slurm CPU job; writes CPU_TEST_R39X.json): (1) code derivation r39xf/r39xc vs the frozen r39f/r39c trees (only the
registered R39X edits); (2) the 15 configs vs their N-shared-cosine reference configs (R38 / R37S) and vs the R39 JOINT template;
(3) real-data bindings per additional parent: cache/prep hash verification (load_cond), checkpoint sha == cache manifest parent sha,
native DEV rSum == reference unit's recorded H rSum, token ids < parent vocab size, reference RESULT/TRAIN_HISTORY hashes.
The reference batch-chain / init identity is asserted in-unit at every readout (train_unit reference check).
Usage: python -I -B tests/cpu_test_x.py"""
from __future__ import annotations
import difflib, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
RES = {}


def check(name, fn):
    t0 = time.time()
    try:
        v = fn(); RES[name] = dict(passed=bool(v.get('passed', True)), detail=v, seconds=time.time() - t0)
    except Exception:
        RES[name] = dict(passed=False, error=traceback.format_exc(), seconds=time.time() - t0)
    print(name, RES[name]['passed'], flush=True)


def sh(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def t_code():
    out = {}
    for x, b in (('r39xf', 'r39f'), ('r39xc', 'r39c')):
        files = sorted(p.name for p in (W / x).glob('*.py'))
        same = {f: sh(W / x / f) == sh(W / b / f) for f in files}
        diffs = {}
        for f in ('common.py', 'r39ext.py', 'train_unit.py'):
            d = [l for l in difflib.unified_diff((W / b / f).read_text().splitlines(), (W / x / f).read_text().splitlines(), lineterm='', n=0)
                 if l[:1] in '+-' and not l.startswith(('+++', '---'))]
            diffs[f] = dict(removed=[l for l in d if l.startswith('-')], n_added=sum(l.startswith('+') for l in d))
        extra = ('data35.py',) if x == 'r39xf' else ()
        ok_same = all(v for f, v in same.items() if f not in ('common.py', 'r39ext.py', 'train_unit.py') + extra)
        if extra:   # R39X n00: load_cond tolerates a parent without a B-TRUST file (only these lines change)
            d35 = [l for l in difflib.unified_diff((W / b / 'data35.py').read_text().splitlines(), (W / x / 'data35.py').read_text().splitlines(), lineterm='', n=0)
                   if l.startswith('-') and not l.startswith('---')]
            diffs['data35.py'] = dict(removed=d35)
            ok_same = ok_same and len(d35) == 4 and all(('trust' in l or l.startswith('-"""')) for l in d35)
        rem_tu = diffs['train_unit.py']['removed']
        ok_tu = len(rem_tu) == 4 and all(('R9.C.PARENT_CKPT' in l or "PROTOCOL_FROZEN.json" in l or l.startswith('-"""R39:')) for l in rem_tu)
        ok_common = len(diffs['common.py']['removed']) == 1 and diffs['common.py']['removed'][0].startswith('-"""R39')
        ok_ext = all(('PARENT_CKPT' in l or 'TARGET_COND' in l or 'DS_KEY' in l or 'load_parent' in l or 'self.dev = device; self.d = d' in l or 'target_cache_manifest_sha256' in l or l.startswith('-"""R39 additions')) for l in diffs['r39ext.py']['removed']) and len(diffs['r39ext.py']['removed']) == 9
        out[x] = dict(byte_identical=same, removed_lines=diffs, ok=bool(ok_same and ok_tu and ok_common and ok_ext))
    out['r39ext_identical_trees'] = sh(W / 'r39xf' / 'r39ext.py') == sh(W / 'r39xc' / 'r39ext.py')
    out['passed'] = out['r39xf']['ok'] and out['r39xc']['ok'] and out['r39ext_identical_trees']
    return out


def t_configs():
    units = json.loads((W / 'configs' / 'UNITS_R39X.json').read_text()); bad = []
    allowed = {'unit_id', 'package_unit', 'block', 'line', 'code_tree', 'runner', 'supervision_label', 'oracle', 'encoder', 'teacher_parents', 'reference_unit',
               'job_bundle', 'optimizer', 'joint_rule', 'evaluation', 'extension', 'architecture', 'architecture_rule', 'label', 'package_block', 'matrix_source', 'reference_note'}
    for u in units:
        c = json.loads((W / 'configs' / f'{u}.json').read_text()); r = c['reference_unit']
        if r is None:   # R39X 0% diagnostic: frozen/JOINT cells built from the R38 s1001 N-shared-cosine config with cond n00
            b = json.loads((Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs') / f"r38_B_f60_R30_s1001_N_shared_{c['objective']}_cosine.json").read_text())
            al = allowed | {'cond', 'noise_percent', 'dataset', 'parent_H', 'parent_key', 'schedule_rule', 'head_init'}
            diff = {k for k in set(c) | set(b) if c.get(k) != b.get(k)} - al
            o_ = dict(c['optimizer']); o_.pop('lr_encoder', None)
            ok = not diff and o_ == b['optimizer'] and c['cond'] == 'n00'
            if c['encoder'] == 'joint_global':
                t = json.loads((W / 'configs' / f"{c['extension']['r39_template']}.json").read_text())
                ok = ok and all(c[k] == t[k] for k in ('encoder', 'joint_rule', 'optimizer', 'steps', 'ckpt_steps', 'n_queries', 'clip_per_peer', 'init_seed', 'sampler_seed', 'supervision', 'structure', 'objective'))
            else:
                ok = ok and 'lr_encoder' not in c['optimizer'] and 'joint_rule' not in c
            if not ok:
                bad.append(dict(unit=u, diff=sorted(diff)))
            continue
        rc = Path(r['result_json']).parent
        cdir = Path('/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/configs') if r['unit_id'].startswith('r37s') else Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs')
        b = json.loads((cdir / f"{r['unit_id']}.json").read_text())
        t = json.loads((W / 'configs' / f"{c['extension']['r39_template']}.json").read_text())
        diff = {k for k in set(c) | set(b) if c.get(k) != b.get(k)} - allowed
        o = dict(c['optimizer']); o.pop('lr_encoder')
        joint_eq = all(c[k] == t[k] for k in ('encoder', 'joint_rule', 'optimizer', 'steps', 'ckpt_steps', 'n_queries', 'clip_per_peer', 'init_seed', 'sampler_seed', 'supervision', 'structure', 'objective'))
        hashes = sh(cdir / f"{r['unit_id']}.json") == r['config_sha256'] and sh(r['result_json']) == r['result_json_sha256'] and sh(r['train_history_json']) == r['train_history_json_sha256']
        if diff or o != b['optimizer'] or not joint_eq or not hashes or not (rc / 'DONE.json').exists():
            bad.append(dict(unit=u, diff=sorted(diff), joint_eq=joint_eq, hashes=hashes))
    return dict(n=len(units), bad=bad, passed=len(units) == 21 and not bad)


def t_bindings(tree):
    def run():
        for m in [m for m in sys.modules if m in ('common', 'data35', 'data36', 'evals', 'heads', 'native_eval', 'objectives', 'r38ext', 'r39ext', 'r39_targets')]:
            del sys.modules[m]
        sys.path[:] = [p for p in sys.path if not p.endswith(('/r39xf', '/r39xc'))]; sys.path.insert(0, str(W / tree))
        import torch, common as C, evals as EV, data36 as D36
        from data35 import load_cond
        out = {}
        units = [u for u in json.loads((W / 'configs' / 'UNITS_R39X.json').read_text()) if json.loads((W / 'configs' / f'{u}.json').read_text())['code_tree'] == tree]
        conds = sorted({json.loads((W / 'configs' / f'{u}.json').read_text())['cond'] for u in units})
        T = np.load(C.TOKENS_FILE)
        for cond in conds:
            jp = C.JOINT_PARENTS[cond]
            d = load_cond(cond, torch.device('cpu'))
            ck = torch.load(jp['ckpt'], map_location='cpu', weights_only=False)
            nat = EV.full_metrics(d['native_dev'])[0]['rsum']
            refs = {}
            for u in units:
                c = json.loads((W / 'configs' / f'{u}.json').read_text())
                if c['cond'] != cond or c['reference_unit'] is None:
                    continue
                res = json.loads(Path(c['reference_unit']['result_json']).read_text()); hist = json.loads(Path(c['reference_unit']['train_history_json']).read_text())
                refs[u] = dict(ref_native_H_rsum=res['native_H_dev']['rsum'], ref_rsum8000=res['final']['dev_retrieval']['rsum'], ref_chain_len=len(hist['batch_hash']),
                               native_matches=abs(res['native_H_dev']['rsum'] - nat) < 1e-9)
            out[cond] = dict(native_dev_rsum=nat, ckpt_sha_matches_cache=__import__('common').sha256(jp['ckpt']) == d['manifest']['parent']['model_best_sha256'],
                             vocab_size=int(ck['opt'].vocab_size), max_token=int(max(T['train_tok'].max(), T['dev_tok'].max())), batch_size=int(ck['opt'].batch_size),
                             code_vse_equal=C.sha256(Path(jp['code']) / 'vse.py') == C.sha256(C.NATIVE_CODE / 'vse.py'), refs=refs)
            o = out[cond]
            o['native_matches_cache_manifest'] = abs(nat - d['manifest']['native_dev']['rsum']) < 1e-9
            o['ok'] = bool(o['native_matches_cache_manifest'] and o['ckpt_sha_matches_cache'] and o['max_token'] < o['vocab_size'] and o['batch_size'] == 128 and o['code_vse_equal'] and all(r['native_matches'] for r in refs.values()))
            del d
        out['passed'] = all(v['ok'] for v in out.values() if isinstance(v, dict))
        return out
    return run


def main():
    check('code_derivation', t_code); check('configs', t_configs)
    check('bindings_r39xf', t_bindings('r39xf')); check('bindings_r39xc', t_bindings('r39xc'))
    out = dict(job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES,
               note='batch-chain identity vs the reference unit is asserted in every formal unit at each readout (train_unit reference check); not duplicated here',
               ALL_PASS=all(v['passed'] for v in RES.values()))
    (W / 'CPU_TEST_R39X.json').write_text(json.dumps(out, indent=1, default=str))
    print('R39X_CPU_TEST ALL_PASS', out['ALL_PASS'])


if __name__ == '__main__':
    main()
