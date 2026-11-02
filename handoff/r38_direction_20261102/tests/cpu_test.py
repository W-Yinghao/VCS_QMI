"""R38 CPU test -> ROOT/CPU_TEST.json (part main) or ROOT/PARENT_ARTIFACT_CHECK.json (part artifacts).
main:
 (1) package CPU reference tests (tests/test_reference.py of the R38 package, run as a separate python -I -B process);
 (2) math equivalence with the package reference: DirGHead.rows == DirectionalResidual (weights copied, float64), list objectives ==
     row_objective for J/BCE/NCE, X.lr_multiplier == package lr_multiplier for all 8000 updates (and identical source);
 (3) directional head init: build_heads('GD') shared keys bitwise == build_heads('G') (R38) == R37 build_heads('G'), same CPU RNG state
     after the build, +131 params per peer, zero extra output, affine_t2i == affine;
 (4) code identity: data35/data36/evals/native_eval/objectives byte-identical to R37 (r38f) / R37C (r38c); heads = R37 heads + DirGHead
     only; r38ext identical in both trees; r38f vs r38c train_unit differ exactly as R37 vs R37C (PER_CAP edits);
 (5) REAL data, CPU, 1 thread: R38 shared/constant path == R37 (Flickr n60) / R37C (CC s1001) path bitwise (init, 60-update params,
     batch chain; N and T); cosine: first update identical to constant, later lr = formula, same batch chain; directional cosine NCE
     250 updates with the in-unit reference assertions (R36/R37C init hash + batch chain at 0/250) and the first-update graph check;
     directional constant J 30 updates with DEV readouts: step-0 native-rank check, per-direction DEV scores;
 (6) isolation (trainer guard refuses test_* / oracle paths) + static scan (no TEST file names in r38 code).
artifacts: new CC152K parents cc_s1002/cc_s1003: cache manifest (exact native DEV reproduction, identity map), prep files hashed,
     locks byte-equal to R37C locks, candidate tables, calibration a0>0, trust subset well-formed.
Usage: python -I -B cpu_test.py main|artifacts"""
import ast, difflib, importlib.util, io, json, os, re, subprocess, sys, tempfile, time, tokenize, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
R37 = Path('/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102/r37'); R37C = Path('/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102/r37c')
R36CFG = Path('/home/infres/yinwang/CS_QMI/handoff/r36_supervision_20261102/configs'); R37CCFG = Path('/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102/configs')
PKG = Path('/home/infres/yinwang/CS_QMI/r38_plan/VCS_R37_Review_R38_Plan_20261102')
sys.path.insert(0, str(ROOT / 'r38c'))
import numpy as np
import torch
from common import install_guards, write_json, read_json, sha256, utc, CONDS, OUT, ORACLE_RUNS


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


checks = {}
def chk(name, fn):
    try:
        info = fn(); ok = info.pop('pass_', True) if isinstance(info, dict) else bool(info)
        checks[name] = dict(pass_=bool(ok), **(info if isinstance(info, dict) else {}))
    except Exception:
        checks[name] = dict(pass_=False, error=traceback.format_exc())
    print(('PASS ' if checks[name]['pass_'] else 'FAIL ') + name, flush=True)


def part_main():
    install_guards(role='trainer'); torch.set_num_threads(4)
    ref = load_mod(PKG / 'reference' / 'directional_reference.py', 'r38pkg_reference')
    X = load_mod(ROOT / 'r38c' / 'r38ext.py', 'r38ext_c')
    H = load_mod(ROOT / 'r38c' / 'heads.py', 'heads_r38c'); H37 = load_mod(R37 / 'heads.py', 'heads_r37')
    O = load_mod(ROOT / 'r38c' / 'objectives.py', 'objectives_r38c')

    def pkg_tests():
        with tempfile.TemporaryDirectory() as td:
            p = subprocess.run([PY, '-I', '-B', str(PKG / 'tests' / 'test_reference.py')], cwd=td, capture_output=True, text=True, timeout=600)
        return dict(pass_=p.returncode == 0, returncode=p.returncode, tail=p.stderr.strip().splitlines()[-3:])
    chk('package_reference_tests', pkg_tests)

    def math_eq():
        torch.manual_seed(3)
        m = H.DirGHead(2.0, -.3).double(); r_ = ref.DirectionalResidual(1024, 256, 128, 2.0, -.3).double()
        with torch.no_grad():
            for q in (m.out, m.out_t2i):
                q.weight.normal_(); q.bias.normal_()
            m.affine_t2i.raw_a.fill_(.7); m.affine_t2i.b.fill_(1.1)
            r_.trunk[0].load_state_dict(m.glob.l1.state_dict()); r_.trunk[2].load_state_dict(m.glob.l2.state_dict())
            r_.outputs[0].load_state_dict(m.out.state_dict()); r_.outputs[1].load_state_dict(m.out_t2i.state_dict())
            r_.raw_scale.copy_(torch.stack([m.affine.raw_a, m.affine_t2i.raw_a])); r_.offset.copy_(torch.stack([m.affine.b, m.affine_t2i.b]))
        x = torch.nn.functional.normalize(torch.randn(50, 1024, dtype=torch.double), dim=-1); y = torch.nn.functional.normalize(torch.randn(50, 1024, dtype=torch.double), dim=-1)
        s = (x * y).sum(-1); hid = m.glob.hidden(x, y, s); dd = torch.tensor([0, 1] * 25)
        mine = m.rows(s, hid, dd); theirs = torch.where(dd.bool(), r_(x, y, 1), r_(x, y, 0))
        err_rows = float((mine - theirs).abs().max())
        mine_ex = m.rows(s.float(), hid, dd, s_exact=s)   # evaluation branch (float64 s_exact)
        err_ex = float((mine_ex - theirs).abs().max())
        g = torch.randn(7, 33, dtype=torch.double); a = torch.nn.functional.one_hot(torch.arange(7), 33).double()
        obj = {o: float((O.loss_list(o, g, a) - ref.row_objective(g, a, o).mean()).abs()) for o in ('J', 'BCE', 'NCE')}
        lr_same = all(X.lr_multiplier(t, 8000, 'cosine', .1) == ref.lr_multiplier(t, 8000, 'cosine', .1) for t in range(8000))
        src = lambda p: [ast.dump(n) for n in ast.parse(Path(p).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'lr_multiplier']
        lr_src = src(ROOT / 'r38c' / 'r38ext.py') == src(PKG / 'reference' / 'directional_reference.py')
        v = [X.lr_multiplier(t, 8000, 'cosine', .1) for t in range(8000)]
        formula = max(abs(1e-4 * v[t] - 1e-4 * (0.1 + 0.45 * (1 + np.cos(np.pi * t / 7999)))) for t in range(8000))
        ok = err_rows < 1e-12 and err_ex < 1e-12 and max(obj.values()) < 1e-12 and lr_same and lr_src and v[0] == 1.0 and abs(v[-1] - .1) < 1e-15 and formula < 1e-18
        return dict(pass_=ok, dir_rows_maxerr=err_rows, dir_rows_exact_branch_maxerr=err_ex, objective_maxerr=obj, lr_identical_to_package=lr_same,
                    lr_source_identical=lr_src, lr0=v[0], lr_last=v[-1], formula_maxerr=formula)
    chk('math_equivalence_with_package_reference', math_eq)

    def head_init():
        out = {}
        torch.manual_seed(0); g = H.build_heads('G', 21.5, -4.3, 1001); rs_g = torch.get_rng_state().clone()
        torch.manual_seed(0); gd = H.build_heads('GD', 21.5, -4.3, 1001); rs_gd = torch.get_rng_state().clone()
        torch.manual_seed(0); g37 = H37.build_heads('G', 21.5, -4.3, 1001)
        sg, sgd, s37 = g.state_dict(), gd.state_dict(), g37.state_dict()
        shared_eq = set(sg) == {k for k in sgd if '.out_t2i.' not in '.' + k and '.affine_t2i.' not in '.' + k} and all(torch.equal(sg[k], sgd[k]) for k in sg)
        r37_eq = set(sg) == set(s37) and all(torch.equal(sg[k], s37[k]) for k in sg)
        extra = {p: sum(q.numel() for n, q in gd[p].named_parameters()) - sum(q.numel() for q in g[p].parameters()) for p in ('A', 'B')}
        zero = all(float(gd[p].out_t2i.weight.abs().max()) == 0 and float(gd[p].out_t2i.bias.abs().max()) == 0 for p in ('A', 'B'))
        aff = all(torch.equal(gd[p].affine.raw_a, gd[p].affine_t2i.raw_a) and torch.equal(gd[p].affine.b, gd[p].affine_t2i.b) for p in ('A', 'B'))
        ok = shared_eq and r37_eq and torch.equal(rs_g, rs_gd) and extra == {'A': 131, 'B': 131} and zero and aff
        return dict(pass_=ok, shared_keys_bitwise_equal=shared_eq, r38_G_equals_r37_G=r37_eq, rng_state_equal=bool(torch.equal(rs_g, rs_gd)),
                    extra_params_per_peer=extra, extra_output_zero=zero, affine_t2i_equals_affine=aff)
    chk('directional_head_init', head_init)

    def code_identity():
        res = {}
        for tree, src in (('r38f', R37), ('r38c', R37C)):
            for f in ('data35.py', 'data36.py', 'evals.py', 'native_eval.py', 'objectives.py'):
                res[f'{tree}/{f}'] = (ROOT / tree / f).read_bytes() == (src / f).read_bytes()
            h = (ROOT / tree / 'heads.py').read_text(); h0 = (src / 'heads.py').read_text()
            cut = h[:h.index('\n\nclass DirGHead(GHead):')]
            cut = cut.replace(", 'GD': lambda: DirGHead(a0, b0)}[structure]", "}[structure]")
            cut = cut[cut.index('R36 (copied from R35 heads.py;'):]; res[f'{tree}/heads.py=R37+DirGHead'] = '"""' + cut == h0
        res['r38ext_identical'] = (ROOT / 'r38f' / 'r38ext.py').read_bytes() == (ROOT / 'r38c' / 'r38ext.py').read_bytes()
        def ch(a_, b_):   # module docstrings and the code-hash directory line (R38: the tree's own directory) excluded
            body = lambda p: p.read_text().split('"""', 2)[2].splitlines()
            d = [l for l in difflib.unified_diff(body(a_), body(b_), lineterm='', n=0) if l[:1] in '+-' and not l.startswith(('+++', '---'))]
            return [l for l in d if 'code differs from frozen protocol' not in l and "sha256(f) for f in sorted(glob.glob(" not in l]
        res['train_unit_tree_diff_equals_R37_vs_R37C'] = ch(ROOT / 'r38f' / 'train_unit.py', ROOT / 'r38c' / 'train_unit.py') == ch(R37 / 'train_unit.py', R37C / 'train_unit.py')
        (ROOT / 'TRAIN_UNIT_R38F.diff').write_text(''.join(difflib.unified_diff((R37 / 'train_unit.py').read_text().splitlines(True), (ROOT / 'r38f' / 'train_unit.py').read_text().splitlines(True), 'r37/train_unit.py', 'r38f/train_unit.py')))
        (ROOT / 'TRAIN_UNIT_R38C.diff').write_text(''.join(difflib.unified_diff((R37C / 'train_unit.py').read_text().splitlines(True), (ROOT / 'r38c' / 'train_unit.py').read_text().splitlines(True), 'r37c/train_unit.py', 'r38c/train_unit.py')))
        return dict(pass_=all(res.values()), **res)
    chk('code_identity', code_identity)

    def real():
        tmp = Path(tempfile.mkdtemp(prefix='r38cpu_'))
        jobs = {'f_base': [str(R37), 'base', 'n60', str(R36CFG), 'r36_{r}_n60_', ''], 'f_r38': [str(ROOT / 'r38f'), 'r38', 'n60', str(R36CFG), 'r36_{r}_n60_', 'r38_{b}_f60_R30_s1001_'],
                'c_base': [str(R37C), 'base', 'cc', str(R37CCFG), 'r37c_pilot_cc_h1001_{r}_', ''], 'c_r38': [str(ROOT / 'r38c'), 'r38', 'cc', str(R37CCFG), 'r37c_pilot_cc_h1001_{r}_', 'r38_{b}_cc_R31_s1001_']}
        procs = {}
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        for k, (tree, mode, cond, rc, rp, p38) in jobs.items():
            (tmp / k).mkdir()
            cmd = [PY, '-I', '-B', str(ROOT / 'tests' / 'cpu_worker.py'), '--tree', tree, '--mode', mode, '--cond', cond, '--refcfg', rc, '--refprefix', rp, '--tmp', str(tmp / k)]
            if p38:
                cmd += ['--r38cfg', str(ROOT / 'configs'), '--r38prefix', p38]
            procs[k] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(tmp))
        R = {}; errs = {}
        for k, p in procs.items():
            so, se = p.communicate(timeout=7200)
            line = [l for l in so.splitlines() if l.startswith('R38_WORKER_JSON ')]
            if p.returncode != 0 or not line:
                errs[k] = se[-4000:]
            else:
                R[k] = json.loads(line[0][len('R38_WORKER_JSON '):])
        if errs:
            return dict(pass_=False, worker_errors=errs)
        out = {}; ok = True
        for ds in ('f', 'c'):
            b, r = R[f'{ds}_base'], R[f'{ds}_r38']; o = {}
            for n in ('N_shared_const_J_60', 'T_shared_const_BCE_60', 'N_shared_const_J_1'):
                o[f'{n}_bitwise_equal_to_R37'] = all(b[n][k] == r[n][k] for k in ('init', 'final', 'chain'))
            o['cos_first_update_equals_const'] = r['N_shared_cos_J_1']['final'] == r['N_shared_const_J_1']['final'] and r['N_shared_cos_J_1']['chain'] == r['N_shared_const_J_1']['chain']
            o['cos_60_same_chain_diff_params'] = r['N_shared_cos_J_60']['chain'] == r['N_shared_const_J_60']['chain'] and r['N_shared_cos_J_60']['final'] != r['N_shared_const_J_60']['final']
            lr60 = r['N_shared_cos_J_60']['r38']['lr_at_readout']['60']
            o['cos_lr_at_update59'] = lr60; o['cos_lr_formula_ok'] = abs(lr60 - 1e-4 * (0.1 + 0.45 * (1 + np.cos(np.pi * 59 / 7999)))) < 1e-18
            o['T_cos_same_chain_diff_params'] = r['T_shared_cos_BCE_60']['chain'] == r['T_shared_const_BCE_60']['chain'] and r['T_shared_cos_BCE_60']['final'] != r['T_shared_const_BCE_60']['final']
            d = r['N_dir_cos_NCE_250_ref']; o['dir_ref'] = {k: d['r38'][k] for k in ('reference_unit', 'reference_init_equal', 'reference_chain_equal_at')}
            o['dir_ref_ok'] = d['r38']['reference_init_equal'] is True and d['r38']['reference_chain_equal_at'] == [0, 250]
            o['dir_graph_check'] = d['r38']['step1_graph_check']; o['dir_graph_ok'] = bool(d['r38']['step1_graph_check']['passed'])
            o['dir_extra_params'] = d['params'] - r['N_shared_const_J_60']['params']
            o['dir_shared_keys_init_equals_shared_init'] = d['r38']['init_state_sha256_shared_keys'] == r['N_shared_const_J_60']['init']
            o['dir_lr_last_readout'] = d['r38']['lr_at_readout']['250']
            ro = r['N_dir_const_J_30_readout']['readout']; o['dir_readout'] = ro
            o['dir_readout_ok'] = bool(ro['step0_native_check']['passed']) and ro['i2t_top_maxabs_vs_dir0'] < 1e-9 and ro['t2i_top_maxabs_vs_dir1'] < 1e-9 and ro['i2t_top_maxabs_vs_dir1'] > 1e-8 and ro['t2i_top_maxabs_vs_dir0'] > 1e-8 and ro['g_gt_vs_g_gt_t2i_maxabs_step0'] == 0.0 and ro['g_gt_vs_g_gt_t2i_maxabs_final'] > 1e-8 and 'g_gt_t2i' in ro['keys']
            o['seconds'] = {k: v['seconds'] for k, v in r.items()}
            bools = [v for k, v in o.items() if isinstance(v, bool)]
            ok &= all(bools) and o['dir_extra_params'] == 262
            out['flickr_n60' if ds == 'f' else 'cc_s1001'] = o
        return dict(pass_=ok, **out)
    chk('real_data_equivalence_and_in_unit_checks', real)

    def isolation():
        res = []
        tdir = Path(tempfile.mkdtemp()) / 'x_precomp'
        for p in (ORACLE_RUNS / 'x', tdir / 'test_caps.tsv', tdir / 'testall_ims.npy'):
            try:
                open(p); res.append('OPENED')
            except PermissionError:
                res.append('REFUSED')
            except FileNotFoundError:
                res.append('MISSING')
        hits = []
        for tree in ('r38f', 'r38c'):
            for f in sorted((ROOT / tree).glob('*.py')):
                for i, line in enumerate(f.read_text().splitlines()):
                    if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line:
                        hits.append(f'{tree}/{f.name}:{i+1}')
        return dict(pass_=res == ['REFUSED', 'REFUSED', 'REFUSED'] and not hits, opens=res, hits=hits)
    chk('isolation_and_static_scan', isolation)


def part_artifacts():
    install_guards(role='trainer')
    r37c_locks = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/prep/locks_records.npz')
    for cond in ('cc_s1002', 'cc_s1003'):
        def f(cond=cond):
            c = CONDS[cond]; cdir = Path(c['cache_dir']); pdir = Path(c['prep_dir'])
            man = read_json(cdir / 'CACHE_MANIFEST.json'); pm = read_json(pdir / 'PREP_MANIFEST.json')
            ok = all(sha256(cdir / n) == r['sha256'] for n, r in man['files'].items()) and all(sha256(pdir / n) == r['sha256'] for n, r in pm['files'].items())
            nd = man['native_dev']; ck = read_json(Path(c['parent_dir']) / 'DONE.json')
            ok &= nd['reproduces_checkpoint_best_rsum_exactly'] and nd['cache_rescore_identical'] and nd['per_captions'] == 1 and nd['rsum'] == ck['best_rsum']
            ok &= man['parent']['unit'] == c['h_unit'] and man['observed_map']['is_identity'] and man['parent_seed'] == c['parent_seed']
            obs = np.load(cdir / 'observed_map.npy'); ok &= bool(np.array_equal(obs, np.arange(150000)))
            L = np.load(pdir / 'locks_records.npz'); L0 = np.load(r37c_locks)
            locks_eq = sorted(L.files) == sorted(L0.files) and all(np.array_equal(L[k], L0[k]) for k in L.files)
            cand = np.load(pdir / f"candidates_{c['prep_key']}.npz"); ci, ct = cand['i2t_top32'], cand['t2i_top32']
            ok &= ci.shape == (150000, 32) and ct.shape == (150000, 32) and ci.min() >= 0 and ci.max() < 150000 and ct.min() >= 0 and ct.max() < 150000
            ok &= bool((np.diff(np.sort(ci[::50], 1), axis=1) > 0).all()) and bool((np.diff(np.sort(ct[::50], 1), axis=1) > 0).all())
            T = np.load(pdir / c['trust_file']); sub = T['subset']
            ok &= bool(np.array_equal(sub, np.nonzero((T['prob_A'] + T['prob_B']) / 2 >= .5)[0]))
            cal = pm['calibration_init'][c['prep_key']]; ok &= cal['a0'] > 0 and cal['dev_rank_invariance']
            return dict(pass_=bool(ok and locks_eq), locks_byte_equal_R37C=locks_eq, native_dev=dict((k, nd[k]) for k in ('rsum', 'checkpoint_best_rsum', 'checkpoint_epoch', 'peer_rsum')),
                        cache_gpu=man['env']['gpu'], a0b0=[cal['a0'], cal['b0']], candidates=pm['candidates'][c['prep_key']], trust=dict(n_subset=int(len(sub)), retention=float(len(sub) / 150000)),
                        cache_manifest_sha256=sha256(cdir / 'CACHE_MANIFEST.json'), prep_manifest_sha256=sha256(pdir / 'PREP_MANIFEST.json'))
        chk(f'parent_artifacts_{cond}', f)


def part_amend():
    """user amendment (54 units on parents f60 s1002/s1003 and CC s1002/s1003): real-data CPU runs through the frozen R38 code of each
    tree: N directional cosine NCE 250 updates (Flickr: in-unit R37 reference init/batch-chain assertions; CC: no reference) and T shared
    cosine BCE 60 updates; first/second-update graph checks; new configs: frozen 30 configs byte-identical."""
    install_guards(role='trainer')
    def run_all():
        tmp = Path(tempfile.mkdtemp(prefix='r38amend_')); env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        jobs = {f'f60_s{s}': ('r38f', f'n60s{s}', str(ROOT / 'configs'), 'ref', f'r38_{{b}}_f60_R30_s{s}_') for s in (1002, 1003)}
        jobs.update({f'cc_s{s}': ('r38c', f'cc_s{s}', str(ROOT / 'configs'), 'none', f'r38_{{b}}_cc_R31_s{s}_') for s in (1002, 1003)})
        procs = {}
        for k, (tree, cond, rc, rp, p38) in jobs.items():
            (tmp / k).mkdir()
            procs[k] = subprocess.Popen([PY, '-I', '-B', str(ROOT / 'tests' / 'cpu_worker.py'), '--tree', str(ROOT / tree), '--mode', 'amend', '--cond', cond, '--refcfg', rc,
                                         '--refprefix', rp, '--tmp', str(tmp / k), '--r38cfg', str(ROOT / 'configs'), '--r38prefix', p38],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(tmp))
        out, ok = {}, True
        for k, p in procs.items():
            so, se = p.communicate(timeout=7200)
            line = [l for l in so.splitlines() if l.startswith('R38_WORKER_JSON ')]
            if p.returncode != 0 or not line:
                out[k] = dict(error=se[-3000:]); ok = False; continue
            r = json.loads(line[0][len('R38_WORKER_JSON '):]); d, t_ = r['N_dir_cos_NCE_250']['r38'], r['T_shared_cos_BCE_60']['r38']
            o = dict(dir_graph_passed=d['step1_graph_check']['passed'], T_graph_passed=t_['step1_graph_check']['passed'], dir_reference=d['reference_unit'],
                     dir_reference_init_equal=d['reference_init_equal'], dir_reference_chain_equal_at=d['reference_chain_equal_at'],
                     T_reference=t_['reference_unit'], T_reference_init_equal=t_['reference_init_equal'], T_reference_chain_equal_at=t_['reference_chain_equal_at'],
                     dir_shared_init=d['init_state_sha256_shared_keys'][:16], T_init=r['T_shared_cos_BCE_60']['init'][:16], dir_lr250=d['lr_at_readout']['250'])
            ok &= bool(o['dir_graph_passed'] and o['T_graph_passed'])
            if k.startswith('f60'):
                ok &= o['dir_reference_init_equal'] is True and o['dir_reference_chain_equal_at'] == [0, 250]
            out[k] = o
        fz = read_json(ROOT / 'PROTOCOL_FROZEN_30.json')
        same30 = all(sha256(ROOT / 'configs' / f'{u}.json') == h for u, h in fz['config_sha256'].items())
        return dict(pass_=bool(ok and same30), frozen30_configs_unchanged=same30, **out)
    chk('amendment_new_parents_real_data', run_all)


if __name__ == '__main__':
    part = sys.argv[1]; t0 = time.time()
    {'main': part_main, 'artifacts': part_artifacts, 'amend': part_amend}[part]()
    allp = all(v['pass_'] for v in checks.values())
    write_json(ROOT / {'main': 'CPU_TEST.json', 'artifacts': 'PARENT_ARTIFACT_CHECK.json', 'amend': 'AMENDMENT_CPU_CHECK.json'}[part], dict(ALL_PASS=allp, checks=checks, created_utc=utc(), job=os.environ.get('SLURM_JOB_ID', 'local'), seconds=time.time() - t0))
    print('R38_CPU', part, 'ALL_PASS' if allp else 'FAIL'); sys.exit(0 if allp else 1)
