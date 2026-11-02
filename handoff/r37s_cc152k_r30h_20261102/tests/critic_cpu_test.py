"""R37S critic CPU test (adapted from R38 tests/cpu_test.py) -> ROOT/CRITIC_CPU_TEST.json.
 (1) code identity: r37s_critic/*.py byte-identical to R38 r38c/*.py except common.py / build_cache.py / prep.py / train_unit.py, and the
     diff of those four is exactly R37S_CRITIC_DIFF_VS_R38C.patch (constants, cond choice, recipe assertion, freeze-file name);
 (2) math equivalence with the R38 package reference (DirGHead rows, J/BCE/NCE list objectives, lr_multiplier for 8000 updates);
 (3) directional head init (shared keys bitwise == G; +131 params/peer; zero extra output; affine_t2i == affine);
 (4) parent artifacts: R37S cache manifest (exact native DEV reproduction, identity map, parent = r37s_cc_s1001_H0, best epoch/rsum ==
     parent EPOCHS DEV-best), prep files hashed, locks byte-equal to R37C locks, candidate tables, calibration a0>0, trust subset;
 (5) REAL data, CPU, 1 thread (critic_cpu_worker.py): cosine first update == constant, same batch chain, lr formula; T cosine vs constant;
     directional cosine NCE 250 updates (graph checks); directional constant J 30 updates with DEV readouts (step-0 native-rank check,
     per-direction DEV scores); every one of the 18 configs: 2 updates, shared-key init and batch chain equal within each route;
 (6) isolation (trainer guard refuses test_* / oracle paths) + static scan (no TEST file names in r37s_critic).
Usage: python -I -B critic_cpu_test.py"""
import ast, difflib, importlib.util, json, os, re, subprocess, sys, tempfile, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TREE = ROOT / 'r37s_critic'; R38C = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/r38c')
PKG = Path('/home/infres/yinwang/CS_QMI/r38_plan/VCS_R37_Review_R38_Plan_20261102')
sys.path.insert(0, str(TREE))
import numpy as np
import torch
from common import install_guards, write_json, read_json, sha256, utc, CONDS, OUT, ORACLE_RUNS, R37C_OUT

checks = {}
def chk(name, fn):
    try:
        info = fn(); ok = info.pop('pass_', True) if isinstance(info, dict) else bool(info)
        checks[name] = dict(pass_=bool(ok), **(info if isinstance(info, dict) else {}))
    except Exception:
        checks[name] = dict(pass_=False, error=traceback.format_exc())
    print(('PASS ' if checks[name]['pass_'] else 'FAIL ') + name, flush=True)


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    install_guards(role='trainer'); torch.set_num_threads(4)
    ref = load_mod(PKG / 'reference' / 'directional_reference.py', 'r38pkg_reference')
    X = load_mod(TREE / 'r38ext.py', 'r38ext_s'); H = load_mod(TREE / 'heads.py', 'heads_s'); O = load_mod(TREE / 'objectives.py', 'objectives_s')

    def code_identity():
        edited = {'common.py', 'build_cache.py', 'prep.py', 'train_unit.py'}
        names = sorted(p.name for p in TREE.glob('*.py')); names38 = sorted(p.name for p in R38C.glob('*.py'))
        same = {f: (TREE / f).read_bytes() == (R38C / f).read_bytes() for f in names if f not in edited}
        p = subprocess.run(['diff', '-u', str(R38C), str(TREE)], capture_output=True, text=True, cwd=str(ROOT))
        norm = lambda t: [l for l in t.splitlines() if not l.startswith(('---', '+++'))]
        patch_eq = norm(p.stdout) == norm((ROOT / 'R37S_CRITIC_DIFF_VS_R38C.patch').read_text())
        tu = [l for l in difflib.unified_diff((R38C / 'train_unit.py').read_text().splitlines(), (TREE / 'train_unit.py').read_text().splitlines(), lineterm='', n=0) if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        return dict(pass_=names == names38 and all(same.values()) and patch_eq and len(tu) == 2, files_equal_r38c=same, patch_file_matches=patch_eq, train_unit_changed_lines=tu)
    chk('code_identity_vs_r38c', code_identity)

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
        err_rows = float((m.rows(s, hid, dd) - torch.where(dd.bool(), r_(x, y, 1), r_(x, y, 0))).abs().max())
        g = torch.randn(7, 33, dtype=torch.double); a = torch.nn.functional.one_hot(torch.arange(7), 33).double()
        obj = {o: float((O.loss_list(o, g, a) - ref.row_objective(g, a, o).mean()).abs()) for o in ('J', 'BCE', 'NCE')}
        lr_same = all(X.lr_multiplier(t, 8000, 'cosine', .1) == ref.lr_multiplier(t, 8000, 'cosine', .1) for t in range(8000))
        v = [X.lr_multiplier(t, 8000, 'cosine', .1) for t in range(8000)]
        formula = max(abs(1e-4 * v[t] - 1e-4 * (0.1 + 0.45 * (1 + np.cos(np.pi * t / 7999)))) for t in range(8000))
        return dict(pass_=err_rows < 1e-12 and max(obj.values()) < 1e-12 and lr_same and v[0] == 1.0 and abs(v[-1] - .1) < 1e-15 and formula < 1e-18,
                    dir_rows_maxerr=err_rows, objective_maxerr=obj, lr_identical_to_package=lr_same, formula_maxerr=formula)
    chk('math_equivalence_with_package_reference', math_eq)

    def head_init():
        torch.manual_seed(0); g = H.build_heads('G', 21.5, -4.3, 1001); rs_g = torch.get_rng_state().clone()
        torch.manual_seed(0); gd = H.build_heads('GD', 21.5, -4.3, 1001); rs_gd = torch.get_rng_state().clone()
        sg, sgd = g.state_dict(), gd.state_dict()
        shared_eq = set(sg) == {k for k in sgd if '.out_t2i.' not in '.' + k and '.affine_t2i.' not in '.' + k} and all(torch.equal(sg[k], sgd[k]) for k in sg)
        extra = {p: sum(q.numel() for q in gd[p].parameters()) - sum(q.numel() for q in g[p].parameters()) for p in ('A', 'B')}
        zero = all(float(gd[p].out_t2i.weight.abs().max()) == 0 and float(gd[p].out_t2i.bias.abs().max()) == 0 for p in ('A', 'B'))
        aff = all(torch.equal(gd[p].affine.raw_a, gd[p].affine_t2i.raw_a) and torch.equal(gd[p].affine.b, gd[p].affine_t2i.b) for p in ('A', 'B'))
        return dict(pass_=shared_eq and torch.equal(rs_g, rs_gd) and extra == {'A': 131, 'B': 131} and zero and aff, shared_keys_bitwise_equal=shared_eq, extra_params_per_peer=extra)
    chk('directional_head_init', head_init)

    def artifacts():
        c = CONDS['cc_r37s']; cdir = Path(c['cache_dir']); pdir = Path(c['prep_dir'])
        man = read_json(cdir / 'CACHE_MANIFEST.json'); pm = read_json(pdir / 'PREP_MANIFEST.json')
        ok = all(sha256(cdir / n) == r['sha256'] for n, r in man['files'].items()) and all(sha256(pdir / n) == r['sha256'] for n, r in pm['files'].items())
        nd = man['native_dev']; done = read_json(Path(c['parent_dir']) / 'DONE.json'); pb = read_json(ROOT / 'PARENT_BINDING.json')
        ok &= nd['reproduces_checkpoint_best_rsum_exactly'] and nd['cache_rescore_identical'] and nd['per_captions'] == 1 and nd['rsum'] == done['best_rsum'] == pb['best_dev_rsum']
        ok &= nd['checkpoint_epoch'] == pb['best_cotrain_epoch'] and nd['six_recalls_equal_logged_epoch'] and man['parent']['model_best_sha256'] == pb['model_best_sha256']
        ok &= man['parent']['unit'] == c['h_unit'] and man['observed_map']['is_identity'] and man['parent_seed'] == c['parent_seed'] and man['env']['gpu'] == pb['best_epoch_gpu']
        obs = np.load(cdir / 'observed_map.npy'); ok &= bool(np.array_equal(obs, np.arange(150000)))
        L = np.load(pdir / 'locks_records.npz'); L0 = np.load(R37C_OUT / 'prep' / 'locks_records.npz')
        locks_eq = sorted(L.files) == sorted(L0.files) and all(np.array_equal(L[k], L0[k]) for k in L.files)
        cand = np.load(pdir / f"candidates_{c['prep_key']}.npz"); ci, ct = cand['i2t_top32'], cand['t2i_top32']
        ok &= ci.shape == (150000, 32) and ct.shape == (150000, 32) and ci.min() >= 0 and ci.max() < 150000 and ct.min() >= 0 and ct.max() < 150000
        ok &= bool((np.diff(np.sort(ci[::50], 1), axis=1) > 0).all()) and bool((np.diff(np.sort(ct[::50], 1), axis=1) > 0).all())
        T = np.load(pdir / c['trust_file']); sub = T['subset']
        ok &= bool(np.array_equal(sub, np.nonzero((T['prob_A'] + T['prob_B']) / 2 >= .5)[0]))
        cal = pm['calibration_init'][c['prep_key']]; ok &= cal['a0'] > 0 and bool(cal['dev_rank_invariance'])
        return dict(pass_=bool(ok and locks_eq), locks_byte_equal_R37C=locks_eq, native_dev={k: nd[k] for k in ('rsum', 'checkpoint_best_rsum', 'checkpoint_epoch', 'peer_rsum')},
                    cache_gpu=man['env']['gpu'], a0b0=[cal['a0'], cal['b0']], candidates=pm['candidates'][c['prep_key']], trust=dict(n_subset=int(len(sub)), retention=float(len(sub) / 150000)),
                    cache_manifest_sha256=sha256(cdir / 'CACHE_MANIFEST.json'), prep_manifest_sha256=sha256(pdir / 'PREP_MANIFEST.json'))
    chk('parent_artifacts_cc_r37s', artifacts)

    def real():
        tmp = Path(tempfile.mkdtemp(prefix='r37scpu_'))
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        p = subprocess.run([PY, '-I', '-B', str(ROOT / 'tests' / 'critic_cpu_worker.py'), '--tree', str(TREE), '--tmp', str(tmp)], capture_output=True, text=True, env=env, cwd=str(tmp), timeout=7200)
        line = [l for l in p.stdout.splitlines() if l.startswith('R37S_WORKER_JSON ')]
        if p.returncode != 0 or not line:
            return dict(pass_=False, error=p.stderr[-4000:])
        r = json.loads(line[0][len('R37S_WORKER_JSON '):]); o = {}
        o['cos_first_update_equals_const'] = r['N_shared_cos_J_1']['final'] == r['N_shared_const_J_1']['final'] and r['N_shared_cos_J_1']['chain'] == r['N_shared_const_J_1']['chain']
        o['cos_60_same_chain_diff_params'] = r['N_shared_cos_J_60']['chain'] == r['N_shared_const_J_60']['chain'] and r['N_shared_cos_J_60']['final'] != r['N_shared_const_J_60']['final']
        lr60 = r['N_shared_cos_J_60']['r38']['lr_at_readout']['60']; o['cos_lr_formula_ok'] = abs(lr60 - 1e-4 * (0.1 + 0.45 * (1 + np.cos(np.pi * 59 / 7999)))) < 1e-18
        o['T_cos_same_chain_diff_params'] = r['T_shared_cos_BCE_60']['chain'] == r['T_shared_const_BCE_60']['chain'] and r['T_shared_cos_BCE_60']['final'] != r['T_shared_const_BCE_60']['final']
        d = r['N_dir_cos_NCE_250']['r38']; o['dir_graph_ok'] = bool(d['step1_graph_check']['passed']); o['dir_reference_none'] = d['reference_unit'] is None
        o['dir_shared_keys_init_equals_shared_init'] = d['init_state_sha256_shared_keys'] == r['N_shared_const_J_60']['init']
        o['dir_extra_params_ok'] = r['N_dir_cos_NCE_250']['params'] - r['N_shared_const_J_60']['params'] == 262
        ro = r['N_dir_const_J_30_readout']['readout']
        o['dir_readout_ok'] = bool(ro['step0_native_check']['passed']) and ro['i2t_top_maxabs_vs_dir0'] < 1e-9 and ro['t2i_top_maxabs_vs_dir1'] < 1e-9 and ro['i2t_top_maxabs_vs_dir1'] > 1e-8 and ro['t2i_top_maxabs_vs_dir0'] > 1e-8 and ro['g_gt_vs_g_gt_t2i_maxabs_step0'] == 0.0 and ro['g_gt_vs_g_gt_t2i_maxabs_final'] > 1e-8
        a18 = {k[len('all18__'):]: v for k, v in r.items() if k.startswith('all18__')}
        o['all18_ran'] = len(a18) == 18 and all(v['r38']['step1_graph_check'] is None or v['r38']['step1_graph_check']['passed'] for v in a18.values())
        for route in ('N', 'T'):
            us = {u: v for u, v in a18.items() if f'_{route}_' in u}
            o[f'all18_{route}_shared_init_equal'] = len({v['r38']['init_state_sha256_shared_keys'] for v in us.values()}) == 1
            o[f'all18_{route}_chain_equal'] = len({json.dumps(v['chain']) for v in us.values()}) == 1
        info = dict(readout=ro, graph_check=d['step1_graph_check'], seconds={k: v['seconds'] for k, v in r.items()}, lr60=lr60)
        return dict(pass_=all(o.values()), **o, **info)
    chk('real_data_in_unit_checks', real)

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
        hits = [f'{f.name}:{i+1}' for f in sorted(TREE.glob('*.py')) for i, line in enumerate(f.read_text().splitlines())
                if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line]
        return dict(pass_=res == ['REFUSED', 'REFUSED', 'REFUSED'] and not hits, opens=res, hits=hits)
    chk('isolation_and_static_scan', isolation)


if __name__ == '__main__':
    t0 = time.time(); main()
    allp = all(v['pass_'] for v in checks.values())
    write_json(ROOT / 'CRITIC_CPU_TEST.json', dict(ALL_PASS=allp, checks=checks, created_utc=utc(), job=os.environ.get('SLURM_JOB_ID', 'local'), seconds=time.time() - t0))
    print('R37S_CRITIC_CPU', 'ALL_PASS' if allp else 'FAIL'); sys.exit(0 if allp else 1)
