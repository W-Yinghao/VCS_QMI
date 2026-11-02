"""R37C CPU test -> ROOT/CPU_TEST.json (CC152K-specific list construction and mapping).
(1) heads/objectives/native_eval byte-identical to R37; data36 functions source-identical to R37 (only doc/rule text differ);
    train_unit/evals differ from R37 only by the PER_CAP/identity edits (diff-line whitelist);
(2) toy identity relation (150000 records): forced (T) lists have exactly one positive = the record's own caption/image,
    N null lists label a real slot iff it is the own partner and A_null iff absent; presence_probs == brute force;
    TRUST anchors only from the pool; no //5 grouping anywhere;
(3) per_captions-1 evaluators: full_metrics == native evaluation.py (R31 code, verbatim upstream) i2t/t2i with per 1;
    two-stage composite with a positive affine of the native score reproduces the native ranks (step-0 property);
    dev_list_positions / dev_tables use the identity;
(4) REAL artifacts (cache + prep, read-only): observed map == arange(150000), native DEV reproduced exactly, cache
    re-score identical, locks in range, candidate tables shape/range/row-unique, trust subset in range/sorted,
    calibration a0 > 0, real-table T/N lists on 64 rows have the identity label structure, presence probs reported;
(5) isolation: trainer role refuses OUT/oracle_runs and a *_precomp/test_* path; static scan (no TEST paths, no //5,
    no f30k, no oracle/audit imports).
Usage: python -I -B cpu_test.py"""
import ast, difflib, importlib.util, inspect, os, re, sys, tempfile, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RC = ROOT / 'r37c'
sys.path.insert(0, str(RC))
import numpy as np
import torch
from common import (install_guards, write_json, read_json, sha256, utc, ORACLE_RUNS, OUT, R37_ROOT, CONDS, CODE_NATIVE,
                    N_TRAIN_CAP, N_TRAIN_IMG, N_DEV_IMG, N_DEV_CAP, PER_CAP)
import objectives as O
import heads as H
import data36 as D36
import data35 as D35
import evals as EV
import native_eval as NE


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    install_guards(role='trainer'); torch.set_num_threads(4)
    checks = {}; t0 = time.time()

    def chk(name, fn):
        try:
            info = fn(); ok = info.pop('pass_', True) if isinstance(info, dict) else bool(info)
            checks[name] = dict(pass_=bool(ok), **(info if isinstance(info, dict) else {}))
        except Exception:
            checks[name] = dict(pass_=False, error=traceback.format_exc())
        print(('PASS ' if checks[name]['pass_'] else 'FAIL ') + name, flush=True)

    # ------------------------------------------------------------------ (1) code identity vs R37
    def same_as_r37():
        byte = {f: (RC / f).read_bytes() == (R37_ROOT / 'r37' / f).read_bytes() for f in ('heads.py', 'objectives.py', 'native_eval.py')}
        def funcs(p):
            t = ast.parse(p.read_text())
            def body(n):
                bb = n.body[1:] if (n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], 'value', None), ast.Constant)) else n.body
                if n.name == 'presence_probs':   # only the human-readable rule= string was edited (150000 CC152K records): blank it
                    for kw in ast.walk(n):
                        if isinstance(kw, ast.keyword) and kw.arg == 'rule':
                            kw.value = ast.Constant('')
                return '\n'.join(ast.dump(x) for x in bb) + ast.dump(n.args)
            return {n.name: body(n) for n in t.body if isinstance(n, ast.FunctionDef)}
        a, b = funcs(RC / 'data36.py'), funcs(R37_ROOT / 'r37' / 'data36.py')
        diff36 = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        # train_unit / evals: every changed line must be a whitelisted PER_CAP / naming edit
        allowed = re.compile(r'PER_CAP|r37c|R37C|R37 unit trainer|R37_UNIT_DONE|R37_JOB_END|assert not need_local|was ROOT/|from common import|from native_eval import|R35 evaluators|R37C port|DEV_GT|// 5|\[::5\]|range\(5\)|5 \* i|reshape\(N_DEV_IMG, 5\)|i2t\(N_DEV_IMG, \w+, 5|t2i\(N_DEV_IMG, \w+, 5|tie_bounded\(|N_DEV_IMG, N_DEV_CAP = 1014|k//5|np.arange\(5 \* q|R37 \(copy|R37 unit|^\s*$|"""')
        bad = {}
        for f in ('train_unit.py', 'evals.py'):
            dl = [l for l in difflib.unified_diff((R37_ROOT / 'r37' / f).read_text().splitlines(), (RC / f).read_text().splitlines(), lineterm='', n=0)
                  if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
            bad[f] = [l for l in dl if not allowed.search(l[1:]) and l[1:].strip() != '' and "'r37'" not in l]
        return dict(pass_=all(byte.values()) and diff36 == [] and not any(bad.values()), byte_identical=byte, data36_function_diffs=diff36,
                    unwhitelisted_diff_lines=bad)
    chk('code_identity_vs_R37', same_as_r37)

    # ------------------------------------------------------------------ (2) toy identity relation lists
    rng0 = np.random.default_rng(5)
    NR = N_TRAIN_CAP
    obs = np.arange(NR, dtype=np.int64)
    ci = np.stack([rng0.choice(NR, 32, replace=False) for _ in range(2000)] * (NR // 2000)).astype(np.int32)
    ct = np.stack([rng0.choice(NR, 32, replace=False) for _ in range(2000)] * (NR // 2000)).astype(np.int32)
    for r in range(0, NR, 3):     # plant own partners in some rows (rank position varies; only if absent -> rows stay unique)
        if not (ci[r] == r).any():
            ci[r, r % 32] = r
    for r in range(0, NR, 4):
        if not (ct[r] == r).any():
            ct[r, (r // 4) % 32] = r
    d = dict(obs_np=obs, cand_i2t=ci, cand_t2i=ct)

    def toy():
        ok = True; info = {}
        # rows must be unique per row for the forced-list rule (31 distinct others)
        rows_unique = all(len(set(ci[r].tolist())) == 32 and len(set(ct[r].tolist())) == 32 for r in range(0, NR, 997))
        pp = D36.presence_probs(d)
        bi = np.array([(ci[r].astype(np.int64) == r).any() for r in range(NR)]).mean()
        bt = np.array([(ct[r].astype(np.int64) == r).any() for r in range(NR)]).mean()
        ok &= abs(pp['p_t2i'] - bt) < 1e-12 and abs(pp['p_i2t'] - bi) < 1e-12 and abs(pp['p_i2t_per_image'] - bi) < 1e-12
        rng = np.random.default_rng([1001, 3]); rngn = np.random.default_rng([1001, 3])
        pool = np.sort(rng0.choice(NR, 5000, replace=False)); pset = set(pool.tolist())
        for _ in range(30):
            (li, lt), (rec_i, rec_t, pi, pt) = D36.sample('TRUST_HARD', rng, None, d, pool=pool)
            qi, caps, ai = li; qc, imgs, at = lt
            ok &= all(x in pset for x in rec_i.tolist() + rec_t.tolist())
            ok &= bool(np.array_equal(qi, rec_i)) and bool(np.array_equal(qc, rec_t))       # identity: query image of record r is image r
            ok &= bool(((ai > 0).sum(1) == 1).all()) and bool(((at > 0).sum(1) == 1).all())    # exactly one positive per row
            ok &= bool(np.array_equal(caps[ai > 0], rec_i)) and bool(np.array_equal(imgs[at > 0], rec_t))
            ok &= bool(np.allclose(ai.sum(1), 1)) and bool(np.allclose(at.sum(1), 1))
            ok &= all(len(set(caps[r].tolist())) == 32 and len(set(imgs[r].tolist())) == 32 for r in range(len(caps)))
            (ni, nt), (ri, rt, pi2, pt2) = D36.sample('OBS_MISSING_NULL', rngn, None, d)
            qi, caps, ai = ni; qc, imgs, at = nt
            real_i, real_t = caps != D36.NULL_ID, imgs != D36.NULL_ID
            own_i = real_i & (caps == ri[:, None]); own_t = real_t & (imgs == rt[:, None])
            ok &= bool(np.array_equal(ai > 0, own_i | (~real_i & ~own_i.any(1)[:, None])))
            ok &= bool(np.array_equal(at > 0, own_t | (~real_t & ~own_t.any(1)[:, None])))
            ok &= bool(np.array_equal(ai[~real_i] == 1, ~own_i.any(1))) and bool(np.array_equal(at[~real_t] == 1, ~own_t.any(1)))
            ok &= bool(np.allclose(ai.sum(1), 1)) and bool(np.allclose(at.sum(1), 1)) and caps.shape[1] == 33 and imgs.shape[1] == 33
            # candidate multiset is the native table row (no insertion/replacement/exclusion)
            ok &= all(sorted(caps[r][real_i[r]].tolist()) == sorted(ci[ri[r]].tolist()) for r in range(len(ri)))
            ok &= all(sorted(imgs[r][real_t[r]].tolist()) == sorted(ct[rt[r]].tolist()) for r in range(len(rt)))
        info.update(presence=pp, toy_rows_unique=rows_unique)
        return dict(pass_=ok and rows_unique, **info)
    chk('toy_identity_T_and_N_lists_presence', toy)

    # ------------------------------------------------------------------ (3) per_captions-1 evaluators
    def evaluators():
        ok = True
        ok &= PER_CAP == 1 and N_DEV_IMG == N_DEV_CAP == 1000 and np.array_equal(EV.DEV_GT, np.arange(1000))
        # upstream i2t/t2i verbatim: our native_eval == R31 evaluation.py source for these two functions
        src = (CODE_NATIVE / 'evaluation.py').read_text(); t = ast.parse(src)
        up = {n.name: ast.dump(n) for n in t.body if isinstance(n, ast.FunctionDef) and n.name in ('i2t', 't2i')}
        mine = {n.name: ast.dump(n) for n in ast.parse((RC / 'native_eval.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('i2t', 't2i')}
        def strip_doc(s):
            return re.sub(r"Expr\(value=Constant\(value='[^']*'\)\), ", '', s)
        same_up = all(strip_doc(up[k]) == strip_doc(mine[k]) for k in ('i2t', 't2i'))
        r = np.random.default_rng(9); s = r.standard_normal((1000, 1000)); s[np.arange(1000), np.arange(1000)] += 1.5
        m, tb, rk = EV.full_metrics(s)
        ri = NE.i2t(1000, s, 1); rt = NE.t2i(1000, s, 1)
        ok &= abs(m['rsum'] - (sum(ri[:3]) + sum(rt[:3]))) < 1e-9
        # brute force ranks per 1
        bi = np.array([(s[i] > s[i, i]).sum() for i in range(1000)]); bt = np.array([(s[:, c] > s[c, c]).sum() for c in range(1000)])
        ok &= bool(np.array_equal(rk['ranks_i'], bi)) and bool(np.array_equal(rk['ranks_t'], bt))
        tab = EV.dev_tables(s)
        ok &= bool(np.array_equal(tab['cov_i'], (tab['i_top'] == np.arange(1000)[:, None]).any(1)))
        ok &= bool(all(i not in tab['rand_i'][i] for i in range(1000))) and bool(all(c not in tab['rand_t'][c] for c in range(1000)))
        gi = 3.7 * s[np.arange(1000)[:, None], tab['i_top']] - 0.4; gt = 3.7 * s[tab['t_top'], np.arange(1000)[:, None]] - 0.4
        mc, tbc, cov, rkc = EV.composite_metrics(s, tab, gi, gt)
        ok &= bool(np.array_equal(rkc['ranks_i'], rk['ranks_i'])) and bool(np.array_equal(rkc['ranks_t'], rk['ranks_t'])) and mc['rsum'] == m['rsum']
        pos_i, pos_t, ai, at = EV.dev_list_positions(tab)
        ok &= bool((ai[:, 0] == 1).all()) and bool((ai[:, 1:] == 0).all()) and bool((at[:, 0] == 1).all())
        caps = tab['i_top'][np.arange(1000)[:, None], pos_i]
        ok &= bool((caps != np.arange(1000)[:, None]).all())
        return dict(pass_=ok and same_up, upstream_i2t_t2i_source_identical=same_up, toy_rsum=m['rsum'])
    chk('per_captions1_evaluators', evaluators)

    # ------------------------------------------------------------------ (4) real artifacts
    def real():
        c = CONDS['cc']; cdir = Path(c['cache_dir']); pdir = Path(c['prep_dir'])
        man = read_json(cdir / 'CACHE_MANIFEST.json'); pm = read_json(pdir / 'PREP_MANIFEST.json')
        ok = True
        for name, rec in man['files'].items():
            ok &= sha256(cdir / name) == rec['sha256']
        for name, rec in pm['files'].items():
            ok &= sha256(pdir / name) == rec['sha256']
        nd = man['native_dev']
        ok &= nd['reproduces_checkpoint_best_rsum_exactly'] and nd['cache_rescore_identical'] and nd['rsum'] == 389.7 and nd['per_captions'] == 1
        obs_ = np.load(cdir / 'observed_map.npy'); ok &= bool(np.array_equal(obs_, np.arange(N_TRAIN_CAP)))
        L = np.load(pdir / 'locks_records.npz')
        for k in ('cal_p', 'ev_p', 'lcal_i', 'lcal_t', 'lev_i', 'lev_t'):
            ok &= bool(L[k].min() >= 0 and L[k].max() < N_TRAIN_CAP)
        ok &= bool((L['cal_q'][:, 0] != L['cal_q'][:, 1]).all()) and L['cal_p'].shape == (5000,) and L['cal_q'].shape == (50000, 2)
        ok &= len(np.intersect1d(L['cal_p'], L['ev_p'])) == 0
        cand = np.load(pdir / 'candidates_cc.npz'); ci_, ct_ = cand['i2t_top32'], cand['t2i_top32']
        ok &= ci_.shape == (N_TRAIN_IMG, 32) and ct_.shape == (N_TRAIN_CAP, 32) and ci_.min() >= 0 and ci_.max() < N_TRAIN_CAP and ct_.min() >= 0 and ct_.max() < N_TRAIN_IMG
        sorted_rows = np.sort(ci_[::50], 1); ok &= bool((np.diff(sorted_rows, axis=1) > 0).all())
        sorted_rows = np.sort(ct_[::50], 1); ok &= bool((np.diff(sorted_rows, axis=1) > 0).all())
        T = np.load(pdir / 'trust_cc.npz'); sub = T['subset']
        ok &= bool(np.all(np.diff(sub) > 0)) and sub.min() >= 0 and sub.max() < N_TRAIN_CAP and len(T['prob_A']) == N_TRAIN_CAP
        ok &= bool(np.array_equal(sub, np.nonzero((T['prob_A'] + T['prob_B']) / 2 >= .5)[0]))
        a0 = pm['calibration_init']['cc']['a0']; ok &= a0 > 0 and pm['calibration_init']['cc']['dev_rank_invariance']
        dd = dict(obs_np=obs_.astype(np.int64), cand_i2t=ci_, cand_t2i=ct_)
        pp = D36.presence_probs(dd)
        rng = np.random.default_rng(77)
        (li, lt), (ri, rt, _, _) = D36.sample('TRUST_HARD', rng, None, dd, n=64, pool=sub.astype(np.int64))
        ok &= bool(np.array_equal(li[1][li[2] > 0], ri)) and bool(np.array_equal(lt[1][lt[2] > 0], rt)) and bool(((li[2] > 0).sum(1) == 1).all())
        (ni, nt), (rn, rtn, _, _) = D36.sample('OBS_MISSING_NULL', rng, None, dd, n=64)
        own = (ni[1] == rn[:, None]); ok &= bool(np.array_equal((ni[2] > 0) & (ni[1] != D36.NULL_ID), own))
        return dict(pass_=ok, native_dev=dict((k, nd[k]) for k in ('rsum', 'checkpoint_best_rsum', 'checkpoint_epoch', 'peer_rsum')), presence=pp,
                    trust=dict(n_subset=int(len(sub)), retention=float(len(sub) / N_TRAIN_CAP)), a0b0=[a0, pm['calibration_init']['cc']['b0']],
                    candidates=pm['candidates']['cc'])
    chk('real_cache_prep_identity_mapping', real)

    def nulls_no_rng():
        torch.manual_seed(7); s0 = torch.get_rng_state().clone(); H.NullSlots(); return bool(torch.equal(s0, torch.get_rng_state()))
    chk('null_slots_no_rng', nulls_no_rng)

    # ------------------------------------------------------------------ (5) isolation + static scan
    def isolation():
        res = []
        tdir = Path(tempfile.mkdtemp()) / 'x_precomp'
        for p in (ORACLE_RUNS / 'x', OUT / 'audit' / 'x.json', tdir / 'test_caps.tsv'):
            try:
                open(p); res.append('OPENED')
            except PermissionError:
                res.append('REFUSED')
            except FileNotFoundError:
                res.append('MISSING')
        hits = []
        import io, tokenize
        for f in sorted(RC.glob('*.py')):
            t = f.read_text()
            for i, line in enumerate(t.splitlines()):
                if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line:
                    hits.append(f'{f.name}:{i+1}:test')
            doc_lines = set()
            for n in ast.walk(ast.parse(t)):
                if isinstance(n, ast.Expr) and isinstance(getattr(n, 'value', None), ast.Constant) and isinstance(n.value.value, str):
                    doc_lines |= set(range(n.lineno, n.end_lineno + 1))
            for tok in tokenize.generate_tokens(io.StringIO(t).readline):
                if tok.type == tokenize.COMMENT or tok.start[0] in doc_lines:
                    continue
                if tok.type == tokenize.STRING:
                    if re.search(r'f30k|coco', tok.string) and 'rule=' not in t.splitlines()[tok.start[0] - 1]:
                        hits.append(f'{f.name}:{tok.start[0]}:flickr_path')
                elif tok.type == tokenize.NUMBER and tok.string in ('145000', '29000', '5070', '1014'):
                    hits.append(f'{f.name}:{tok.start[0]}:flickr_constant')
            for i, line in enumerate(t.splitlines()):
                code = line.split('#')[0]
                if (i + 1) not in doc_lines and re.search(r"//\s*5\b|\[::5\]", code):
                    hits.append(f'{f.name}:{i+1}:five_caption_rule')
            if re.search(r'^\s*(import|from)\s+(oracle_truth|audit_export|audit_candidate_lists)', t, re.M):
                hits.append(f'{f.name}:oracle/audit import')
        return dict(pass_=res == ['REFUSED', 'REFUSED', 'REFUSED'] and not hits, opens=res, hits=hits)
    chk('isolation_and_static_scan', isolation)

    allp = all(v['pass_'] for v in checks.values())
    write_json(ROOT / 'CPU_TEST.json', dict(ALL_PASS=allp, checks=checks, created_utc=utc(), job=os.environ.get('SLURM_JOB_ID', 'local'), seconds=time.time() - t0))
    print('R37C_CPU', 'ALL_PASS' if allp else 'FAIL'); sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
