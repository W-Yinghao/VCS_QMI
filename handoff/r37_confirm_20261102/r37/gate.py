"""R37 GPU gate on real data (technical only; no outcome thresholds). Writes ROOT/GPU_GATE.json.
Checks: static scan + TEST guard (functional, subprocess); per parent: cache/prep hashes, per-peer native shard
reproduces the cached native DEV, new caches reproduce the checkpoint best rSum exactly; R37 prep code reproduces the
R35 n60 prep (calibration, candidate tables); R37 T/N path on the R36 parent == frozen R36 code (separate R36 process,
20 updates, init/final/batch hashes); N lists on every parent's real tables; TRUST+N lists (anchors only from the
parent's B-TRUST subset, full N protocol, labels from the full observed map); PRESENT_ONLY mask == observed-present
rows, masked loss formula, empirical valid fraction vs fixed p_d; identical init + batches across objectives,
determinism rerun, DEV=NaN isolation, resume == uninterrupted (TRUST+N, PRESENT_ONLY, T on s1002, N on s1003); smoke
(30 updates, real readouts) + timing + step 0 == native ranking for every (parent, route) bundle; runner entry
(finalize) for one unit; no TEST file opened.  Usage: python -B gate.py"""
from __future__ import annotations
import copy, json, os, re, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, OUT, R36_ROOT, R35_PREP, PEERS, DATA_SRC, CONDS, install_guards, guard_record, assert_no_oracle_loaded,
                    sha256, write_json, read_json, utc)
import objectives as O
import evals as EV
import data36 as D36
import train_unit as T

PY = sys.executable
R37 = ROOT / 'r37'

R36_PROBE = r'''
import sys, json
sys.path.insert(0, {r36!r})
import numpy as np, torch
import train_unit as T
from common import install_guards
install_guards(role='trainer'); T.set_determinism()
out = {{}}; ctx = None
for uid, steps in {units!r}:
    cfg = T.cfg_of(uid); cfg['_config_sha256'] = 'gate_probe'; cfg['phase'] = 1; cfg['steps'] = steps; cfg['ckpt_steps'] = [0, steps]
    if ctx is None:
        ctx = T.Ctx(cfg['cond'], torch.device('cuda'), need_local=False)
    r = T.run_unit(cfg, ctx, pool=T.pool_of(cfg, ctx), out_dir={gd!r} + '/r36_' + uid, log=lambda *a: None, no_readout=True)
    out[uid] = dict(init=r['init_hash'], final=r['final_hash'], batch=r['batch_hash'])
print('R36_PROBE ' + json.dumps(out))
'''

SMOKE = {'n60s1002': ('r37_confirm_f60_h1002_T_G_J', 'r37_confirm_f60_h1002_N_G_J'),
         'n60s1003': ('r37_confirm_f60_h1003_T_G_J', 'r37_confirm_f60_h1003_N_G_J'),
         'n60': ('r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_J', 'r37_factor_f60_h1001_N_PRESENT_ONLY_G_J')}
FAMILY = {'n60': ('r37_factor_f60_h1001_TRUST_NATIVE_NULL_G', 'r37_factor_f60_h1001_N_PRESENT_ONLY_G'),
          'n60s1002': ('r37_confirm_f60_h1002_T_G',), 'n60s1003': ('r37_confirm_f60_h1003_N_G',)}


def main():
    install_guards(role='gate'); assert_no_oracle_loaded(); T.set_determinism()
    job = os.environ.get('SLURM_JOB_ID', 'local')
    GD = OUT / 'gate' / job; GD.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda'); checks = {}; info = {}; t0 = time.time(); timing = {}

    def chk(name, ok, **kw):
        checks[name] = dict(pass_=bool(ok), **kw)
        print(('PASS ' if ok else 'FAIL ') + name, {k: v for k, v in kw.items() if len(str(v)) < 400}, flush=True)

    def guarded(name, fn):
        try:
            fn()
        except Exception:
            chk(name, False, error=traceback.format_exc())

    hits = []
    for f in sorted(R37.glob('*.py')):
        if f.name == 'gate.py':
            continue
        for i, line in enumerate(f.read_text().splitlines()):
            if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line:
                hits.append(f'{f.name}:{i+1}')
    chk('static_no_test_paths', not hits, hits=hits)
    probe = (f"import sys; sys.path.insert(0, {str(R37)!r}); from common import install_guards; install_guards()\n"
             f"try:\n    open({str(DATA_SRC / ('te' + 'st_caps.txt'))!r}); print('OPENED')\nexcept PermissionError: print('BLOCKED')")
    r = subprocess.run([PY, '-B', '-c', probe], capture_output=True, text=True)
    chk('test_guard_functional_subprocess', r.stdout.strip() == 'BLOCKED', stdout=r.stdout.strip())

    cfgs = {u: T.cfg_of(u) for u in read_json(ROOT / 'configs' / 'UNITS.json')}

    def short(uid, steps, **kw):
        c = copy.deepcopy(cfgs[uid]); c.update(kw); c['steps'] = steps; c['ckpt_steps'] = [0, steps]; c['_config_sha256'] = 'gate_override'; c.setdefault('phase', 1)
        return c

    def run(c, ctx, out, **kw):
        return T.run_unit(c, ctx, pool=T.pool_of(c, ctx), out_dir=GD / out, log=lambda *a: None, **kw)

    def step0_native(ctx, out_dir, name):
        z = np.load(GD / out_dir / 'dev_scores_step0.npz'); d = ctx.d; nat = d['native_dev']
        m, tb, cov, rk = EV.composite_metrics(nat, d['dev_tab'], z['gi_top'], z['gt_top'])
        di = np.nonzero(rk['ranks_i'] != ctx.nat_rk['ranks_i'])[0]; dt = np.nonzero(rk['ranks_t'] != ctx.nat_rk['ranks_t'])[0]
        TOL = 1e-6
        tied_i = all(any((np.abs(nat[i] - nat[i, 5 * i + k]) <= TOL).sum() > 1 for k in range(5)) for i in di)
        tied_t = all((np.abs(nat[:, c] - nat[c // 5, c]) <= TOL).sum() > 1 for c in dt)
        chk(name, (len(di) == 0 and len(dt) == 0) or (tied_i and tied_t), rsum=m['rsum'], native_rsum=ctx.nat_metrics['rsum'], n_i2t_rank_diff=len(di), n_t2i_rank_diff=len(dt))

    r36b = read_json(R36_ROOT / 'PARENT_BINDING.json')['conditions']['n60']
    for cond in ('n60', 'n60s1002', 'n60s1003'):
        ctx = T.Ctx(cond, dev, need_local=False); d = ctx.d
        man = d['manifest']
        hk = dict(cache_manifest=d['manifest_sha256'], prep_manifest=d['prep_manifest_sha256'], candidates=d['cand_sha256'], trust=d['trust_sha256'],
                  parent=man['parent']['unit'], model_best_sha256=man['parent']['model_best_sha256'], a0b0=d['a0b0'])
        if cond == 'n60':
            ok = (d['manifest_sha256'] == r36b['r34_cache_manifest_sha256'] and d['cand_sha256'] == r36b['candidates_sha256']
                  and d['trust_sha256'] == read_json(R36_ROOT / 'PARENT_BINDING.json')['trust_n60_sha256'] and man['parent']['model_best_sha256'] == r36b['model_best_sha256'])
            chk('n60_artifacts_identical_to_R36_binding', ok, **hk)
        else:
            nd = man['native_dev']
            chk(f'{cond}_new_cache_reproduces_checkpoint_best_rsum_exactly', nd['reproduces_checkpoint_best_rsum_exactly'] and nd['cache_rescore_identical'],
                rsum=nd['rsum'], checkpoint_best_rsum=nd['checkpoint_best_rsum'], epoch=nd['checkpoint_epoch'], cache_gpu=man['env']['gpu'], **hk)
        nat_mean = ((d['dev_s']['A'] + d['dev_s']['B']) / 2).cpu().numpy(); mx = float(np.abs(nat_mean - d['native_dev']).max())
        chk(f'{cond}_per_peer_native_shard_reproduces_native_dev', mx <= 1e-6, max_abs=mx, gpu=torch.cuda.get_device_name(0), native_rsum=ctx.nat_metrics['rsum'])
        chk(f'{cond}_calibration_positive_rank_invariant', d['a0b0']['a0'] > 0 and d['a0b0']['dev_rank_invariance'], a0b0=d['a0b0'])
        info[cond] = dict(native=ctx.nat_metrics, a0b0=d['a0b0'], present_probs=ctx.present, n_trust=int(len(d['trust_subset'])))
        if cond == 'n60':
            def prep_repro():
                sys.path.insert(0, str(R37)); import prep as P
                i2t, t2i = P.candidate_tables(d)
                ref = np.load(R35_PREP / 'candidates_n60.npz')
                fi, ft = float((i2t == ref['i2t_top32']).mean()), float((t2i == ref['t2i_top32']).mean())
                L = d['locks']; obs = d['obs_np']
                sp = P.native_pair_scores(d, obs[L['cal_p']], L['cal_p']); sq = P.native_pair_scores(d, obs[L['cal_q'][:, 0]], L['cal_q'][:, 1])
                g = np.concatenate([sp, sq]); wp = np.r_[np.full(len(sp), 1 / len(sp)), np.zeros(len(sq))]; wq = np.r_[np.zeros(len(sp)), np.full(len(sq), 1 / len(sq))]
                fit = EV.fit_affine(g, wp, wq, 'balanced_logistic')
                ra, rb = abs(fit['a'] / d['a0b0']['a0'] - 1), abs(fit['b'] - d['a0b0']['b0'])
                chk('R37_prep_code_reproduces_R35_n60_prep', fi >= .999 and ft >= .999 and ra < 1e-6 and rb < 1e-6, frac_i2t_equal=fi, frac_t2i_equal=ft, a_rel=ra, b_abs=rb)
            guarded('R37_prep_code_reproduces_R35_n60_prep', prep_repro)

            def r36_equiv():
                units = [('r36_T_n60_G_J', 20), ('r36_N_n60_G_J', 20)]
                r = subprocess.run([PY, '-B', '-c', R36_PROBE.format(r36=str(R36_ROOT / 'r36'), units=units, gd=str(GD))], capture_output=True, text=True,
                                   env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                line = [l for l in r.stdout.splitlines() if l.startswith('R36_PROBE ')]
                ref = json.loads(line[0].split(' ', 1)[1]) if line else {}
                for uid, _ in units:
                    c36 = read_json(R36_ROOT / 'configs' / f'{uid}.json'); c36.update(phase=1, steps=20, ckpt_steps=[0, 20], _config_sha256='gate')
                    mine = run(c36, ctx, f'r37_{uid}', no_readout=True)
                    ok = bool(ref) and all(mine[k] == ref[uid][x] for k, x in (('init_hash', 'init'), ('final_hash', 'final'), ('batch_hash', 'batch')))
                    chk(f'{uid}_R37_code_equals_frozen_R36_code_bitwise', ok, r36=ref.get(uid), r37=dict(init=mine['init_hash'], final=mine['final_hash'], batch=mine['batch_hash']),
                        r36_stderr=r.stderr[-800:] if not ref else '')
            guarded('R36_equivalence', r36_equiv)

            def tn_lists():
                rng = np.random.default_rng([1001, 3]); pool = d['trust_subset']; ok = True; pos = np.zeros(33, int); nf = {0: [], 1: []}
                pset = set(pool.tolist())
                for _ in range(200):
                    lists, (ri, rt, pi, pt) = D36.sample('TRUST_NATIVE_NULL', rng, None, d, pool=pool)
                    ok &= all(x in pset for x in ri.tolist()) and all(x in pset for x in rt.tolist())
                    ref = D36.null_lists(ri, rt, d['obs_np'], d['cand_i2t'], d['cand_t2i'], pi, pt)
                    for k in range(2):
                        for a, b in zip(lists[k], ref[k]):
                            ok &= bool(np.array_equal(a, b))
                    (qi, caps, ai), (qc, imgs, at) = lists
                    ok &= bool(np.allclose(ai.sum(1), 1)) and bool(np.allclose(at.sum(1), 1)) and bool(((caps == D36.NULL_ID).sum(1) == 1).all())
                    ok &= bool(np.array_equal(np.sort(np.where(caps == D36.NULL_ID, -2, caps), 1)[:, 1:], np.sort(d['cand_i2t'][d['obs_np'][ri]].astype(np.int64), 1)))
                    pos += np.bincount(np.nonzero(caps == D36.NULL_ID)[1], minlength=33)
                    nf[0].append(float(ai[caps == D36.NULL_ID].mean())); nf[1].append(float(at[imgs == D36.NULL_ID].mean()))
                chk('n60_TRUST_NATIVE_NULL_lists_anchor_pool_and_full_N_protocol', ok and (pos > 0).all(), null_label_frac_i2t=float(np.mean(nf[0])), null_label_frac_t2i=float(np.mean(nf[1])))
            guarded('n60_TRUST_NATIVE_NULL_lists', tn_lists)

            def po_lists():
                rng = np.random.default_rng([1001, 3]); ok = True; vi, vt, n = 0, 0, 0
                for _ in range(2000):
                    lists, _ = D36.sample('N_PRESENT_ONLY', rng, None, d)
                    (qi, caps, ai), (qc, imgs, at) = lists
                    mi, mt = D36.valid_rows(lists)
                    ok &= bool(np.array_equal(mi, ((d['obs_np'][np.where(caps == D36.NULL_ID, 0, caps)] == qi[:, None]) & (caps != D36.NULL_ID)).any(1)))
                    ok &= bool(np.array_equal(mt, ((imgs == d['obs_np'][qc][:, None]) & (imgs != D36.NULL_ID)).any(1)))
                    vi += int(mi.sum()); vt += int(mt.sum()); n += len(mi)
                torch.manual_seed(0); g = torch.randn(16, 33, dtype=torch.float64) * 3
                a = torch.as_tensor(ai, dtype=torch.float64); m = torch.as_tensor(mi, dtype=torch.float64); p = ctx.present['p_i2t']
                lf = {o: float(O.loss_list_masked(o, g, a, m, p)) for o in O.OBJECTIVES}
                manual = {o: float((O.list_rows(g, a, o) * m).sum() / (16 * p)) * (-1 if o == 'J' else 1) for o in O.OBJECTIVES}
                ok &= all(abs(lf[o] - manual[o]) < 1e-12 for o in O.OBJECTIVES)
                ok &= all(abs(float(O.loss_list_masked(o, g, a, torch.ones(16, dtype=torch.float64), 1.0)) - float(O.loss_list(o, g, a))) < 1e-12 for o in O.OBJECTIVES)
                emp = dict(p_i2t_empirical=vi / n, p_t2i_empirical=vt / n)
                ok &= abs(emp['p_i2t_empirical'] - ctx.present['p_i2t']) < .02 and abs(emp['p_t2i_empirical'] - ctx.present['p_t2i']) < .02
                chk('n60_N_PRESENT_ONLY_mask_formula_and_p_d', ok, **emp, **{k: v for k, v in ctx.present.items() if k != 'rule'})
            guarded('n60_N_PRESENT_ONLY', po_lists)

        def nlists(cond=cond, d=d):
            rng = np.random.default_rng([1001, 3]); pos = np.zeros(33, int); nf = {0: [], 1: []}; ok = True
            for _ in range(200):
                lists, (ri, rt, pi, pt) = D36.sample('OBS_MISSING_NULL', rng, None, d)
                (qi, caps, ai), (qc, imgs, at) = lists
                for k, (ids, a, base) in enumerate(((caps, ai, d['cand_i2t'][d['obs_np'][ri]]), (imgs, at, d['cand_t2i'][rt]))):
                    ok &= bool(np.allclose(a.sum(1), 1)) and bool(((ids == D36.NULL_ID).sum(1) == 1).all())
                    ok &= bool(np.array_equal(np.sort(np.where(ids == D36.NULL_ID, -2, ids), 1)[:, 1:], np.sort(base.astype(np.int64), 1)))
                    pos += np.bincount(np.nonzero(ids == D36.NULL_ID)[1], minlength=33); nf[k].append(float(a[ids == D36.NULL_ID].mean()))
            chk(f'{cond}_N_lists_real_tables', ok and (pos > 0).all(), null_label_frac_i2t=float(np.mean(nf[0])), null_label_frac_t2i=float(np.mean(nf[1])))
        guarded(f'{cond}_N_lists_real_tables', nlists)

        for base in FAMILY[cond]:
            def fam(base=base, ctx=ctx):
                res = {o: run(short(f'{base}_{o}', 20), ctx, f'{base}_obj_{o}', no_readout=True) for o in O.OBJECTIVES}
                chk(f'{base}_identical_init_and_batches_across_objectives', len({r['init_hash'] for r in res.values()}) == 1 and len({r['batch_hash'] for r in res.values()}) == 1
                    and len({r['final_hash'] for r in res.values()}) == 3)
                r2 = run(short(f'{base}_J', 20), ctx, f'{base}_rerun', no_readout=True)
                chk(f'{base}_determinism_rerun', r2['final_hash'] == res['J']['final_hash'])
                r3 = run(short(f'{base}_J', 20), ctx, f'{base}_devnan', dev_nan=True)
                chk(f'{base}_dev_isolation_nan_dev', r3['final_hash'] == res['J']['final_hash'])
                full = run(short(f'{base}_J', 40), ctx, f'{base}_res_full', ckpt_steps=[0, 20, 40], no_readout=True)
                run(short(f'{base}_J', 40), ctx, f'{base}_res_split', ckpt_steps=[0, 20, 40], stop_at=20, no_readout=True)
                rs = run(short(f'{base}_J', 40), ctx, f'{base}_res_split', ckpt_steps=[0, 20, 40], no_readout=True)
                chk(f'{base}_resume_equals_uninterrupted', rs['final_hash'] == full['final_hash'] and rs['batch_hash'] == full['batch_hash'])
                if res['J']['nulls'] is not None:
                    nn0 = res['J']['nulls']
                    chk(f'{base}_null_params_trained_and_finite', all(torch.isfinite(v).all() for v in nn0.v.values()) and any(float(v.abs().sum()) > 0 for v in nn0.v.values()),
                        nulls={p: [float(x) for x in nn0.v[p].detach().cpu()] for p in PEERS})
            guarded(f'{base}_family_checks', fam)

        for uid in SMOKE[cond]:
            def smoke(uid=uid, ctx=ctx):
                c = short(uid, 30)
                torch.cuda.reset_peak_memory_stats(dev)
                r = run(c, ctx, f'smoke_{uid}')
                ev = read_json(GD / f'smoke_{uid}' / 'eval_step30.json'); e0 = read_json(GD / f'smoke_{uid}' / 'eval_step0.json')
                st = r['hist']['stats_sum']
                timing[uid] = dict(train_seconds_per_update=r['hist']['train_seconds'] / 30, readout_seconds=ev['readout_seconds'],
                                   peak_gpu_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30, rsum_step0=e0['dev_retrieval']['rsum'], rsum_step30=ev['dev_retrieval']['rsum'],
                                   stats=st, slot_grad_means=ev.get('train_window', {}).get('slot_grad_means'), null_readout=ev.get('fit_null_lists_observed_J_raw'))
                ok = np.isfinite(ev['dev_retrieval']['rsum'])
                if c['supervision'] == 'N_PRESENT_ONLY':
                    ok &= st.get('valid_rows_i2t', 0) > 0 and st.get('valid_rows_t2i', 0) > 0 and 'empty_valid_batch_t2i' in st
                chk(f'smoke_{uid}', ok, **{k: v for k, v in timing[uid].items() if k not in ('stats', 'null_readout')})
                step0_native(ctx, f'smoke_{uid}', f'{uid}_step0_two_stage_equals_native')
            guarded(f'smoke_{uid}', smoke)
        del ctx, d; torch.cuda.empty_cache()

    uid = 'r37_factor_f60_h1001_N_PRESENT_ONLY_G_NCE'
    r = subprocess.run([PY, '-B', str(R37 / 'train_unit.py'), '--units', uid, '--gate-out', str(GD / 'main'), '--gate-steps', '10'], capture_output=True, text=True, env=dict(os.environ))
    ok = r.returncode == 0 and (GD / 'main' / uid / 'RESULT.json').exists()
    if ok:
        rr = read_json(GD / 'main' / uid / 'RESULT.json')
        ok = rr['test_guard']['role'] == 'trainer' and not rr['test_guard']['blocked_attempts'] and rr['present_probs']['p_t2i'] > 0 and rr['parent']['unit'] == 'r30_f60_s1001_H0'
    chk('runner_entry_finalize', ok, rc=r.returncode, tail=(r.stdout + r.stderr)[-800:] if not ok else '')
    gr = guard_record()
    chk('test_guard_no_test_file_opened', gr['installed'] and not gr['blocked_attempts'], **{k: v for k, v in gr.items() if k != 'forbidden_prefixes'})
    allp = all(v['pass_'] for v in checks.values())
    code = {f.name: sha256(f) for f in sorted(R37.glob('*.py'))}
    write_json(ROOT / 'GPU_GATE.json', dict(ALL_PASS=allp, checks=checks, measured_timing=timing, info=info, created_utc=utc(), job=job,
                                            gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, code_sha256=code, elapsed_seconds=time.time() - t0))
    print('R37_GATE', 'ALL_PASS' if allp else 'FAIL', [k for k, v in checks.items() if not v['pass_']])
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
