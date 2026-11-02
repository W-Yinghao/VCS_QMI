"""R36 GPU gate on real data (technical only; no outcome thresholds). Writes ROOT/GPU_GATE.json and
ROOT/ORACLE_ISOLATION.json. Checks: static scan + TEST guard (functional); R34 cache / R35 prep hashes; native
shard reproduction; R36 forced-list path == frozen R35 C code (20 updates, C-G and C-L, final state bitwise, run in
a separate R35 process); step 0 == native ranking for every block path (fresh heads; null ignored at inference);
identical init + batches across objectives; determinism rerun; DEV=NaN isolation; resume == uninterrupted (N with
null params + 33-slot perms, M-RANDOM with its second stream, T-L, V phase 2); V phase boundary (optimizer reset,
own phase-1 head, phase-2 step 0 == phase-1 final scores bitwise); N lists on the real tables; smoke + timing of
every path; oracle runner in a separate process (SHIELD/RELABEL, same batches as C), refusals both ways, trainer
refuses opening oracle_runs/audit outputs.
Usage: python -B gate.py"""
from __future__ import annotations
import copy, json, os, re, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, OUT, RUNS, ORACLE_RUNS, R35_ROOT, PEERS, DATA_SRC, N_TRAIN_CAP, install_guards, guard_record, assert_no_oracle_loaded,
                    sha256, write_json, read_json, utc)
import objectives as O
import evals as EV
import data36 as D36
import train_unit as T

PY = sys.executable
R36 = ROOT / 'r36'
OPT = dict(name='AdamW', lr_head=1e-4, weight_decay=1e-4, betas=[0.9, 0.999], eps=1e-8, schedule='constant')

R35_PROBE = r'''
import sys, json, os
sys.path.insert(0, {r35!r})
import numpy as np, torch
import train_unit as T
from common import install_guards
install_guards(); T.set_determinism()
out = {{}}
ctxs = {{}}
for uid, steps in {units!r}:
    cfg = T.cfg_of(uid); cfg['_config_sha256'] = 'gate_probe'
    key = (cfg['cond'], cfg['structure'] == 'L')
    if key not in ctxs:
        ctxs[key] = T.Ctx(cfg['cond'], torch.device('cuda'), need_local=key[1])
    r = T.run_unit(cfg, ctxs[key], imap=ctxs[key].d['obs_np'], pool=np.arange(145000), out_dir={gd!r} + '/r35_' + uid, steps=steps, ckpt_steps=[0, steps],
                   log=lambda *a: None, no_readout=True)
    out[uid] = dict(init=r['init_hash'], final=r['final_hash'])
print('R35_PROBE ' + json.dumps(out))
'''


def main():
    install_guards(role='gate'); assert_no_oracle_loaded(); T.set_determinism()
    job = os.environ.get('SLURM_JOB_ID', 'local')
    GD = OUT / 'gate' / job; GD.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda'); checks = {}; info = {}; t0 = time.time(); timing = {}

    def chk(name, ok, **kw):
        checks[name] = dict(pass_=bool(ok), **kw)
        print(('PASS ' if ok else 'FAIL ') + name, {k: v for k, v in kw.items() if len(str(v)) < 300}, flush=True)

    def guarded(name, fn):
        try:
            fn()
        except Exception:
            chk(name, False, error=traceback.format_exc())

    # ---- static scan + TEST guard
    hits = []
    for f in sorted(R36.glob('*.py')):
        if f.name == 'gate.py':
            continue
        for i, line in enumerate(f.read_text().splitlines()):
            if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line:
                hits.append(f'{f.name}:{i+1}')
    chk('static_no_test_paths', not hits, hits=hits)
    probe = (f"import sys; sys.path.insert(0, {str(R36)!r}); from common import install_guards; install_guards()\n"
             f"try:\n    open({str(DATA_SRC / ('te' + 'st_caps.txt'))!r}); print('OPENED')\nexcept PermissionError: print('BLOCKED')")
    r = subprocess.run([PY, '-B', '-c', probe], capture_output=True, text=True)
    chk('test_guard_functional_subprocess', r.stdout.strip() == 'BLOCKED', stdout=r.stdout.strip())

    cfgs = {u: T.cfg_of(u) for u in read_json(ROOT / 'configs' / 'UNITS.json')}

    def short(uid, steps, **kw):
        c = copy.deepcopy(cfgs[uid]); c.update(kw); c['steps'] = steps; c['ckpt_steps'] = [0, steps]; c['_config_sha256'] = 'gate_override'; c.setdefault('phase', 1)
        return c

    def c_equiv(structure, steps):
        return dict(unit_id=f'gate_C_equiv_{structure}', block='C', phase=1, cond='n60', structure=structure, objective='J', supervision='OBS_HARD', oracle=False,
                    steps=steps, ckpt_steps=[0, steps], init_seed=1001, sampler_seed=[1001, 3], n_queries=16, optimizer=OPT, clip_per_peer=2.0, _config_sha256='gate')

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

    for cond in ('n60', 'n00'):
        ctxL = T.Ctx(cond, dev, need_local=True)
        ctxG = T._shallow(ctxL, need_local=False)
        d = ctxL.d
        chk(f'{cond}_hashes_R34_cache_and_R35_prep', True, cache_manifest=d['manifest_sha256'], prep_manifest=d['prep_manifest_sha256'], candidates=d['cand_sha256'])
        nat_mean = ((d['dev_s']['A'] + d['dev_s']['B']) / 2).cpu().numpy(); mx = float(np.abs(nat_mean - d['native_dev']).max())
        chk(f'{cond}_per_peer_native_shard_reproduces_native_dev', mx <= 1e-6, max_abs=mx, gpu=torch.cuda.get_device_name(0), native_rsum=ctxL.nat_metrics['rsum'])
        if cond == 'n60':
            # ---- R36 forced-list path == frozen R35 C code
            def equiv():
                units = [('r35_C_n60_s1001_G_J', 20), ('r35_C_n60_s1001_L_J', 20)]
                r = subprocess.run([PY, '-B', '-c', R35_PROBE.format(r35=str(R35_ROOT / 'r35'), units=units, gd=str(GD))], capture_output=True, text=True,
                                   env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                line = [l for l in r.stdout.splitlines() if l.startswith('R35_PROBE ')]
                ref = json.loads(line[0].split(' ', 1)[1]) if line else {}
                for st, ctx, uid in (('G', ctxG, 'r35_C_n60_s1001_G_J'), ('L', ctxL, 'r35_C_n60_s1001_L_J')):
                    mine = run(c_equiv(st, 20), ctx, f'c_equiv_{st}', no_readout=True)
                    chk(f'n60_C{st}_forced_path_equals_frozen_R35_code_bitwise', bool(ref) and mine['final_hash'] == ref[uid]['final'] and mine['init_hash'] == ref[uid]['init'],
                        r35=ref.get(uid), r36=dict(init=mine['init_hash'], final=mine['final_hash']), r35_stderr=r.stderr[-800:] if not ref else '')
            guarded('n60_forced_path_equals_frozen_R35_code', equiv)
            # ---- N lists on the real tables
            def nlists():
                rng = np.random.default_rng([1001, 3]); pos = np.zeros(33, int); nf = {0: [], 1: []}; ok = True
                for _ in range(200):
                    lists, (ri, rt, pi, pt) = D36.sample('OBS_MISSING_NULL', rng, None, d)
                    (qi, caps, ai), (qc, imgs, at) = lists
                    for k, (ids, a, base) in enumerate(((caps, ai, d['cand_i2t'][d['obs_np'][ri]]), (imgs, at, d['cand_t2i'][rt]))):
                        ok &= bool(np.allclose(a.sum(1), 1)) and bool(((ids == D36.NULL_ID).sum(1) == 1).all())
                        ok &= bool(np.array_equal(np.sort(np.where(ids == D36.NULL_ID, -2, ids), 1)[:, 1:], np.sort(base.astype(np.int64), 1)))
                        pos += np.bincount(np.nonzero(ids == D36.NULL_ID)[1], minlength=33); nf[k].append(float(a[ids == D36.NULL_ID].mean()))
                chk('n60_N_lists_real_tables', ok and (pos > 0).all(), null_position_min=int(pos.min()), null_label_frac_i2t=float(np.mean(nf[0])), null_label_frac_t2i=float(np.mean(nf[1])))
            guarded('n60_N_lists_real_tables', nlists)
            fam = dict(M=('r36_M_n60_G_RANDOM', ctxG), TL=('r36_T_n60_L', ctxL), NG=('r36_N_n60_G', ctxG), NL=('r36_N_n60_L', ctxL), TG=('r36_T_n60_G', ctxG))
            for k, (base, ctx) in fam.items():
                def fam_checks(k=k, base=base, ctx=ctx):
                    res = {o: run(short(f'{base}_{o}', 20), ctx, f'{k}_obj_{o}', no_readout=True) for o in O.OBJECTIVES}
                    chk(f'{k}_identical_init_and_batches_across_objectives', len({r['init_hash'] for r in res.values()}) == 1 and len({r['batch_hash'] for r in res.values()}) == 1
                        and len({r['final_hash'] for r in res.values()}) == 3)
                    if k in ('M', 'TL', 'NG'):
                        r2 = run(short(f'{base}_J', 20), ctx, f'{k}_rerun', no_readout=True)
                        chk(f'{k}_determinism_rerun', r2['final_hash'] == res['J']['final_hash'])
                        r3 = run(short(f'{base}_J', 20), ctx, f'{k}_devnan', dev_nan=True)
                        chk(f'{k}_dev_isolation_nan_dev', r3['final_hash'] == res['J']['final_hash'])
                        full = run(short(f'{base}_J', 40), ctx, f'{k}_res_full', ckpt_steps=[0, 20, 40], no_readout=True)
                        run(short(f'{base}_J', 40), ctx, f'{k}_res_split', ckpt_steps=[0, 20, 40], stop_at=20, no_readout=True)
                        rs = run(short(f'{base}_J', 40), ctx, f'{k}_res_split', ckpt_steps=[0, 20, 40], no_readout=True)
                        chk(f'{k}_resume_equals_uninterrupted', rs['final_hash'] == full['final_hash'] and rs['batch_hash'] == full['batch_hash'])
                    if k == 'NG':
                        nn0 = res['J']['nulls']
                        chk('NG_null_params_trained_and_finite', all(torch.isfinite(v).all() for v in nn0.v.values()) and any(float(v.abs().sum()) > 0 for v in nn0.v.values()),
                            nulls={p: [float(x) for x in nn0.v[p].detach().cpu()] for p in PEERS})
                guarded(f'{k}_family_checks', fam_checks)
            # ---- smoke + timing + step-0 native, every n60 path (J), 30 updates, real readouts
            for uid in ('r36_M_n60_G_RANDOM_J', 'r36_T_n60_G_J', 'r36_T_n60_L_J', 'r36_N_n60_G_J', 'r36_N_n60_L_J'):
                def smoke(uid=uid):
                    c = short(uid, 30); ctx = ctxL if c['structure'] == 'L' else ctxG
                    torch.cuda.reset_peak_memory_stats(dev)
                    r = run(c, ctx, f'smoke_{uid}')
                    ev = read_json(GD / f'smoke_{uid}' / 'eval_step30.json'); e0 = read_json(GD / f'smoke_{uid}' / 'eval_step0.json')
                    timing[uid] = dict(train_seconds_per_update=r['hist']['train_seconds'] / 30, readout_seconds=ev['readout_seconds'],
                                       peak_gpu_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30, rsum_step0=e0['dev_retrieval']['rsum'], rsum_step30=ev['dev_retrieval']['rsum'],
                                       stats=r['hist']['stats_sum'], slot_grad_means=ev.get('train_window', {}).get('slot_grad_means'),
                                       null_readout=ev.get('fit_null_lists_observed_J_raw'))
                    chk(f'smoke_{uid}', np.isfinite(ev['dev_retrieval']['rsum']), **{k: v for k, v in timing[uid].items() if k != 'stats'})
                    step0_native(ctx, f'smoke_{uid}', f'{uid}_step0_two_stage_equals_native')
                guarded(f'smoke_{uid}', smoke)
            # ---- C-equivalent 30-update batch hash (compared with the oracle runner's SHIELD/RELABEL batches)
            info['c_equiv_G_30_batch_hash'] = run(c_equiv('G', 30), ctxG, 'c_equiv_G_30', ckpt_steps=[0, 30], no_readout=True)['batch_hash']
        else:
            # ---- V: phase 1 -> phase 2 boundary, resume, step 0 native
            def vcheck():
                uid = 'r36_V_n00_h2002_L_J'
                p1 = T.phase_cfgs(short(uid, 20))[0]; p1['steps'] = 20; p1['ckpt_steps'] = [0, 20]
                torch.cuda.reset_peak_memory_stats(dev)
                r1 = run(p1, ctxL, 'V/phase1')
                step0_native(ctxL, 'V/phase1', f'{uid}_phase1_step0_two_stage_equals_native')
                p2 = T.phase_cfgs(cfgs[uid])[1]; p2.update(steps=40, ckpt_steps=[0, 20, 40], parent_step=20, parent_ckpt=str(GD / 'V/phase1' / 'ckpt_step20.pt'), _config_sha256='gate')
                full = run(p2, ctxL, 'V/phase2_full')
                rec = read_json(GD / 'V/phase2_full' / 'PHASE_BOUNDARY_RECEIPT.json')
                z1 = np.load(GD / 'V/phase1' / 'dev_scores_step20.npz'); z2 = np.load(GD / 'V/phase2_full' / 'dev_scores_step0.npz')
                same = all(np.array_equal(z1[k], z2[k]) for k in ('gi_top', 'gt_top', 'g_gt'))
                chk('V_phase2_step0_equals_phase1_final_bitwise', same)
                chk('V_phase_boundary_optimizer_reset_and_new_stream', rec['optimizer_state_entries_at_reset'] == 0 and rec['sampler_seed'] == [2002, 4]
                    and rec['parent_sampler_seed'] == [2002, 3] and rec['parent_step'] == 20, receipt=rec)
                run(p2, ctxL, 'V/phase2_split', stop_at=20, no_readout=True)
                rs = run(p2, ctxL, 'V/phase2_split', no_readout=True)
                chk('V_phase2_resume_equals_uninterrupted', rs['final_hash'] == full['final_hash'] and rs['batch_hash'] == full['batch_hash'])
                enc_same = all(torch.equal(q, q0) for p in PEERS for q, q0 in zip(full['encs'][p].parameters(), ctxL.enc0[p].parameters()))
                chk('V_local_encoder_unchanged', enc_same)
                h = read_json(GD / 'V/phase1' / 'TRAIN_HISTORY.json'); h2 = read_json(GD / 'V/phase2_full' / 'TRAIN_HISTORY.json')
                timing[uid] = dict(train_seconds_per_update_phase1=h['train_seconds'] / 20, train_seconds_per_update_phase2=h2['train_seconds'] / 40,
                                   readout_seconds=read_json(GD / 'V/phase1' / 'eval_step20.json')['readout_seconds'], peak_gpu_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30)
            guarded('V_checks', vcheck)
            # whole-unit V via the runner entry (both phases, finalize)
            r = subprocess.run([PY, '-B', str(R36 / 'train_unit.py'), '--units', 'r36_V_n00_h2003_L_NCE', '--gate-out', str(GD / 'Vmain'), '--gate-steps', '10'],
                               capture_output=True, text=True, env=dict(os.environ))
            ok = r.returncode == 0 and (GD / 'Vmain' / 'r36_V_n00_h2003_L_NCE' / 'RESULT.json').exists()
            if ok:
                rr = read_json(GD / 'Vmain' / 'r36_V_n00_h2003_L_NCE' / 'RESULT.json')
                ok = rr['phase2_step0_equals_phase1_final_scores']['bitwise'] and len(rr['phases']) == 2 and rr['test_guard']['role'] == 'trainer'
            chk('V_runner_entry_two_phases', ok, rc=r.returncode, tail=(r.stdout + r.stderr)[-800:] if not ok else '')
        info[cond] = dict(native=ctxL.nat_metrics, a0b0=d['a0b0'])
        del ctxL, ctxG, d; torch.cuda.empty_cache()

    # ---- oracle runner (separate process) + isolation
    iso = {}
    ou = 'r36_M_n60_G_SHIELD_ORACLE_J,r36_M_n60_G_SHIELD_ORACLE_NCE,r36_M_n60_G_RELABEL_AVAILABLE_ORACLE_J'
    r = subprocess.run([PY, '-B', str(R36 / 'train_oracle.py'), '--units', ou, '--gate-out', str(GD / 'oracle'), '--gate-steps', '30'], capture_output=True, text=True)
    iso['oracle_runner_rc'] = r.returncode; iso['oracle_runner_tail'] = (r.stdout + r.stderr)[-1500:]
    ok = r.returncode == 0
    for u in ou.split(','):
        f = GD / 'oracle' / u / 'RESULT.json'
        if not f.exists():
            ok = False; continue
        rr = read_json(f); st = rr['batch_stats']['phase1']
        iso[u] = dict(guard=rr['test_guard'], rsum_step30=rr['final']['dev_retrieval']['rsum'], rsum_step0=rr['phases'][0]['curves'][0]['dev_retrieval']['rsum'],
                      batch_hash=rr['phases'][0]['identity']['batch_index_sha256'], oracle_counts={k: v for k, v in st.items() if k.startswith('oracle_')},
                      slot_grad_means=rr['final'].get('train_window', {}).get('slot_grad_means'))
        timing[u] = dict(train_seconds_per_update=rr['cost']['train_seconds'] / 30, readout_seconds=rr['final']['readout_seconds'])
        ok &= rr['test_guard']['oracle_process'] and not rr['test_guard']['blocked_attempts'] and np.isfinite(rr['final']['dev_retrieval']['rsum'])
        ok &= iso[u]['batch_hash'] == info.get('c_equiv_G_30_batch_hash')
        if 'SHIELD' in u:
            ok &= st.get('oracle_i2t_rows_with_shielded', 0) > 0 and st.get('oracle_t2i_rows_with_shielded', 0) > 0
        else:
            ok &= st.get('oracle_i2t_rows_relabelled', 0) > 0 and st.get('oracle_t2i_rows_unchanged_no_true', -1) >= 0
    chk('oracle_runner_separate_process_smoke_same_batches_as_C', ok, rc=r.returncode)
    r = subprocess.run([PY, '-B', str(R36 / 'train_unit.py'), '--units', 'r36_M_n60_G_SHIELD_ORACLE_J', '--gate-out', str(GD / 'refuse1'), '--gate-steps', '1'], capture_output=True, text=True)
    iso['non_oracle_runner_refuses_oracle_unit'] = r.returncode != 0 and 'refused in the non-oracle runner' in (r.stdout + r.stderr)
    chk('non_oracle_runner_refuses_oracle_unit', iso['non_oracle_runner_refuses_oracle_unit'])
    r = subprocess.run([PY, '-B', str(R36 / 'train_oracle.py'), '--units', 'r36_N_n60_G_J', '--gate-out', str(GD / 'refuse2'), '--gate-steps', '1'], capture_output=True, text=True)
    iso['oracle_runner_refuses_non_oracle_unit'] = r.returncode != 0 and 'refused in the oracle runner' in (r.stdout + r.stderr)
    chk('oracle_runner_refuses_non_oracle_unit', iso['oracle_runner_refuses_non_oracle_unit'])
    probe = f"import os,sys; os.environ['R36_ORACLE_PROCESS']='1'; sys.path.insert(0,{str(R36)!r})\ntry:\n    import oracle_truth; print('IMPORTED')\nexcept ImportError: print('REFUSED')"
    r = subprocess.run([PY, '-B', '-c', probe], capture_output=True, text=True)
    chk('oracle_module_refuses_foreign_entry_even_with_flag', r.stdout.strip() == 'REFUSED', stdout=r.stdout.strip())
    try:
        import importlib
        importlib.import_module('oracle_truth'); iso['import_in_gate_process'] = 'NOT_REFUSED'
    except ImportError:
        iso['import_in_gate_process'] = 'refused'
    chk('oracle_import_refused_in_gate_process', iso['import_in_gate_process'] == 'refused' and 'oracle_truth' not in sys.modules)
    probe = (f"import sys; sys.path.insert(0,{str(R36)!r}); from common import install_guards, ORACLE_RUNS, OUT; install_guards(role='trainer')\nres=[]\n"
             f"for p in (ORACLE_RUNS / 'x', OUT / 'audit' / 'AUDIT_EXPORT.json'):\n    try:\n        open(p); res.append('OPENED')\n    except PermissionError: res.append('REFUSED')\n    except FileNotFoundError: res.append('MISSING')\n"
             f"sys.path.insert(0,{str(ROOT / 'audit')!r})\ntry:\n    import audit_export; res.append('IMPORTED')\nexcept ImportError: res.append('REFUSED')\nprint(' '.join(res))")
    r = subprocess.run([PY, '-B', '-c', probe], capture_output=True, text=True)
    chk('trainer_role_refuses_oracle_runs_audit_and_eval_module', r.stdout.strip() == 'REFUSED REFUSED REFUSED', stdout=r.stdout.strip(), stderr=r.stderr[-300:])
    importers = [f.name for f in R36.glob('*.py') if f.name not in ('train_oracle.py', 'oracle_truth.py') and re.search(r'^\s*(import|from)\s+oracle_truth', f.read_text(), re.M)]
    aud = [f.name for f in R36.glob('*.py') if re.search(r'^\s*(import|from)\s+(audit_export|audit_candidate_lists)', f.read_text(), re.M)]
    chk('static_only_oracle_runner_imports_truth_no_audit_imports', not importers and not aud, importers=importers, audit_importers=aud)

    gr = guard_record()
    chk('test_guard_no_test_file_opened', gr['installed'] and not gr['blocked_attempts'], **{k: v for k, v in gr.items() if k != 'forbidden_prefixes'})
    allp = all(v['pass_'] for v in checks.values())
    code = {f.name: sha256(f) for f in sorted(R36.glob('*.py'))}
    write_json(ROOT / 'ORACLE_ISOLATION.json', dict(created_utc=utc(), job=job, ALL_PASS=all(checks[k]['pass_'] for k in checks if 'oracle' in k or 'trainer_role' in k), **iso,
                                                   design=dict(truth_module='r36/oracle_truth.py (ImportError unless R36_ORACLE_PROCESS=1 and argv[0] is train_oracle*)',
                                                               blocker='common._OracleBlocker meta-path finder in every R36 entry; trainer/oracle roles also refuse audit_export/audit_candidate_lists',
                                                               file_guard='trainer role: audit hook refuses opening OUT/oracle_runs/* and OUT/audit/*; oracle role refuses OUT/audit/*',
                                                               runner='r36/train_oracle.py (own process; refuses non-oracle units; outputs OUT/oracle_runs)'), code_sha256=code))
    write_json(ROOT / 'GPU_GATE.json', dict(ALL_PASS=allp, checks=checks, measured_timing=timing, info=info, created_utc=utc(), job=job,
                                            gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, code_sha256=code, elapsed_seconds=time.time() - t0))
    print('R36_GATE', 'ALL_PASS' if allp else 'FAIL', [k for k, v in checks.items() if not v['pass_']])
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
