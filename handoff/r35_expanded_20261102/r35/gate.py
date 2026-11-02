"""R35 GPU gate on real data (technical only; no outcome thresholds). Writes ROOT/GPU_GATE.json and
ROOT/ORACLE_ISOLATION.json. Checks: static scan; TEST guard (functional, subprocess) and no test file opened;
objective values/gradients GPU fp32 vs CPU fp64; R34 cache/prep hashes; per-peer native shard reproduction of the
native DEV score; step-0 native equivalence (A-ID, A-G, C-G, C-L: ranks identical to frozen H); D step 0 ==
C-L parent scores (bitwise); identical init + batches across objectives; determinism rerun; DEV=NaN isolation;
checkpoint resume == uninterrupted (A-G, C-L, D joint); smoke of every structure/supervision path incl. B-ORACLE
in a separate process; oracle refusals."""
from __future__ import annotations
import copy, os, re, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, OUT, PEERS, DATA_SRC, N_TRAIN_CAP, install_guards, guard_record, assert_no_oracle_loaded, sha256, write_json, read_json, utc
import objectives as O
import evals as EV
from heads import build_heads
import train_unit as T

PY = sys.executable
R35 = ROOT / 'r35'


def main():
    install_guards(); assert_no_oracle_loaded(); T.set_determinism()
    job = os.environ.get('SLURM_JOB_ID', 'local')
    GD = OUT / 'gate' / job; GD.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda'); checks = {}; info = {}; t0 = time.time()

    def chk(name, ok, **kw):
        checks[name] = dict(pass_=bool(ok), **kw); print(('PASS ' if ok else 'FAIL ') + name, {k: v for k, v in kw.items() if not isinstance(v, (list, dict)) or len(str(v)) < 400}, flush=True)

    def guarded(name, fn):
        try:
            fn()
        except Exception:
            chk(name, False, error=traceback.format_exc())

    # ---- static scan
    hits = []
    for f in sorted(R35.glob('*.py')):
        if f.name == 'gate.py':
            continue
        for i, line in enumerate(f.read_text().splitlines()):
            if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line:
                hits.append(f'{f.name}:{i+1}')
    chk('static_no_test_paths', not hits, hits=hits)
    # ---- TEST guard functional (subprocess; the hook raises before any byte is read)
    probe = (f"import sys; sys.path.insert(0, {str(R35)!r}); from common import install_guards; install_guards()\n"
             f"try:\n    open({str(DATA_SRC / ('te' + 'st_caps.txt'))!r}); print('OPENED')\nexcept PermissionError: print('BLOCKED')")
    r = subprocess.run([PY, '-B', '-c', probe], capture_output=True, text=True)
    chk('test_guard_functional_subprocess', r.stdout.strip() == 'BLOCKED', stdout=r.stdout.strip(), stderr=r.stderr[-500:])

    # ---- objectives GPU fp32 vs CPU fp64
    torch.manual_seed(0)
    g64 = torch.randn(128, 128, dtype=torch.float64) * 3; p64, q64 = O.pq_weights(128, 'cpu', torch.float64); p32, q32 = O.pq_weights(128, dev)
    a64 = torch.zeros(16, 32, dtype=torch.float64); a64[:, 0] = .5; a64[:, 7] = .5; l64 = torch.randn(16, 32, dtype=torch.float64) * 3
    for obj in O.OBJECTIVES:
        a = g64.clone().requires_grad_(); la = O.loss_records(obj, a, p64, q64); (ga,) = torch.autograd.grad(la, a)
        b = g64.float().to(dev).requires_grad_(); lb = O.loss_records(obj, b, p32, q32); (gb,) = torch.autograd.grad(lb, b)
        c = l64.clone().requires_grad_(); lc = O.loss_list(obj, c, a64); (gc,) = torch.autograd.grad(lc, c)
        d_ = l64.float().to(dev).requires_grad_(); ld = O.loss_list(obj, d_, a64.float().to(dev)); (gd,) = torch.autograd.grad(ld, d_)
        ok = abs(float(lb) - float(la)) <= 1e-5 * max(1, abs(float(la))) and torch.allclose(gb.double().cpu(), ga, atol=1e-7, rtol=1e-4) \
            and abs(float(ld) - float(lc)) <= 1e-5 * max(1, abs(float(lc))) and torch.allclose(gd.double().cpu(), gc, atol=1e-6, rtol=1e-4)
        chk(f'objective_gpu_vs_fp64_{obj}', ok, records_abs=abs(float(lb) - float(la)), lists_abs=abs(float(ld) - float(lc)))

    cfgs = {u: T.cfg_of(u) for u in read_json(ROOT / 'configs' / 'UNITS.json')}
    timing = {}

    def short(uid, steps, **kw):
        c = copy.deepcopy(cfgs[uid]); c.update(kw); c['steps'] = steps; c['ckpt_steps'] = [0, steps]; c['_config_sha256'] = 'gate_override'
        return c

    def run(c, ctx, out, **kw):
        sup = c['supervision']
        pool = ctx.d['trust_subset'] if sup == 'TRUST' else np.arange(N_TRAIN_CAP)
        return T.run_unit(c, ctx, imap=ctx.d['obs_np'], pool=pool, out_dir=GD / out, log=lambda *a: None, **kw)

    for cond in ('n00', 'n60'):
        ctxL = T.Ctx(cond, dev, need_local=True)          # asserts every R34 cache and R35 prep hash
        ctxG = T._shallow(ctxL, need_local=False)
        d = ctxL.d
        chk(f'{cond}_hashes_R34_cache_and_prep', True, cache_manifest=d['manifest_sha256'], prep_manifest=d['prep_manifest_sha256'])
        nat_mean = ((d['dev_s']['A'] + d['dev_s']['B']) / 2).cpu().numpy()
        TOL = 1e-6   # explicit cross-hardware tolerance (R34 caches built on L40S; fp32 shard matmul may differ in the last bits elsewhere)
        mx = float(np.abs(nat_mean - d['native_dev']).max())
        chk(f'{cond}_per_peer_native_shard_reproduces_native_dev', mx <= TOL, bitwise=bool(np.array_equal(nat_mean, d['native_dev'])), max_abs=mx, tol=TOL,
            gpu=torch.cuda.get_device_name(0), native_rsum=ctxL.nat_metrics['rsum'], manifest_rsum=d['manifest']['native_dev']['rsum'])
        tb_nat = EV.tie_bounded(d['native_dev'])
        # ---- step-0 native equivalence
        a0, b0 = d['a0b0']['a0'], d['a0b0']['b0']
        for st, lists in (('ID', False), ('G', False), ('G', True), ('L', True)):
            heads = build_heads(st, a0, b0, 1001).to(dev)
            if lists:
                encs = torch.nn.ModuleDict({p: T._copy_enc(ctxL.enc0[p]) for p in PEERS}) if st == 'L' else None
                rec, arrs, rk = T.readout_lists(heads, encs, ctxL if st == 'L' else ctxG, final=True)
                name = f'{cond}_C{st}_step0_two_stage_equals_native'
            else:
                rec, s_np, rk = T.readout_records(heads, ctxG, final=True)
                name = f'{cond}_A{st}_step0_full_dev_equals_native'
            nat = d['native_dev']
            di = np.nonzero(rk['ranks_i'] != ctxL.nat_rk['ranks_i'])[0]; dt = np.nonzero(rk['ranks_t'] != ctxL.nat_rk['ranks_t'])[0]
            ni, nt = len(di), len(dt)
            # only queries whose native GT score is tied with a competitor within TOL (exact ties, or cross-hardware
            # last-bit differences of the per-peer skip score) may move; counts recorded
            tied_i = all(any((np.abs(nat[i] - nat[i, 5 * i + k]) <= TOL).sum() > 1 for k in range(5)) for i in di)
            tied_t = all((np.abs(nat[:, c] - nat[c // 5, c]) <= TOL).sum() > 1 for c in dt)
            ok = (ni == 0 and nt == 0) or (tied_i and tied_t)
            extra = dict(rsum=rec['dev_retrieval']['rsum'], native_rsum=ctxL.nat_metrics['rsum'], n_i2t_rank_diff=ni, n_t2i_rank_diff=nt, tol=TOL,
                         native_gt_ties=(tb_nat['n_i2t_queries_gt_tied'], tb_nat['n_t2i_queries_gt_tied']), readout_s=rec['readout_seconds'])
            if not lists:
                extra['max_abs_vs_a0_native_plus_b0'] = float(np.abs(s_np - (a0 * d['native_dev'] + b0)).max())
            else:
                extra['coverage'] = rec['candidate_coverage']
            chk(name, ok, **extra)
            del heads
        # ---- identical init / batches across objectives; determinism; DEV isolation; resume
        for st, base in (('G', f'r35_A_{cond}_s1001_G'), ('L', f'r35_C_{cond}_s1001_L')):
            ctx = ctxL if st == 'L' else ctxG
            res = {o: run(short(f'{base}_{o}', 20), ctx, f'{cond}_{st}_obj_{o}', no_readout=True) for o in O.OBJECTIVES}
            chk(f'{cond}_{st}_identical_init_and_batches_across_objectives',
                len({r['init_hash'] for r in res.values()}) == 1 and len({r['batch_hash'] for r in res.values()}) == 1 and len({r['final_hash'] for r in res.values()}) == 3,
                init=res['J']['init_hash'][:16], batches=res['J']['batch_hash'][:16])
            r2 = run(short(f'{base}_J', 20), ctx, f'{cond}_{st}_rerun', no_readout=True)
            chk(f'{cond}_{st}_determinism_rerun', r2['final_hash'] == res['J']['final_hash'])
            r3 = run(short(f'{base}_J', 20), ctx, f'{cond}_{st}_devnan', dev_nan=True)
            chk(f'{cond}_{st}_dev_isolation_nan_dev', r3['final_hash'] == res['J']['final_hash'])
            full = run(short(f'{base}_J', 40), ctx, f'{cond}_{st}_res_full', ckpt_steps=[0, 20, 40], no_readout=True)
            run(short(f'{base}_J', 40), ctx, f'{cond}_{st}_res_split', ckpt_steps=[0, 20, 40], stop_at=20, no_readout=True)
            rs = run(short(f'{base}_J', 40), ctx, f'{cond}_{st}_res_split', ckpt_steps=[0, 20, 40], no_readout=True)
            chk(f'{cond}_{st}_resume_equals_uninterrupted', rs['final_hash'] == full['final_hash'] and rs['batch_hash'] == full['batch_hash'])
        # ---- D step 0 == C-L parent (bitwise), D determinism/resume, D smoke
        par = run(short(f'r35_C_{cond}_s1001_L_J', 40), ctxL, f'{cond}_D_parent')
        pck = str(GD / f'{cond}_D_parent' / 'ckpt_step40.pt')
        pz = np.load(GD / f'{cond}_D_parent' / 'dev_scores_step40.npz')
        for mode, uid in (('joint', f'r35_D_{cond}_s1001_L_J'), ('head_control', f'r35_D_{cond}_s1001_L_J_HEAD_CONTROL')):
            c = short(uid, 20, parent_ckpt=pck, parent_step=40, parent_unit=f'r35_C_{cond}_s1001_L_J')
            r = run(c, ctxL, f'{cond}_D_{mode}')
            z = np.load(GD / f'{cond}_D_{mode}' / 'dev_scores_step0.npz')
            eq = all(np.array_equal(z[k], pz[k]) for k in ('gi_top', 'gt_top', 'g_gt'))
            chk(f'{cond}_D_{mode}_step0_equals_C_L_parent_bitwise', eq, max_abs=float(max(np.abs(z[k] - pz[k]).max() for k in ('gi_top', 'gt_top', 'g_gt'))))
            if mode == 'joint':
                r2 = run(c, ctxL, f'{cond}_D_joint_rerun', no_readout=True)
                chk(f'{cond}_D_joint_determinism_rerun', r2['final_hash'] == r['final_hash'])
                c40 = short(uid, 40, parent_ckpt=pck, parent_step=40, parent_unit=f'r35_C_{cond}_s1001_L_J')
                f1 = run(c40, ctxL, f'{cond}_D_res_full', ckpt_steps=[0, 20, 40], no_readout=True)
                run(c40, ctxL, f'{cond}_D_res_split', ckpt_steps=[0, 20, 40], stop_at=20, no_readout=True)
                f2 = run(c40, ctxL, f'{cond}_D_res_split', ckpt_steps=[0, 20, 40], no_readout=True)
                chk(f'{cond}_D_joint_resume_equals_uninterrupted', f1['final_hash'] == f2['final_hash'] and f1['batch_hash'] == f2['batch_hash'])
                enc_changed = any(not torch.equal(dict(r['encs'][p].named_parameters())[n], dict(ctxL.enc0[p].named_parameters())[n])
                                  for p in PEERS for n in ('fc.weight', 'rnn.weight_hh_l0'))
                emb_same = all(torch.equal(r['encs'][p].embed.weight, ctxL.enc0[p].embed.weight) for p in PEERS)
                bn_same = all(torch.equal(r['encs'][p].mlp.bns[0].running_mean, ctxL.enc0[p].mlp.bns[0].running_mean) for p in PEERS)
                chk(f'{cond}_D_joint_updates_local_encoder_only_unfrozen', enc_changed and emb_same and bn_same)
            else:
                same = all(torch.equal(q, q0) for p in PEERS for q, q0 in zip(r['encs'][p].parameters(), ctxL.enc0[p].parameters()))
                chk(f'{cond}_D_head_control_encoder_unchanged', same)
        # ---- smoke + measured per-update time of every structure/supervision path (30 updates, real readouts)
        uids = [u for u, c in cfgs.items() if c['cond'] == cond and c['objective'] == 'J' and not c['oracle'] and c['block'] in ('A', 'B', 'C')]
        for u in uids:
            c = short(u, 30); ctx = ctxL if c['structure'] == 'L' else ctxG
            torch.cuda.reset_peak_memory_stats(dev)
            r = run(c, ctx, f'smoke_{u}')
            h = r['hist']; ev = read_json(GD / f'smoke_{u}' / 'eval_step30.json')
            timing[u] = dict(train_seconds_per_update=h['train_seconds'] / 30, readout_seconds=ev['readout_seconds'],
                             peak_gpu_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30, rsum_step30=ev['dev_retrieval']['rsum'], stats=h['stats_sum'])
            chk(f'smoke_{u}', np.isfinite(ev['dev_retrieval']['rsum']), **timing[u])
        for u, mode in ((f'r35_D_{cond}_s1001_L_J', 'joint'), (f'r35_D_{cond}_s1001_L_J_HEAD_CONTROL', 'head_control')):
            timing[u] = dict(train_seconds_per_update=read_json(GD / f'{cond}_D_{mode}' / 'TRAIN_HISTORY.json')['train_seconds'] / 20,
                             readout_seconds=read_json(GD / f'{cond}_D_{mode}' / 'eval_step20.json')['readout_seconds'])
        if cond == 'n60':
            info['trust'] = read_json(OUT / 'prep' / 'PREP_MANIFEST.json').get('trust_n60')
        info[cond] = dict(native=ctxL.nat_metrics, a0b0=d['a0b0'])
        del ctxL, ctxG, d; torch.cuda.empty_cache()

    # ---- oracle isolation (separate processes)
    iso = {}
    env = dict(os.environ)
    r = subprocess.run([PY, '-B', str(R35 / 'train_oracle.py'), '--units', 'r35_B_n60_s1001_ORACLE_J', '--gate-out', str(GD / 'oracle'), '--gate-steps', '30'],
                       capture_output=True, text=True, env=env)
    iso['oracle_runner_smoke_rc'] = r.returncode; iso['oracle_runner_tail'] = r.stdout[-1500:] + r.stderr[-1500:]
    ok_smoke = r.returncode == 0 and (GD / 'oracle' / 'r35_B_n60_s1001_ORACLE_J' / 'RESULT.json').exists()
    if ok_smoke:
        rr = read_json(GD / 'oracle' / 'r35_B_n60_s1001_ORACLE_J' / 'RESULT.json')
        iso['oracle_smoke_guard'] = rr['test_guard']; iso['oracle_smoke_rsum_step30'] = rr['final']['dev_retrieval']['rsum']
        iso['oracle_fit_truth_J_step30'] = rr['final'].get('fit_truth_J_raw')
        ok_smoke = rr['test_guard']['oracle_process'] and not rr['test_guard']['blocked_attempts']
        timing['r35_B_n60_s1001_ORACLE_J'] = dict(train_seconds_per_update=rr['cost']['train_seconds'] / 30, readout_seconds=rr['final']['readout_seconds'])
    chk('oracle_runner_separate_process_smoke', ok_smoke, rc=r.returncode)
    r = subprocess.run([PY, '-B', str(R35 / 'train_unit.py'), '--units', 'r35_B_n60_s1001_ORACLE_J', '--gate-out', str(GD / 'refuse1'), '--gate-steps', '1'], capture_output=True, text=True)
    iso['non_oracle_runner_refuses_oracle_unit'] = r.returncode != 0 and 'refused in the non-oracle runner' in (r.stdout + r.stderr)
    chk('non_oracle_runner_refuses_oracle_unit', iso['non_oracle_runner_refuses_oracle_unit'])
    r = subprocess.run([PY, '-B', str(R35 / 'train_oracle.py'), '--units', 'r35_A_n60_s1001_G_J', '--gate-out', str(GD / 'refuse2'), '--gate-steps', '1'], capture_output=True, text=True)
    iso['oracle_runner_refuses_non_oracle_unit'] = r.returncode != 0 and 'refused in the oracle runner' in (r.stdout + r.stderr)
    chk('oracle_runner_refuses_non_oracle_unit', iso['oracle_runner_refuses_non_oracle_unit'])
    probe = f"import os,sys; os.environ['R35_ORACLE_PROCESS']='1'; sys.path.insert(0,{str(R35)!r})\ntry:\n    import oracle_truth; print('IMPORTED')\nexcept ImportError: print('REFUSED')"
    r = subprocess.run([PY, '-B', '-c', probe], capture_output=True, text=True)
    iso['oracle_module_refuses_foreign_entry_even_with_flag'] = r.stdout.strip() == 'REFUSED'
    chk('oracle_module_refuses_foreign_entry', iso['oracle_module_refuses_foreign_entry_even_with_flag'], stdout=r.stdout.strip())
    try:
        import importlib
        importlib.import_module('oracle_truth')   # negative test: must be refused in this (non-oracle) process
        iso['import_in_gate_process'] = 'NOT_REFUSED'
    except ImportError as e:
        iso['import_in_gate_process'] = 'refused'
    chk('oracle_import_refused_in_gate_process', iso['import_in_gate_process'] == 'refused' and 'oracle_truth' not in sys.modules)
    importers = [f.name for f in R35.glob('*.py') if f.name not in ('train_oracle.py', 'oracle_truth.py') and re.search(r'^\s*(import|from)\s+oracle_truth', f.read_text(), re.M)]
    iso['static_importers_of_oracle_truth'] = importers
    chk('static_only_oracle_runner_imports_truth', not importers, importers=importers)
    if (OUT / 'runs' / 'ORACLE_DIAGNOSTICS.json').exists():
        iso['oracle_diagnostics'] = read_json(OUT / 'runs' / 'ORACLE_DIAGNOSTICS.json')

    gr = guard_record()
    chk('test_guard_no_test_file_opened', gr['installed'] and not gr['blocked_attempts'], **gr)
    allp = all(v['pass_'] for v in checks.values())
    code = {f.name: sha256(f) for f in sorted(R35.glob('*.py'))}
    write_json(ROOT / 'ORACLE_ISOLATION.json', dict(created_utc=utc(), job=job, ALL_PASS=all(checks[k]['pass_'] for k in checks if 'oracle' in k), **iso,
                                                   design=dict(truth_module='r35/oracle_truth.py (raises ImportError unless R35_ORACLE_PROCESS=1 and argv[0] is train_oracle*)',
                                                               blocker='common._OracleBlocker meta-path finder installed by every R35 entry; non-oracle runner asserts the module is not loaded and refuses ORACLE units',
                                                               runner='r35/train_oracle.py (own process; refuses non-ORACLE units)'), code_sha256=code))
    write_json(ROOT / 'GPU_GATE.json', dict(ALL_PASS=allp, checks=checks, measured_timing=timing, info=info, created_utc=utc(), job=job,
                                            gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, code_sha256=code, elapsed_seconds=time.time() - t0))
    print('R35_GATE', 'ALL_PASS' if allp else 'FAIL', [k for k, v in checks.items() if not v['pass_']])
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
