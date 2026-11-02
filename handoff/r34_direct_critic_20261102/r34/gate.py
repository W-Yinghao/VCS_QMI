"""R34 GPU gate (technical only; no outcome thresholds). Writes ROOT/GPU_GATE.json.
Checks: cache/lock hashes; native H full-DEV reproduction; GPU value/gradient vs CPU fp64; D/X forward
GPU vs CPU fp64 and X blockwise==concat on real features; J/BCE/NCE identical init + batches within a
structure; DEV isolation (DEV=NaN leaves training bitwise unchanged); determinism rerun; 12-config smoke
through the full unit path (50 updates); TEST guard (no test_* file opened) + static scan."""
from __future__ import annotations
import copy, os, re, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONDS, OUT, ROOT, PEERS, install_test_guard, guard_record, sha256, write_json, read_json, utc
import critics as C
from native_eval import upstream_metrics
from train_unit import load_cond, train, run_unit, state_hash

R32_SRC = '/home/infres/yinwang/CS_QMI/r32_native_ratio/VCS_NATIVE_RATIO_v1_Server_20261101/src'


def main():
    install_test_guard()
    dev = torch.device('cuda'); checks = {}; t0 = time.time()

    def chk(name, ok, **info):
        checks[name] = dict(pass_=bool(ok), **info); print(('PASS ' if ok else 'FAIL ') + name, info, flush=True)

    # static scan: no test-split path literals in the line's code (guard regex excepted)
    hits = []
    for f in sorted(p for p in (ROOT / 'r34').glob('*.py') if p.name != 'gate.py'):
        for i, line in enumerate(f.read_text().splitlines()):
            if re.search(r"test_(ims|caps|ids|tags)|testall|'test'|\"test\"", line) and '_TEST_RE' not in line:
                hits.append(f'{f.name}:{i+1}')
    chk('static_no_test_paths', not hits, hits=hits)

    # GPU vs CPU fp64 objective values / gradients
    torch.manual_seed(0)
    g64 = torch.randn(128, 128, dtype=torch.float64) * 3
    p64, q64 = C.pq_weights(128, 'cpu', torch.float64); p32, q32 = C.pq_weights(128, dev)
    for obj in C.OBJECTIVES:
        a = g64.clone().requires_grad_(); l64 = C.loss_of(obj, a, p64, q64); (ga,) = torch.autograd.grad(l64, a)
        b = g64.float().to(dev).requires_grad_(); l32 = C.loss_of(obj, b, p32, q32); (gb,) = torch.autograd.grad(l32, b)
        chk(f'value_grad_gpu_vs_fp64_{obj}', abs(float(l32) - float(l64)) <= 1e-5 * max(1, abs(float(l64))) and torch.allclose(gb.double().cpu(), ga, atol=1e-7, rtol=1e-4),
            loss_abs_diff=abs(float(l32) - float(l64)), grad_max_abs_diff=float((gb.double().cpu() - ga).abs().max()))
    sys.path.insert(1, R32_SRC)
    from native_ratio.objectives import original_j, inbatch_pair_weights
    P, Q = inbatch_pair_weights(128, device=dev)
    gg = g64.float().to(dev)
    chk('J_equals_R32_original_j_gpu', abs(float(C.j_value(gg, p32, q32)) - float(original_j(torch.tanh(gg / 2), P, Q))) < 1e-6)

    cond_info = {}
    for cond in CONDS:
        d = load_cond(cond, dev)  # asserts every cache/lock sha256
        man = d['manifest']; nat = man['native_dev']
        chk(f'{cond}_cache_and_lock_hashes', True, n_files=len(man['files']))
        chk(f'{cond}_native_H_dev_reproduction', nat['cache_rescore_identical'] and nat['abs_diff_vs_checkpoint'] <= 0.1,
            rsum=nat['rsum'], checkpoint_best_rsum=nat['checkpoint_best_rsum'], exact=nat['reproduces_checkpoint_best_rsum_exactly'],
            peer_rsum=nat['peer_rsum'], cache_rescore_identical=nat['cache_rescore_identical'])
        up, _, _ = upstream_metrics(d['native_dev'])
        chk(f'{cond}_native_sims_file_metrics', abs(up['rsum'] - nat['rsum']) < 1e-9, rsum=up['rsum'], native_rsum=nat['rsum'])  # sum-order float: 1e-9, was exact ==
        # real-feature forward GPU fp32 vs CPU fp64, X blockwise vs concat
        idx = torch.arange(0, 64 * 5, 5, device=dev)
        x = d[('fit', 'img', 'A')][d['obs'][idx]]; y = d[('fit', 'cap', 'A')][idx]
        for s in C.STRUCTURES:
            torch.manual_seed(1001); m = C.build(s).to(dev); m64 = copy.deepcopy(m).double().cpu()
            g32 = m.pair_scores(x, y); g64r = m64.pair_scores(x.double().cpu(), y.double().cpu())
            chk(f'{cond}_{s}_forward_gpu_vs_fp64', torch.allclose(g32.double().cpu(), g64r, atol=1e-4, rtol=1e-4), max_abs=float((g32.double().cpu() - g64r).abs().max()))
            if s == 'X':
                gc = m64.concat_scores(x.double().cpu(), y.double().cpu())
                chk(f'{cond}_X_blockwise_equals_concat', torch.allclose(g64r, gc, atol=1e-10), max_abs=float((g64r - gc).abs().max()))
        # init/batch identity across objectives, determinism, DEV isolation (short)
        for s in C.STRUCTURES:
            res = {}
            for o in C.OBJECTIVES:
                cfg = read_json(ROOT / 'configs' / f"direct_{cond}_s1001_{'D_dual_cosine' if s == 'D' else 'X_joint_mlp'}_{'InfoNCE' if o == 'NCE' else o}.json")
                r = train(cfg, d, dev, steps=30, eval_every=15, with_rank=False, log=lambda *a: None)
                res[o] = r
            chk(f'{cond}_{s}_identical_init_and_batches', len({r['init_hash'] for r in res.values()}) == 1 and len({r['batch_hash'] for r in res.values()}) == 1
                and len({r['final_hash'] for r in res.values()}) == 3, init=res['J']['init_hash'][:16], batches=res['J']['batch_hash'][:16])
            cfg = read_json(ROOT / 'configs' / f"direct_{cond}_s1001_{'D_dual_cosine' if s == 'D' else 'X_joint_mlp'}_J.json")
            r2 = train(cfg, d, dev, steps=30, eval_every=15, with_rank=False, log=lambda *a: None)
            r3 = train(cfg, d, dev, steps=30, eval_every=15, with_rank=False, dev_nan=True, log=lambda *a: None)
            chk(f'{cond}_{s}_determinism_rerun', r2['final_hash'] == res['J']['final_hash'])
            chk(f'{cond}_{s}_dev_isolation_nan_dev', r3['final_hash'] == res['J']['final_hash'],
                nan_dev_readout_J=r3['curves'][-1]['dev_clean']['J_mean_of_peers'])
        # smoke: 6 units of the condition through the full unit path, 50 updates
        smoke = {}
        for u in [x for x in read_json(ROOT / 'configs' / 'UNITS.json') if f'_{cond}_' in x]:
            cfg = dict(read_json(ROOT / 'configs' / f'{u}.json')); cfg['steps'] = 50; cfg['eval_every'] = 25
            out = OUT / 'gate' / os.environ.get('SLURM_JOB_ID', 'local')
            r = run_unit(cfg, d, dev, log=lambda *a: None, runs_dir=out)
            fin = all(np.isfinite(v) for v in (r['dev_retrieval_step2000']['rsum'], r['J_step2000']['fit_observed']['J_mean_of_peers'],
                                               r['J_step2000']['dev_clean']['J_mean_of_peers']))
            smoke[u] = dict(rsum_step50=r['dev_retrieval_step2000']['rsum'], wall_s=r['cost']['unit_wall_seconds'],
                            train_s_50=r['cost']['train_seconds_2000_updates'], peak_mem_gb=r['cost']['peak_gpu_memory_bytes'] / 2**30,
                            readout_s=r['curves'][-1]['readout_seconds'], params=r['cost']['params'], finite=fin)
            chk(f'{cond}_smoke_{u}', fin and r['test_guard']['blocked_attempts'] == [], **smoke[u])
        cond_info[cond] = dict(native=nat, smoke=smoke)
        del d; torch.cuda.empty_cache()
    gr = guard_record()
    chk('test_guard_no_test_file_opened', gr['installed'] and not gr['blocked_attempts'], **gr)
    allp = all(v['pass_'] for v in checks.values())
    write_json(ROOT / 'GPU_GATE.json', dict(ALL_PASS=allp, checks=checks, conditions=cond_info, created_utc=utc(),
                                            job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(0), host=os.uname().nodename,
                                            code_sha256={f.name: sha256(f) for f in sorted((ROOT / 'r34').glob('*.py'))},
                                            elapsed_seconds=time.time() - t0))
    print('R34_GATE', 'ALL_PASS' if allp else 'FAIL', [k for k, v in checks.items() if not v['pass_']])
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
