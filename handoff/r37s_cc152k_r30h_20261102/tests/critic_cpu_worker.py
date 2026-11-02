"""R37S critic CPU worker (adapted from R38 tests/cpu_worker.py): short real-data trainings with r37s_critic/train_unit.py on CPU
(1 thread, deterministic) on the R37S parent cache/prep; prints one line 'R37S_WORKER_JSON <json>'. Outputs only under --tmp.
Usage: python -I -B critic_cpu_worker.py --tree DIR --tmp DIR"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--tree', required=True); ap.add_argument('--tmp', required=True)
a = ap.parse_args()
sys.path.insert(0, a.tree)
import torch
torch.set_num_threads(1)
import train_unit as TU
from common import install_guards, read_json, ROOT
install_guards(role='trainer')
TU.set_determinism()
ctx = TU.Ctx('cc_r37s', torch.device('cpu'), need_local=False)
tmp = Path(a.tmp); res = {}
P = 'r37s_cc_r30h_s1001_'


def cfg(rest):
    c = read_json(ROOT / 'configs' / f'{P}{rest}.json'); c['_config_sha256'] = None; return c


def run(name, c, steps, ckpt, readout=False):
    t0 = time.time()
    r = TU.run_unit(c, ctx, pool=TU.pool_of(c, ctx), out_dir=tmp / name, steps=steps, ckpt_steps=ckpt, no_readout=not readout, log=lambda *x: None)
    o = dict(init=r['init_hash'], final=r['final_hash'], chain=r['batch_hash'], seconds=time.time() - t0, params=int(sum(p.numel() for p in r['heads'].parameters())))
    h = r['hist'].get('r38', {})
    o['r38'] = {k: h.get(k) for k in ('schedule', 'directional', 'reference_unit', 'lr_at_readout', 'step1_graph_check', 'init_state_sha256_shared_keys')}
    if readout:
        z = np.load(tmp / name / f'dev_scores_step{steps}.npz'); z0 = np.load(tmp / name / 'dev_scores_step0.npz')
        e0 = read_json(tmp / name / 'eval_step0.json'); e1 = read_json(tmp / name / f'eval_step{steps}.json')
        d = ctx.d; tab = d['dev_tab']; heads = r['heads']
        q = np.arange(0, tab['i_top'].shape[0], 37)
        with torch.no_grad():
            def g(img, cap, dd):
                tot = 0
                for p in TU.PEERS:
                    s_ex = d['dev_s'][p][torch.as_tensor(img), torch.as_tensor(cap)].numpy()
                    tot = tot + TU.g_pairs_lists(heads[p], None, ctx, p, 'dev', img, cap, s_exact=s_ex, direction=dd)
                return (tot / 2).numpy()
            ii = np.repeat(q, 100); cc = tab['i_top'][q].ravel()
            gi0, gi1 = g(ii, cc, 0).reshape(len(q), 100), g(ii, cc, 1).reshape(len(q), 100)
            qc = np.arange(0, tab['t_top'].shape[0], 151); tim = tab['t_top'][qc].ravel(); tcc = np.repeat(qc, 100)
            gt0, gt1 = g(tim, tcc, 0).reshape(len(qc), 100), g(tim, tcc, 1).reshape(len(qc), 100)
        o['readout'] = dict(step0_native_check=e0.get('step0_native_check'), rsum0=e0['dev_retrieval']['rsum'], rsum_final=e1['dev_retrieval']['rsum'],
                            native_rsum=ctx.nat_metrics['rsum'],
                            i2t_top_maxabs_vs_dir0=float(np.abs(z['gi_top'][q] - gi0).max()), i2t_top_maxabs_vs_dir1=float(np.abs(z['gi_top'][q] - gi1).max()),
                            t2i_top_maxabs_vs_dir1=float(np.abs(z['gt_top'][qc] - gt1).max()), t2i_top_maxabs_vs_dir0=float(np.abs(z['gt_top'][qc] - gt0).max()),
                            g_gt_vs_g_gt_t2i_maxabs_final=float(np.abs(z['g_gt'] - z['g_gt_t2i']).max()), g_gt_vs_g_gt_t2i_maxabs_step0=float(np.abs(z0['g_gt'] - z0['g_gt_t2i']).max()),
                            keys=sorted(z.files))
    res[name] = o


run('N_shared_const_J_60', cfg('N_shared_J_constant'), 60, [0, 60])
run('N_shared_const_J_1', cfg('N_shared_J_constant'), 1, [0, 1])
run('N_shared_cos_J_1', cfg('N_shared_J_cosine'), 1, [0, 1])
run('N_shared_cos_J_60', cfg('N_shared_J_cosine'), 60, [0, 60])
run('T_shared_const_BCE_60', cfg('T_shared_BCE_constant'), 60, [0, 60])
run('T_shared_cos_BCE_60', cfg('T_shared_BCE_cosine'), 60, [0, 60])
run('N_dir_cos_NCE_250', cfg('N_directional_NCE_cosine'), 250, [0, 250])
run('N_dir_const_J_30_readout', cfg('N_directional_J_constant'), 30, [0, 30], readout=True)
for u in read_json(ROOT / 'configs' / 'UNITS.json'):   # every frozen config: builds, 2 updates (first-update graph check runs at update 1)
    run('all18__' + u, cfg(u[len(P):]), 2, [0, 2])
print('R37S_WORKER_JSON ' + json.dumps(res, default=float), flush=True)
