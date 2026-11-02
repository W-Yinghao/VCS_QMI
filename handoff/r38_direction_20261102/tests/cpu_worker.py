"""R38 CPU worker: short real-data trainings with the train_unit.py of one code tree (--tree DIR) on CPU (1 thread, deterministic),
prints one line 'R38_WORKER_JSON <json>'. --mode base: the R37 / R37C tree (imported read-only, nothing written there) running
the R36/R37C reference configs; --mode r38: the r38f / r38c tree running the same reference configs (shared/constant path) and
actual R38 unit configs (directional / cosine). Outputs only under --tmp.
Usage: python -I -B cpu_worker.py --tree DIR --mode base|r38 --cond COND --refcfg DIR --refprefix PREFIX --tmp DIR [--r38cfg DIR --r38prefix P]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
for k in ('--tree', '--mode', '--cond', '--refcfg', '--refprefix', '--tmp'):
    ap.add_argument(k, required=True)
ap.add_argument('--r38cfg', default=''); ap.add_argument('--r38prefix', default='')
a = ap.parse_args()
sys.path.insert(0, a.tree)
import torch
torch.set_num_threads(1)
import train_unit as TU
from common import install_guards, read_json
install_guards(role='trainer')
TU.set_determinism()
ctx = TU.Ctx(a.cond, torch.device('cpu'), need_local=False)
tmp = Path(a.tmp); res = {}


def ref_cfg(route, obj):
    c = read_json(Path(a.refcfg) / (a.refprefix.format(r=route) + f'G_{obj}.json')); c['_config_sha256'] = None
    return c


def r38_cfg(rest, keep_ref=False):
    c = read_json(Path(a.r38cfg) / (a.r38prefix.format(b='C' if rest.startswith('T_') else 'B') + f'{rest}.json')); c['_config_sha256'] = None
    if not keep_ref:
        c['reference_unit'] = None
    return c


def run(name, cfg, steps, ckpt, readout=False):
    t0 = time.time()
    r = TU.run_unit(cfg, ctx, pool=TU.pool_of(cfg, ctx), out_dir=tmp / name, steps=steps, ckpt_steps=ckpt, no_readout=not readout, log=lambda *x: None)
    o = dict(init=r['init_hash'], final=r['final_hash'], chain=r['batch_hash'], hist_chain=r['hist']['batch_hash'], seconds=time.time() - t0,
             params=int(sum(p.numel() for p in r['heads'].parameters())))
    if 'r38' in r['hist']:
        h = r['hist']['r38']
        o['r38'] = {k: h.get(k) for k in ('schedule', 'directional', 'reference_unit', 'reference_init_equal', 'reference_chain_equal_at', 'lr_at_readout',
                                          'step1_graph_check', 'init_state_sha256_shared_keys')}
        o['r38']['dir_grad_diag_steps'] = sorted(h.get('dir_grad_diag', {}))
    if readout:   # directional DEV readout: per-direction scores vs an independent recomputation from the final heads
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
                            i2t_top_maxabs_vs_dir0=float(np.abs(z['gi_top'][q] - gi0).max()), i2t_top_maxabs_vs_dir1=float(np.abs(z['gi_top'][q] - gi1).max()),
                            t2i_top_maxabs_vs_dir1=float(np.abs(z['gt_top'][qc] - gt1).max()), t2i_top_maxabs_vs_dir0=float(np.abs(z['gt_top'][qc] - gt0).max()),
                            g_gt_vs_g_gt_t2i_maxabs_final=float(np.abs(z['g_gt'] - z['g_gt_t2i']).max()), g_gt_vs_g_gt_t2i_maxabs_step0=float(np.abs(z0['g_gt'] - z0['g_gt_t2i']).max()),
                            heads_ab=e1.get('heads_ab'), heads_ab_t2i=e1.get('heads_ab_t2i'), keys=sorted(z.files))
    res[name] = o


if a.mode == 'amend':   # user amendment: new parents (Flickr R37 s1002/s1003 conds in r38f; CC s1002/s1003 T in r38c)
    keep = (Path(a.refcfg) / 'UNITS.json').exists() and a.refprefix != 'none'
    run('N_dir_cos_NCE_250', r38_cfg('N_directional_NCE_cosine', keep_ref=keep), 250, [0, 250])
    run('T_shared_cos_BCE_60', r38_cfg('T_shared_BCE_cosine'), 60, [0, 60])   # 60 updates: reference chain (250-step grid) not comparable
    print('R38_WORKER_JSON ' + json.dumps(res, default=float), flush=True); sys.exit(0)
run('N_shared_const_J_60', ref_cfg('N', 'J'), 60, [0, 60])
run('T_shared_const_BCE_60', ref_cfg('T', 'BCE'), 60, [0, 60])
run('N_shared_const_J_1', ref_cfg('N', 'J'), 1, [0, 1])
if a.mode == 'r38':
    run('N_shared_cos_J_1', r38_cfg('N_shared_J_cosine'), 1, [0, 1])
    run('N_shared_cos_J_60', r38_cfg('N_shared_J_cosine'), 60, [0, 60])
    run('T_shared_cos_BCE_60', r38_cfg('T_shared_BCE_cosine'), 60, [0, 60])
    run('N_dir_cos_NCE_250_ref', r38_cfg('N_directional_NCE_cosine', keep_ref=True), 250, [0, 250])
    run('N_dir_const_J_30_readout', r38_cfg('N_directional_J_constant'), 30, [0, 30], readout=True)
print('R38_WORKER_JSON ' + json.dumps(res, default=float), flush=True)
