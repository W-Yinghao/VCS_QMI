"""CPU simulation of the R35 trainer code paths (code check only, synthetic data, shrunken DEV): A-ID/A-G/B-TRUST
records, C-G/C-L lists, D joint/head-control from a C-L parent, step-0 native equivalence, D step0 == parent,
resume == uninterrupted, determinism. Not evidence about the real experiment. Usage: python -B cpu_sim.py"""
import sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'r35'))
import numpy as np
import torch
import common as CM
import evals as EV
import data35 as D35
import train_unit as T

NI, NC = 200, 1000                    # shrunken DEV
EV.N_DEV_IMG, EV.N_DEV_CAP = NI, NC; EV.DEV_GT = np.arange(NC) // 5
T.N_DEV_IMG, T.N_DEV_CAP = NI, NC


def unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def fake_ctx(local):
    torch.manual_seed(0); rng = np.random.default_rng(0); dev = torch.device('cpu')
    d = dict(cond='n60', manifest=dict(parent={'unit': 'fake'}), manifest_sha256='x', prep_manifest_sha256='y', cand_sha256='z')
    obs = np.arange(CM.N_TRAIN_CAP) // 5; m = rng.random(CM.N_TRAIN_CAP) < .6; obs[m] = rng.permutation(obs[m])
    d['obs_np'] = obs; d['obs'] = torch.from_numpy(obs)
    base_i = torch.randn(CM.N_TRAIN_IMG, 64)
    for p in CM.PEERS:
        W = torch.randn(64, 1024)
        d[('fit', 'img', p)] = unit(base_i @ W + .5 * torch.randn(CM.N_TRAIN_IMG, 1024))
        d[('fit', 'cap', p)] = unit(base_i[np.arange(CM.N_TRAIN_CAP) // 5] @ W + .8 * torch.randn(CM.N_TRAIN_CAP, 1024))
        bi = torch.randn(NI, 64)
        d[('dev', 'img', p)] = unit(bi @ W + .5 * torch.randn(NI, 1024)); d[('dev', 'cap', p)] = unit(bi[np.arange(NC) // 5] @ W + 6.0 * torch.randn(NC, 1024))
    d['dev_s'] = {p: D35.shard_mm(d[('dev', 'img', p)], d[('dev', 'cap', p)]) for p in CM.PEERS}
    d['native_dev'] = ((d['dev_s']['A'] + d['dev_s']['B']) / 2).numpy()
    d['dev_obs'] = np.arange(NC) // 5; d['dev_clean'] = np.arange(NC) // 5
    d['dev_tab'] = EV.dev_tables(d['native_dev'])
    d['a0b0'] = dict(a0=12.0, b0=-3.0)
    perm = rng.permutation(CM.N_TRAIN_CAP)
    d['locks'] = dict(cal_p=perm[:500], ev_p=perm[500:1000], cal_q=rng.integers(0, CM.N_TRAIN_CAP, (5000, 2)), ev_q=rng.integers(0, CM.N_TRAIN_CAP, (5000, 2)),
                      lcal_i=perm[1000:1064], lcal_t=perm[1064:1128], lev_i=perm[1128:1192], lev_t=perm[1192:1256])
    for k in ('lcal_i', 'lcal_t', 'lev_i', 'lev_t'):
        d['locks'][k + '_perm'] = np.stack([rng.permutation(32) for _ in range(64)])
    xs = [d[('fit', 'img', p)] for p in CM.PEERS]; ys = [d[('fit', 'cap', p)] for p in CM.PEERS]
    d['cand_i2t'] = np.concatenate([(sum(x[i:i + 2000] @ y.T for x, y in zip(xs, ys))).topk(32, 1).indices.numpy() for i in range(0, CM.N_TRAIN_IMG, 2000)])
    d['cand_t2i'] = np.concatenate([(sum(y[j:j + 8000] @ x.T for x, y in zip(xs, ys))).topk(32, 1).indices.numpy() for j in range(0, CM.N_TRAIN_CAP, 8000)])
    d['trust_subset'] = np.sort(rng.choice(CM.N_TRAIN_CAP, 60000, replace=False))
    ctx = T.Ctx.__new__(T.Ctx); ctx.cond = 'n60'; ctx.dev = dev; ctx.d = d; ctx.fit_maps = {'observed': obs}
    ctx.pos_i, ctx.pos_t, ctx.ai_dev, ctx.at_dev = EV.dev_list_positions(d['dev_tab'])
    ctx.need_local = local
    nat, _, rk = EV.full_metrics(d['native_dev']); ctx.nat_metrics, ctx.nat_rk = nat, rk
    if local:
        R = 4
        ctx.raw = dict(train_ims=torch.randn(CM.N_TRAIN_IMG, R, 2048) * .05, dev_ims=torch.randn(NI, R, 2048) * .05,
                       train_tok=torch.randint(1, 8000, (CM.N_TRAIN_CAP, 9)), train_len=torch.from_numpy(rng.integers(3, 10, CM.N_TRAIN_CAP)),
                       dev_tok=torch.randint(1, 8000, (NC, 9)), dev_len=torch.from_numpy(rng.integers(3, 10, NC)))
        ctx.enc0, ctx.enc_info = D35.load_local_encoders('n60', dev)
    return ctx


def main():
    CM.install_guards(); torch.set_num_threads(8); T.set_determinism()
    tmp = Path(tempfile.mkdtemp(prefix='r35sim_', dir='/tmp/claude-34987' if Path('/tmp/claude-34987').exists() else None))
    out = {}
    cG = fake_ctx(False); t0 = time.time()

    def cfg(u, steps, **kw):
        c = T.cfg_of(u); c.update(kw); c['steps'] = steps; c['ckpt_steps'] = [0, steps]; c['_config_sha256'] = 'sim'; return c

    def step0(c, ctx, name):
        r = T.run_unit(cfg(c, 2), ctx, imap=ctx.d['obs_np'], pool=np.arange(CM.N_TRAIN_CAP), out_dir=tmp / name, log=lambda *a: None)
        ev = CM.read_json(tmp / name / 'eval_step0.json')
        out[name] = dict(step0_rsum=ev['dev_retrieval']['rsum'], native=ctx.nat_metrics['rsum'], equal=abs(ev['dev_retrieval']['rsum'] - ctx.nat_metrics['rsum']) < 1e-9,
                         step2=CM.read_json(tmp / name / 'eval_step2.json')['dev_retrieval']['rsum'])
        print(name, out[name], f'{time.time() - t0:.0f}s', flush=True)
    step0('r35_A_n60_s1001_ID_J', cG, 'A_ID')
    step0('r35_A_n60_s1001_G_NCE', cG, 'A_G')
    r = T.run_unit(cfg('r35_B_n60_s1001_TRUST_BCE', 2), cG, imap=cG.d['obs_np'], pool=cG.d['trust_subset'], out_dir=tmp / 'B', log=lambda *a: None)
    print('B ok', flush=True)
    step0('r35_C_n60_s1001_G_J', cG, 'C_G')
    cL = fake_ctx(True)
    step0('r35_C_n60_s1001_L_J', cL, 'C_L')
    # resume / determinism on C-L
    a = T.run_unit(cfg('r35_C_n60_s1001_L_BCE', 4), cL, out_dir=tmp / 'res_full', ckpt_steps=[0, 2, 4], no_readout=True, log=lambda *a: None)
    T.run_unit(cfg('r35_C_n60_s1001_L_BCE', 4), cL, out_dir=tmp / 'res_split', ckpt_steps=[0, 2, 4], stop_at=2, no_readout=True, log=lambda *a: None)
    b = T.run_unit(cfg('r35_C_n60_s1001_L_BCE', 4), cL, out_dir=tmp / 'res_split', ckpt_steps=[0, 2, 4], no_readout=True, log=lambda *a: None)
    out['resume_CL'] = a['final_hash'] == b['final_hash'] and a['batch_hash'] == b['batch_hash']
    # D from the C_L parent (step 2)
    pck = str(tmp / 'C_L' / 'ckpt_step2.pt'); pz = np.load(tmp / 'C_L' / 'dev_scores_step2.npz')
    for mode, u in (('joint', 'r35_D_n60_s1001_L_J'), ('head_control', 'r35_D_n60_s1001_L_J_HEAD_CONTROL')):
        c = cfg(u, 2, parent_ckpt=pck, parent_step=2, parent_unit='r35_C_n60_s1001_L_J')
        r = T.run_unit(c, cL, out_dir=tmp / f'D_{mode}', log=lambda *a: None)
        z = np.load(tmp / f'D_{mode}' / 'dev_scores_step0.npz')
        out[f'D_{mode}_step0_eq_parent'] = all(np.array_equal(z[k], pz[k]) for k in ('gi_top', 'gt_top', 'g_gt'))
        out[f'D_{mode}_enc_changed'] = any(not torch.equal(q, q0) for q, q0 in zip(r['encs']['A'].parameters(), cL.enc0['A'].parameters()))
        if mode == 'joint':
            T.run_unit(c, cL, out_dir=tmp / 'D_jsplit', ckpt_steps=[0, 1, 2], stop_at=1, no_readout=True, log=lambda *a: None)
            b = T.run_unit(c, cL, out_dir=tmp / 'D_jsplit', ckpt_steps=[0, 1, 2], no_readout=True, log=lambda *a: None)
            f = T.run_unit(c, cL, out_dir=tmp / 'D_jfull', ckpt_steps=[0, 1, 2], no_readout=True, log=lambda *a: None)
            out['resume_D_joint'] = b['final_hash'] == f['final_hash']
        print(mode, out, flush=True)
    # finalize path
    res = T.run_unit(cfg('r35_A_n60_s1001_ID_BCE', 2), cG, imap=cG.d['obs_np'], pool=np.arange(CM.N_TRAIN_CAP), out_dir=tmp / 'fin', log=lambda *a: None)
    T.finalize(cfg('r35_A_n60_s1001_ID_BCE', 2), cG, res, 1.0)
    out['finalize'] = (tmp / 'fin' / 'RESULT.json').exists()
    print('CPU_SIM', out, 'tmp', tmp)


if __name__ == '__main__':
    main()
