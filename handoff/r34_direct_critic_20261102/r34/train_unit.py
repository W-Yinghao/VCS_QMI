"""R34 critic-only fit on frozen R30-H caches + fixed step-2000 full-DEV direct retrieval.
Usage: python -B train_unit.py --cond n60 [--units id1,id2,...]   (default: all 6 units of the condition)
Only cache files are read (no raw data, no test files). DEV enters only no-grad readouts."""
from __future__ import annotations
import argparse, hashlib, io, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONDS, OUT, ROOT, PEERS, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, install_test_guard, guard_record, sha256, write_json, read_json, utc
from critics import build, n_params, pq_weights, loss_of, j_value
from native_eval import upstream_metrics, tie_bounded

RUNS = OUT / 'runs'


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- data

def load_cond(cond, device, verify=True):
    cdir = OUT / 'cache' / cond
    man = read_json(cdir / 'CACHE_MANIFEST.json')
    locks = read_json(OUT / 'locks' / 'LOCKS.json')
    if verify:
        for name, rec in man['files'].items():
            assert sha256(cdir / name) == rec['sha256'], f'cache hash mismatch {name}'
        for rec in (locks['dev_observed_maps'][cond], locks['fit_eval_subset']):
            assert sha256(rec['path']) == rec['sha256']
        assert sha256(man['observed_map']['path']) == man['observed_map']['sha256']
    d = {'cond': cond, 'manifest': man, 'locks': locks, 'manifest_sha256': sha256(cdir / 'CACHE_MANIFEST.json')}
    obs = np.load(man['observed_map']['path']).astype(np.int64)
    assert obs.shape == (N_TRAIN_CAP,)
    d['obs'] = torch.from_numpy(obs).to(device)
    d['obs_np'] = obs
    for p in PEERS:
        st = np.load(cdir / f'standardization_peer{p}.npz')
        mx, sx, my, sy = (torch.from_numpy(st[k]).to(device) for k in ('image_mean', 'image_std', 'text_mean', 'text_std'))
        for split in ('fit', 'dev'):
            x = torch.from_numpy(np.load(cdir / f'{split}_img_peer{p}.npy')).to(device)
            y = torch.from_numpy(np.load(cdir / f'{split}_cap_peer{p}.npy')).to(device)
            d[(split, 'img', p)] = (x - mx) / sx
            d[(split, 'cap', p)] = (y - my) / sy
    d['native_dev'] = np.load(cdir / 'native_dev_sims.npy')
    d['dev_clean'] = np.arange(N_DEV_CAP, dtype=np.int64) // 5
    d['dev_obs'] = np.load(locks['dev_observed_maps'][cond]['path']).astype(np.int64)
    sub = np.load(locks['fit_eval_subset']['path']).astype(np.int64)
    imgs, m_local = np.unique(obs[sub], return_inverse=True)
    d['fit_sub'] = dict(records=torch.from_numpy(sub).to(device), images=torch.from_numpy(imgs).to(device),
                        map=m_local.astype(np.int64), n_images=int(len(imgs)))
    return d


# ----------------------------------------------------------------------------- readouts

@torch.no_grad()
def score_matrix(model, x, y, chunk=64):
    model.eval()
    if hasattr(model, 'first'):
        return model.pair_scores(x, y, chunk=chunk)
    return torch.cat([model.pair_scores(x[i:i + 1024], y) for i in range(0, len(x), 1024)], 0)


@torch.no_grad()
def j_from_matrix(g, m):
    """record U-statistic: P over records (x_m(r), y_r); Q over r != s (x_m(r), y_s); same estimator as training."""
    n = g.shape[1]
    mt = torch.as_tensor(m, device=g.device)
    t = torch.tanh(g / 2).double()
    tp = t[mt, torch.arange(n, device=g.device)]
    ep_lin, ep_quad = tp.mean(), (tp * tp).mean()
    counts = torch.bincount(mt, minlength=g.shape[0]).double()
    lin_all = (counts[:, None] * t).sum(); quad_all = (counts[:, None] * t * t).sum()
    eq_lin = (lin_all - tp.sum()) / (n * (n - 1)); eq_quad = (quad_all - (tp * tp).sum()) / (n * (n - 1))
    j = ep_lin - .5 * ep_quad - eq_lin - .5 * eq_quad
    return dict(J=float(j), E_P_T=float(ep_lin), E_P_T2=float(ep_quad), E_Q_T=float(eq_lin), E_Q_T2=float(eq_quad))


def j_readout(gs, m):
    per = {p: j_from_matrix(g, m) for p, g in gs.items()}
    out = {f'peer{p}': v for p, v in per.items()}
    out['J_mean_of_peers'] = float(np.mean([v['J'] for v in per.values()]))
    out['J_of_peer_mean_score'] = j_from_matrix(sum(gs.values()) / len(gs), m)['J']
    return out


@torch.no_grad()
def readout(model, d, with_rank=True):
    t0 = time.time()
    fs = d['fit_sub']
    g_fit = {p: score_matrix(model, d[('fit', 'img', p)][fs['images']], d[('fit', 'cap', p)][fs['records']], chunk=32) for p in PEERS}
    rec = {'fit_observed': j_readout(g_fit, fs['map'])}
    del g_fit
    g_dev = {p: score_matrix(model, d[('dev', 'img', p)], d[('dev', 'cap', p)]) for p in PEERS}
    rec['dev_observed'] = j_readout(g_dev, d['dev_obs'])
    rec['dev_clean'] = j_readout(g_dev, d['dev_clean'])
    if with_rank:
        s = sum(g.double() for g in g_dev.values()).cpu().numpy() / len(g_dev)
        rec['dev_retrieval'], _, _ = upstream_metrics(s)
    rec['readout_seconds'] = time.time() - t0
    return rec, g_dev


def saturation(g_dev, m):
    out = {}
    for p, g in g_dev.items():
        t = torch.tanh(g / 2)
        pos = t[torch.as_tensor(m, device=g.device), torch.arange(g.shape[1], device=g.device)]
        out[f'peer{p}'] = dict(frac_absT_ge_0p999=float((t.abs() >= .999).double().mean()),
                               frac_absT_eq_1_fp32=float((t.abs() == 1).double().mean()),
                               frac_pos_absT_ge_0p999=float((pos.abs() >= .999).double().mean()),
                               g_quantiles={f'p{int(q*100)}': float(torch.quantile(g.flatten()[::97].float(), q)) for q in (.01, .5, .99)})
    return out


def mechanism(s, nat):
    """§6 tables on DEV (clean annotation). s: critic peer-mean g [1014,5070]; nat: native H cosine (peer mean)."""
    n_img, n_cap = s.shape
    gt = np.arange(n_cap) // 5
    res = {}
    _, ri_c, rt_c = upstream_metrics(s); _, ri_n, rt_n = upstream_metrics(nat)
    for name, rc, rn in (('i2t', ri_c, ri_n), ('t2i', rt_c, rt_n)):
        d = rc - rn
        res[f'{name}_rank_change'] = dict(rescued_r1=int(((rn >= 1) & (rc < 1)).sum()), lost_r1=int(((rn < 1) & (rc >= 1)).sum()),
                                          rescued_r5=int(((rn >= 5) & (rc < 5)).sum()), lost_r5=int(((rn < 5) & (rc >= 5)).sum()),
                                          n_improved=int((d < 0).sum()), n_worsened=int((d > 0).sum()), n_same=int((d == 0).sum()),
                                          mean_rank_delta=float(d.mean()), median_rank_delta=float(np.median(d)), n_queries=len(d))
    # native hard negatives
    gt_mask = np.zeros_like(s, dtype=bool); gt_mask[gt, np.arange(n_cap)] = True
    nat_neg = np.where(gt_mask, -np.inf, nat)
    t_hard = np.argsort(-nat_neg, axis=0, kind='stable')[:10]          # [10, n_cap] image ids
    i_hard = np.argsort(-nat_neg, axis=1, kind='stable')[:, :10]       # [n_img, 10] caption ids
    cols = np.arange(n_cap)
    def side(score):
        pos_t = score[gt, cols]
        hard_t = score[t_hard, cols[None, :]]                         # [10, n_cap]
        pos_i = score[np.arange(n_img)[:, None], np.arange(n_img)[:, None] * 5 + np.arange(5)[None, :]].max(1)
        hard_i = score[np.arange(n_img)[:, None], i_hard]               # [n_img,10]
        rnd_t = ((score < pos_t[None, :]) & ~gt_mask).sum(0) / (n_img - 1)
        rnd_i = ((score < pos_i[:, None]) & ~gt_mask).sum(1) / (n_cap - 5)
        return dict(
            t2i_win_vs_native_hardest=float((pos_t > hard_t[0]).mean()), i2t_win_vs_native_hardest=float((pos_i > hard_i[:, 0]).mean()),
            t2i_pairacc_native_hard10=float((pos_t[None, :] > hard_t).mean()), i2t_pairacc_native_hard10=float((pos_i[:, None] > hard_i).mean()),
            t2i_pairacc_random=float(rnd_t.mean()), i2t_pairacc_random=float(rnd_i.mean()),
            score_mean_pos=float(pos_t.mean()), score_mean_all_pairs=float(score.mean()),
            score_mean_native_hardest_t2i=float(hard_t[0].mean()), score_mean_native_hardest_i2t=float(hard_i[:, 0].mean()),
            margin_vs_native_hardest_t2i=dict(mean=float((pos_t - hard_t[0]).mean()), median=float(np.median(pos_t - hard_t[0])), frac_pos=float((pos_t > hard_t[0]).mean())),
            margin_vs_native_hardest_i2t=dict(mean=float((pos_i - hard_i[:, 0]).mean()), median=float(np.median(pos_i - hard_i[:, 0])), frac_pos=float((pos_i > hard_i[:, 0]).mean())))
    res['critic'] = side(s); res['native'] = side(nat)
    # Top-100 rerank on native candidates (diagnostic, not the primary full-matrix result)
    K = 100
    t_top = np.argsort(-nat, axis=0, kind='stable')[:K]               # [K, n_cap]
    i_top = np.argsort(-nat, axis=1, kind='stable')[:, :K]            # [n_img, K]
    def rerank(score):
        sc_t = score[t_top, cols[None, :]]; order_t = np.argsort(-sc_t, axis=0, kind='stable')
        hit_t = (t_top[order_t, cols[None, :]] == gt[None, :])             # [K, n_cap]
        r_t = np.where(hit_t.any(0), hit_t.argmax(0), K + 1)
        sc_i = score[np.arange(n_img)[:, None], i_top]; order_i = np.argsort(-sc_i, axis=1, kind='stable')
        ids = np.take_along_axis(i_top, order_i, 1); hit_i = (ids // 5) == np.arange(n_img)[:, None]
        r_i = np.where(hit_i.any(1), hit_i.argmax(1), K + 1)
        return {f'{n}_r{k}': 100.0 * float((r < k).mean()) for n, r in (('i2t', r_i), ('t2i', r_t)) for k in (1, 5, 10)}
    cov_t = float((t_top == gt[None, :]).any(0).mean()); cov_i = float(((i_top // 5) == np.arange(n_img)[:, None]).any(1).mean())
    res['top100'] = dict(coverage_t2i=cov_t, coverage_i2t=cov_i, critic_rerank=rerank(s), native_rerank=rerank(nat),
                         note='candidates = native H cosine top-100 per query; GT outside top-100 counts as miss')
    return res


# ----------------------------------------------------------------------------- training

def train(cfg, d, device, steps=None, eval_every=None, dev_nan=False, with_rank=True, log=print):
    steps = cfg['steps'] if steps is None else steps
    eval_every = cfg['eval_every'] if eval_every is None else eval_every
    if dev_nan:  # isolation probe: DEV replaced by NaN must not change training
        d = dict(d)
        for p in PEERS:
            for k in ('img', 'cap'):
                d[('dev', k, p)] = torch.full_like(d[('dev', k, p)], float('nan'))
    torch.manual_seed(cfg['seed'])
    model = build(cfg['structure']).to(device)
    init_hash = state_hash(model)
    o = cfg['optimizer']
    opt = torch.optim.AdamW(model.parameters(), lr=o['lr'], betas=tuple(o['betas']), eps=o['eps'], weight_decay=o['weight_decay'], amsgrad=o['amsgrad'])
    rng = np.random.default_rng(cfg['seed'])
    B = cfg['batch_size']; p_w, q_w = pq_weights(B, device)
    bh = hashlib.sha256(); curves = []; win = []; t_train = 0.0; dup_same_image = 0; dup_record = 0
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            rec, _ = readout(model, d, with_rank=with_rank and not dev_nan)
            rec['step'] = step
            if win:
                w = np.array(win); rec['train_window'] = dict(n=len(win), loss_mean=float(w[:, 0].mean()), batch_J_mean=float(w[:, 1].mean()),
                                                               grad_norm_mean=float(w[:, 2].mean()), loss_last=float(w[-1, 0]))
            curves.append(rec); win = []
            log(f"step {step} fitJ {rec['fit_observed']['J_mean_of_peers']:.4f} devobsJ {rec['dev_observed']['J_mean_of_peers']:.4f} "
                f"devcleanJ {rec['dev_clean']['J_mean_of_peers']:.4f} rsum {rec.get('dev_retrieval', {}).get('rsum', float('nan')):.2f}")
        if step == steps:
            break
        t0 = time.time()
        model.train()
        idx = rng.integers(0, len(d['obs_np']), size=B); bh.update(idx.tobytes())
        oi = d['obs_np'][idx]
        dup_record += int(len(idx) - len(np.unique(idx))); dup_same_image += int(((oi[:, None] == oi[None, :]).sum() - B) // 2)
        it = torch.from_numpy(idx).to(device); ii = d['obs'][it]
        loss = 0.0; jb = 0.0
        for p in PEERS:
            g = model.pair_scores(d[('fit', 'img', p)][ii], d[('fit', 'cap', p)][it])
            loss = loss + .5 * loss_of(cfg['objective'], g, p_w, q_w)
            jb += .5 * float(j_value(g.detach(), p_w, q_w))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.stack([q.grad.detach().square().sum() for q in model.parameters() if q.grad is not None]).sum().sqrt())
        if not (np.isfinite(gn) and torch.isfinite(loss)):
            raise FloatingPointError(f'nonfinite at step {step}')
        if cfg['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step()
        win.append((float(loss), jb, gn)); t_train += time.time() - t0
    return dict(model=model, init_hash=init_hash, batch_hash=bh.hexdigest(), final_hash=state_hash(model), curves=curves,
                train_seconds=t_train, dup_record_pairs=dup_record, same_observed_image_offdiag_pairs=dup_same_image,
                n_updates=steps)


def run_unit(cfg, d, device, log=print, runs_dir=None):
    out = (RUNS if runs_dir is None else Path(runs_dir)) / cfg['unit_id']
    if (out / 'DONE.json').exists():
        log(f"skip DONE {cfg['unit_id']}"); return read_json(out / 'RESULT.json')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    tr = train(cfg, d, device, log=log)
    model = tr.pop('model')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    te = time.time()
    _, g_dev = readout(model, d, with_rank=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    eval_seconds = time.time() - te
    s = sum(g.double() for g in g_dev.values()).cpu().numpy() / len(g_dev)
    final = tr['curves'][-1]
    retr, ri, rt = upstream_metrics(s)
    nat, _, _ = upstream_metrics(d['native_dev'])
    res = dict(unit_id=cfg['unit_id'], config=cfg, status='COMPLETE', step=cfg['steps'],
               dev_retrieval_step2000=retr, dev_retrieval_tie_bounded=tie_bounded(s),
               native_H_dev=nat, diff_vs_native={k: retr[k] - nat[k] for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')},
               J_step2000={k: final[k] for k in ('fit_observed', 'dev_observed', 'dev_clean')},
               J_fit_minus_devobs=final['fit_observed']['J_mean_of_peers'] - final['dev_observed']['J_mean_of_peers'],
               T_saturation_dev=saturation(g_dev, d['dev_clean']),
               mechanism=mechanism(s, d['native_dev']), curves=tr['curves'],
               identity=dict(init_state_sha256=tr['init_hash'], batch_index_sha256=tr['batch_hash'], final_state_sha256=tr['final_hash'],
                             cache_manifest_sha256=d.get('manifest_sha256'),
                             code_sha256={f.name: sha256(f) for f in sorted((ROOT / 'r34').glob('*.py'))},
                             config_sha256=sha256(ROOT / 'configs' / f"{cfg['unit_id']}.json") if runs_dir is None else 'gate_override'),
               batch_stats=dict(dup_record_pairs=tr['dup_record_pairs'], same_observed_image_offdiag_pairs=tr['same_observed_image_offdiag_pairs']),
               cost=dict(params=n_params(model), train_seconds_2000_updates=tr['train_seconds'], unit_wall_seconds=time.time() - t0,
                         dev_full_scoring_seconds_both_peers=eval_seconds, dev_pairs_scored_per_second=2 * N_DEV_IMG * N_DEV_CAP / max(eval_seconds, 1e-9),
                         peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0),
               env=dict(gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu', host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__,
                        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
               test_guard=guard_record(), finished_utc=utc())
    torch.save({'state_dict': model.state_dict(), 'config': cfg, 'step': cfg['steps']}, out / 'critic_step2000.pt')
    np.save(out / 'dev_scores_step2000.npy', s)
    np.savez(out / 'dev_ranks_step2000.npz', i2t=ri, t2i=rt)
    res['artifacts'] = {n: sha256(out / n) for n in ('critic_step2000.pt', 'dev_scores_step2000.npy', 'dev_ranks_step2000.npz')}
    write_json(out / 'RESULT.json', res)
    write_json(out / 'DONE.json', dict(status='COMPLETE', finished_utc=utc(), rsum=retr['rsum'], job=os.environ.get('SLURM_JOB_ID')))
    log(f"R34_UNIT_DONE {cfg['unit_id']} rsum {retr['rsum']:.2f} native {nat['rsum']:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True, choices=sorted(CONDS))
    ap.add_argument('--units', default=''); a = ap.parse_args()
    install_test_guard()
    frozen = read_json(ROOT / 'PROTOCOL_FROZEN.json')
    for f, h in frozen['code_sha256'].items():
        if f in ('summarize.py', 'freeze.py'):
            continue
        assert sha256(ROOT / 'r34' / f) == h, f'code differs from frozen protocol: {f}'
    units = [u for u in read_json(ROOT / 'configs' / 'UNITS.json') if f"_{a.cond}_" in u]
    if a.units:
        units = [u for u in units if u in a.units.split(',')]
    for u in units:
        assert sha256(ROOT / 'configs' / f'{u}.json') == frozen['config_sha256'][u], f'config differs from frozen: {u}'
    gate = read_json(ROOT / 'GPU_GATE.json'); assert gate['ALL_PASS'], 'GPU gate not passed'
    device = torch.device('cuda')
    d = load_cond(a.cond, device)
    failed = []
    for u in units:
        cfg = read_json(ROOT / 'configs' / f'{u}.json')
        try:
            run_unit(cfg, d, device)
        except Exception:
            tb = traceback.format_exc(); print(tb, flush=True)
            write_json(RUNS / u / f"FAILED_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(unit=u, traceback=tb, utc=utc()))
            failed.append(u)
    print('R34_COND_END', a.cond, 'failed', failed)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
