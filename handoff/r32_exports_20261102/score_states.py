#!/usr/bin/env python3
"""R32 read-only DEV scoring of existing W2/E2/C2 states (review package 2026-11-02, SERVER_PLAN §1 R32).
Same full-DEV pool as the formal training readout (roles/dev.npz: 1014 images / 5070 captions, clean annotation), same
candidates and truth for every score. Three scores per state:
  h_cos     cosine of the pointwise encoder outputs h (eval mode, frozen BN, no centering)
  head_cos  cosine of the linear head outputs (image_head(sqrt(1024) h), text_head(sqrt(1024) h)), i.e. the raw head logits
  formal    the unit's formal readout a.b (ratio arms: (u/sqrt pi).(v/sqrt pi); cosine arms: cosine of the head outputs)
Each with two tie rules: package retrieval_metrics (descending score, increasing candidate index = the training readout) and
upstream IN2R i2t/t2i (argsort). States: W2 parent (epoch 5), E2 zero-update (child built from its W2 parent exactly as
train_gpo.Runner.init_model, before any step), E2/C2 epoch 20 and SELECT-J selected. Unfinished units are skipped (NA).
Formal-score rsum is checked against the unit's own logged DEV readout of the same checkpoint. TEST is never loaded.
Writes out/score/<unit>__<tag>.json (+ ranks npz). Nothing is written under the R32 dirs.
Usage: score_states.py [--only UNIT ...]
"""
import argparse, glob, hashlib, json, math, os, sys, time
import numpy as np

W = '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101'
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101'
IMS = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp'   # dev_ims.npy only (train mmap unused)
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (f'{W}/nrcs', f'{W}/upstream', f'{W}/package/src'):
    sys.path.insert(0, p)
import torch
import torch.nn.functional as F
import train_gpo as TG
import gpo_adapter as ga
import test_entry as TE
from native_ratio.frozen import evaluate as pkg_evaluate
from native_ratio.retrieval import retrieval_metrics

KEYS = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def code_stats(z):
    """softmax code of linear head outputs z [N,K]"""
    p = z.double().softmax(-1)
    ent = -(p * p.clamp_min(1e-300).log()).sum(-1)
    mu = p.mean(0)
    hmu = float(-(mu * mu.clamp_min(1e-300).log()).sum())
    return {'mean_per_sample_entropy_nats': float(ent.mean()), 'entropy_of_mean_posterior_nats': hmu,
            'effective_components_of_mean_posterior': math.exp(hmu), 'mean_per_sample_effective_components': float(ent.exp().mean()),
            'hard_occupied_components': int(p.argmax(-1).unique().numel()), 'logit_std': float(z.std(unbiased=False)),
            'mean_logit_vector_norm': float(z.norm(dim=-1).mean()), 'max_posterior_mean': float(p.max(-1)[0].mean())}


def metrics(s, mapping, up_i2t, up_t2i):
    pk = retrieval_metrics(s, mapping)
    n = s.shape[0]
    r, ri = up_i2t(n, s, 5), up_t2i(n, s, 5)
    up = dict(zip(KEYS, map(float, (r[0], r[1], r[2], ri[0], ri[1], ri[2]))))
    up['rsum'] = float(sum(up[k] for k in KEYS)); up['i2t_medr'], up['t2i_medr'] = float(r[3]), float(ri[3])
    return {'package_stable_ties': {k: pk[k] for k in KEYS + ('rsum', 'i2t_median_rank', 't2i_median_rank')},
            'upstream_argsort': up, 'unique_scores': int(np.unique(s).size), 'n_scores': int(s.size)}


def score_model(model, pool, w2i, up):
    with torch.no_grad():
        hi, ht = ga.encode_pool(model, pool, len(w2i), w2i['<mask>'])
        xi, yt = model.heads_in(hi), model.heads_in(ht)
        za, zb = model.critic.image_head(xi), model.critic.text_head(yt)
        a, b = model.critic.retrieval_embeddings(xi, yt)
        def chunked(x, y):                                    # same 256-row chunking as native_ratio.frozen.evaluate
            return np.concatenate([(x[lo:lo + 256] @ y.T).float().cpu().numpy() for lo in range(0, len(x), 256)])
        nh_i, nh_t = F.normalize(hi, dim=-1), F.normalize(ht, dim=-1)
        nz_a, nz_b = F.normalize(za, dim=-1), F.normalize(zb, dim=-1)
        S = {'h_cos': chunked(nh_i, nh_t), 'head_cos': chunked(nz_a, nz_b), 'formal': chunked(a, b)}
        geom = {}
        for nm, (u_, v_) in (('h', (nh_i, nh_t)), ('head', (nz_a, nz_b))):
            mu, mv = u_.mean(0), v_.mean(0)
            geom[nm] = {'image_mean_unit_vector_norm': float(mu.norm()), 'text_mean_unit_vector_norm': float(mv.norm()),
                        'cos_image_mean_text_mean': float(F.cosine_similarity(mu, mv, dim=0))}
        res = {k: metrics(v, pool.mapping, *up) for k, v in S.items()}
        ev = pkg_evaluate(model.critic, {'x': xi, 'y': yt, 'mapping': pool.mapping, 'role': 'dev'}, 'cuda', 256)
        extra = {'formal_pkg_evaluate': {k: ev[k] for k in ev if k in ('J', 'J_terms', 'marginals', 'ratio_capacity', 'affine_scale', 'affine_offset')},
                 'formal_pkg_evaluate_rsum': ev['retrieval']['rsum'],
                 'h_norm_mean': {'image': float(hi.norm(dim=-1).mean()), 'text': float(ht.norm(dim=-1).mean())},
                 'mean_direction_geometry': geom, 'score_path': 'all three scores in 256-image-row chunks (native_ratio.frozen.evaluate chunking)',
                 'head_code_image': code_stats(za), 'head_code_text': code_stats(zb)}
        ranks = {}
        for k, v in S.items():
            i2t, t2i = TG.ranks_from_scores(v, pool.mapping)
            ranks[f'{k}_i2t'] = i2t; ranks[f'{k}_t2i'] = t2i
    return res, extra, ranks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*'); a = ap.parse_args()
    TG.set_precision()
    pools = TG.load_pools(f'{O}/data', IMS, 0, 'cuda', roles=('dev',))
    pool = pools['dev']
    assert np.array_equal(pool.mapping, np.arange(pool.n_cap) // 5) and pool.n_img == 1014 and pool.n_cap == 5070
    w2i = ga.load_vocab()
    up = TE.upstream_metric_fns()
    rows = TG.matrix_rows()
    od = f'{HERE}/out/score'; os.makedirs(od, exist_ok=True)
    env = {'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename, 'job': os.environ.get('SLURM_JOB_ID'), 'torch': torch.__version__}
    code = {f: sha(f) for f in (os.path.abspath(__file__), f'{W}/nrcs/train_gpo.py', f'{W}/nrcs/gpo_adapter.py', f'{W}/nrcs/test_entry.py')}
    todo = []
    for u, r in sorted(rows.items()):
        if r['stage'] not in ('W2', 'E2', 'C2') or (a.only and u not in a.only):
            continue
        rd = f'{O}/runs/{u}'
        done = json.load(open(f'{rd}/DONE.json')) if os.path.exists(f'{rd}/DONE.json') else None
        if done is None:
            print('NA (unfinished)', u, flush=True); continue
        if r['stage'] == 'W2':
            todo.append((u, r, 'W2_E5', f'{rd}/parent.pt', done['epoch5']['dev']['rsum'], 'DONE.epoch5.dev'))
            continue
        if r['stage'] == 'E2':
            h0 = json.load(open(f'{rd}/history.json'))[0]
            assert h0['epoch'] == 0
            todo.append((u, r, 'E2_zero_update', None, h0['eval']['dev']['retrieval']['rsum'], 'history[epoch0].eval.dev (before any step)'))
        todo.append((u, r, 'E20', f"{rd}/epoch{done['config']['epochs']}.pt", done['last']['dev']['retrieval']['rsum'], 'DONE.last.dev'))
        todo.append((u, r, 'selected', f'{rd}/selected.pt', done['selected']['dev']['retrieval']['rsum'], f"DONE.selected.dev (epoch {done['selected']['epoch']})"))
    print(len(todo), 'states', flush=True)
    for u, r, tag, path, logged, src in todo:
        dest = f'{od}/{u}__{tag}.json'
        if os.path.exists(dest):
            continue
        t0 = time.time()
        if tag == 'E2_zero_update':
            pdir = f"{O}/runs/{r['parent_id']}"
            psha = sha(f'{pdir}/parent.pt')
            assert psha == json.load(open(f'{pdir}/DONE.json'))['parent_checkpoint_sha256']
            torch.manual_seed(int(r['seed']))                      # train_gpo.Runner.init_model
            model = ga.GPOTwoTower(r['arm'], w2i)
            transfer = ga.transfer_state(torch.load(f'{pdir}/parent.pt', map_location='cpu', weights_only=False)['model'], model)
            ck_info = {'built_from_parent': r['parent_id'], 'parent_sha256': psha, 'transfer_kept': len(transfer[0]), 'transfer_fresh': transfer[1]}
        else:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            arm = ck.get('arm', r['arm'])
            assert arm == r['arm'], (arm, r['arm'])
            model = ga.GPOTwoTower(arm, w2i); model.load_state_dict(ck['model'])
            ck_info = {'checkpoint': path, 'sha256': sha(path), 'checkpoint_epoch': ck.get('epoch')}
        model.cuda().eval()
        res, extra, ranks = score_model(model, pool, w2i, up)
        out = {'unit_id': u, 'tag': tag, 'stage': r['stage'], 'arm': r['arm'], 'seed': int(r['seed']), 'noise_percent': int(r['noise_percent']),
               'balance_weight': float(r['balance_weight']), 'intra_weight': float(r['intra_weight']), 'parent_id': r['parent_id'] or None,
               'state': ck_info, 'scores': res, **extra,
               'check_formal_vs_logged': {'logged_rsum': logged, 'logged_source': src, 'recomputed_package_rsum': res['formal']['package_stable_ties']['rsum'],
                                          'abs_diff': abs(res['formal']['package_stable_ties']['rsum'] - logged)},
               'env': env, 'code_sha256': code, 'seconds': time.time() - t0, 'test_loaded': False}
        np.savez_compressed(f'{od}/{u}__{tag}__ranks.npz', **ranks)
        json.dump(out, open(dest + '.tmp', 'w'), indent=1, default=float); os.replace(dest + '.tmp', dest)
        print(u, tag, f"formal {res['formal']['package_stable_ties']['rsum']:.3f} logged {logged:.3f} h {res['h_cos']['package_stable_ties']['rsum']:.2f} "
              f"head {res['head_cos']['package_stable_ties']['rsum']:.2f} {time.time()-t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    print('SCORE_DONE', flush=True)


if __name__ == '__main__':
    main()
