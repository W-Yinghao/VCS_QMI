"""R27 audit 3: per-query full-DEV ranks of the selected (best full-DEV) checkpoints. DEV only; TEST is never read.

Usage: audit_ranks.py OUT_ROOT DATA_PATH
Units: every R27 unit runs/<cell>/<arm> with DONE.json, plus the six A0 = R25 H0 runs (groups/<cell>/H0, DONE.json).
Incremental: a unit whose OUT_ROOT/<cell>/<arm>/RESULT.json exists (same model_best sha256) is skipped.
Per unit: load model_best.pth.tar (model_A, model_B), encode full Flickr DEV (1,014 images / 5,070 captions) with the
host validate path (r25_co_train.validate: encode_data per peer, image rows every 5th, shard_attn_scores shard 100,
similarity summed over peers then / 2), host evaluation.i2t / t2i with return_ranks=True.
Saved (int32 .npy): i2t_rank [1014] (0-based rank of the best-ranked GT caption of image i), i2t_top1 [1014] (caption
index ranked first), t2i_rank [5070] (0-based rank of the GT image of caption j, GT image = j // 5), t2i_top1 [5070].
Check: six recalls recomputed from the saved ranks vs the best-epoch entry of EPOCHS.json (epoch = model_best['epoch']).
"""
import os, sys, json, glob, time
import numpy as np
import torch

import audit_common as C
import r25_run
from data import get_dataset, get_loader
from vse import VSE
from vocab import deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores

R27_RUNS = '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/runs'
R25_GROUPS = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups'
CELLS = ('f40_s1001', 'f40_s1002', 'f40_s1003', 'f60_s1001', 'f60_s1002', 'f60_s1003')
ARMS = ('a1_iface', 'a2_mean', 'a3_vcs', 'a4_bce', 'a5_snce', 'a6_delta')
TOL_PP = 0.1   # documented tolerance per recall (pp); one i2t query = 100/1014 = 0.0986 pp


def units():
    out = []
    for cell in CELLS:
        d = f'{R25_GROUPS}/{cell}/H0'
        out.append((cell, 'H0_A0', d))
        for arm in ARMS:
            out.append((cell, arm, f'{R27_RUNS}/{cell}/{arm}'))
    return [(c, a, d) for c, a, d in out if os.path.exists(f'{d}/DONE.json') and os.path.exists(f'{d}/model_best.pth.tar')]


def peer_sims(m, val_loader, opt):
    img_embs, cap_embs, cap_lens = encode_data(m, val_loader, 10 ** 9)
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    return shard_attn_scores(m, img_embs, cap_embs, cap_lens, opt, shard_size=100), img_embs.shape[0], cap_embs.shape[0]


def main():
    out_root, dp = sys.argv[1], sys.argv[2]
    opt = r25_run.parse(['--noise_file', 'unused', '--output_dir', '/tmp/unused', '--data_path', dp])
    r25_run.seed_all(1001)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "dev")
    val_loader = get_loader(caps, ims, vocab, "dev", opt.batch_size, opt.workers)
    idx = dict(split='dev', data_name=opt.data_name, n_images=int(len(ims) // 5 if len(ims) == len(caps) else len(ims)), n_captions=len(caps),
               dev_caps_sha256=C.sha(os.path.join(dp, opt.data_name, 'dev_caps.txt')),
               dev_ims_sha256=C.sha(os.path.join(dp, opt.data_name, 'dev_ims.npy')),
               image_order='row i of the dev image list after taking every 5th encoded image row (= dev_ims.npy order as loaded by host get_dataset)',
               caption_order='line j of dev_caps.txt; GT image of caption j is j // 5 (5 captions per image)',
               arrays=dict(i2t_rank='int32 [n_images], 0-based rank of the best-ranked GT caption (host evaluation.i2t ranks)',
                           i2t_top1='int32 [n_images], caption index ranked first',
                           t2i_rank='int32 [n_captions], 0-based rank of the GT image (host evaluation.t2i ranks)',
                           t2i_top1='int32 [n_captions], image index ranked first'),
               similarity='peer-averaged: (sim_A + sim_B) / 2 as r25_co_train.validate; per peer encode_data + shard_attn_scores(shard 100)',
               recall_definition='R@k = 100 * mean(rank < k)', tolerance_pp=TOL_PP)
    os.makedirs(out_root, exist_ok=True)
    json.dump(idx, open(os.path.join(out_root, 'INDEX.json'), 'w'), indent=1)
    for cell, arm, d in units():
        od = os.path.join(out_root, cell, arm)
        ck = f'{d}/model_best.pth.tar'
        csha = C.sha(ck)
        rp = os.path.join(od, 'RESULT.json')
        if os.path.exists(rp) and json.load(open(rp)).get('model_best_sha256') == csha:
            print('skip (done)', cell, arm, flush=True); continue
        os.makedirs(od, exist_ok=True)
        t0 = time.time()
        ckpt = torch.load(ck, map_location='cpu', weights_only=False)
        A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
        A.load_state_dict(ckpt['model_A']); B.load_state_dict(ckpt['model_B'])
        sims_mean, peers = 0, {}
        with torch.no_grad():
            for ind, m in enumerate((A, B)):
                sim, ni, nc = peer_sims(m, val_loader, opt)
                r = i2t(ni, sim, 5); ri = t2i(ni, sim, 5)
                peers[f'peer{ind}_rsum'] = float(sum(r[:3]) + sum(ri[:3]))
                sims_mean = sims_mean + sim
        sims_mean = sims_mean / 2
        (r1, r5, r10, _, _), (rk_i, top_i, _, _) = i2t(ni, sims_mean, 5, return_ranks=True)
        (q1, q5, q10, _, _), (rk_t, top_t, _, _) = t2i(ni, sims_mean, 5, return_ranks=True)
        arrs = dict(i2t_rank=rk_i.astype(np.int32), i2t_top1=top_i.astype(np.int32), t2i_rank=rk_t.astype(np.int32), t2i_top1=top_t.astype(np.int32))
        for k, v in arrs.items():
            np.save(os.path.join(od, k + '.npy'), v)
        rec = C.recalls_from_ranks(np.load(os.path.join(od, 'i2t_rank.npy'))) + C.recalls_from_ranks(np.load(os.path.join(od, 't2i_rank.npy')))
        keys = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10')
        recomputed = dict(zip(keys, map(float, rec)))
        recomputed['rsum'] = float(sum(rec))
        host_now = dict(zip(keys, map(float, (r1, r5, r10, q1, q5, q10))))
        hist = json.load(open(f'{d}/EPOCHS.json'))['cotrain']
        ep = int(ckpt['epoch'])
        ent = [e for e in hist if e['epoch'] == ep]
        best = max(hist, key=lambda e: e['dev']['rsum'])
        logged = ent[-1]['dev'] if ent else None
        diffs = {k: recomputed[k] - logged[k] for k in keys + ('rsum',)} if logged else None
        diffs_peer = {k: peers[k] - logged[k] for k in peers} if logged else None
        mx = max(abs(v) for k, v in diffs.items() if k != 'rsum') if diffs else None
        res = dict(cell=cell, arm=arm, run_dir=d, model_best_sha256=csha, model_best_epoch=ep, epochs_best_epoch=int(best['epoch']),
                   epochs_best_rsum=float(best['dev']['rsum']), logged_dev_at_model_best_epoch=logged, recomputed_from_saved_ranks=recomputed,
                   host_recalls_now=host_now, peer_rsum_now=peers, diff_recomputed_minus_logged=diffs, diff_peer_rsum_now_minus_logged=diffs_peer,
                   max_abs_recall_diff_pp=mx, within_tolerance=(mx is not None and mx <= TOL_PP), tolerance_pp=TOL_PP,
                   recomputed_equals_host_now=bool(all(abs(recomputed[k] - host_now[k]) == 0 for k in keys)),
                   n_images=int(ni), n_captions=int(nc), gpu=torch.cuda.get_device_name(0), host=os.uname().nodename,
                   job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - t0)
        json.dump(res, open(rp, 'w'), indent=1)
        print(json.dumps(dict(cell=cell, arm=arm, epoch=ep, rsum=recomputed['rsum'], logged=logged['rsum'] if logged else None, maxdiff=mx)), flush=True)
        del A, B, ckpt
        torch.cuda.empty_cache()
    json.dump(dict(audit_sha256=C.own_hashes(), code_v11_sha256=C.code_hashes(), job=os.environ.get('SLURM_JOB_ID')),
              open(os.path.join(out_root, f'PROVENANCE_{os.environ.get("SLURM_JOB_ID")}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
