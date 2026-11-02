"""Build the frozen native embedding caches of one R30 H parent (both peers), FIT(train) + DEV, and
reproduce the parent's logged full-DEV rSum with the native validate() path.

Native embedding = VSE.forward_emb in eval mode (dropout off, BN running stats): encoder output,
per-batch mean subtraction (separately per modality) and L2 norm, batch_size 128, sequential order.
DEV: native dev loader exactly as R25 validate (rows = dev_ims.npy rows = image k//5, caption k).
FIT: the same layout applied to train (row k -> image k//5, caption k, no augmentation), image row
taken at k=5j like native img_embs[::5]. The layout is file order; the observed (noisy) map is NOT used
for encoding, it only forms critic pairs later. No test file is opened (audit guard).
Usage: python -B build_cache.py --cond n00|n60 --data-path <staged dir containing f30k_precomp>
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CODE_R30, CONDS, OUT, R30_RUNS, VOCAB, PEERS, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP,
                    install_test_guard, guard_record, sha256, write_json, read_json, utc)


def standardization(x_img, y_cap, obs_map, floor=1e-6):
    """R32 F1 standardize_caches rule: image mean/std under the observed-caption-induced image marginal,
    text mean/std uniform over captions; float64 compute, float32 stored; FIT only."""
    x, y = x_img.astype(np.float64), y_cap.astype(np.float64)
    w = np.bincount(obs_map, minlength=len(x)).astype(np.float64) / len(y)
    mx = (w[:, None] * x).sum(0); my = y.mean(0)
    sx = np.sqrt((w[:, None] * (x - mx) ** 2).sum(0)).clip(min=floor)
    sy = np.sqrt(((y - my) ** 2).mean(0)).clip(min=floor)
    return dict(image_mean=mx.astype(np.float32), image_std=sx.astype(np.float32),
                text_mean=my.astype(np.float32), text_std=sy.astype(np.float32), std_floor=np.float64(floor))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True, choices=sorted(CONDS))
    ap.add_argument('--data-path', required=True); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    install_test_guard()
    t_start = time.time()
    import torch
    sys.path.insert(0, str(CODE_R30))
    import vse as V, data as D, evaluation as E
    from vocab import deserialize_vocab
    torch.backends.cudnn.benchmark = True  # native (r25_run.py)

    c = CONDS[a.cond]; hdir = R30_RUNS / c['h_unit']; cdir = OUT / 'cache' / a.cond
    if (cdir / 'CACHE_MANIFEST.json').exists():
        print('cache already built', cdir); return
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = hdir / 'model_best.pth.tar'
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    opt = ck['opt']; run = read_json(hdir / c['run_json'])
    assert float(opt.noise_ratio) == c['noise_ratio'] and run['opt']['noise_file'] == c['noise_file']
    assert sha256(c['noise_file']) == c['noise_sha256']
    obs = np.load(c['noise_file']); assert obs.shape == (N_TRAIN_CAP,) and obs.dtype == np.int64
    if c['noise_ratio'] == 0:
        # native data.py does not load the file at ratio 0: t2i_index = arange//5; the file must equal it
        assert np.array_equal(obs, np.arange(N_TRAIN_CAP) // 5)
    opt.data_path = a.data_path; opt.vocab_path = str(VOCAB); opt.workers = 0
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    models = []
    for key in ('model_A', 'model_B'):
        m = V.VSE(opt, vocab.word2idx); m.load_state_dict(ck[key]); m.val_start(); models.append(m)

    # ---------------- DEV: exact native validate() (r25_co_train.validate) -----------------
    caps_dev, ims_dev = D.get_dataset(opt.data_path, opt.data_name, 'dev')
    val_loader = D.get_loader(caps_dev, ims_dev, vocab, 'dev', opt.batch_size, 0)
    sims_mean, peer_rsum, dev = 0, {}, {}
    for p, m in zip(PEERS, models):
        img, cap, cap_lens = E.encode_data(m, val_loader, opt.log_step)
        img = np.array([img[i] for i in range(0, len(img), 5)])
        sim = E.shard_attn_scores(m, img, cap, cap_lens, opt, shard_size=100)
        r = E.i2t(img.shape[0], sim, 5); ri = E.t2i(img.shape[0], sim, 5)
        peer_rsum[p] = float(sum(r[:3]) + sum(ri[:3]))
        sims_mean = sims_mean + sim
        for nm, arr in (('img', img), ('cap', cap)):
            f32 = arr.astype(np.float32); assert np.array_equal(f32.astype(np.float64), arr), 'non-lossless float32'
            dev[(nm, p)] = f32
    sims_mean = sims_mean / 2
    r = E.i2t(N_DEV_IMG, sims_mean, 5); ri = E.t2i(N_DEV_IMG, sims_mean, 5)
    native = dict(i2t_r1=r[0], i2t_r5=r[1], i2t_r10=r[2], t2i_r1=ri[0], t2i_r5=ri[1], t2i_r10=ri[2],
                  rsum=float(sum(r[:3]) + sum(ri[:3])), peer_rsum=peer_rsum)
    # re-score from the float32 cache with the same native shard function (what the critic jobs read)
    sims_cache = sum(E.shard_attn_scores(m, dev[('img', p)].astype(np.float64), dev[('cap', p)].astype(np.float64), [0] * N_DEV_CAP, opt, shard_size=100)
                     for p, m in zip(PEERS, models)) / 2
    native['cache_rescore_identical'] = bool(np.array_equal(sims_cache, sims_mean))
    native['checkpoint_best_rsum'] = float(ck['best_rsum']); native['checkpoint_epoch'] = int(ck['epoch'])
    native['reproduces_checkpoint_best_rsum_exactly'] = native['rsum'] == float(ck['best_rsum'])
    native['abs_diff_vs_checkpoint'] = abs(native['rsum'] - float(ck['best_rsum']))
    print('NATIVE DEV', native, flush=True)
    files = {}
    def save(name, arr):
        path = cdir / name; np.save(path, arr); files[name] = dict(sha256=sha256(path), shape=list(arr.shape), dtype=str(arr.dtype))
    for (nm, p), arr in dev.items():
        save(f'dev_{nm}_peer{p}.npy', arr)
    save('native_dev_sims.npy', sims_mean.astype(np.float64))

    # ---------------- FIT (train pool), same eval-mode layout -----------------
    caps_tr, ims_tr = D.get_dataset(opt.data_path, opt.data_name, 'train')
    assert len(caps_tr) == N_TRAIN_CAP and ims_tr.shape[0] == N_TRAIN_IMG

    class TrainEvalLayout(torch.utils.data.Dataset):
        def __len__(self): return N_TRAIN_CAP
        def __getitem__(self, k):
            return torch.Tensor(ims_tr[k // 5]), D.process_caption(vocab, caps_tr[k], False), k, k // 5
    loader = torch.utils.data.DataLoader(TrainEvalLayout(), batch_size=opt.batch_size, shuffle=False, pin_memory=True,
                                         collate_fn=D.collate_fn, num_workers=a.workers)
    stats = {}
    for p, m in zip(PEERS, models):
        t0 = time.time()
        img, cap, _ = E.encode_data(m, loader, 200)
        img = img[::5]
        fi, fc = img.astype(np.float32), cap.astype(np.float32)
        assert np.array_equal(fi.astype(np.float64), img) and np.array_equal(fc.astype(np.float64), cap)
        assert fi.shape == (N_TRAIN_IMG, 1024) and fc.shape == (N_TRAIN_CAP, 1024) and np.isfinite(fi).all() and np.isfinite(fc).all()
        save(f'fit_img_peer{p}.npy', fi); save(f'fit_cap_peer{p}.npy', fc)
        st = standardization(fi, fc, obs)
        np.savez(cdir / f'standardization_peer{p}.npz', **st); files[f'standardization_peer{p}.npz'] = dict(sha256=sha256(cdir / f'standardization_peer{p}.npz'))
        stats[p] = dict(encode_seconds=time.time() - t0, img_norm_mean=float(np.linalg.norm(fi, axis=1).mean()))
        del img, cap
    manifest = dict(
        line='R34 DIRECT_CRITIC_RETRIEVAL_v1', cond=a.cond, created_utc=utc(),
        parent=dict(unit=c['h_unit'], model_best=str(ckpt_path), model_best_sha256=sha256(ckpt_path),
                    epoch=int(ck['epoch']), best_rsum=float(ck['best_rsum']), run_json=str(hdir / c['run_json']),
                    code_dir=str(CODE_R30), code_sha256={f: sha256(CODE_R30 / f) for f in ('vse.py', 'data.py', 'evaluation.py', 'vocab.py', 'glove_shim.py')},
                    run_code_sha256_matches=all(run['env']['code_sha256'][f] == sha256(CODE_R30 / f) for f in ('vse.py', 'data.py', 'evaluation.py'))),
        observed_map=dict(path=c['noise_file'], sha256=c['noise_sha256'], noise_ratio=c['noise_ratio'],
                          realized_mismatch=float((obs != np.arange(N_TRAIN_CAP) // 5).mean()),
                          note='same file the H parent trained with; identity at 0% (native data.py uses arange//5 there)'),
        encoding=dict(mode='VSE.forward_emb, model.val_start() (eval: dropout off, BN running stats)',
                      centering='per-batch mean subtraction per modality, then L2 (native forward_emb)',
                      batch_size=int(opt.batch_size), order='sequential, shuffle=False',
                      dev_layout='native dev loader: row k = dev_ims[k] (= image k//5), caption k; img = rows[::5]',
                      fit_layout='row k = train_ims[k//5], caption k (process_caption drop=False), img = rows[::5]; pairing-free',
                      dtype='float32 encoder outputs stored float32 (lossless: checked float64 round trip)',
                      peers='model_A, model_B encoded separately; never cross-peer products'),
        standardization='R32 F1 rule (FIT only): image mean/std under observed-caption-induced image marginal, text uniform, floor 1e-6',
        native_dev=native, files=files, encode_stats=stats,
        env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
        test_guard=guard_record(), elapsed_seconds=time.time() - t_start)
    write_json(cdir / 'CACHE_MANIFEST.json', manifest)
    print('R34_CACHE_DONE', a.cond, native['rsum'], native['reproduces_checkpoint_best_rsum_exactly'])


if __name__ == '__main__':
    main()
