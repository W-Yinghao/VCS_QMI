"""R37S: byte copy of R38 r38c build_cache.py (R37S edits: --cond choice cc_r37s, recipe assertion = the R37S R30-style parent recipe
(common.PARENT_RECIPE) instead of the R31 author defaults, series/manifest text; exact native DEV reproduction rule unchanged).
R38: byte copy of R37C build_cache.py (R38 edits: --cond required (cc_s1002/cc_s1003), manifest line text). R37C (port of R37/R34 build_cache.py to CC152K). Build the frozen native embedding caches of the R31 CC152K parent
(H0_default cc_s1001, both peers), FIT(train) + DEV, and reproduce the parent's logged full-DEV rSum EXACTLY with the
native validate() path (r25_co_train.validate: per_captions = 1 for cc152k_precomp). Refuses to write the manifest if
the reproduction is not exact (float equality with the checkpoint's best_rsum).

Native embedding = VSE.forward_emb in eval mode (dropout off, BN running stats): encoder output, per-batch mean
subtraction (separately per modality) and L2 norm, batch_size 128, sequential order.
DEV: native dev loader exactly as R25 validate (row k = dev_ims[k] = image k, caption k, im_div = 1; img = rows[::1]).
FIT: the same layout applied to train (row k -> train_ims[k] (im_div = 1), caption k, no augmentation).
Observed relation: re-derived from the native PrecompDataset('train', noise_ratio 0) itself (t2i_index), asserted to
be arange(150000) and stored as observed_map.npy. No truth exists for CC152K. No test file is opened (audit guard).
Usage: python -B build_cache.py --data-path <dir containing cc152k_precomp> [--require-gpu NAME]
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PARENT_RECIPE, SERIES, CODE_NATIVE, CONDS, OUT, VOCAB, DATA_NAME, PEERS, PER_CAP, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP,
                    install_guards, guard_record, sha256, write_json, read_json, utc)


def standardization(x_img, y_cap, obs_map, floor=1e-6):
    """R32 F1 standardize_caches rule (kept from R34/R37 for completeness; not used by G units)."""
    x, y = x_img.astype(np.float64), y_cap.astype(np.float64)
    w = np.bincount(obs_map, minlength=len(x)).astype(np.float64) / len(y)
    mx = (w[:, None] * x).sum(0); my = y.mean(0)
    sx = np.sqrt((w[:, None] * (x - mx) ** 2).sum(0)).clip(min=floor)
    sy = np.sqrt(((y - my) ** 2).mean(0)).clip(min=floor)
    return dict(image_mean=mx.astype(np.float32), image_std=sx.astype(np.float32),
                text_mean=my.astype(np.float32), text_std=sy.astype(np.float32), std_floor=np.float64(floor))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True, choices=('cc_r37s',))
    ap.add_argument('--data-path', required=True); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--require-gpu', default='')
    a = ap.parse_args()
    install_guards(role='gate')
    t_start = time.time()
    import torch
    sys.path.insert(0, str(CODE_NATIVE))
    import vse as V, data as D, evaluation as E
    from vocab import deserialize_vocab
    torch.backends.cudnn.benchmark = True  # native (r25_run.py)

    c = CONDS[a.cond]; hdir = Path(c['parent_dir']); cdir = Path(c['cache_dir'])
    assert str(cdir).startswith(str(OUT))
    if a.require_gpu:
        assert torch.cuda.get_device_name(0) == a.require_gpu, (torch.cuda.get_device_name(0), a.require_gpu)
    if (cdir / 'CACHE_MANIFEST.json').exists():
        print('cache already built', cdir); return
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = hdir / 'model_best.pth.tar'
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    opt = ck['opt']; run = read_json(hdir / c['run_json'])
    assert opt.data_name == DATA_NAME and float(opt.noise_ratio) == 0 and run['opt']['data_name'] == DATA_NAME
    assert all(run['opt'][k] == v for k, v in PARENT_RECIPE.items()) and all(getattr(opt, k) == v for k, v in PARENT_RECIPE.items()), (PARENT_RECIPE, run['opt'])
    assert int(opt.seed) == c['parent_seed']
    run_code_match = {f: run['env']['code_sha256'][f] == sha256(CODE_NATIVE / f) for f in run['env']['code_sha256']}
    assert all(run_code_match.values()), ('native code differs from the parent run', run_code_match)
    opt.data_path = a.data_path; opt.vocab_path = str(VOCAB); opt.workers = 0
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    models = []
    for key in ('model_A', 'model_B'):
        m = V.VSE(opt, vocab.word2idx); m.load_state_dict(ck[key]); m.val_start(); models.append(m)

    # ---------------- DEV: exact native validate() (r25_co_train.validate, per_captions = 1 for cc152k) -------------
    per = 1 if opt.data_name == 'cc152k_precomp' else 5
    assert per == PER_CAP
    caps_dev, ims_dev = D.get_dataset(opt.data_path, opt.data_name, 'dev')
    assert len(caps_dev) == N_DEV_CAP and ims_dev.shape[0] == N_DEV_IMG
    val_loader = D.get_loader(caps_dev, ims_dev, vocab, 'dev', opt.batch_size, 0)
    assert val_loader.dataset.im_div == 1 and np.array_equal(val_loader.dataset.t2i_index, np.arange(N_DEV_CAP))
    sims_mean, peer_rsum, dev = 0, {}, {}
    for p, m in zip(PEERS, models):
        img, cap, cap_lens = E.encode_data(m, val_loader, opt.log_step)
        img = np.array([img[i] for i in range(0, len(img), per)])
        sim = E.shard_attn_scores(m, img, cap, cap_lens, opt, shard_size=100)
        r = E.i2t(img.shape[0], sim, per); ri = E.t2i(img.shape[0], sim, per)
        peer_rsum[p] = float(sum(r[:3]) + sum(ri[:3]))
        sims_mean = sims_mean + sim
        for nm, arr in (('img', img), ('cap', cap)):
            f32 = arr.astype(np.float32); assert np.array_equal(f32.astype(np.float64), arr), 'non-lossless float32'
            dev[(nm, p)] = f32
    sims_mean = sims_mean / 2
    r = E.i2t(N_DEV_IMG, sims_mean, per); ri = E.t2i(N_DEV_IMG, sims_mean, per)
    native = dict(i2t_r1=r[0], i2t_r5=r[1], i2t_r10=r[2], t2i_r1=ri[0], t2i_r5=ri[1], t2i_r10=ri[2],
                  rsum=float(sum(r[:3]) + sum(ri[:3])), peer_rsum=peer_rsum, per_captions=per)
    sims_cache = sum(E.shard_attn_scores(m, dev[('img', p)].astype(np.float64), dev[('cap', p)].astype(np.float64), [0] * N_DEV_CAP, opt, shard_size=100)
                     for p, m in zip(PEERS, models)) / 2
    native['cache_rescore_identical'] = bool(np.array_equal(sims_cache, sims_mean))
    native['checkpoint_best_rsum'] = float(ck['best_rsum']); native['checkpoint_epoch'] = int(ck['epoch'])
    native['reproduces_checkpoint_best_rsum_exactly'] = native['rsum'] == float(ck['best_rsum'])
    native['abs_diff_vs_checkpoint'] = abs(native['rsum'] - float(ck['best_rsum']))
    ep = [e for e in read_json(hdir / 'EPOCHS.json')['cotrain'] if e['epoch'] == int(ck['epoch'])]
    native['logged_epoch_dev'] = ep[0]['dev'] if ep else None
    native['six_recalls_equal_logged_epoch'] = bool(ep) and all(abs(native[k] - ep[0]['dev'][k]) < 1e-9 for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10'))
    print('NATIVE DEV', native, flush=True)
    if not (native['reproduces_checkpoint_best_rsum_exactly'] and native['cache_rescore_identical']):
        write_json(cdir / f"FAILED_REPRO_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(native=native, gpu=torch.cuda.get_device_name(0), utc=utc()))
        raise SystemExit('R37S: native DEV not reproduced exactly; cache NOT written')
    files = {}
    def save(name, arr):
        path = cdir / name; np.save(path, arr); files[name] = dict(sha256=sha256(path), shape=list(arr.shape), dtype=str(arr.dtype))
    for (nm, p), arr in dev.items():
        save(f'dev_{nm}_peer{p}.npy', arr)
    save('native_dev_sims.npy', sims_mean.astype(np.float64))

    # ---------------- FIT (train pool): observed relation from the native loader, eval-mode layout ---------------
    caps_tr, ims_tr = D.get_dataset(opt.data_path, opt.data_name, 'train')
    assert len(caps_tr) == N_TRAIN_CAP and ims_tr.shape[0] == N_TRAIN_IMG
    nds = D.PrecompDataset(caps_tr, ims_tr, vocab, 'train', 0, 'none')   # native train dataset (warmup/eval_train layout)
    assert nds.im_div == 1
    obs = np.asarray(nds.t2i_index, dtype=np.int64)
    assert obs.shape == (N_TRAIN_CAP,) and np.array_equal(obs, np.arange(N_TRAIN_CAP)), 'native CC152K observed relation must be the identity'
    save('observed_map.npy', obs)
    ids = [l[0] for l in csv.reader(open(os.path.join(opt.data_path, opt.data_name, 'train_caps.tsv')), delimiter='\t')]
    cap_dup = collections.Counter(caps_tr)

    class TrainEvalLayout(torch.utils.data.Dataset):
        def __len__(self): return N_TRAIN_CAP
        def __getitem__(self, k):
            return torch.Tensor(ims_tr[k]), D.process_caption(vocab, caps_tr[k], False), k, k
    loader = torch.utils.data.DataLoader(TrainEvalLayout(), batch_size=opt.batch_size, shuffle=False, pin_memory=True,
                                         collate_fn=D.collate_fn, num_workers=a.workers)
    stats = {}
    for p, m in zip(PEERS, models):
        t0 = time.time()
        img, cap, _ = E.encode_data(m, loader, 200)
        img = img[::PER_CAP]
        fi, fc = img.astype(np.float32), cap.astype(np.float32)
        assert np.array_equal(fi.astype(np.float64), img) and np.array_equal(fc.astype(np.float64), cap)
        assert fi.shape == (N_TRAIN_IMG, 1024) and fc.shape == (N_TRAIN_CAP, 1024) and np.isfinite(fi).all() and np.isfinite(fc).all()
        save(f'fit_img_peer{p}.npy', fi); save(f'fit_cap_peer{p}.npy', fc)
        st = standardization(fi, fc, obs)
        np.savez(cdir / f'standardization_peer{p}.npz', **st); files[f'standardization_peer{p}.npz'] = dict(sha256=sha256(cdir / f'standardization_peer{p}.npz'))
        hi = collections.Counter(hashlib.sha1(r.tobytes()).digest() for r in fi)
        stats[p] = dict(encode_seconds=time.time() - t0, img_norm_mean=float(np.linalg.norm(fi, axis=1).mean()),
                        n_image_rows_with_bitwise_duplicate_embedding=int(sum(v for v in hi.values() if v > 1)))
        del img, cap
    manifest = dict(
        line='R37S CC152K cache of parent %s (R37C/R38 build_cache rule unchanged; development experiment, no TEST read)' % a.cond, cond=a.cond,
        parent_seed=c['parent_seed'], created_utc=utc(),
        parent=dict(unit=c['h_unit'], series=SERIES,
                    model_best=str(ckpt_path), model_best_sha256=sha256(ckpt_path),
                    epoch=int(ck['epoch']), best_rsum=float(ck['best_rsum']), run_json=str(hdir / c['run_json']),
                    code_dir=str(CODE_NATIVE), code_sha256={f: sha256(CODE_NATIVE / f) for f in ('vse.py', 'data.py', 'evaluation.py', 'vocab.py', 'glove_shim.py', 'r25_co_train.py')},
                    run_code_sha256_matches=all(run_code_match.values())),
        observed_map=dict(path=str(cdir / 'observed_map.npy'), sha256=files['observed_map.npy']['sha256'], noise_ratio=0.0,
                          rule='native PrecompDataset(train, noise_ratio=0): images.shape[0] == len(captions) -> im_div = 1, t2i_index = arange(150000); record k = image row k <-> caption row k',
                          is_identity=True, train_ids_unique=len(set(ids)) == len(ids), n_train_ids=len(ids),
                          n_caption_rows_with_duplicate_text=int(sum(v for v in cap_dup.values() if v > 1)), n_distinct_caption_texts=len(cap_dup),
                          note='CC152K real (uninjected) noise: the observed relation is the only relation; no truth exists, none is used'),
        encoding=dict(mode='VSE.forward_emb, model.val_start() (eval: dropout off, BN running stats)',
                      centering='per-batch mean subtraction per modality, then L2 (native forward_emb)',
                      batch_size=int(opt.batch_size), order='sequential, shuffle=False',
                      dev_layout='native dev loader: row k = dev_ims[k] (= image k, im_div 1), caption k; per_captions = 1',
                      fit_layout='row k = train_ims[k], caption k (process_caption drop=False); pairing-free',
                      dtype='float32 encoder outputs stored float32 (lossless: checked float64 round trip)',
                      peers='model_A, model_B encoded separately; never cross-peer products'),
        native_dev=native, files=files, encode_stats=stats,
        env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
        test_guard=guard_record(), elapsed_seconds=time.time() - t_start)
    write_json(cdir / 'CACHE_MANIFEST.json', manifest)
    print('R37S_CACHE_DONE', a.cond, native['rsum'], native['reproduces_checkpoint_best_rsum_exactly'])


if __name__ == '__main__':
    main()
