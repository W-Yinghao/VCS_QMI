"""R43 A (MS-COCO): tokens (once) + frozen native caches + prep + JOINT prep of one IN2R-native R33 COCO parent (H0, R25 IN2R code
configuration) on the r43o tree. Same pipeline rules as tools/r42_build_parent.py (Flickr) and the R37C CC152K port (no new rule):
 0. tokens = data.process_caption(vocab, caption, drop=False) of the COCO train / dev captions with the native coco vocab + '<mask>'
             (R35 tokens.npz rule, R39 CC152K tokens rule); built once into R43_PARENTS/coco/tokens_coco.npz.
 1. cache  = R37 build_cache rule (R34): VSE.forward_emb in eval mode, per-batch centring + L2, batch 128, sequential native layout;
             DEV by the exact native validate() path (native dev loader, img = rows[::5], per_captions 5 -> 1000 images x 5000 captions);
             FIT row k -> train_ims[k // 5], caption k; both peers. REFUSES to write the manifest unless the DEV rSum equals the
             checkpoint's best_rsum exactly and the float32-cache re-score is identical (R37C rule).
 2. prep   = R37C prep rules: record locks rebuilt for the 566435 COCO records with R35 prep.record_locks and the SAME seeds
             (20261110..20261113); common positive affine a0/b0 (balanced logistic, cal_p vs cal_q, observed map of the parent); natural
             top-32 FIT candidate tables (peer-mean native fp32 score over the full pool, torch.topk). No B-TRUST (N route only).
 3. JOINT  = R39X / R42 JOINT prep rule: parent per-native-batch centring vectors, student-vs-cache equivalence, DEV retrieval from the
             student vectors.
No test file is opened (TEST guard, role gate). Usage:
  python -B tools/r43_build_coco.py --cond c60in_s1001 --data-path <staged dir containing coco_precomp> [--require-gpu NAME]"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r43o'))
import common as C
from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, PER_CAP, VOCAB, install_guards, guard_record, sha256, write_json, read_json, utc

N_CAL_P, N_CAL_Q, N_LIST = 5000, 50000, 1024


def native_modules():
    if str(C.CODE_IN2R) not in sys.path:
        sys.path.insert(1, str(C.CODE_IN2R))
    import vse as V, data as D, evaluation as E
    from vocab import deserialize_vocab
    return V, D, E, deserialize_vocab


def build_tokens(data_path, any_ckpt):
    import torch
    if C.TOKENS_FILE.exists():
        print('tokens already built', C.TOKENS_FILE); return
    V, D, E, deserialize_vocab = native_modules()
    vocab = deserialize_vocab(os.path.join(str(VOCAB), f'{C.DATA_NAME}_vocab.json')); vocab.add_word('<mask>')
    ck = torch.load(any_ckpt, map_location='cpu', weights_only=False)
    assert len(vocab) == int(ck['opt'].vocab_size) == ck['model_A'][1]['embed.weight'].shape[0], (len(vocab), ck['opt'].vocab_size)
    out = {}
    for split in ('train', 'dev'):
        caps, _ = D.get_dataset(data_path, C.DATA_NAME, split)
        toks = [D.process_caption(vocab, c, False).long().numpy() for c in caps]
        L = max(len(t) for t in toks); arr = np.zeros((len(toks), L), np.int64)
        for k, t in enumerate(toks):
            arr[k, :len(t)] = t
        out[f'{split}_tok'] = arr; out[f'{split}_len'] = np.array([len(t) for t in toks], np.int64)
    assert len(out['train_len']) == N_TRAIN_CAP and len(out['dev_len']) == N_DEV_CAP
    C.TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.TOKENS_FILE.with_name('tokens_coco.tmp.npz'); np.savez(tmp, **out); os.replace(tmp, C.TOKENS_FILE)
    write_json(C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json'), dict(path=str(C.TOKENS_FILE), sha256=sha256(C.TOKENS_FILE), vocab_size=len(vocab), created_utc=utc(),
                                                                    rule='data.process_caption(vocab, caption, drop=False), native coco_precomp vocab + <mask> (R35 / R39 rule)',
                                                                    shapes={k: list(v.shape) for k, v in out.items()}, test_guard=guard_record()))
    print('R43_TOKENS_DONE', {k: v.shape for k, v in out.items()}, flush=True)


def build_cache(cond, data_path, require_gpu):
    import torch
    c = C.CONDS[cond]; hdir = Path(c['parent_dir']); cdir = Path(c['cache_dir'])
    if (cdir / 'CACHE_MANIFEST.json').exists():
        print('cache already built', cdir); return
    V, D, E, deserialize_vocab = native_modules()
    torch.backends.cudnn.benchmark = True  # native (r25_run.py)
    if require_gpu:
        assert torch.cuda.get_device_name(0) == require_gpu, (torch.cuda.get_device_name(0), require_gpu)
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = hdir / 'model_best.pth.tar'
    done = read_json(hdir / 'DONE.json'); assert done['status'] == 'COMPLETE' and done['model_best_sha256'] == sha256(ckpt_path)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    opt = ck['opt']; run = read_json(hdir / c['run_json'])
    assert float(opt.noise_ratio) == c['noise_ratio'] and run['opt']['noise_file'] == c['noise_file'] == opt.noise_file and opt.data_name == C.DATA_NAME
    assert sha256(c['noise_file']) == c['noise_sha256']
    run_code_match = {f: run['env']['code_sha256'][f] == sha256(C.CODE_IN2R / f) for f in ('vse.py', 'data.py', 'evaluation.py', 'vocab.py')}
    assert all(run_code_match.values()), run_code_match
    obs = np.load(c['noise_file']); assert obs.shape == (N_TRAIN_CAP,) and obs.dtype == np.int64
    opt.data_path = data_path; opt.vocab_path = str(VOCAB); opt.workers = 0
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    models = []
    for key in ('model_A', 'model_B'):
        m = V.VSE(opt, vocab.word2idx); m.load_state_dict(ck[key]); m.val_start(); models.append(m)
    caps_dev, ims_dev = D.get_dataset(opt.data_path, opt.data_name, 'dev')
    assert len(caps_dev) == N_DEV_CAP and ims_dev.shape[0] == N_DEV_CAP   # COCO dev: one image row per caption (native im_div 1 on dev)
    val_loader = D.get_loader(caps_dev, ims_dev, vocab, 'dev', opt.batch_size, 0)
    sims_mean, peer_rsum, dev = 0, {}, {}
    for p, m in zip(PEERS, models):
        img, cap, cap_lens = E.encode_data(m, val_loader, opt.log_step)
        img = np.array([img[i] for i in range(0, len(img), PER_CAP)])   # native validate(): per_captions 5 for coco_precomp
        sim = E.shard_attn_scores(m, img, cap, cap_lens, opt, shard_size=100)
        r = E.i2t(img.shape[0], sim, PER_CAP); ri = E.t2i(img.shape[0], sim, PER_CAP)
        peer_rsum[p] = float(sum(r[:3]) + sum(ri[:3]))
        sims_mean = sims_mean + sim
        for nm, arr in (('img', img), ('cap', cap)):
            f32 = arr.astype(np.float32); assert np.array_equal(f32.astype(np.float64), arr), 'non-lossless float32'
            dev[(nm, p)] = f32
    assert dev[('img', 'A')].shape == (N_DEV_IMG, 1024) and dev[('cap', 'A')].shape == (N_DEV_CAP, 1024)
    sims_mean = sims_mean / 2
    r = E.i2t(N_DEV_IMG, sims_mean, PER_CAP); ri = E.t2i(N_DEV_IMG, sims_mean, PER_CAP)
    native = dict(i2t_r1=r[0], i2t_r5=r[1], i2t_r10=r[2], t2i_r1=ri[0], t2i_r5=ri[1], t2i_r10=ri[2], rsum=float(sum(r[:3]) + sum(ri[:3])), peer_rsum=peer_rsum)
    sims_cache = sum(E.shard_attn_scores(m, dev[('img', p)].astype(np.float64), dev[('cap', p)].astype(np.float64), [0] * N_DEV_CAP, opt, shard_size=100)
                     for p, m in zip(PEERS, models)) / 2
    native['cache_rescore_identical'] = bool(np.array_equal(sims_cache, sims_mean))
    native['checkpoint_best_rsum'] = float(ck['best_rsum']); native['checkpoint_epoch'] = int(ck['epoch'])
    native['done_json_selected_dev_rsum'] = float(done['selected_dev_rsum'])
    native['reproduces_checkpoint_best_rsum_exactly'] = native['rsum'] == float(ck['best_rsum'])
    native['abs_diff_vs_checkpoint'] = abs(native['rsum'] - float(ck['best_rsum']))
    print('NATIVE DEV', cond, native, flush=True)
    if not (native['reproduces_checkpoint_best_rsum_exactly'] and native['cache_rescore_identical']):
        write_json(cdir / f"FAILED_REPRO_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(native=native, gpu=torch.cuda.get_device_name(0), utc=utc()))
        raise SystemExit(f'R43: native DEV of {cond} not reproduced exactly; cache NOT written')
    files = {}

    def save(name, arr):
        path = cdir / name; np.save(path, arr); files[name] = dict(sha256=sha256(path), shape=list(arr.shape), dtype=str(arr.dtype))
    for (nm, p), arr in dev.items():
        save(f'dev_{nm}_peer{p}.npy', arr)
    save('native_dev_sims.npy', sims_mean.astype(np.float64))
    caps_tr, ims_tr = D.get_dataset(opt.data_path, opt.data_name, 'train')
    assert len(caps_tr) == N_TRAIN_CAP and ims_tr.shape[0] == N_TRAIN_IMG

    class TrainEvalLayout(torch.utils.data.Dataset):
        def __len__(self): return N_TRAIN_CAP

        def __getitem__(self, k):
            return torch.Tensor(ims_tr[k // PER_CAP]), D.process_caption(vocab, caps_tr[k], False), k, k // PER_CAP
    loader = torch.utils.data.DataLoader(TrainEvalLayout(), batch_size=opt.batch_size, shuffle=False, pin_memory=True, collate_fn=D.collate_fn, num_workers=4)
    stats = {}
    for p, m in zip(PEERS, models):
        t0 = time.time()
        img, cap, _ = E.encode_data(m, loader, 500)
        img = img[::PER_CAP]
        fi, fc = img.astype(np.float32), cap.astype(np.float32)
        assert np.array_equal(fi.astype(np.float64), img) and np.array_equal(fc.astype(np.float64), cap)
        assert fi.shape == (N_TRAIN_IMG, 1024) and fc.shape == (N_TRAIN_CAP, 1024) and np.isfinite(fi).all() and np.isfinite(fc).all()
        save(f'fit_img_peer{p}.npy', fi); save(f'fit_cap_peer{p}.npy', fc)
        stats[p] = dict(encode_seconds=time.time() - t0, img_norm_mean=float(np.linalg.norm(fi, axis=1).mean()))
    del ims_tr
    man = dict(line='R43 COCO cache (R37/R34 build_cache rule; IN2R-native R33 H0 parent)', cond=cond, parent_seed=c['parent_seed'], created_utc=utc(),
               parent=dict(unit=c['h_unit'], model_best=str(ckpt_path), model_best_sha256=sha256(ckpt_path), epoch=int(ck['epoch']), best_rsum=float(ck['best_rsum']),
                           run_json=str(hdir / c['run_json']), code_dir=str(C.CODE_IN2R), code_sha256={f: sha256(C.CODE_IN2R / f) for f in ('vse.py', 'data.py', 'evaluation.py', 'vocab.py')},
                           run_code_sha256_matches=run_code_match, run_opt={k: run['opt'].get(k) for k in ('warmup_epoch', 'k', 'i2t', 't2i', 'noise_ratio', 'seed')}),
               observed_map=dict(path=c['noise_file'], sha256=c['noise_sha256'], noise_ratio=c['noise_ratio'], realized_mismatch=float((obs != np.arange(N_TRAIN_CAP) // PER_CAP).mean()),
                                 is_identity=False, note='the file the parent trained with (run opt noise_file)'),
               encoding=dict(mode='VSE.forward_emb, model.val_start()', centering='per-batch mean subtraction per modality, then L2', batch_size=int(opt.batch_size),
                             dev_layout='native dev loader (5000 rows, one image row per caption), img = rows[::5] (validate per_captions 5)',
                             fit_layout='row k = train_ims[k//5], caption k (drop=False), img = rows[::5]'),
               native_dev=native, files=files, encode_stats=stats,
               env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
               test_guard=guard_record())
    write_json(cdir / 'CACHE_MANIFEST.json', man)
    print('R43_CACHE_DONE', cond, native['rsum'], flush=True)


def record_locks(n_rec=N_TRAIN_CAP):
    """R35 prep.record_locks verbatim (R37C copy), parametrised by the number of records (same seeds 20261110..20261113)."""
    rng = np.random.default_rng(20261110)
    perm = rng.permutation(n_rec)
    cal_p, ev_p = np.sort(perm[:N_CAL_P]), np.sort(perm[N_CAL_P:2 * N_CAL_P])
    def qpairs(r):
        a = r.integers(0, n_rec, N_CAL_Q); b = r.integers(0, n_rec, N_CAL_Q)
        keep = a != b
        while not keep.all():
            b[~keep] = r.integers(0, n_rec, int((~keep).sum())); keep = a != b
        return np.stack([a, b], 1)
    cal_q = qpairs(np.random.default_rng(20261111)); ev_q = qpairs(np.random.default_rng(20261112))
    r = np.random.default_rng(20261113)
    lp = r.permutation(n_rec)[:4 * N_LIST]
    L = dict(cal_p=cal_p, cal_q=cal_q, ev_p=ev_p, ev_q=ev_q,
             lcal_i=lp[:N_LIST], lcal_t=lp[N_LIST:2 * N_LIST], lev_i=lp[2 * N_LIST:3 * N_LIST], lev_t=lp[3 * N_LIST:])
    for k in ('lcal_i', 'lcal_t', 'lev_i', 'lev_t'):
        L[k + '_perm'] = np.stack([r.permutation(32) for _ in range(N_LIST)])
    return L


def build_prep(cond):
    import torch
    import evals as EV
    from data35 import load_cond
    c = C.CONDS[cond]; pdir = Path(c['prep_dir']); pdir.mkdir(parents=True, exist_ok=True)
    if (pdir / 'PREP_MANIFEST.json').exists():
        print('prep already built', cond); return
    files = {}

    def save_npz(name, **arrs):
        np.savez(pdir / name, **arrs); files[name] = dict(sha256=sha256(pdir / name), keys=sorted(arrs))
    L = record_locks(); save_npz('locks_records.npz', **L)
    d = load_cond(cond, torch.device('cuda'), need_prep=False); obs = d['obs_np']

    def nps(img_idx, cap_idx):
        out = 0
        with torch.no_grad():
            for p in PEERS:
                s = (d[('fit', 'img', p)][torch.as_tensor(img_idx, device='cuda')] * d[('fit', 'cap', p)][torch.as_tensor(cap_idx, device='cuda')]).sum(-1)
                out = out + s.double().cpu().numpy()
        return out / 2
    sp = nps(obs[L['cal_p']], L['cal_p']); sq = nps(obs[L['cal_q'][:, 0]], L['cal_q'][:, 1])
    g = np.concatenate([sp, sq]); wp = np.r_[np.full(len(sp), 1 / len(sp)), np.zeros(len(sq))]; wq = np.r_[np.zeros(len(sp)), np.full(len(sq), 1 / len(sq))]
    fit = EV.fit_affine(g, wp, wq, 'balanced_logistic'); assert fit['a'] > 0 and fit['finite']
    cal = {cond: dict(a0=fit['a'], b0=fit['b'], balanced_logistic_value=fit['value'], fit_set='R43 COCO locks cal_p (P) + cal_q (Q: image of record a, caption of record b), observed map of the parent',
                      score='native peer-mean fp32 dot of the R43 cache', J_raw_fit=EV.j_weighted(g, wp, wq), J_affine_fit=EV.j_weighted(fit['a'] * g + fit['b'], wp, wq),
                      dev_rank_invariance=EV.rank_invariance(d['native_dev'], fit['a'], fit['b']))}
    t1 = time.time()
    xs = [d[('fit', 'img', p)] for p in PEERS]; ys = [d[('fit', 'cap', p)] for p in PEERS]
    i2t = np.zeros((N_TRAIN_IMG, 32), np.int32); t2i = np.zeros((N_TRAIN_CAP, 32), np.int32)
    with torch.no_grad():
        for i in range(0, N_TRAIN_IMG, 500):
            i2t[i:i + 500] = (sum(x[i:i + 500] @ y.T for x, y in zip(xs, ys)) / 2).topk(32, dim=1).indices.int().cpu().numpy()
        for j in range(0, N_TRAIN_CAP, 4000):
            t2i[j:j + 4000] = (sum(y[j:j + 4000] @ x.T for x, y in zip(xs, ys)) / 2).topk(32, dim=1).indices.int().cpu().numpy()
    name = f'candidates_{cond}.npz'; save_npz(name, i2t_top32=i2t, t2i_top32=t2i)
    cands = {cond: dict(seconds=time.time() - t1, rule=f'peer-mean native fp32 score over the full FIT pool ({N_TRAIN_IMG} x {N_TRAIN_CAP}), torch.topk(32) (R35/R37 rule)',
                        i2t_frac_images_with_an_observed_caption_in_top32=float((obs[i2t.astype(np.int64)] == np.arange(N_TRAIN_IMG)[:, None]).any(1).mean()),
                        t2i_frac_captions_observed_image_in_top32=float((t2i.astype(np.int64) == obs[:, None]).any(1).mean()))}
    man = dict(line='R43 COCO prep (R37C prep rules rebuilt for MS-COCO; development experiment, no TEST read)', created_utc=utc(),
               locks=dict(rule=f'R35 prep.record_locks with n_records={N_TRAIN_CAP} (seeds 20261110/1/2/3); Q pairs a != b', sizes={k: list(v.shape) for k, v in L.items()}),
               calibration_init=cal, candidates=cands, files=files, test_guard=guard_record(),
               env=dict(gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__))
    write_json(pdir / 'PREP_MANIFEST.json', man)
    print('R43_PREP_DONE', cond, cal[cond]['a0'], cands[cond], flush=True)


def build_joint(cond):
    import torch
    import r39ext as R9
    from data35 import load_cond, shard_mm
    import evals as EV
    jp = C.JOINT_PARENTS[cond]; pdir = Path(jp['prep_dir']); pdir.mkdir(parents=True, exist_ok=True)
    if (pdir / 'JOINT_PREP_MANIFEST.json').exists():
        print('joint prep already built', cond); return
    tm = C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json'); tw = time.time()
    while not tm.exists():   # written by the c60in_s1001 build job (one writer)
        assert time.time() - tw < 4 * 3600, 'R43: tokens_coco.npz not built (c60in_s1001 build job)'
        time.sleep(60)
    assert sha256(C.TOKENS_FILE) == read_json(tm)['sha256']
    dev = torch.device('cuda'); t0 = time.time()
    d = load_cond(cond, dev)
    ck = R9.load_parent(jp['ckpt']); stud = R9.build_students(ck, dev)
    assert int(ck['opt'].vocab_size) == ck['model_A'][1]['embed.weight'].shape[0] and int(np.load(C.TOKENS_FILE)['train_tok'].max()) < int(ck['opt'].vocab_size)
    tr = np.load(C.RAW_FIT_IMS, mmap_mode='r'); dv = np.load(C.RAW_DEV_IMS, mmap_mode='r'); T = np.load(C.TOKENS_FILE)
    assert tr.shape == (N_TRAIN_IMG, 36, 2048) and dv.shape == (N_DEV_CAP, 36, 2048)
    ims = dict(fit=torch.from_numpy(np.array(tr)), dev=torch.from_numpy(np.ascontiguousarray(dv)))
    tok = dict(fit=torch.from_numpy(T['train_tok']).to(dev), dev=torch.from_numpy(T['dev_tok']).to(dev))
    ln = dict(fit=torch.from_numpy(T['train_len']).to(dev), dev=torch.from_numpy(T['dev_len']).to(dev))
    nrow = dict(fit=N_TRAIN_CAP, dev=N_DEV_CAP); files = {}; eq = {}; devu = {}
    for p in PEERS:
        sp = stud[p]
        for split in ('fit', 'dev'):
            with torch.no_grad():
                src = ims[split]; n_src = src.shape[0]; E_img = torch.empty(n_src, 1024, device=dev)
                for i in range(0, n_src, 512):
                    E_img[i:i + 512] = sp.enc_img(src[i:i + 512].to(dev))
                E_cap = torch.empty(nrow[split], 1024, device=dev)
                for i in range(0, nrow[split], 2048):
                    E_cap[i:i + 2048] = sp.enc_cap(tok[split][i:i + 2048], ln[split][i:i + 2048])
            rows = np.arange(nrow[split]); row_img_src = rows // R9.PER_CAP if split == 'fit' else rows
            ei = E_img[torch.as_tensor(row_img_src, device=dev)].double().cpu().numpy(); ec = E_cap.double().cpu().numpy()
            starts = np.arange(0, nrow[split], R9.NATIVE_BATCH); cnt = np.diff(np.append(starts, nrow[split]))[:, None]
            mu_i = (np.add.reduceat(ei, starts, axis=0) / cnt).astype(np.float32); mu_c = (np.add.reduceat(ec, starts, axis=0) / cnt).astype(np.float32)
            del ei, ec
            for kind, mu in (('img', mu_i), ('cap', mu_c)):
                fn = pdir / f'center_{split}_{kind}_peer{p}.npy'; np.save(fn, mu); files[fn.name] = dict(sha256=sha256(fn), shape=list(mu.shape))
            n_img = N_TRAIN_IMG if split == 'fit' else N_DEV_IMG
            item_img_row = np.arange(n_img) * (1 if split == 'fit' else C.DEV_IMS_ROW_STRIDE)
            with torch.no_grad():
                for kind, E, src_rows, n in (('img', E_img, item_img_row, n_img), ('cap', E_cap, np.arange(nrow[split]), nrow[split])):
                    ids = torch.arange(n, device=dev)
                    mu = torch.from_numpy(np.load(pdir / f'center_{split}_{kind}_peer{p}.npy')).to(dev)
                    u = R9.native_vse().l2norm(E[torch.as_tensor(src_rows, device=dev)] - mu[R9.batch_index(kind, ids)])
                    ref = d[(split, kind, p)]; dl = (u - ref).abs()
                    eq[f'{split}_{kind}_{p}'] = dict(n=int(n), max_abs=float(dl.max()), mean_abs=float(dl.mean()), frac_rows_bitwise=float((dl == 0).all(1).float().mean()), min_cos=float((u * ref).sum(1).min()))
                    if split == 'dev':
                        devu[(kind, p)] = u
            del E_img, E_cap
    sims = sum(shard_mm(devu[('img', p)], devu[('cap', p)]) for p in PEERS).cpu().numpy() / 2
    m_stu, _, rk_stu = EV.full_metrics(sims); m_nat, _, rk_nat = EV.full_metrics(d['native_dev'])
    eq['dev_native_from_student_vectors'] = dict(rsum=m_stu['rsum'], native_rsum=m_nat['rsum'], max_abs_sim_diff=float(np.abs(sims - d['native_dev']).max()),
                                                 n_i2t_rank_diff=int((rk_stu['ranks_i'] != rk_nat['ranks_i']).sum()), n_t2i_rank_diff=int((rk_stu['ranks_t'] != rk_nat['ranks_t']).sum()))
    summ = dict(max_abs_over_all=max(v['max_abs'] for k, v in eq.items() if isinstance(v, dict) and 'max_abs' in v), encode_seconds=time.time() - t0)
    man = dict(line='R43 COCO JOINT prep (R39X prep_joint_x / R42 rule)', created_utc=utc(), ds=C.DS_KEY, cond=cond, target_cond=cond, prep_key=cond, extension='R43',
               parent=dict(ckpt=str(jp['ckpt']), model_best_sha256=sha256(jp['ckpt']), vocab_size=int(ck['opt'].vocab_size), code_vse_sha256=sha256(Path(jp['code']) / 'vse.py'),
                           epoch=int(ck['epoch']), best_rsum=float(ck['best_rsum'])),
               target_cache_manifest_sha256=d['manifest_sha256'], native_code=dict(dir=str(C.NATIVE_CODE), vse_sha256=sha256(C.NATIVE_CODE / 'vse.py')),
               raw=dict(fit=str(C.RAW_FIT_IMS), dev=str(C.RAW_DEV_IMS), dev_row_stride=C.DEV_IMS_ROW_STRIDE),
               tokens=dict(path=str(C.TOKENS_FILE), sha256=sha256(C.TOKENS_FILE), rule='R43 tokens_coco.npz: data.process_caption(vocab, caption, drop=False), coco vocab + <mask>'), files=files,
               centring_rule='parent per-native-batch mean (128 rows, per modality), held fixed for the student (R39 registered patch)',
               equivalence=eq, equivalence_summary=summ, test_guard=guard_record(), env=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__))
    write_json(pdir / 'JOINT_PREP_MANIFEST.json', man)
    print('R43_JOINT_PREP', cond, json.dumps(summ), json.dumps(eq['dev_native_from_student_vectors']), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True, choices=tuple(C.CONDS)); ap.add_argument('--data-path', required=True)
    ap.add_argument('--require-gpu', default='')
    a = ap.parse_args()
    install_guards(role='gate')
    if a.cond == C.TARGET_COND:   # one writer: the tokens are shared by the five COCO parents (same vocab; asserted against each checkpoint at JOINT prep)
        build_tokens(a.data_path, Path(C.CONDS[a.cond]['parent_dir']) / 'model_best.pth.tar')
    build_cache(a.cond, a.data_path, a.require_gpu)   # native path exactly as R37 build_cache (no determinism flags, cudnn.benchmark True)
    import train_unit as TU
    TU.set_determinism()
    build_prep(a.cond)
    build_joint(a.cond)
    print('R43_BUILD_COCO_DONE', a.cond)


if __name__ == '__main__':
    main()
