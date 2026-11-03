"""R41 C: frozen native caches + prep + JOINT prep of one IN2R-native Flickr parent (f40in = R31 H0_default f40_s1001, f80in = R33 H0
f80_s1001), with the existing pipeline rules (no new rule):
 1. cache  = R37 build_cache.py rule (R34): VSE.forward_emb in eval mode, per-batch centring + L2, batch 128, sequential native layout;
             DEV by the exact native validate() path; FIT row k -> train_ims[k//5], caption k; both peers. Native code = the parent's own
             code (R25 IN2R copy; byte-identical vse/data/evaluation to the R30 code). REFUSES to write the manifest unless the DEV rSum
             equals the checkpoint's best_rsum exactly and the float32-cache re-score is identical (R37C rule).
 2. prep   = R37 prep.py rule (R35): common positive affine a0/b0 (balanced logistic on the R35 FIT calibration records, observed map of
             the parent), natural top-32 FIT candidate tables (peer-mean native score); R35 record locks / tokens shared (hash-bound).
             No B-TRUST (N route only).
 3. JOINT  = R39X prep_joint_x.py rule: parent per-native-batch centring vectors, student-vs-cache equivalence, DEV retrieval from
             student vectors.
No test file is opened (audit guard). Usage: python -B tools/r41_build_parent.py --cond f40in|f80in --data-path <staged dir> [--require-gpu NAME]"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r41f'))
import common as C
from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, VOCAB, R35_PREP, install_guards, guard_record, sha256, write_json, read_json, utc


def build_cache(cond, data_path, require_gpu):
    import torch
    c = C.CONDS[cond]; code = Path(c['code']); hdir = Path(c['parent_dir']); cdir = Path(c['cache_dir'])
    if (cdir / 'CACHE_MANIFEST.json').exists():
        print('cache already built', cdir); return
    sys.path.insert(0, str(code))
    import vse as V, data as D, evaluation as E
    from vocab import deserialize_vocab
    torch.backends.cudnn.benchmark = True  # native (r25_run.py)
    if require_gpu:
        assert torch.cuda.get_device_name(0) == require_gpu, (torch.cuda.get_device_name(0), require_gpu)
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = hdir / 'model_best.pth.tar'
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    opt = ck['opt']; run = read_json(hdir / c['run_json'])
    assert float(opt.noise_ratio) == c['noise_ratio'] and run['opt']['noise_file'] == c['noise_file'] and opt.data_name == 'f30k_precomp'
    assert sha256(c['noise_file']) == c['noise_sha256']
    run_code_match = {f: run['env']['code_sha256'][f] == sha256(code / f) for f in ('vse.py', 'data.py', 'evaluation.py', 'vocab.py')}
    assert all(run_code_match.values()), run_code_match
    obs = np.load(c['noise_file']); assert obs.shape == (N_TRAIN_CAP,) and obs.dtype == np.int64
    opt.data_path = data_path; opt.vocab_path = str(VOCAB); opt.workers = 0
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    models = []
    for key in ('model_A', 'model_B'):
        m = V.VSE(opt, vocab.word2idx); m.load_state_dict(ck[key]); m.val_start(); models.append(m)
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
    native = dict(i2t_r1=r[0], i2t_r5=r[1], i2t_r10=r[2], t2i_r1=ri[0], t2i_r5=ri[1], t2i_r10=ri[2], rsum=float(sum(r[:3]) + sum(ri[:3])), peer_rsum=peer_rsum)
    sims_cache = sum(E.shard_attn_scores(m, dev[('img', p)].astype(np.float64), dev[('cap', p)].astype(np.float64), [0] * N_DEV_CAP, opt, shard_size=100)
                     for p, m in zip(PEERS, models)) / 2
    native['cache_rescore_identical'] = bool(np.array_equal(sims_cache, sims_mean))
    native['checkpoint_best_rsum'] = float(ck['best_rsum']); native['checkpoint_epoch'] = int(ck['epoch'])
    native['reproduces_checkpoint_best_rsum_exactly'] = native['rsum'] == float(ck['best_rsum'])
    native['abs_diff_vs_checkpoint'] = abs(native['rsum'] - float(ck['best_rsum']))
    print('NATIVE DEV', cond, native, flush=True)
    if not (native['reproduces_checkpoint_best_rsum_exactly'] and native['cache_rescore_identical']):
        write_json(cdir / f"FAILED_REPRO_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(native=native, gpu=torch.cuda.get_device_name(0), utc=utc()))
        raise SystemExit(f'R41: native DEV of {cond} not reproduced exactly; cache NOT written')
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
            return torch.Tensor(ims_tr[k // 5]), D.process_caption(vocab, caps_tr[k], False), k, k // 5
    loader = torch.utils.data.DataLoader(TrainEvalLayout(), batch_size=opt.batch_size, shuffle=False, pin_memory=True, collate_fn=D.collate_fn, num_workers=4)
    stats = {}
    for p, m in zip(PEERS, models):
        t0 = time.time()
        img, cap, _ = E.encode_data(m, loader, 200)
        img = img[::5]
        fi, fc = img.astype(np.float32), cap.astype(np.float32)
        assert np.array_equal(fi.astype(np.float64), img) and np.array_equal(fc.astype(np.float64), cap)
        assert fi.shape == (N_TRAIN_IMG, 1024) and fc.shape == (N_TRAIN_CAP, 1024) and np.isfinite(fi).all() and np.isfinite(fc).all()
        save(f'fit_img_peer{p}.npy', fi); save(f'fit_cap_peer{p}.npy', fc)
        stats[p] = dict(encode_seconds=time.time() - t0, img_norm_mean=float(np.linalg.norm(fi, axis=1).mean()))
    man = dict(line='R41 C cache (R37/R34 build_cache rule; IN2R-native parent)', cond=cond, parent_seed=c['parent_seed'], created_utc=utc(),
               parent=dict(unit=c['h_unit'], model_best=str(ckpt_path), model_best_sha256=sha256(ckpt_path), epoch=int(ck['epoch']), best_rsum=float(ck['best_rsum']),
                           run_json=str(hdir / c['run_json']), code_dir=str(code), code_sha256={f: sha256(code / f) for f in ('vse.py', 'data.py', 'evaluation.py', 'vocab.py')},
                           run_code_sha256_matches=run_code_match, run_opt={k: run['opt'].get(k) for k in ('warmup_epoch', 'k', 'i2t', 't2i', 'noise_ratio', 'seed')}),
               observed_map=dict(path=c['noise_file'], sha256=c['noise_sha256'], noise_ratio=c['noise_ratio'], realized_mismatch=float((obs != np.arange(N_TRAIN_CAP) // 5).mean()),
                                 note='the file the parent trained with (run opt noise_file)'),
               encoding=dict(mode='VSE.forward_emb, model.val_start()', centering='per-batch mean subtraction per modality, then L2', batch_size=int(opt.batch_size),
                             dev_layout='native dev loader, img = rows[::5]', fit_layout='row k = train_ims[k//5], caption k (drop=False), img = rows[::5]'),
               native_dev=native, files=files, encode_stats=stats,
               env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
               test_guard=guard_record())
    write_json(cdir / 'CACHE_MANIFEST.json', man)
    print('R41_CACHE_DONE', cond, native['rsum'], flush=True)


def build_prep(cond):
    import torch
    import evals as EV
    from data35 import load_cond
    c = C.CONDS[cond]; pdir = Path(c['prep_dir']); pdir.mkdir(parents=True, exist_ok=True)
    mf = pdir / 'PREP_MANIFEST.json'
    man = read_json(mf) if mf.exists() else None
    r35m = read_json(R35_PREP / 'PREP_MANIFEST.json')
    if man is None:
        man = dict(line='R41 C prep (R35/R37 prep rules, new IN2R-native parents)', created_utc=utc(), r35_prep_manifest_sha256=sha256(R35_PREP / 'PREP_MANIFEST.json'),
                   shared_from_r35={k: r35m['files'][k] for k in ('locks_records.npz', 'tokens.npz')}, calibration_init={}, candidates={}, files={})
    if cond in man['calibration_init']:
        print('prep already built', cond); return
    for name in ('locks_records.npz', 'tokens.npz'):
        assert sha256(R35_PREP / name) == r35m['files'][name]['sha256']
    L = np.load(R35_PREP / 'locks_records.npz'); L = {k: L[k] for k in L.files}
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
    man['calibration_init'][cond] = dict(a0=fit['a'], b0=fit['b'], balanced_logistic_value=fit['value'], fit_set='R35 locks cal_p (P) + cal_q (Q), observed map of the parent',
                                         score='native peer-mean fp32 dot of the R41 cache', J_raw_fit=EV.j_weighted(g, wp, wq), J_affine_fit=EV.j_weighted(fit['a'] * g + fit['b'], wp, wq),
                                         dev_rank_invariance=EV.rank_invariance(d['native_dev'], fit['a'], fit['b']))
    xs = [d[('fit', 'img', p)] for p in PEERS]; ys = [d[('fit', 'cap', p)] for p in PEERS]
    i2t = np.zeros((N_TRAIN_IMG, 32), np.int32); t2i = np.zeros((N_TRAIN_CAP, 32), np.int32)
    with torch.no_grad():
        for i in range(0, N_TRAIN_IMG, 1000):
            i2t[i:i + 1000] = (sum(x[i:i + 1000] @ y.T for x, y in zip(xs, ys)) / 2).topk(32, dim=1).indices.int().cpu().numpy()
        for j in range(0, N_TRAIN_CAP, 4000):
            t2i[j:j + 4000] = (sum(y[j:j + 4000] @ x.T for x, y in zip(xs, ys)) / 2).topk(32, dim=1).indices.int().cpu().numpy()
    name = f'candidates_{cond}.npz'; np.savez(pdir / name, i2t_top32=i2t, t2i_top32=t2i); man['files'][name] = dict(sha256=sha256(pdir / name), keys=['i2t_top32', 't2i_top32'])
    man['candidates'][cond] = dict(rule='peer-mean native fp32 score over the full FIT pool, torch.topk(32) (R35/R37 rule)',
                                   i2t_frac_images_with_an_observed_caption_in_top32=float((obs[i2t.astype(np.int64)] == np.arange(N_TRAIN_IMG)[:, None]).any(1).mean()),
                                   t2i_frac_captions_observed_image_in_top32=float((t2i.astype(np.int64) == obs[:, None]).any(1).mean()))
    man.setdefault('env', {})[cond] = dict(gpu=torch.cuda.get_device_name(0), job=os.environ.get('SLURM_JOB_ID'), utc=utc())
    write_json(mf, man)
    print('R41_PREP_DONE', cond, man['calibration_init'][cond]['a0'], man['candidates'][cond], flush=True)


def build_joint(cond):
    import torch
    import r39ext as R9
    from data35 import load_cond, shard_mm
    import evals as EV
    jp = C.JOINT_PARENTS[cond]; pdir = Path(jp['prep_dir']); pdir.mkdir(parents=True, exist_ok=True)
    if (pdir / 'JOINT_PREP_MANIFEST.json').exists():
        print('joint prep already built', cond); return
    dev = torch.device('cuda'); t0 = time.time()
    d = load_cond(cond, dev)
    ck = R9.load_parent(jp['ckpt']); stud = R9.build_students(ck, dev)
    assert int(ck['opt'].vocab_size) == ck['model_A'][1]['embed.weight'].shape[0] and int(np.load(C.TOKENS_FILE)['train_tok'].max()) < int(ck['opt'].vocab_size)
    tr = np.load(C.RAW_FIT_IMS, mmap_mode='r'); dv = np.load(C.RAW_DEV_IMS, mmap_mode='r'); T = np.load(C.TOKENS_FILE)
    ims = dict(fit=torch.from_numpy(np.ascontiguousarray(tr)), dev=torch.from_numpy(np.ascontiguousarray(dv)))
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
    sims = sum(shard_mm(devu[('img', p)], devu[('cap', p)]) for p in PEERS).cpu().numpy() / 2
    m_stu, _, rk_stu = EV.full_metrics(sims); m_nat, _, rk_nat = EV.full_metrics(d['native_dev'])
    eq['dev_native_from_student_vectors'] = dict(rsum=m_stu['rsum'], native_rsum=m_nat['rsum'], max_abs_sim_diff=float(np.abs(sims - d['native_dev']).max()),
                                                 n_i2t_rank_diff=int((rk_stu['ranks_i'] != rk_nat['ranks_i']).sum()), n_t2i_rank_diff=int((rk_stu['ranks_t'] != rk_nat['ranks_t']).sum()))
    summ = dict(max_abs_over_all=max(v['max_abs'] for k, v in eq.items() if isinstance(v, dict) and 'max_abs' in v), encode_seconds=time.time() - t0)
    man = dict(line='R41 C JOINT prep (R39X prep_joint_x rule)', created_utc=utc(), ds='f60', cond=cond, target_cond=cond, prep_key=cond, extension='R41',
               parent=dict(ckpt=str(jp['ckpt']), model_best_sha256=sha256(jp['ckpt']), vocab_size=int(ck['opt'].vocab_size), code_vse_sha256=sha256(Path(jp['code']) / 'vse.py'),
                           epoch=int(ck['epoch']), best_rsum=float(ck['best_rsum'])),
               target_cache_manifest_sha256=d['manifest_sha256'], native_code=dict(dir=str(C.NATIVE_CODE), vse_sha256=sha256(C.NATIVE_CODE / 'vse.py')),
               tokens=dict(path=str(C.TOKENS_FILE), sha256=sha256(C.TOKENS_FILE), rule='R35 prep tokens.npz (read-only)'), files=files,
               centring_rule='parent per-native-batch mean (128 rows, per modality), held fixed for the student (R39 registered patch)',
               equivalence=eq, equivalence_summary=summ, test_guard=guard_record(), env=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__))
    write_json(pdir / 'JOINT_PREP_MANIFEST.json', man)
    print('R41_JOINT_PREP', cond, json.dumps(summ), json.dumps(eq['dev_native_from_student_vectors']), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True, choices=('f40in', 'f80in')); ap.add_argument('--data-path', required=True)
    ap.add_argument('--require-gpu', default='')
    a = ap.parse_args()
    install_guards(role='gate')
    build_cache(a.cond, a.data_path, a.require_gpu)   # native path exactly as R37 build_cache (no determinism flags, cudnn.benchmark True)
    import train_unit as TU
    TU.set_determinism()
    build_prep(a.cond)
    build_joint(a.cond)
    print('R41_BUILD_PARENT_DONE', a.cond)


if __name__ == '__main__':
    main()
