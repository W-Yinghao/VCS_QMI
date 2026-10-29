"""R26 fixed-graph diagnostic export (adjustments package 20261028, SERVER_INSTRUCTIONS P1 item 3).

Isolated replay process, no optimizer.step, nothing written back to any production checkpoint. For each listed
checkpoint (frozen-checkpoint reconstruction, NOT historical training-time measurement):
  1. load_full_state into two fresh VSE peers (production loader, pinned R26 code);
  2. rebuild the GMM split the next epoch would use (eval_train on the FIT train pool only);
  3. per peer, first NB valid unlabeled FIT batches of a fixed generator (seed 20261027, as r26_calibrate): current peer
     in train mode (as production), other peer val mode, build_graph for i2t and t2i from the other peer's frozen queue
     -> NPZ with X, A, Y (+ sample ids / slot cols as metadata);
  4. cross-peer coordinate check on NC fixed FIT batches (all samples, both peers val mode, identical batch tensors):
     same-sample cross-peer cosine, within/cross-peer score separation, neighbour stability, and an orthogonal map fitted
     on one half of the samples and checked on the other half (diagnostic only, never used in training);
  5. queue facts (fill, exact duplicate rows); queue age / write version are not stored by the host -> not recoverable.
No DEV/TEST read. Usage: r26_graph_export.py OUT_DIR CELL LABEL=CKPT [LABEL=CKPT ...] -- <r25_run args>
"""
import os, sys, json, copy, hashlib, time
import numpy as np
import torch
import torch.nn.functional as Fn

CODE = '/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/code'
sys.path.insert(0, CODE)
import r25_run
import r25_co_train as R
from data import get_loader, get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob
from r26_graph import build_graph
from r26_co_train import bank

NB, NC, GEN_SEED = 8, 96, 20261027


def sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def fixed_loader(ds, collate, bs, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, generator=g, pin_memory=True, collate_fn=collate, num_workers=0)


def topk_overlap(S1, S2, k=5):
    a = torch.topk(S1, k, dim=1).indices; b = torch.topk(S2, k, dim=1).indices
    return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(a, b)]))


def sep(S):
    d = S.diagonal(); n = S.shape[0]
    off = (S.sum() - d.sum()) / (n * n - n)
    r1 = float((S.argmax(1) == torch.arange(n, device=S.device)).float().mean())
    return {'diag_mean': float(d.mean()), 'offdiag_mean': float(off), 'diag_minus_off': float(d.mean() - off), 'inbatch_diag_top1_noisy_pairs': r1}


def procrustes(src, dst):
    u, _, vt = torch.linalg.svd(src.double().t() @ dst.double())
    return (u @ vt).float()


def cross_peer(A, B, loader):
    A.val_start(); B.val_start()
    feats = {k: [] for k in ('iA', 'tA', 'iB', 'tB')}
    img_ids = []
    t2i = getattr(loader.dataset, 't2i_index', None)
    per_batch = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= NC:
                break
            imgs, ilen, txt, tlen = batch[0], batch[1], batch[2], batch[3]
            if imgs.size(0) < 2:
                continue
            iA, tA, _ = A.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            iB, tB, _ = B.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            for k, v in (('iA', iA), ('tA', tA), ('iB', iB), ('tB', tB)):
                feats[k].append(v.float())
            img_ids += [int(t2i[i]) if t2i is not None else int(i) // 5 for i in batch[4]]
            per_batch.append({'AA': sep(iA @ tA.t()), 'BB': sep(iB @ tB.t()), 'AB': sep(iA @ tB.t()), 'BA': sep(iB @ tA.t()),
                              'nbr_overlap_i2t_AA_vs_BB': topk_overlap(iA @ tA.t(), iB @ tB.t()),
                              'nbr_overlap_t2i_AA_vs_BB': topk_overlap(tA @ iA.t(), tB @ iB.t()),
                              'nbr_overlap_i2t_AA_vs_AB': topk_overlap(iA @ tA.t(), iA @ tB.t()),
                              'nbr_overlap_t2i_AA_vs_AB': topk_overlap(tA @ iA.t(), tA @ iB.t())})
            n += 1
    F = {k: torch.cat(v) for k, v in feats.items()}
    N = F['iA'].shape[0]
    agg = {}
    for key in per_batch[0]:
        if isinstance(per_batch[0][key], dict):
            agg[key] = {m: float(np.mean([b[key][m] for b in per_batch])) for m in per_batch[0][key]}
        else:
            agg[key] = float(np.mean([b[key] for b in per_batch]))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(GEN_SEED)).to(F['iA'].device)
    same = {}
    for mod in ('i', 't'):
        a, b = F[mod + 'A'], F[mod + 'B']
        cs = (a * b).sum(1)
        same[mod] = {'same_sample_cross_peer_cos_mean': float(cs.mean()), 'same_sample_cross_peer_cos_p10': float(cs.quantile(.1)),
                     'other_sample_cross_peer_cos_mean': float((a * b[perm]).sum(1).mean()),
                     'same_sample_within_peer_ref_cos': 1.0}
    # split by presented image id so captions of one image never straddle fit/check (review 20261029 §7.4)
    ids = torch.tensor(img_ids, device=F['iA'].device)
    uniq = torch.unique(ids)
    gperm = uniq[torch.randperm(len(uniq), generator=torch.Generator().manual_seed(GEN_SEED)).to(ids.device)]
    fit_imgs = gperm[: len(gperm) // 2]
    in_fit = torch.isin(ids, fit_imgs)
    fit, chk = in_fit.nonzero().flatten(), (~in_fit).nonzero().flatten()
    half = int(fit.numel())
    proc = {}
    for mod in ('i', 't'):
        Rm = procrustes(F[mod + 'B'][fit], F[mod + 'A'][fit])
        b_chk = F[mod + 'B'][chk]; a_chk = F[mod + 'A'][chk]
        mapped = Fn.normalize(b_chk @ Rm, dim=-1)
        proc[mod + '_B_to_A'] = {'fit_n': int(half), 'check_n': int(N - half), 'fit_images': int(len(fit_imgs)), 'check_images': int(len(uniq) - len(fit_imgs)), 'dim': int(a_chk.shape[1]),
                                 'check_same_sample_cos_before': float((a_chk * b_chk).sum(1).mean()),
                                 'check_same_sample_cos_after': float((a_chk * mapped).sum(1).mean()),
                                 'fit_same_sample_cos_after': float((F[mod + 'A'][fit] * Fn.normalize(F[mod + 'B'][fit] @ Rm, dim=-1)).sum(1).mean())}
        if mod == 't':  # does the map help the actual cross-peer i2t score A-image vs B-caption on held-out samples?
            ic, tc = F['iA'][chk], F['tB'][chk]
            m = min(len(ic), 4096)
            proc['i2t_AB_check_sep_before'] = sep(ic[:m] @ tc[:m].t())
            proc['i2t_AB_check_sep_after'] = sep(ic[:m] @ Fn.normalize(tc[:m] @ Rm, dim=-1).t())
            proc['i2t_AA_check_sep_ref'] = sep(ic[:m] @ F['tA'][chk][:m].t())
    return {'n_batches': n, 'n_samples': int(N), 'n_images': int(len(uniq)), 'split': 'image-grouped (presented image id), fit/check disjoint images',
            'label_note': 'diagonal of a noisy FIT batch is not all true correspondences; inbatch_diag_top1_noisy_pairs is not a retrieval R@1', 'batch_context': 'identical batch tensors for both peers, val mode (no dropout), host batch centering applies within each batch',
            'per_batch_mean': agg, 'same_sample': same, 'orthogonal_map_diagnostic_only': proc}


def queue_facts(q):
    n = q.queue_size if q.is_full else q.ptr
    ib, tb = q.img_queue[:n], q.txt_queue[:n]
    out = {'fill': int(n), 'queue_size': int(q.queue_size), 'is_full': bool(q.is_full), 'ptr': int(q.ptr),
           'age_and_write_version': 'NOT_RECOVERABLE (host queue stores no record id / write version / step)'}
    for name, t in (('img', ib), ('txt', tb)):
        if n == 0:
            continue
        h = [hashlib.sha1(r.tobytes()).hexdigest() for r in t.cpu().numpy()]
        out[f'{name}_exact_duplicate_rows'] = int(len(h) - len(set(h)))
        out[f'{name}_norm_dev_max'] = float((t.norm(dim=1) - 1).abs().max())
    return out


def main():
    argv = sys.argv[1:]
    cut = argv.index('--')
    out_dir, cell, ckpts, rest = argv[0], argv[1], argv[2:cut], argv[cut + 1:]
    os.makedirs(out_dir, exist_ok=True)
    opt = r25_run.parse(rest)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    code_sha = {f: sha(os.path.join(CODE, f)) for f in sorted(os.listdir(CODE)) if f.endswith('.py')}
    lock = {'cell': cell, 'role': 'FIT train pool only (no DEV/TEST read)', 'data_name': opt.data_name, 'data_path': opt.data_path,
            'noise_file': opt.noise_file, 'noise_file_sha256': sha(opt.noise_file), 'seed': opt.seed,
            'generator_seed': GEN_SEED, 'NB_graph_batches_per_peer': NB, 'NC_cross_peer_batches': NC,
            'export_code_sha256': sha(os.path.abspath(__file__)), 'r26_code_sha256': code_sha,
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename,
            'job': os.environ.get('SLURM_JOB_ID'), 'checkpoints': {}}
    summary = {'cell': cell, 'checkpoints': {}}
    for spec in ckpts:
        label, path = spec.split('=', 1)
        t0 = time.time()
        entry = {'path': path}
        if not os.path.exists(path):
            entry['status'] = 'MISSING'
            lock['checkpoints'][label] = entry; summary['checkpoints'][label] = entry
            continue
        entry['sha256'] = sha(path)
        r25_run.seed_all(opt.seed)
        A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
        st = R.load_full_state(path, opt, A, B)
        entry['state_epoch'] = int(st['epoch'])
        split_epoch = st['epoch'] + 1
        noisy_loader, data_size, clean_labels = get_loader(caps, ims, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
        prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, split_epoch)
        pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
        for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
            cp = prob[pred]; m.avg_clean_prob = cp.mean() if len(cp) > 0 else 0.0
        entry['split_epoch_reconstructed'] = int(split_epoch)
        entry['labeled_frac'] = {'A': float(np.mean(pred_A)), 'B': float(np.mean(pred_B))}
        entry['queues'] = {'A': queue_facts(A.memory_queue), 'B': queue_facts(B.memory_queue)}
        snaps = []
        torch.manual_seed(GEN_SEED); np.random.seed(GEN_SEED)
        for name, net, net2, pred, prob in (('A', A, B, pred_B, prob_B), ('B', B, A, pred_A, prob_A)):
            _, ut = get_loader(caps, ims, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred, prob=prob)
            loader = fixed_loader(ut.dataset, ut.collate_fn, opt.batch_size, GEN_SEED)
            net.train_start(); net2.val_start()
            ib, tb = bank(net2.memory_queue)
            n_ok = n_skip = 0
            with torch.no_grad():
                for batch in loader:
                    if n_ok >= NB:
                        break
                    imgs, ilen, txt, tlen, ids = batch[0], batch[1], batch[2], batch[3], batch[4]
                    if imgs.size(0) < 2:
                        n_skip += 1; continue
                    xi, xt, _ = net.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
                    xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
                    Gi = build_graph(xi2, ib, tb, opt.k, 'i2t'); Gt = build_graph(xt2, ib, tb, opt.k, 't2i')
                    if Gi is None or Gt is None:
                        n_skip += 1; continue
                    for branch, X, G in (('i2t', xi, Gi), ('t2i', xt, Gt)):
                        fn = os.path.join(out_dir, f'{cell}__{label}__peer{name}__{branch}__b{n_ok:02d}.npz')
                        if os.path.exists(fn):
                            raise SystemExit(f'refusing to overwrite {fn}')
                        np.savez_compressed(fn, X=X.detach().cpu().numpy(), A=G['A'].detach().cpu().numpy(), Y=G['Y'].detach().cpu().numpy(),
                                            sample_ids=np.asarray(ids, dtype=np.int64), slot_cols=G['cols'].cpu().numpy())
                        snaps.append({'file': os.path.basename(fn), 'sha256': sha(fn), 'peer': name, 'branch': branch, 'batch': n_ok,
                                      'B': int(X.shape[0]), 'M': int(G['n_atoms']), 'queue_snapshot': f'peer{"B" if name == "A" else "A"} queue of {label}, frozen (ptr={net2.memory_queue.ptr}, full={net2.memory_queue.is_full})',
                                      'query_mode': 'current peer train mode (dropout on, as production); other peer val mode'})
                    n_ok += 1
            entry[f'peer{name}_valid_batches'] = n_ok; entry[f'peer{name}_skipped_batches'] = n_skip
        entry['snapshots'] = snaps
        all_ds = get_loader(caps, ims, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)[0].dataset
        cp_loader = fixed_loader(all_ds, noisy_loader.collate_fn, opt.batch_size, GEN_SEED + 1)
        cpr = cross_peer(A, B, cp_loader)
        entry['seconds'] = time.time() - t0
        lock['checkpoints'][label] = {k: v for k, v in entry.items() if k != 'queues'}
        summary['checkpoints'][label] = {'state_epoch': entry['state_epoch'], 'labeled_frac': entry['labeled_frac'],
                                         'queues': entry['queues'], 'cross_peer': cpr, 'n_snapshots': len(snaps)}
        json.dump(lock, open(os.path.join(out_dir, f'SOURCE_LOCK_{cell}.json'), 'w'), indent=1)
        json.dump(summary, open(os.path.join(out_dir, f'EXPORT_SUMMARY_{cell}.json'), 'w'), indent=1)
        print(f'{label}: {len(snaps)} snapshots, {entry["seconds"]:.0f}s', flush=True)
        del A, B, st; torch.cuda.empty_cache()
    print('EXPORT_DONE', cell)


if __name__ == '__main__':
    main()
