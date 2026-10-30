"""R27-D candidate-source audit: frozen FIT feature exporter (read-only replay; no training, no optimizer step).

Usage: export_candidates.py OUT_ROOT CELL NB LABEL=CKPT [LABEL=CKPT ...] -- <r25_run args>
  Writes OUT_ROOT/CELL/LABEL/peer_<T>/{features.npz, labels.npz, metadata.json}; a set whose metadata.json exists is skipped.
  T = training peer (A or B). The SEARCHING peer is the other one (net2): exactly as production
  r27_co_train.noisy_branches, net2 (val mode, no grad) encodes T's noisy FIT batch with native forward_emb and searches
  net2.memory_queue (I.search, K=5). Queries and bank therefore come from the same frozen peer net2.

Loading / split / queue / batch flow mirror handoff/r27_audit_20261029/audit_graphs.py line for line (imported read-only):
  production R27 load_full_state -> seed_all(cell seed) -> eval_train / split_prob (next-epoch GMM split) -> co-teaching
  loaders -> per peer: saved ID-bearing queue (final states) or IN-MEMORY reconstruction exactly as audit_graphs
  (r27_calibrate.rebuild_queue, generator QUEUE_SEED, torch seed QUEUE_SEED+pi, enc state restored) for the prefix ->
  first NB unlabeled FIT batches of the fixed generator GEN_SEED (B >= 2). The current peer's own forward (needed only
  by the graph audit) is not run here; it does not draw from the numpy/python RNG that drives the loader augmentation.
Truth (pre-injection pairing: caption j <-> image j // 5) is written ONLY to labels.npz.
Nothing is written back to production files. No DEV/TEST read (only the train split is loaded).
"""
import os, sys, json, copy, time, hashlib
import numpy as np
import torch

AUD = '/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029'
sys.path.insert(0, AUD)
import audit_common as C            # noqa: E402  (puts frozen code_v11 on sys.path)
import audit_graphs as AG           # noqa: E402  (constants GEN_SEED / QUEUE_SEED / QUERY_SEED reused)
import r25_run                      # noqa: E402
import r25_co_train as R            # noqa: E402
import r27_co_train as R27          # noqa: E402
import r27_interface as I           # noqa: E402
import r27_calibrate as RC          # noqa: E402
from data import get_dataset        # noqa: E402
from vse import VSE                 # noqa: E402
from vocab import deserialize_vocab  # noqa: E402
from co_train import eval_train, split_prob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = '/home/infres/yinwang/CS_QMI/r27d_server/VCS_Retrieval_R27D_Server_20261030'
K = 5
CENTERING = ('query_img/query_txt = searching peer net2 native VSE.forward_emb in val mode (eval, no grad) of the training '
             'peer\'s noisy FIT batch: each modality centered by THAT batch\'s own mean (per batch of 128, batches '
             'concatenated after centering) then l2-normalized -- identical to the production search query (xi2/xt2). '
             'bank_img/bank_txt = net2.memory_queue.img_queue/txt_queue rows exactly as stored (each written as the '
             'normalized train-mode forward_emb of a labeled batch, centered by that labeled batch\'s mean at write time); '
             'no re-centering, no re-normalization. Score = dot product, as I.search.')


def own_hashes():
    return {f: C.sha(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith('.py')}


def main():
    sep = sys.argv.index('--')
    out_root, cell, nb = sys.argv[1], sys.argv[2], int(sys.argv[3])
    specs = [a.split('=', 1) for a in sys.argv[4:sep]]
    opt = r25_run.parse(sys.argv[sep + 1:])
    assert opt.data_name == 'f30k_precomp', 'truth labels: Flickr only (caption j <-> image j // 5)'
    noise_obs = np.load(opt.noise_file)
    opt.arm, opt.lam = 'a1_iface', 0.0        # not used: noisy_branches is never called
    R27.install()
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    n_caps, n_imgs = len(caps), int(ims.shape[0])
    assert n_caps == 5 * n_imgs
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
    lock = json.load(open(os.path.join(PKG, 'MANIFEST_SHA256.json')))
    for label, ck in specs:
        od = os.path.join(out_root, cell, label)
        if all(os.path.exists(os.path.join(od, f'peer_{p}', 'metadata.json')) for p in 'AB'):
            print('skip (done)', label, flush=True); continue
        if ck.startswith('MISSING'):
            os.makedirs(od, exist_ok=True)
            json.dump(dict(cell=cell, label=label, status='MISSING', reason=ck.split(':', 1)[-1]), open(os.path.join(od, 'MISSING.json'), 'w'), indent=1)
            print('MISSING', label, flush=True); continue
        t_ck = time.time()
        base = dict(cell=cell, label=label, split='FIT', checkpoint=ck, checkpoint_sha256=C.sha(ck),
                    gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                    nb_per_peer=nb, k=K, seeds=dict(cell=opt.seed, fit_batches=AG.GEN_SEED, queue_rebuild=AG.QUEUE_SEED,
                    query_torch_seed_base=AG.QUERY_SEED), code_v11_sha256=C.code_hashes(), exporter_sha256=own_hashes(),
                    audit_graphs_sha256=C.sha(os.path.join(AUD, 'audit_graphs.py')),
                    package_source_lock=dict(manifest_sha256=C.sha(os.path.join(PKG, 'MANIFEST_SHA256.json')),
                                             tool=lock['files']['tools/candidate_source_audit.py'] if 'files' in lock else None),
                    centering_rule=CENTERING)
        A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
        st = R.load_full_state(ck, opt, A, B)
        has_ids = st['model_A'].get('r27_queue_ids') is not None and st['model_B'].get('r27_queue_ids') is not None
        base.update(state_epoch=int(st['epoch']), state_has_queue_ids=has_ids, next_epoch=int(st['epoch']) + 1)
        r25_run.seed_all(opt.seed)
        epoch = st['epoch'] + 1
        t = time.time()
        prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, epoch)
        base['eval_train_seconds'] = time.time() - t
        pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
        for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
            cp = prob[pred]; m.avg_clean_prob = cp.mean() if len(cp) > 0 else 0.0
        base['gmm_split'] = dict(n_clean_pred_A=int(np.sum(pred_A)), n_clean_pred_B=int(np.sum(pred_B)), n=int(len(pred_A)))
        loaders = {'A': R.get_loader(caps, ims, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B),
                   'B': R.get_loader(caps, ims, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred_A, prob=prob_A)}
        for pi, (name, net, net2) in enumerate((('A', A, B), ('B', B, A))):
            meta = dict(base, train_peer=name, search_peer='B' if name == 'A' else 'A')
            if has_ids:
                q2 = net2.memory_queue
                meta['queue'] = 'saved ID-bearing queue of the searching peer (state file)'
                meta['queue_reconstructed'] = False
            else:
                g0 = torch.Generator(); g0.manual_seed(AG.QUEUE_SEED)
                torch.manual_seed(AG.QUEUE_SEED + pi); torch.cuda.manual_seed(AG.QUEUE_SEED + pi)
                ltd = loaders['B' if name == 'A' else 'A'][0].dataset
                lq = torch.utils.data.DataLoader(ltd, batch_size=opt.batch_size, shuffle=True, generator=g0, pin_memory=True,
                                                 collate_fn=loaders[name][0].collate_fn, num_workers=0)
                t = time.time()
                q2, nbq = RC.rebuild_queue(opt, net2, lq)
                meta['queue'] = ('prefix_reconstructed: in-memory ID-bearing queue rebuilt exactly as audit_graphs.py '
                                 '(r27_calibrate.rebuild_queue; prefix slots carry no IDs); never written back')
                meta['queue_reconstructed'] = True
                meta['queue_rebuild'] = dict(batches=nbq, seconds=time.time() - t, is_full=bool(q2.is_full))
            n, valid = I.usable(q2)
            ids = q2.r27_ids
            cap_all, img_all = ids['cap'][:n], ids['img'][:n]
            elig = (cap_all >= 0) & (img_all >= 0)
            assert torch.equal(elig, valid), 'production valid slots (cap>=0) must equal slots with both IDs'
            sl = torch.nonzero(elig).flatten()
            bank = dict(bank_img=q2.img_queue[:n][sl].float().cpu().numpy(), bank_txt=q2.txt_queue[:n][sl].float().cpu().numpy(),
                        bank_caption_ids=cap_all[sl].cpu().numpy().astype(np.int64),
                        bank_observed_img_ids=img_all[sl].cpu().numpy().astype(np.int64),
                        bank_slot_index=sl.cpu().numpy().astype(np.int64),
                        bank_write_epoch=ids['wepoch'][:n][sl].cpu().numpy().astype(np.int64),
                        bank_write_iter=ids['witer'][:n][sl].cpu().numpy().astype(np.int64),
                        bank_write_peer=ids['peer'][:n][sl].cpu().numpy().astype(np.int64))
            assert (noise_obs[bank['bank_caption_ids']] == bank['bank_observed_img_ids']).all(), 'bank IDs vs noise lock'
            meta.update(queue_filled=int(n), queue_id_bearing=int(valid.sum()), bank_n=int(sl.numel()))
            ut = loaders[name][1]
            gen = torch.Generator(); gen.manual_seed(AG.GEN_SEED)
            utl = torch.utils.data.DataLoader(ut.dataset, batch_size=opt.batch_size, shuffle=True, generator=gen, pin_memory=True,
                                              collate_fn=ut.collate_fn, num_workers=0)
            net.train_start(); net2.val_start()                       # as audit_graphs / production (net2 val mode)
            k, bh, Q = 0, [], {x: [] for x in ('qi', 'qt', 'cap', 'obs', 'bidx', 'si', 'st', 'wi', 'wt', 'pi', 'pt')}
            for batch in utl:
                if k >= nb:
                    break
                imgs, ilen, txt, tlen, bids = batch[0], batch[1], batch[2], batch[3], batch[4]
                if imgs.size(0) < 2:
                    continue
                ids_np = np.asarray(bids, dtype=np.int64)
                cap_g = ut.dataset.r27_gidx[ids_np]
                obs_img = np.asarray(ut.dataset.t2i_index, dtype=np.int64)[ids_np]
                assert (noise_obs[cap_g] == obs_img).all()
                bh.append(hashlib.sha256(np.asarray(cap_g).tobytes()).hexdigest()[:16])
                s = AG.QUERY_SEED + 1000 * pi + k
                torch.manual_seed(s); torch.cuda.manual_seed(s)
                with torch.no_grad():
                    xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
                S_i, qs_i = I.search(q2, xi2, 'i2t', K)
                S_t, qs_t = I.search(q2, xt2, 't2i', K)
                assert S_i is not None and S_t is not None, (qs_i, qs_t)
                Q['qi'].append(xi2.float().cpu().numpy()); Q['qt'].append(xt2.float().cpu().numpy())
                Q['cap'].append(cap_g); Q['obs'].append(obs_img); Q['bidx'].append(np.full(len(cap_g), k, dtype=np.int64))
                Q['si'].append(S_i['cap'].cpu().numpy()); Q['st'].append(S_t['img'].cpu().numpy())
                Q['wi'].append(S_i['wepoch'].cpu().numpy()); Q['wt'].append(S_t['wepoch'].cpu().numpy())
                Q['pi'].append(S_i['slots'].cpu().numpy()); Q['pt'].append(S_t['slots'].cpu().numpy())
                k += 1
            cat = {x: np.concatenate(v) for x, v in Q.items()}
            feats = dict(query_img=cat['qi'], query_txt=cat['qt'], query_caption_ids=cat['cap'].astype(np.int64),
                         query_observed_img_ids=cat['obs'].astype(np.int64), query_batch_index=cat['bidx'],
                         A_slot_i2t=cat['si'].astype(np.int64), A_slot_t2i=cat['st'].astype(np.int64),
                         A_slot_write_epoch_i2t=cat['wi'].astype(np.int64), A_slot_write_epoch_t2i=cat['wt'].astype(np.int64),
                         A_slot_queue_slot_i2t=cat['pi'].astype(np.int64), A_slot_queue_slot_t2i=cat['pt'].astype(np.int64), **bank)
            assert not any(x.startswith(('gt_', 'true_', 'clean_')) for x in feats)
            capg = cat['cap'].astype(np.int64)
            labels = dict(gt_i2t=(cat['obs'].astype(np.int64)[:, None] * 5 + np.arange(5, dtype=np.int64)[None]),
                          gt_t2i=(capg // 5)[:, None], noisy_query=(cat['obs'].astype(np.int64) != capg // 5))
            age = epoch - bank['bank_write_epoch']
            meta.update(batches=k, batch_capid_sha16=bh, n_queries=int(len(capg)),
                        n_distinct_query_observed_images=int(len(np.unique(cat['obs']))),
                        n_distinct_query_captions=int(len(np.unique(capg))),
                        batch_caption_ids_sha256=hashlib.sha256(capg.tobytes()).hexdigest(),
                        cache_age_epochs=(dict(note='reconstructed queue: write epoch is a placeholder (1); age not meaningful')
                                          if meta['queue_reconstructed'] else
                                          dict(mean=float(age.mean()), min=int(age.min()), max=int(age.max()),
                                               median=float(np.median(age)))),
                        model_mode=dict(searching_peer='val (eval) mode, torch.no_grad, native forward_emb',
                                        training_peer='not forwarded by the exporter',
                                        queue_rebuild='train-mode forward, no grad, encoder state restored (r27_calibrate.rebuild_queue)'),
                        id_ranges=dict(n_captions=n_caps, n_images=n_imgs),
                        files=dict(features='features.npz (no gt/true/clean keys)', labels='labels.npz (evaluator only)'))
            pd = os.path.join(od, f'peer_{name}')
            os.makedirs(pd, exist_ok=True)
            np.savez(os.path.join(pd, 'features.npz'), **feats)
            np.savez(os.path.join(pd, 'labels.npz'), **labels)
            meta['features_sha256'] = C.sha(os.path.join(pd, 'features.npz'))
            meta['labels_sha256'] = C.sha(os.path.join(pd, 'labels.npz'))
            json.dump(meta, open(os.path.join(pd, 'metadata.json'), 'w'), indent=1, default=float)
            print(json.dumps(dict(label=label, peer=name, n_queries=meta['n_queries'], bank_n=meta['bank_n'], batches=bh)), flush=True)
            if not has_ids:
                del q2
        print('DONE', label, round(time.time() - t_ck, 1), flush=True)
        del A, B, st
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
