"""C1 link and c_D scale calibration at a common prefix (04_MATH_INTERFACE_CN.md §3 C1, §6).

Separate process: loads prefix_state.pt, rebuilds the epoch-(prefix+1) GMM split exactly as the arms will,
then for each peer n in (A,B) takes the first 32 unlabeled FIT batches (fixed generator) whose two branches
both have prototypes and B>=2. No parameter update, no DEV/TEST read, nothing written back to the prefix.
Pass 1: one forward per batch -> native SCE, VCS-C0, BCE-C0 query-gradient norms + all branch scores.
C1: mP = median diagonal score, mQ = median of all B^2 scores (both peers, both branches);
    admissible iff mP > mQ; b=(mP+mQ)/2, a=2/max(mP-mQ,0.1).
Pass 2 (if C1 admissible): per-batch CPU/CUDA RNG restored, same cached batch -> SCE, VCS-C1, BCE-C1 norms.
c_D = median over the 64 (peer,batch) ratios of ||g_SCE|| / (||g_D|| + 1e-12),
with ell_D = 0.5 L_D(S_i2t) + 0.5 L_D(S_t2i) and g = grad wrt the two original query embedding tensors.
"""
import os, sys, json, copy
import numpy as np
import torch

import r25_run
import r25_co_train as R
from data import get_loader, get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob
from noisy_loss import noisy_loss

NB = 32


def qgrad(loss_fn, Si, St, xi, xt):
    l = 0.5 * loss_fn(Si) + 0.5 * loss_fn(St)
    g = torch.autograd.grad(l, [xi, xt], retain_graph=True, allow_unused=True)
    if any(x is None for x in g):
        return 'disconnected', None
    n = float(torch.sqrt(sum((x.double() ** 2).sum() for x in g)))
    if not np.isfinite(n):
        return 'nonfinite', n
    if n == 0.0:
        return 'zero', n
    return 'ok', n


def param_gnorm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x.double() ** 2).sum() for x in g if x is not None)))


def main():
    opt = r25_run.parse(sys.argv[1:])
    assert opt.resume, '--resume prefix_state.pt required'
    out_path = os.path.join(opt.output_dir, 'CALIBRATION.json')
    os.makedirs(opt.output_dir, exist_ok=True)
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    captions_train, images_train = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = get_loader(captions_train, images_train, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(opt.resume, opt, A, B)
    epoch = st['epoch'] + 1
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, epoch)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)

    cache, scores_d, scores_q = [], [], []
    rows = []
    for name, net, net2, pred, prob in (('A', A, B, pred_B, prob_B), ('B', B, A, pred_A, prob_A)):
        lt, ut = get_loader(captions_train, images_train, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred, prob=prob)
        lgen = torch.Generator(); lgen.manual_seed(20261028)
        lt_iter = iter(torch.utils.data.DataLoader(lt.dataset, batch_size=opt.batch_size, shuffle=True, generator=lgen,
                                                   pin_memory=True, collate_fn=lt.collate_fn, num_workers=0))
        gen = torch.Generator(); gen.manual_seed(20261027)
        ut = torch.utils.data.DataLoader(ut.dataset, batch_size=opt.batch_size, shuffle=True, generator=gen,
                                         pin_memory=True, collate_fn=ut.collate_fn, num_workers=0)
        net.train_start(); net2.val_start()
        n_ok, n_skip = 0, 0
        for batch in ut:
            if n_ok >= NB:
                break
            imgs, ilen, txt, tlen = batch[0], batch[1], batch[2], batch[3]
            if imgs.size(0) < 2:
                n_skip += 1; continue
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            xi, xt, _ = net.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            pt = net2.memory_queue.retrieve_nearest_text(xi, k=opt.k, refiner_node=net.i2t_refiner_node)
            pi = net2.memory_queue.retrieve_nearest_img(xt, k=opt.k, refiner_node=net.t2i_refiner_node)
            if pt is None or pi is None:
                n_skip += 1; continue
            Si, St = xi.mm(pt.t()), xt.mm(pi.t())
            r = {'peer': name, 'batch': n_ok, 'B': int(imgs.size(0))}
            for key, obj, link in (('sce', 'sce', 'c0'), ('vcs_c0', 'vcs', 'c0'), ('bce_c0', 'bce', 'c0')):
                status, nrm = qgrad(lambda S: noisy_loss(S, obj, link)[0], Si, St, xi, xt)
                r[key] = nrm; r[key + '_status'] = status
            enc = net.params; ref = list(net.i2t_refiner_node.parameters()) + list(net.t2i_refiner_node.parameters())
            for key, obj in (('sce', 'sce'), ('vcs_c0', 'vcs')):
                l = 0.5 * noisy_loss(Si, obj, 'c0')[0] + 0.5 * noisy_loss(St, obj, 'c0')[0]
                r[key + '_enc_gnorm'] = param_gnorm(l, enc); r[key + '_ref_gnorm'] = param_gnorm(l, ref)
            lb = next(lt_iter)  # clean (labeled) loss gradient norm on encoder params, for the noisy/clean ratio report
            ie1, ce1, cl1 = net.forward_emb(lb[0], lb[2], lb[3], imgae_lengths=lb[1])
            ie2, ce2, _ = net.forward_emb(lb[0], lb[2], lb[3], imgae_lengths=lb[1])
            crit = net.criterion
            loss_l = crit(net.forward_sim(ie1, ce1, cl1), labels=None) + crit(ie1.mm(ie2.t()), margin=0.3) + crit(ce1.mm(ce2.t()), margin=0.3)
            r['clean_enc_gnorm'] = param_gnorm(loss_l, enc)
            del ie1, ce1, ie2, ce2, loss_l
            for S in (Si, St):
                scores_d.append(S.detach().diagonal().cpu().double().numpy()); scores_q.append(S.detach().flatten().cpu().double().numpy())
            cache.append((name, batch, rng))
            rows.append(r); n_ok += 1
            del xi, xt, pt, pi, Si, St
        rows.append({'peer': name, 'valid_batches': n_ok, 'skipped_batches': n_skip})
    mP = float(np.median(np.concatenate(scores_d))); mQ = float(np.median(np.concatenate(scores_q)))
    c1_ok = mP > mQ
    a = 2.0 / max(mP - mQ, 0.1) if c1_ok else None; b = (mP + mQ) / 2 if c1_ok else None
    if c1_ok:
        k = 0
        for name, batch, rng in cache:
            net, net2 = (A, B) if name == 'A' else (B, A)
            net.train_start(); net2.val_start()
            torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
            imgs, ilen, txt, tlen = batch[0], batch[1], batch[2], batch[3]
            xi, xt, _ = net.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            pt = net2.memory_queue.retrieve_nearest_text(xi, k=opt.k, refiner_node=net.i2t_refiner_node)
            pi = net2.memory_queue.retrieve_nearest_img(xt, k=opt.k, refiner_node=net.t2i_refiner_node)
            Si, St = xi.mm(pt.t()), xt.mm(pi.t())
            while rows[k].get('peer') != name or 'batch' not in rows[k]:
                k += 1
            r = rows[k]; k += 1
            for key, obj, link in (('sce_pass2', 'sce', 'c0'), ('vcs_c1', 'vcs', 'c1'), ('bce_c1', 'bce', 'c1')):
                status, nrm = qgrad(lambda S: noisy_loss(S, obj, link, a, b)[0], Si, St, xi, xt)
                r[key] = nrm; r[key + '_status'] = status
            del xi, xt, pt, pi, Si, St
    batch_rows = [r for r in rows if 'batch' in r]
    out = {'prefix': opt.resume, 'prefix_epoch': st['epoch'], 'split_epoch': epoch, 'fit_batches_per_peer': NB,
           'mP': mP, 'mQ': mQ, 'C1_available': bool(c1_ok), 'a': a, 'b': b, 'c_D': {}, 'ratio_quantiles': {}, 'status_counts': {}}
    for key, num in (('vcs_c0', 'sce'), ('bce_c0', 'sce'), ('vcs_c1', 'sce_pass2'), ('bce_c1', 'sce_pass2')):
        if key not in batch_rows[0]:
            continue
        st_counts = {}
        for r in batch_rows:
            st_counts[r[key + '_status']] = st_counts.get(r[key + '_status'], 0) + 1
        out['status_counts'][key] = st_counts
        if any(r[key + '_status'] != 'ok' or r[num + '_status'] != 'ok' for r in batch_rows):
            out['c_D'][key] = None  # pause/diagnose: never convert to a huge scale
            continue
        ratios = np.array([r[num] / (r[key] + 1e-12) for r in batch_rows])
        out['c_D'][key] = float(np.median(ratios))
        out['ratio_quantiles'][key] = {q: float(np.quantile(ratios, q)) for q in (0, .1, .25, .5, .75, .9, 1)}
    out['n_ratio_rows'] = len(batch_rows)
    out['rows'] = rows
    json.dump(out, open(out_path, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
