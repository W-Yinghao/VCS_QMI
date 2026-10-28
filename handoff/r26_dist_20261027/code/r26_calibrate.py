"""R26 outer-scale calibration c_J (DESIGN §1, §4.3), once per dataset at a common native prefix.

Same protocol as r25_calibrate (pinned): load prefix_state.pt, rebuild the epoch-(prefix+1) GMM split as the arms
will, for each peer take the first 32 unlabeled FIT batches (generator seed 20261027) with B>=2 and usable banks.
Per batch: query-gradient norm of the native correction reference (0.5 SCE(i2t) + 0.5 SCE(t2i) through the current
GraphRefiners, as native) and of each R26 arm (0.5 L(i2t) + 0.5 L(t2i), frozen real candidates), wrt the two current
query embedding tensors. c_J[arm] = median over the 64 (peer,batch) ratios ||g_native|| / ||g_arm||.
Also logged: raw J, scaled losses, encoder-gradient ratio scaled-noisy / clean on a paired labeled FIT batch.
No parameter update, no DEV/TEST read, nothing written back to the prefix. Usage: r26_calibrate.py OUT.json <r25_run args incl --resume>.
"""
import os, sys, json, copy
import numpy as np
import torch

import r25_run
import r25_co_train as R
from data import get_loader, get_dataset
from vse import VSE, infonce
from vocab import deserialize_vocab
from co_train import eval_train, split_prob
from r26_graph import ARMS, build_graph, arm_loss
from r26_co_train import bank

NB = 32


def gnorm(loss, tensors):
    g = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    if any(x is None for x in g):
        return None
    return float(torch.sqrt(sum((x.double() ** 2).sum() for x in g)))


def pnorm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x.double() ** 2).sum() for x in g if x is not None)))


def main():
    out_path = sys.argv[1]
    opt = r25_run.parse(sys.argv[2:])
    assert opt.resume
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = get_loader(caps, ims, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(opt.resume, opt, A, B)
    epoch = st['epoch'] + 1
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, epoch)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        cp = prob[pred]; m.avg_clean_prob = cp.mean() if len(cp) > 0 else 0.0
    rows = []
    for name, net, net2, pred, prob in (('A', A, B, pred_B, prob_B), ('B', B, A, pred_A, prob_A)):
        lt, ut = get_loader(caps, ims, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred, prob=prob)
        gen = torch.Generator(); gen.manual_seed(20261027)
        ut = torch.utils.data.DataLoader(ut.dataset, batch_size=opt.batch_size, shuffle=True, generator=gen, pin_memory=True, collate_fn=ut.collate_fn, num_workers=0)
        gen2 = torch.Generator(); gen2.manual_seed(20261028)
        lt = iter(torch.utils.data.DataLoader(lt.dataset, batch_size=opt.batch_size, shuffle=True, generator=gen2, pin_memory=True, collate_fn=lt.collate_fn, num_workers=0))
        net.train_start(); net2.val_start()
        ib, tb = bank(net2.memory_queue)
        n_ok = n_skip = 0
        for batch in ut:
            if n_ok >= NB:
                break
            imgs, ilen, txt, tlen = batch[0], batch[1], batch[2], batch[3]
            if imgs.size(0) < 2:
                n_skip += 1; continue
            xi, xt, _ = net.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            with torch.no_grad():
                xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            Gi = build_graph(xi2, ib, tb, opt.k, 'i2t'); Gt = build_graph(xt2, ib, tb, opt.k, 't2i')
            pt = net2.memory_queue.retrieve_nearest_text(xi, k=opt.k, refiner_node=net.i2t_refiner_node)
            pi = net2.memory_queue.retrieve_nearest_img(xt, k=opt.k, refiner_node=net.t2i_refiner_node)
            if Gi is None or Gt is None or pt is None or pi is None:
                n_skip += 1; continue
            r = {'peer': name, 'batch': n_ok, 'B': int(imgs.size(0)), 'n_atoms_i2t': Gi['n_atoms'], 'n_atoms_t2i': Gt['n_atoms'],
                 'shared_frac_i2t': Gi['shared_frac'], 'shared_frac_t2i': Gt['shared_frac']}
            nat = 0.5 * infonce(xi.mm(pt.t())) + 0.5 * infonce(xt.mm(pi.t()))
            r['native'] = gnorm(nat, [xi, xt])
            # clean reference on a paired labeled batch (encoder-parameter gradient norm)
            bl = next(lt)
            il, cl, cll = net.forward_emb(bl[0], bl[2], bl[3], imgae_lengths=bl[1])
            il2, cl2, _ = net.forward_emb(bl[0], bl[2], bl[3], imgae_lengths=bl[1])
            crit = net.criterion
            lclean = crit(net.forward_sim(il, cl, cll), labels=None) + crit(il.mm(il2.t()), margin=0.3) + crit(cl.mm(cl2.t()), margin=0.3)
            r['clean_enc'] = pnorm(lclean, list(net.params))
            for arm in ARMS:
                li, di = arm_loss(arm, xi, Gi); lt_, dt_ = arm_loss(arm, xt, Gt)
                l = 0.5 * li + 0.5 * lt_
                r[arm] = gnorm(l, [xi, xt])
                r[arm + '_enc'] = pnorm(l, list(net.params))
                r[arm + '_loss'] = float(l); r[arm + '_rawJ'] = 0.5 * float(di['raw_J']) + 0.5 * float(dt_['raw_J'])
            rows.append(r); n_ok += 1
            del xi, xt, xi2, xt2, il, cl, il2, cl2
        rows.append({'peer': name, 'valid_batches': n_ok, 'skipped_batches': n_skip})
    br = [r for r in rows if 'batch' in r]
    out = {'prefix': opt.resume, 'prefix_epoch': st['epoch'], 'split_epoch': epoch, 'data_name': opt.data_name,
           'noise_file': opt.noise_file, 'fit_batches_per_peer': NB, 'c_J': {}, 'ratio_quantiles': {}, 'rawJ_median': {},
           'scaled_noisy_to_clean_enc_median': {}, 'native_to_clean_enc_note': 'native ref enc-grad not computed (refiner path)'}
    for arm in ARMS:
        bad = [r for r in br if not r.get(arm) or not r.get('native') or not np.isfinite(r[arm])]
        if bad:
            out['c_J'][arm] = None; continue
        ratios = np.array([r['native'] / r[arm] for r in br])
        c = float(np.median(ratios))
        out['c_J'][arm] = c
        out['ratio_quantiles'][arm] = {q: float(np.quantile(ratios, q)) for q in (0, .1, .25, .5, .75, .9, 1)}
        out['rawJ_median'][arm] = float(np.median([r[arm + '_rawJ'] for r in br]))
        out['scaled_noisy_to_clean_enc_median'][arm] = float(np.median([c * r[arm + '_enc'] / r['clean_enc'] for r in br if r.get('clean_enc')]))
    out['n_ratio_rows'] = len(br)
    out['rows'] = rows
    json.dump(out, open(out_path, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    sys.exit(0 if all(v is not None for v in out['c_J'].values()) and len(br) == 2 * NB else 1)


if __name__ == '__main__':
    main()
