"""R26 driver: R25 driver (pinned copy, unchanged) with the noisy correction branch replaced.

Everything outside the noisy branch is r25_co_train (warmup, eval_train/GMM split, clean ranking + intra losses,
FIFO enqueue of high-confidence clean pairs, grad clip, LR, full-state save/resume, full-DEV validation, no TEST).
Noisy branch (all R26 arms identical except the objective):
  * no GraphRefiner: candidates are real cached opposite-modality embeddings from the OTHER peer's FIFO, selected by
    the other peer's no-grad forward of the same noisy batch (r26_graph.build_graph), frozen for the update;
  * loss = 0.5 * c_J * arm_loss per direction, separate backward per direction (as native);
  * shared skip rule: unlabeled batch < 2 or empty split, or bank smaller than K -> no noisy loss, logged.
GraphRefiner modules still exist in the loaded prefix state (created by the native prefix epoch); they receive no
gradient and AdamW skips parameters with grad None, so they are inert.
"""
import time
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

import r25_co_train as R
from r26_graph import build_graph, arm_loss

logger = R.logger


def bank(q):
    n = q.queue_size if q.is_full else q.ptr
    return q.img_queue[:n], q.txt_queue[:n]


def noisy_branches(opt, net, net2, imgs, ilen, txt, tlen, img_emb_u, cap_emb_u, stats, probe=None):
    with torch.no_grad():
        xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)  # net2 is in val mode (net2.val_start())
    ib, tb = bank(net2.memory_queue)
    for branch, on in (('i2t', opt.i2t), ('t2i', opt.t2i)):
        if not on:
            continue
        G = build_graph(xi2 if branch == 'i2t' else xt2, ib, tb, opt.k, branch)
        if G is None:
            stats[f'{branch}_no_proto'] += 1
            continue
        X = img_emb_u if branch == 'i2t' else cap_emb_u
        loss, diag = arm_loss(opt.arm, X, G)
        if probe is not None:
            probe(branch, X, G, loss)
        (0.5 * opt.c_J * loss).backward()
        stats[f'{branch}_n'] += 1
        stats[f'{branch}_raw_loss'] += float(loss.detach())
        stats[f'{branch}_n_atoms'] += G['n_atoms']
        stats[f'{branch}_shared_frac'] += G['shared_frac']
        stats[f'{branch}_max_col_mass'] += G['max_col_mass']
        stats[f'cand_norm_dev_max_{branch}'] = max(stats.get(f'cand_norm_dev_max_{branch}', 0.0), G['norm_dev'])
        for k, v in diag.items():
            stats[f'{branch}_{k}'] += float(v)


def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader, epoch, stats, max_iters=None, probe=None):
    """r25_co_train.train with the R26 noisy branch (identical clean part, line for line)."""
    max_len = len(labeled_trainloader)
    if max_iters:
        max_len = min(max_len, max_iters)
    R.ensure_refiners(opt, net, net2)  # keeps the native param-group layout of the loaded prefix
    criterion = net.criterion
    net.train_start()
    net2.val_start()
    have_u = len(unlabeled_trainloader.dataset) > 0
    unlabeled_train_iter = iter(unlabeled_trainloader) if have_u else None
    labeled_train_iter = iter(labeled_trainloader)
    t0 = time.time()
    for i in range(max_len):
        try:
            (batch_images_l, batch_img_len_l, batch_text_l, batch_lengths_l, _, batch_labels_l, batch_prob_l, batch_clean_labels_l) = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            (batch_images_l, batch_img_len_l, batch_text_l, batch_lengths_l, _, batch_labels_l, batch_prob_l, batch_clean_labels_l) = next(labeled_train_iter)
        if have_u:
            try:
                (batch_images_u, batch_img_len_u, batch_text_u, batch_lengths_u, _, batch_clean_labels_u) = next(unlabeled_train_iter)
            except StopIteration:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (batch_images_u, batch_img_len_u, batch_text_u, batch_lengths_u, _, batch_clean_labels_u) = next(unlabeled_train_iter)
        if batch_images_l.size(0) < 2:
            # native bug (R25 NATIVE_FIDELITY §3): batch centering turns a 1-sample batch into a zero vector; l2norm
            # backward gives NaN that the optimizer writes into the parameters. Skipped (all arms) and counted.
            stats['skip_labeled_lt2'] += 1
            continue
        batch_prob_l = batch_prob_l.cuda(); batch_labels_l = batch_labels_l.cuda()

        net.optimizer.zero_grad()
        img_emb_l, cap_emb_l, cap_len_l = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)
        sims_l = net.forward_sim(img_emb_l, cap_emb_l, cap_len_l)
        loss_1 = criterion(sims_l, labels=None)
        img_emb_l2, cap_emb_l2, cap_len_l2 = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)
        loss_2 = criterion(img_emb_l.mm(img_emb_l2.t()), margin=0.3)
        loss_3 = criterion(cap_emb_l.mm(cap_emb_l2.t()), margin=0.3)
        loss_l = loss_1 + loss_2 + loss_3
        loss_l.backward()

        high_conf_idx = (batch_prob_l > net.avg_clean_prob).nonzero(as_tuple=True)[0]
        if high_conf_idx.numel() > 0:
            net.memory_queue.enqueue(img_emb_l[high_conf_idx], cap_emb_l[high_conf_idx])
        stats['clean_loss'] += loss_l.item(); stats['steps'] += 1

        if not have_u or batch_images_u.size(0) < 2:
            stats['skip_small_or_empty'] += 1
        else:
            img_emb_u, cap_emb_u, cap_len_u = net.forward_emb(batch_images_u, batch_text_u, batch_lengths_u, imgae_lengths=batch_img_len_u)
            noisy_branches(opt, net, net2, batch_images_u, batch_img_len_u, batch_text_u, batch_lengths_u,
                           img_emb_u, cap_emb_u, stats, probe=probe)
        if net.grad_clip > 0:
            clip_grad_norm_(net.params, net.grad_clip)
        net.optimizer.step()
        if i % opt.log_step == 0:
            logger.info(f"Epoch[{epoch}] step {i}/{max_len} loss_l {loss_l.item():.4f} {time.time() - t0:.1f}s")


def summarize(stats):
    out = R.summarize(stats)
    return out


def main(opt):
    R.train = train  # R.main resolves train() from its module globals at call time
    R.main(opt)
