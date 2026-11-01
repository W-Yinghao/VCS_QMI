"""R29 driver: R27 driver (r27_co_train, arm a1_iface, lam 0 -> noisy branch = A1 interface for EVERY arm) on top of the R25
driver (r25_co_train), run from the common UNTRAINED seed-determined initialization (warmup + 40 co-train epochs, no prefix).

Changes, all confined to the image-text alignment term of the clean main learning (r29_align.align):
  * warmup(): line-for-line co_train.warmup + VSE.train(mode='warmup'); the cross-modal term infonce(sims) -> align(...,'warmup').
  * train():  line-for-line r27_co_train.train; loss_1 = criterion(sims_l, labels=None) -> align(..., 'cotrain').
  * P arm only: one scalar b per peer (T = tanh(s/tau + b)), own AdamW param group (lr = model lr schedule, weight_decay 0,
    not clipped), saved/restored with the full state.
  * report-only diagnostics every R27.TIME_EVERY steps: alignment-term encoder gradient norm (torch.autograd.grad on the
    retained graph; .grad untouched) and r29_align.readout of the batch; P's b at the end of every stage.
H (h_native) returns the native expressions unchanged in both phases (the gate checks bitwise identity vs the original
co_train.warmup / VSE.train and r27_co_train.train). eval_train / GMM, intra-modal terms, enqueue, noisy branch, grad clip,
LR schedule, validation and save/resume are the unchanged R25/R27 code.
"""
import os, json, time
from collections import defaultdict
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

import vse as V
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
from r27_interface import forward_emb_ctx, enqueue_with_ids
from vse import infonce
import r29_align as AL

logger = R.logger
DIAG_EVERY = R27.TIME_EVERY
_WARM = {}          # warmup diagnostics, keyed '<epoch>_<peer>'
_WPEER = {}


# ----------------------------------------------------------------------------- model factory (P: learnable b)

def make_vse(opt, word2idx):
    m = V.VSE(opt, word2idx)
    if opt.r29_arm == 'p_pointwise_vcs':
        m.r29_b = torch.nn.Parameter(torch.full((), AL.P_B0, dtype=torch.float32, device='cuda'))
        m.optimizer.add_param_group({'params': [m.r29_b], 'lr': opt.learning_rate, 'weight_decay': 0.0})
    return m


def _gnorm_align(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return sum(float((x.double() ** 2).sum()) for x in g if x is not None) ** 0.5


def _obs(dataset, ids):
    return torch.as_tensor(np.asarray(dataset.t2i_index, dtype=np.int64)[np.asarray(ids, dtype=np.int64)], device='cuda')


# ----------------------------------------------------------------------------- warmup

def warmup_step(opt, model, images, captions, lengths, image_lengths, obs, st, diag):
    """VSE.train(mode='warmup') line for line; only the cross-modal term goes through AL.align."""
    model.Eiters += 1
    img_embs, cap_embs, cap_lens = model.forward_emb(images, captions, lengths, imgae_lengths=image_lengths)
    sims = model.forward_sim(img_embs, cap_embs, cap_lens)
    model.optimizer.zero_grad()
    img_embs2, cap_embs2, cap_lens2 = model.forward_emb(images, captions, lengths, imgae_lengths=image_lengths)
    sims1 = model.forward_sim(img_embs2, cap_embs2, cap_lens2)
    img_sims = img_embs.mm(img_embs2.t())
    cap_sims = cap_embs.mm(cap_embs2.t())
    loss = AL.align(opt, model, 'warmup', sims, obs)
    loss2 = infonce(img_sims)
    loss3 = infonce(cap_sims)
    if diag:
        st['gnorm_align'] += _gnorm_align(loss, model.params); st['align_loss'] += float(loss.detach())
        for k, v in AL.readout(sims, obs, getattr(model, 'r29_b', None)).items():
            st[k] += v
        st['diag_n'] += 1
    loss = loss + loss2 + loss3
    loss.backward()
    if model.grad_clip > 0:
        clip_grad_norm_(model.params, model.grad_clip)
    model.optimizer.step()
    return loss.item()


def _warm_record(opt, model, epoch, st, n):
    if not _WPEER:
        _WPEER[id(model)] = 'A'
    _WPEER.setdefault(id(model), 'B')
    out = {k: v / st['diag_n'] for k, v in st.items() if k != 'diag_n' and st['diag_n']}
    out.update(steps=n, diag_n=st['diag_n'])
    if hasattr(model, 'r29_b'):
        out['r29_b_end'] = float(model.r29_b.detach())
    _WARM[f'{epoch}_{_WPEER[id(model)]}'] = out
    json.dump(_WARM, open(os.path.join(opt.output_dir, 'R29_WARMUP_DIAG.json'), 'w'), indent=1)


def warmup(opt, train_loader, model, epoch):
    """co_train.warmup (meters/progress printing dropped; no numerical effect)."""
    st, n = defaultdict(float), 0
    ds = train_loader.dataset
    for i, (images, image_lengths, captions, lengths, ids) in enumerate(train_loader):
        if images.size(0) == 1:
            break
        model.train_start()
        warmup_step(opt, model, images, captions, lengths, image_lengths, _obs(ds, ids), st, i % DIAG_EVERY == 0)
        n += 1
        if i % opt.log_step == 0:
            logger.info(f"Warmup[{epoch}] step {i}/{len(train_loader)}")
    _warm_record(opt, model, epoch, st, n)


def limited_warmup(opt, train_loader, model, epoch):
    """r25_co_train._limited_warmup (smoke only)."""
    st, n = defaultdict(float), 0
    ds = train_loader.dataset
    for i, (images, image_lengths, captions, lengths, ids) in enumerate(train_loader):
        if i >= opt.max_iters or images.size(0) == 1:
            break
        model.train_start()
        warmup_step(opt, model, images, captions, lengths, image_lengths, _obs(ds, ids), st, i % DIAG_EVERY == 0)
        n += 1
    _warm_record(opt, model, epoch, st, n)


# ----------------------------------------------------------------------------- co-training (copy of r27_co_train.train)

def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader, epoch, stats, max_iters=None, probe=None):
    """r27_co_train.train; the only numerical change is loss_1 (R29 marks)."""
    max_len = len(labeled_trainloader)
    if max_iters:
        max_len = min(max_len, max_iters)
    R.ensure_refiners(opt, net, net2)  # keeps the native param-group layout of the loaded prefix
    I.ensure_ids(net.memory_queue); I.ensure_ids(net2.memory_queue)
    peer = R27.peer_of(net, net2)
    opt.r27_epoch = epoch
    ld, ud = labeled_trainloader.dataset, unlabeled_trainloader.dataset
    l_gidx, l_obs = ld.r27_gidx, np.asarray(ld.t2i_index, dtype=np.int64)
    criterion = net.criterion
    net.train_start()
    net2.val_start()
    have_u = len(unlabeled_trainloader.dataset) > 0
    unlabeled_train_iter = iter(unlabeled_trainloader) if have_u else None
    labeled_train_iter = iter(labeled_trainloader)
    u_gidx, u_obs = (ud.r27_gidx, np.asarray(ud.t2i_index, dtype=np.int64)) if have_u else (None, None)
    t0 = time.time()
    for i in range(max_len):
        try:
            (batch_images_l, batch_img_len_l, batch_text_l, batch_lengths_l, ids_l, batch_labels_l, batch_prob_l, batch_clean_labels_l) = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            (batch_images_l, batch_img_len_l, batch_text_l, batch_lengths_l, ids_l, batch_labels_l, batch_prob_l, batch_clean_labels_l) = next(labeled_train_iter)
        if have_u:
            try:
                (batch_images_u, batch_img_len_u, batch_text_u, batch_lengths_u, ids_u, batch_clean_labels_u) = next(unlabeled_train_iter)
            except StopIteration:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (batch_images_u, batch_img_len_u, batch_text_u, batch_lengths_u, ids_u, batch_clean_labels_u) = next(unlabeled_train_iter)
        if batch_images_l.size(0) < 2:
            # native bug (R25 NATIVE_FIDELITY §3): 1-sample batch centering -> zero vector -> NaN. Skipped (all arms).
            stats['skip_labeled_lt2'] += 1
            continue
        timed = (i % R27.TIME_EVERY == 0)
        ts = R27._sync() if timed else None
        batch_prob_l = batch_prob_l.cuda(); batch_labels_l = batch_labels_l.cuda()

        net.optimizer.zero_grad()
        img_emb_l, cap_emb_l, cap_len_l = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)
        sims_l = net.forward_sim(img_emb_l, cap_emb_l, cap_len_l)
        obs_l = torch.as_tensor(l_obs[np.asarray(ids_l, dtype=np.int64)], device=sims_l.device)          # R29
        loss_1 = AL.align(opt, net, 'cotrain', sims_l, obs_l)                                             # R29 (H: criterion(sims_l, labels=None))
        img_emb_l2, cap_emb_l2, cap_len_l2 = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)
        loss_2 = criterion(img_emb_l.mm(img_emb_l2.t()), margin=0.3)
        loss_3 = criterion(cap_emb_l.mm(cap_emb_l2.t()), margin=0.3)
        if timed:                                                                                         # R29 report-only
            stats['r29_gnorm_align'] += _gnorm_align(loss_1, net.params); stats['r29_align_loss'] += float(loss_1.detach())
            for k, v in AL.readout(sims_l, obs_l, getattr(net, 'r29_b', None)).items():
                stats['r29_' + k] += v
            stats['r29_diag_n'] += 1
        loss_l = loss_1 + loss_2 + loss_3
        loss_l.backward()

        high_conf_idx = (batch_prob_l > net.avg_clean_prob).nonzero(as_tuple=True)[0]
        if high_conf_idx.numel() > 0:
            hc = np.asarray(ids_l, dtype=np.int64)[high_conf_idx.cpu().numpy()]
            enqueue_with_ids(net.memory_queue, img_emb_l[high_conf_idx], cap_emb_l[high_conf_idx],
                             l_gidx[hc], l_obs[hc], epoch, i, peer)
        stats['clean_loss'] += loss_l.item(); stats['steps'] += 1

        if not have_u or batch_images_u.size(0) < 2:
            stats['skip_small_or_empty'] += 1
        else:
            img_emb_u, cap_emb_u, cap_len_u, mi, mt = forward_emb_ctx(net, batch_images_u, batch_text_u, batch_lengths_u, imgae_lengths=batch_img_len_u)
            iu = np.asarray(ids_u, dtype=np.int64)
            dev = img_emb_u.device
            uids = (torch.as_tensor(u_gidx[iu], device=dev), torch.as_tensor(u_obs[iu], device=dev))
            gsnap = [p.grad.detach().clone() if p.grad is not None else None for p in net.params] if timed else None
            R27.noisy_branches(opt, net, net2, batch_images_u, batch_img_len_u, batch_text_u, batch_lengths_u,
                               img_emb_u, cap_emb_u, mi, mt, stats, uids, timed, probe=probe)
            if timed:  # gradient diagnostics (no extra autograd): clean = grads before the noisy branch, noisy = increment
                gc = sum(float((g.double() ** 2).sum()) for g in gsnap if g is not None)
                gn = sum(float(((p.grad.double() - (g.double() if g is not None else 0)) ** 2).sum()) for p, g in zip(net.params, gsnap) if p.grad is not None)
                gr = sum(float((p.grad.double() ** 2).sum()) for m in (net.i2t_refiner_node, net.t2i_refiner_node) for p in m.parameters() if p.grad is not None)
                stats['gnorm_enc_clean'] += gc ** 0.5; stats['gnorm_enc_noisy'] += gn ** 0.5; stats['gnorm_refiner'] += gr ** 0.5
                stats['gnorm_n'] += 1
                del gsnap
            if i % R27.SELFCOS_EVERY == 0:
                # diagnostic: eval-mode re-encoding of the query batch itself, same centering, vs native train-mode query
                with torch.no_grad():
                    was = net.training; net.val_start()
                    ri = net.img_enc(batch_images_u.cuda(), batch_img_len_u.cuda()); rt = net.txt_enc(batch_text_u.cuda(), batch_lengths_u)
                    if was:
                        net.train_start()
                    stats['selfcos_img'] += float((R.N.F.normalize(ri - mi, dim=-1) * img_emb_u.detach()).sum(1).mean())
                    stats['selfcos_txt'] += float((R.N.F.normalize(rt - mt, dim=-1) * cap_emb_u.detach()).sum(1).mean())
                    stats['selfcos_n'] += 1
        if net.grad_clip > 0:
            clip_grad_norm_(net.params, net.grad_clip)
        net.optimizer.step()
        if timed:
            stats['time_step_total'] += R27._sync() - ts
            stats['time_n'] += 1
        if i % opt.log_step == 0:
            logger.info(f"Epoch[{epoch}] step {i}/{max_len} loss_l {loss_l.item():.4f} {time.time() - t0:.1f}s")
    stats['train_seconds'] += time.time() - t0
    if hasattr(net, 'r29_b'):                                                                             # R29
        stats['r29_b_end'] = float(net.r29_b.detach())


def summarize(stats):
    s = dict(stats)
    out = R27.summarize(stats)
    n = s.get('r29_diag_n', 0)
    for k in list(out):
        if k.startswith('r29_') and k not in ('r29_diag_n', 'r29_b_end') and n:
            out[k + '_mean'] = out.pop(k) / n
    return out


# ----------------------------------------------------------------------------- state hooks (P's b)

def model_full_state(m):
    s = R27.model_full_state(m)
    if hasattr(m, 'r29_b'):
        s['r29_b'] = m.r29_b.detach().clone()
    return s


def load_full_state(path, opt, model_A, model_B):
    st = R27.load_full_state(path, opt, model_A, model_B)
    for key, m in (('model_A', model_A), ('model_B', model_B)):
        has_m, has_s = hasattr(m, 'r29_b'), 'r29_b' in st[key]
        assert has_m == has_s, 'R29: P-arm bias present in model / state mismatch (cross-arm resume?)'
        if has_m:
            with torch.no_grad():
                m.r29_b.copy_(st[key]['r29_b'].to(m.r29_b.device))
    assert st['opt'].get('r29_arm') == opt.r29_arm, ('R29: cross-arm resume refused', st['opt'].get('r29_arm'), opt.r29_arm)
    return st


def install():
    R27.install()
    R.VSE = make_vse
    R.warmup = warmup
    R._limited_warmup = limited_warmup
    R.train = train
    R.summarize = summarize
    R.model_full_state = model_full_state
    R.load_full_state = load_full_state


def main(opt):
    assert opt.r29_arm in AL.ARMS and opt.arm == 'a1_iface' and opt.lam == 0.0
    install()
    R.main(opt)
