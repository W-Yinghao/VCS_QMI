"""R27 driver: pinned R25 driver (r25_co_train, unchanged file) with the noisy branch fed by the R27 shared interface.

Outside the noisy branch everything is r25_co_train line for line (warmup, eval_train/GMM split, clean ranking + intra
losses, FIFO enqueue of high-confidence clean pairs, grad clip, LR, full-state save/resume, full-DEV validation, no
TEST). Changes common to A1-A5 (the interface, r27_interface):
  * enqueue also writes the training-visible IDs (caption id, observed image id, write step, peer);
  * candidate identities: other peer's eval/no-grad query search of its OWN queue, ID-bearing slots only, K=5;
  * candidates re-encoded by the CURRENT peer (eval, no grad), centered with the current noisy batch's pre-centering
    modality mean, unit-normalized, stop-grad;
  * the native GraphRefiner + SCE (infonce, correction weight 0.5, separate backward per direction) consumes these
    K re-encoded neighbours instead of the other peer's cached queue embeddings.
A2-A5 add, inside the same per-direction backward, 0.5 * lam * SCE(query, stopgrad(target)) with the generator target
(mean / vcs / bce / snce, r27_targets). A1 = lam 0, no generator.
"""
import time
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

import r25_co_train as R
import r27_interface as I
from r27_interface import Ctx, forward_emb_ctx, search, reencode, enqueue_with_ids
from r27_targets import generate, f_vcs
from r27_delta_control import generate_centered_mean
from vse import infonce

logger = R.logger
ARMS = {'a1_iface': None, 'a2_mean': 'mean', 'a3_vcs': 'vcs', 'a4_bce': 'bce', 'a5_snce': 'snce', 'a6_delta': 'delta'}  # v1.1 supplement: linear background-subtracted control (not full VCS)
TIME_EVERY = 25
SELFCOS_EVERY = 50
_PEER = {}


def peer_of(net, net2):
    if not _PEER:
        _PEER[id(net)] = 0   # model_A always trains first in an epoch (R.main order)
        _PEER[id(net2)] = 1
    return _PEER[id(net)]


def _sync():
    torch.cuda.synchronize()
    return time.time()


def acc_info(stats, b, info, valid):
    """Accumulate generator diagnostics for one direction (means per step; extremes / counts unnormalized)."""
    def f(t):  # mean over finite entries (BCE objective is -inf when a normalized target hits a candidate exactly)
        t = t.double()
        m = torch.isfinite(t)
        return float(t[m].mean()) if bool(m.any()) else float('nan')
    for k in ('obj_raw', 'obj_norm', 'obj_mean'):
        if k in info:
            stats[f'cnt_{b}_{k}_nonfinite'] += int((~torch.isfinite(info[k])).sum())
    stats[f'{b}_Jv_raw'] += f(info['Jv_raw']); stats[f'{b}_Jv_norm'] += f(info['Jv_norm']); stats[f'{b}_Jv_mean'] += f(info['Jv_mean'])
    stats[f'{b}_cos_to_mean'] += f(info['cos_to_mean'])
    stats[f'cnt_{b}_fallback'] += int(info['fallback'].sum())
    stats[f'cnt_{b}_invalid_rows'] += int((~valid).sum())
    stats[f'cnt_{b}_rows'] += int(valid.numel())
    if 'z_norm' in info:
        stats[f'{b}_z_norm'] += f(info['z_norm'])
        stats[f'min_znorm_{b}'] = min(stats.get(f'min_znorm_{b}', 9.0), float(info['z_norm'].min()))
        stats[f'{b}_obj_raw'] += f(info['obj_raw']); stats[f'{b}_obj_norm'] += f(info['obj_norm'])
        stats[f'{b}_obj_mean'] += f(info['obj_mean']); stats[f'{b}_obj_zero'] += f(info['obj_zero'])
        stats[f'{b}_iters'] += f(info['iters'])
        stats[f'max_iters_{b}'] = max(stats.get(f'max_iters_{b}', 0), int(info['iters'].max()))
        stats[f'{b}_rel_res'] += f(info['rel'])
        stats[f'max_rel_res_{b}'] = max(stats.get(f'max_rel_res_{b}', 0.0), float(info['rel'].max()))
        stats[f'cnt_{b}_cap_hits'] += int(info['cap'].sum())
        stats[f'cnt_{b}_interior'] += int(info['interior'].sum())
        stats[f'{b}_nu_hat'] += f(info['nu_hat'])
        if 'near_log_boundary' in info:
            stats[f'cnt_{b}_near_log_boundary'] += int(info['near_log_boundary'].sum())
        if 'lost' in info:
            stats[f'max_range_loss_{b}'] = max(stats.get(f'max_range_loss_{b}', 0.0), float(info['lost'].max()))


def noisy_branches(opt, net, net2, imgs, ilen, txt, tlen, img_u, cap_u, mi, mt, stats, uids, timed, probe=None):
    gen = ARMS[opt.arm]
    t0 = _sync() if timed else None
    with torch.no_grad():
        xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)  # net2 is in val mode (net2.val_start())
    if timed:
        stats['time_peer_fwd'] += _sync() - t0
    for branch, on in (('i2t', opt.i2t), ('t2i', opt.t2i)):
        if not on:
            continue
        tb = _sync() if timed else None
        S, qs = search(net2.memory_queue, xi2 if branch == 'i2t' else xt2, branch, opt.k)
        stats[f'{branch}_usable_frac_sum'] += qs['nv'] / max(qs['n'], 1)
        stats[f'{branch}_usable_cnt'] += 1
        if S is None:
            stats[f'{branch}_no_proto'] += 1
            continue
        t1 = _sync() if timed else None
        Y = reencode(net, branch, S['cols'], mt if branch == 'i2t' else mi)          # [M,d] stop-grad
        neigh = Y[S['inv']]                                                          # [B,K,d]
        t2 = _sync() if timed else None
        X = img_u if branch == 'i2t' else cap_u
        refiner = net.i2t_refiner_node if branch == 'i2t' else net.t2i_refiner_node
        proto = refiner(neigh)                                                       # native GraphRefiner (trainable)
        l_nat = infonce(X.mm(proto.t()))
        loss = 0.5 * l_nat
        Xd = X.detach()
        c = S['A'].mean(0)
        stats[f'{branch}_J_query'] += float(f_vcs(Xd.double(), S['A'].double(), c.double(), Y.double()).mean())
        l_aux = None
        if gen is not None:
            if gen == 'delta':
                Tg, valid, info = generate_centered_mean(S['A'], Y)
            else:
                Tg, valid, info = generate(gen, S['A'], Y)
            t3 = _sync() if timed else None
            if int(valid.sum()) >= 2:
                if bool(valid.all()):
                    l_aux = infonce(X.mm(Tg.t()))
                else:
                    l_aux = infonce(X[valid].mm(Tg[valid].t()))
                loss = loss + 0.5 * opt.lam * l_aux
            else:
                stats[f'cnt_{branch}_aux_skipped'] += 1
            acc_info(stats, branch, info, valid)
            if timed:
                stats[f'time_gen_{branch}'] += t3 - t2
        if probe is not None:
            probe(branch, dict(X=X, Y=Y, neigh=neigh, proto=proto, S=S, l_nat=l_nat, l_aux=l_aux, loss=loss,
                               target=Tg if gen is not None else None, mean=mt if branch == 'i2t' else mi))
        loss.backward()  # native: separate backward per direction
        stats[f'{branch}_n'] += 1
        stats[f'{branch}_raw_loss'] += float(l_nat.detach())
        if l_aux is not None:
            stats[f'{branch}_aux_loss'] += float(l_aux.detach())
        stats[f'{branch}_n_atoms'] += S['cols'].numel()
        stats[f'{branch}_shared_frac'] += float((S['A'] > 0).sum(0).gt(1).float().mean())
        stats[f'{branch}_max_col_mass'] += float(S['A'].mean(0).max())
        stats[f'{branch}_cand_age_epochs'] += float((opt.r27_epoch - S['wepoch']).double().mean())
        # training-visible self-hit diagnostic: candidate == the query's own observed caption / image
        if uids is not None:
            own = uids[0] if branch == 'i2t' else uids[1]
            atom = S['cap'] if branch == 'i2t' else S['img']
            stats[f'{branch}_self_hit'] += float((atom == own.unsqueeze(1)).any(1).float().mean())
        if timed:
            stats[f'time_search_{branch}'] += t1 - tb
            stats[f'time_reenc_{branch}'] += t2 - t1
    if timed:
        stats['time_noisy_total'] += _sync() - t0


def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader, epoch, stats, max_iters=None, probe=None):
    """r25_co_train.train with the R27 noisy branch (identical clean part, line for line, plus ID writes)."""
    max_len = len(labeled_trainloader)
    if max_iters:
        max_len = min(max_len, max_iters)
    R.ensure_refiners(opt, net, net2)  # keeps the native param-group layout of the loaded prefix
    I.ensure_ids(net.memory_queue); I.ensure_ids(net2.memory_queue)
    peer = peer_of(net, net2)
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
        timed = (i % TIME_EVERY == 0)
        ts = _sync() if timed else None
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
            noisy_branches(opt, net, net2, batch_images_u, batch_img_len_u, batch_text_u, batch_lengths_u,
                           img_emb_u, cap_emb_u, mi, mt, stats, uids, timed, probe=probe)
            if i % SELFCOS_EVERY == 0:
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
            stats['time_step_total'] += _sync() - ts
            stats['time_n'] += 1
        if i % opt.log_step == 0:
            logger.info(f"Epoch[{epoch}] step {i}/{max_len} loss_l {loss_l.item():.4f} {time.time() - t0:.1f}s")
    stats['train_seconds'] += time.time() - t0


def summarize(stats):
    s = dict(stats)
    out = _orig_summarize(stats)
    for b in ('i2t', 't2i'):
        n = s.get(f'{b}_usable_cnt', 0)
        if n:
            out[f'{b}_usable_frac_mean'] = s[f'{b}_usable_frac_sum'] / n
            out.pop(f'{b}_usable_frac_sum_mean', None); out.pop(f'{b}_usable_cnt_mean', None)
            out.pop(f'{b}_usable_frac_sum', None); out.pop(f'{b}_usable_cnt', None)
            out[f'{b}_usable_steps'] = n
    tn = s.get('time_n', 0)
    for k in list(out):
        if k.startswith('time_') and k != 'time_n' and tn:
            out[k + '_mean_s'] = out.pop(k) / tn
    if s.get('selfcos_n'):
        out['selfcos_img_mean'] = out.pop('selfcos_img') / s['selfcos_n']
        out['selfcos_txt_mean'] = out.pop('selfcos_txt') / s['selfcos_n']
    return out


# ----------------------------------------------------------------------------- state / loader hooks

_orig_mfs = R.model_full_state
_orig_lfs = R.load_full_state
_orig_get_loader = R.get_loader
_orig_summarize = R.summarize


def model_full_state(m):
    s = _orig_mfs(m)
    if hasattr(m, 'memory_queue'):
        s['r27_queue_ids'] = I.ids_state(m.memory_queue)
    return s


def load_full_state(path, opt, model_A, model_B):
    st = _orig_lfs(path, opt, model_A, model_B)
    info = {}
    for key, m in (('model_A', model_A), ('model_B', model_B)):
        if hasattr(m, 'memory_queue'):
            had = I.load_ids(m.memory_queue, st[key].get('r27_queue_ids'))
            info[key] = 'restored' if had else 'pre-ID prefix queue: all slots cap=-1 (excluded until overwritten)'
    logger.info(f"R27 queue IDs: {info}")
    _PEER.clear()
    return st


def get_loader(captions, images, vocab, data_split, batch_size, workers, noise_ratio=0, noise_file="", pred=[], prob=[]):
    Ctx.set(captions, images, vocab)
    out = _orig_get_loader(captions, images, vocab, data_split, batch_size, workers, noise_ratio, noise_file, pred=pred, prob=prob)
    if data_split == 'train':
        lt, ut = out
        lt.dataset.r27_gidx = np.nonzero(np.asarray(pred))[0].astype(np.int64)
        ut.dataset.r27_gidx = np.nonzero(1 - np.asarray(pred).astype(np.int64))[0].astype(np.int64)
        for d in (lt.dataset, ut.dataset):
            assert len(d.r27_gidx) == len(d), (len(d.r27_gidx), len(d))
            assert all(d.captions[j] == captions[d.r27_gidx[j]] for j in range(0, len(d), 997))
    return out


def install():
    R.model_full_state = model_full_state
    R.load_full_state = load_full_state
    R.get_loader = get_loader
    R.train = train            # R.main resolves these from its module globals at call time
    R.summarize = summarize


def main(opt):
    install()
    R.main(opt)
