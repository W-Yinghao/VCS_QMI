"""R25 driver around IN2R-code (commit 909e5329).

Reuses the author's GraphRefiner, MemoryQueue, warmup, eval_train, split_prob, data loaders, VSE and infonce
unchanged. Differences from co_train.main/train (all arms share them, see NATIVE_FIDELITY_CN.md):
  * noisy correction loss is a hook (native = infonce(sims_u)); VCS/BCE are scaled by a frozen c_D;
  * noisy correction is skipped for every arm when the unlabeled batch has B<2 or the unlabeled split is empty;
  * full co-training state is saved at every epoch boundary and can be resumed (common prefix);
  * validate() returns all six recalls (native returned only rSum); per-epoch records are written as JSON;
  * per-epoch checkpoint_{e}.pth.tar files are replaced by model_best.pth.tar (native format) + state_last.pt;
  * no automatic TEST at the end of training (explicit tools/test_eval.py).
"""
import os, sys, time, json, random, logging
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

import co_train as N  # author module (GraphRefiner, MemoryQueue, warmup, eval_train, split_prob, logger)
from co_train import GraphRefiner, MemoryQueue, warmup, eval_train, split_prob
from data import get_loader, get_dataset
from vse import VSE
from vocab import deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores
from utils import AverageMeter, ProgressMeter, adjust_learning_rate
from noisy_loss import noisy_loss

logger = N.logger


# ----------------------------------------------------------------------------- state

def ensure_refiners(opt, net, net2):
    """Same creation order as co_train.train(): net.i2t, net.t2i, net2.i2t, net2.t2i (+ queues)."""
    for m in (net, net2):
        if not hasattr(m, 'memory_queue'):
            m.memory_queue = MemoryQueue(queue_size=opt.queue_size, dim=opt.embed_size)
        for name in ('i2t_refiner_node', 't2i_refiner_node'):
            if not hasattr(m, name):
                r = GraphRefiner(dim=opt.embed_size, k=opt.k).cuda()
                r.train()
                setattr(m, name, r)
                m.optimizer.add_param_group({"params": r.parameters(), "lr": opt.learning_rate})


def model_full_state(m):
    s = {'img_enc': m.img_enc.state_dict(), 'txt_enc': m.txt_enc.state_dict(),
         'optimizer': m.optimizer.state_dict(), 'Eiters': m.Eiters,
         'avg_clean_prob': getattr(m, 'avg_clean_prob', None)}
    if hasattr(m, 'i2t_refiner_node'):
        s['i2t_refiner'] = m.i2t_refiner_node.state_dict()
        s['t2i_refiner'] = m.t2i_refiner_node.state_dict()
        s['n_param_groups'] = len(m.optimizer.param_groups)
    if hasattr(m, 'memory_queue'):
        q = m.memory_queue
        s['queue'] = {'img': q.img_queue.clone(), 'txt': q.txt_queue.clone(), 'ptr': q.ptr, 'is_full': q.is_full}
    return s


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def set_rng_state(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])


def save_full_state(path, opt, model_A, model_B, epoch, best_rsum, all_loss, history):
    st = {'epoch': epoch, 'best_rsum': best_rsum, 'model_A': model_full_state(model_A), 'model_B': model_full_state(model_B),
          'all_loss': all_loss, 'history': history, 'opt': vars(opt).copy(), 'rng': rng_state(),
          'cudnn_benchmark': torch.backends.cudnn.benchmark}
    tmp = path + '.tmp'
    torch.save(st, tmp)
    os.replace(tmp, path)


def load_full_state(path, opt, model_A, model_B):
    st = torch.load(path, map_location='cpu', weights_only=False)  # RNG states must stay CPU ByteTensors
    for m, s in ((model_A, st['model_A']), (model_B, st['model_B'])):
        m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc'])
        m.Eiters = s['Eiters']
        if s['avg_clean_prob'] is not None:
            m.avg_clean_prob = s['avg_clean_prob']
    if 'i2t_refiner' in st['model_A']:
        ensure_refiners(opt, model_A, model_B)  # refiners + param groups in the native order
        for m, s in ((model_A, st['model_A']), (model_B, st['model_B'])):
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            assert len(m.optimizer.param_groups) == s['n_param_groups']
    for m, s in ((model_A, st['model_A']), (model_B, st['model_B'])):
        m.optimizer.load_state_dict(s['optimizer'])
        if 'queue' in s:
            q = m.memory_queue
            q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt'])
            q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
    set_rng_state(st['rng'])  # last: refiner construction above consumed RNG
    return st


# ----------------------------------------------------------------------------- validation

def validate(opt, val_loader, models):
    """Native validate(); returns the six recalls of the peer-averaged similarity (native readout)."""
    per_captions = 1 if opt.data_name == "cc152k_precomp" else 5
    sims_mean, out = 0, {}
    for ind, m in enumerate(models):
        img_embs, cap_embs, cap_lens = encode_data(m, val_loader, opt.log_step)
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        sim = shard_attn_scores(m, img_embs, cap_embs, cap_lens, opt, shard_size=100)
        r = i2t(img_embs.shape[0], sim, per_captions); ri = t2i(img_embs.shape[0], sim, per_captions)
        out[f'peer{ind}_rsum'] = float(sum(r[:3]) + sum(ri[:3]))
        sims_mean = sims_mean + sim
    sims_mean = sims_mean / len(models)
    r = i2t(img_embs.shape[0], sims_mean, per_captions); ri = t2i(img_embs.shape[0], sims_mean, per_captions)
    out.update(i2t_r1=r[0], i2t_r5=r[1], i2t_r10=r[2], t2i_r1=ri[0], t2i_r5=ri[1], t2i_r10=ri[2],
               rsum=float(sum(r[:3]) + sum(ri[:3])), n_images=int(img_embs.shape[0]), n_captions=int(cap_embs.shape[0]))
    logger.info(f"DEV {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()})}")
    return out


# ----------------------------------------------------------------------------- training

def scaled_noisy(opt, S):
    loss, diag = noisy_loss(S, opt.noisy_obj, opt.link, opt.c1_a, opt.c1_b)
    scale = 1.0 if opt.noisy_obj == 'sce' else opt.c_D
    return 0.5 * scale * loss, loss.detach(), diag


def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader, epoch, stats, max_iters=None):
    """co_train.train() with the noisy-loss hook and the shared B<2 / empty-split skip."""
    max_len = len(labeled_trainloader)
    if max_iters:
        max_len = min(max_len, max_iters)
    ensure_refiners(opt, net, net2)
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
            # native bug: batch centering turns a 1-sample batch into a zero vector; l2norm backward then gives NaN
            # that the optimizer writes into the parameters. Skipped (all arms) and counted; no other effect.
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
            for branch, on in (('i2t', opt.i2t), ('t2i', opt.t2i)):
                if not on:
                    continue
                if branch == 'i2t':
                    proto = net2.memory_queue.retrieve_nearest_text(img_emb_u, k=opt.k, refiner_node=net.i2t_refiner_node, img_refiner_node=net.t2i_refiner_node)
                    query = img_emb_u
                else:
                    proto = net2.memory_queue.retrieve_nearest_img(cap_emb_u, k=opt.k, refiner_node=net.t2i_refiner_node, txt_refiner_node=net.i2t_refiner_node)
                    query = cap_emb_u
                if proto is None:
                    stats[f'{branch}_no_proto'] += 1
                    continue
                S = query.mm(proto.t())
                loss_u, raw, diag = scaled_noisy(opt, S)
                loss_u.backward()  # native: separate backward per branch (disjoint graphs)
                stats[f'{branch}_n'] += 1
                stats[f'{branch}_raw_loss'] += raw.item()
                for k, v in diag.items():
                    stats[f'{branch}_{k}'] += float(v)
        if net.grad_clip > 0:
            clip_grad_norm_(net.params, net.grad_clip)  # native: encoder params only (refiners excluded)
        net.optimizer.step()
        if i % opt.log_step == 0:
            logger.info(f"Epoch[{epoch}] step {i}/{max_len} loss_l {loss_l.item():.4f} {time.time() - t0:.1f}s")


def new_stats():
    from collections import defaultdict
    return defaultdict(float)


def summarize(stats):
    out = dict(stats)
    for b in ('i2t', 't2i'):
        n = stats.get(f'{b}_n', 0)
        if n:
            for k in list(out):
                if k.startswith(b + '_') and k != f'{b}_n' and not k.endswith('no_proto'):
                    out[k + '_mean'] = out.pop(k) / n
    if stats.get('steps'):
        out['clean_loss_mean'] = out.pop('clean_loss') / stats['steps']
    return out


def split_quality(pred, clean_labels):
    """Post-hoc only (never fed to training): precision/recall of the GMM 'clean' split vs synthetic truth."""
    pred = np.asarray(pred).astype(bool); clean = np.asarray(clean_labels).astype(bool)
    tp = (pred & clean).sum()
    return dict(n_pred_clean=int(pred.sum()), precision=float(tp / max(pred.sum(), 1)), recall=float(tp / max(clean.sum(), 1)))


def main(opt):
    os.makedirs(opt.output_dir, exist_ok=True)
    fh = logging.FileHandler(f'{opt.output_dir}/log.txt', encoding='utf-8'); fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'); fh.setFormatter(fmt)
    ch = logging.StreamHandler(); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name))
    vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    captions_train, images_train = get_dataset(opt.data_path, opt.data_name, "train")
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev")
    assert os.path.exists(opt.noise_file) or opt.noise_ratio == 0, opt.noise_file  # never generate a mapping here
    noisy_trainloader, data_size, clean_labels = get_loader(captions_train, images_train, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
    val_loader = get_loader(captions_dev, images_dev, vocab, "dev", opt.batch_size, opt.workers)

    model_A = VSE(opt, vocab.word2idx)
    model_B = VSE(opt, vocab.word2idx)
    model_B.load_state_dict(model_A.state_dict())

    rec_path = f'{opt.output_dir}/EPOCHS.json'
    history = {'warmup': [], 'cotrain': []}
    best_rsum, start_epoch, all_loss = 0, 0, [[], []]
    if opt.resume:
        st = load_full_state(opt.resume, opt, model_A, model_B)
        best_rsum, start_epoch, all_loss, history = st['best_rsum'], st['epoch'] + 1, st['all_loss'], st['history']
        history['resumed_from'] = {'path': opt.resume, 'epoch': st['epoch']}
        logger.info(f"resumed full state {opt.resume} at co-training epoch {st['epoch']} (best_rsum {best_rsum})")
    else:
        for epoch in range(opt.warmup_epoch):
            t = time.time()
            for m in (model_A, model_B):
                if opt.max_iters:  # smoke only
                    _limited_warmup(opt, noisy_trainloader, m, epoch)
                else:
                    warmup(opt, noisy_trainloader, m, epoch)
            dev = validate(opt, val_loader, [model_A, model_B])
            history['warmup'].append({'epoch': epoch, 'dev': dev, 'seconds': time.time() - t})
            json.dump(history, open(rec_path, 'w'), indent=1)

    for epoch in range(start_epoch, opt.num_epochs):
        t = time.time()
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)
        prob_A, prob_B, all_loss = eval_train(opt, model_A, model_B, noisy_trainloader, data_size, all_loss, clean_labels, epoch)
        pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
        for m, prob, pred in ((model_A, prob_A, pred_A), (model_B, prob_B, pred_B)):
            cp = prob[pred]
            m.avg_clean_prob = cp.mean() if len(cp) > 0 else 0.0
        stats = {}
        for name, net, net2, pred, prob in (('A', model_A, model_B, pred_B, prob_B), ('B', model_B, model_A, pred_A, prob_A)):
            lt, ut = get_loader(captions_train, images_train, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred, prob=prob)
            s = new_stats()
            train(opt, net, net2, lt, ut, epoch, s, max_iters=opt.max_iters)
            stats[name] = summarize(s)
        dev = validate(opt, val_loader, [model_A, model_B])
        is_best = dev['rsum'] > best_rsum
        best_rsum = max(dev['rsum'], best_rsum)
        history['cotrain'].append({'epoch': epoch, 'dev': dev, 'is_best': is_best, 'train': stats,
                                   'split_A': split_quality(pred_A, clean_labels), 'split_B': split_quality(pred_B, clean_labels),
                                   'lr': model_A.optimizer.param_groups[0]['lr'], 'seconds': time.time() - t,
                                   'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9})
        if is_best:
            torch.save({"epoch": epoch, "model_A": model_A.state_dict(), "model_B": model_B.state_dict(), "best_rsum": best_rsum, "opt": opt},
                       f'{opt.output_dir}/model_best.pth.tar')
        save_full_state(f'{opt.output_dir}/state_last.pt', opt, model_A, model_B, epoch, best_rsum, all_loss, history)
        if epoch == opt.prefix_epoch:
            save_full_state(f'{opt.output_dir}/prefix_state.pt', opt, model_A, model_B, epoch, best_rsum, all_loss, history)
        json.dump(history, open(rec_path, 'w'), indent=1)
        if opt.stop_after_prefix and epoch >= opt.prefix_epoch:
            logger.info("prefix saved; stopping as requested")
            break
    json.dump({'status': 'COMPLETE', 'best_rsum': best_rsum, 'epochs_done': len(history['cotrain'])}, open(f'{opt.output_dir}/DONE.json', 'w'))


def _limited_warmup(opt, train_loader, model, epoch):
    for i, (images, image_lengths, captions, lengths, _) in enumerate(train_loader):
        if i >= opt.max_iters or images.size(0) == 1:
            break
        model.train_start()
        model.train(images, captions, lengths, image_lengths=image_lengths, mode="warmup")
