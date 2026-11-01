"""R30 v1.1 driver (docs/SERVER_PLAN_CN.md §3, §5, §8; docs/STATE_AND_REUSE_CN.md): parent states and forked branches.

Stage 'warmup' (one per parent: 4 H, 3 L): r25_co_train.main up to the end of its warmup loop, line for line, with the R29
install, from the common untrained seed-determined init. Source H = R29 h_native warmup (native VSE.train(mode='warmup') +
report-only diagnostics); source L = R29 l_listwise_vcs warmup (R29 listwise-VCS warmup recipe). The boundary is after the
5th warmup epoch of both peers and its validation, before the first co-train adjust_learning_rate / eval_train (GMM) /
data-loader / sampling; saved as parent_state.pt (read-only) with epoch -1 through the unchanged save_full_state (both encoders
with buffers, both AdamW states incl. step / moments / param groups, Eiters, python/numpy/torch/cuda RNG, warmup history,
best_rsum 0, all_loss [[], []], opt). Refiners / memory queues / queue IDs do not exist yet (created lazily, consuming RNG, by
the first co-train train() call = their first-creation point); the LR schedule is the stateless adjust_learning_rate(co-train
epoch) and the co-train counter restarts at 0 from the boundary; loaders keep no state (workers 0, RandomSampler seeded from the
torch RNG at each iter()); the GMM uses the numpy global RNG (saved). PARENT.json records sha256, training-state fingerprint
(r30_state), data / noise / code / environment digests.

Stage 'unit' (one MATRIX.csv row): r25_co_train.main with --resume <parent> (mode fork: epoch -1 -> co-train epochs
0..num_epochs-1, native resume branch, no warmup) or --resume <own state_last.pt> (mode resume, same branch only).
  * train(): line-for-line r27_co_train.train; loss_1 -> r30_align.main_term(continuation objective, scale) (R30 marks);
    report-only diagnostics as in R30 v1.
  * fork: parent sha256 / id / source / cell / smoke-or-formal identity checked before loading; after loading, the live
    training-state fingerprint must equal the parent file's component by component (FORK_REPORT.json: allowed differences
    are metadata, continuation objective, registered scale, output/budget fields only).
  * resume: only this unit's own state (unit id, parent sha + fingerprint, source, objective, scale, coefficient, budget, cell).
  * every co-train epoch: history entry annotated with raw_epoch / completed_cotrain_epochs / updates (optimizer steps of both
    peers) and METRICS.jsonl (templates/METRICS_SCHEMA.json fields, boundary row completed 0) rewritten; at
    completed_cotrain_epochs == r30_milestone (10) the state is hard-linked to state_completed10.pt (read-only, never
    overwritten) and E10_RECORD.json + per-query DEV ranks are written (extra eval pass; it consumes torch RNG through the
    DataLoader base seed, so the full RNG state is restored afterwards and verified). FULL40 units
    continue to 40; SHORT10/CONTROL10 units have num_epochs 10 (native LR schedule, decay epoch 15 unchanged).
"""
import os, json, time, hashlib, logging
from collections import defaultdict
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r29_co_train as R29
import r29_align as AL
import r30_align as A30
import r30_state as ST
import vse as V
from r27_interface import forward_emb_ctx, enqueue_with_ids
from vocab import deserialize_vocab

logger = R.logger
DIAG_EVERY = R27.TIME_EVERY
SOURCE_ARM = {'H': 'h_native', 'L': 'l_listwise_vcs'}
# opt keys that must be identical between a parent and every branch / resume (scientific configuration and data identity)
STRICT_KEYS = ('data_name', 'vocab_path', 'batch_size', 'lr_update', 'learning_rate', 'workers', 'grad_clip', 'margin', 'img_dim',
               'dropout_rate', 'word_dim', 'embed_size', 'num_layers', 'bi_gru', 'no_imgnorm', 'no_txtnorm', 'noise_file', 'noise_ratio',
               'no_co_training', 'warmup_epoch', 'p_threshold', 'soft_margin', 'seed', 'queue_size', 'k', 'i2t', 't2i', 'noisy_obj',
               'link', 'c1_a', 'c1_b', 'c_D', 'max_iters', 'arm', 'lam', 'r30_tau', 'vocab_size')
ALLOWED_FORK_DIFF = ('data_path', 'output_dir', 'resume', 'num_epochs', 'log_step', 'prefix_epoch', 'stop_after_prefix',
                     'warmup_model_path', 'r29_arm', 'r27_epoch')          # + every r30_* metadata key
IDENTITY_KEYS = STRICT_KEYS        # (v1 name, used by the runner records)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- stage 'warmup' (parent state)

def warmup_main(opt):
    """r25_co_train.main up to the end of the warmup loop (no resume), then the boundary save (R30 marks)."""
    os.makedirs(opt.output_dir, exist_ok=True)
    fh = logging.FileHandler(f'{opt.output_dir}/log.txt', encoding='utf-8'); fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'); fh.setFormatter(fmt)
    ch = logging.StreamHandler(); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name))
    vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    captions_train, images_train = R.get_dataset(opt.data_path, opt.data_name, "train")
    captions_dev, images_dev = R.get_dataset(opt.data_path, opt.data_name, "dev")
    assert os.path.exists(opt.noise_file) or opt.noise_ratio == 0, opt.noise_file  # never generate a mapping here
    noisy_trainloader, data_size, clean_labels = R.get_loader(captions_train, images_train, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
    val_loader = R.get_loader(captions_dev, images_dev, vocab, "dev", opt.batch_size, opt.workers)

    model_A = R.VSE(opt, vocab.word2idx)
    model_B = R.VSE(opt, vocab.word2idx)
    model_B.load_state_dict(model_A.state_dict())

    rec_path = f'{opt.output_dir}/EPOCHS.json'
    history = {'warmup': [], 'cotrain': []}
    best_rsum, start_epoch, all_loss = 0, 0, [[], []]
    assert not opt.resume                                                                             # R30
    for epoch in range(opt.warmup_epoch):
        t = time.time()
        for m in (model_A, model_B):
            if opt.max_iters:  # smoke only
                R._limited_warmup(opt, noisy_trainloader, m, epoch)
            else:
                R.warmup(opt, noisy_trainloader, m, epoch)
        dev = R.validate(opt, val_loader, [model_A, model_B])
        history['warmup'].append({'epoch': epoch, 'dev': dev, 'seconds': time.time() - t})
        json.dump(history, open(rec_path, 'w'), indent=1)
    save_boundary(opt, model_A, model_B, best_rsum, all_loss, history)                              # R30


def data_digests(opt):
    """sha256 of the train/dev caption files and image arrays actually read (staged copies) and of the noise file."""
    d = os.path.join(opt.data_path, opt.data_name)
    files = {f: sha256(os.path.join(d, f)) for f in ('train_caps.txt', 'dev_caps.txt', 'train_ims.npy', 'dev_ims.npy')}
    noise = sha256(opt.noise_file) if os.path.isfile(opt.noise_file) else None
    return dict(files=files, data_split_digest=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), noise_digest=noise)


def env_record():
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'))


def save_boundary(opt, model_A, model_B, best_rsum, all_loss, history):
    """epoch -1 full state through the unchanged save_full_state (read-only parent_state.pt); PARENT.json with sha256 and the
    training-state fingerprint (live == re-read file, asserted)."""
    opt.r30_stage = 'boundary'
    path = f'{opt.output_dir}/parent_state.pt'
    assert not os.path.exists(path), 'parent state exists; parents are built once and never overwritten'
    live = ST.fingerprint(ST.live_state(R, model_A, model_B, -1, best_rsum, all_loss))
    R.save_full_state(path, opt, model_A, model_B, -1, best_rsum, all_loss, history)
    os.chmod(path, 0o444)
    filefp = ST.fingerprint(torch.load(path, map_location='cpu', weights_only=False))
    assert ST.compare(live, filefp)['equal'], 'parent file differs from the live boundary state'
    here = os.path.dirname(os.path.abspath(__file__))
    env = env_record()
    rec = dict(parent_id=opt.r30_parent_id, warmup_source=opt.r30_source_warmup, state_path=path, sha256=sha256(path),
               bytes=os.path.getsize(path), epoch=-1, training_state_fingerprint=filefp['overall'], fingerprint_components=filefp['components'],
               warmup_dev=[w['dev'] for w in history['warmup']], warmup_seconds=[w['seconds'] for w in history['warmup']],
               identity={k: getattr(opt, k) for k in STRICT_KEYS}, max_iters=opt.max_iters, data=data_digests(opt), env=env,
               env_digest=hashlib.sha256(json.dumps({k: env[k] for k in ('torch', 'cuda', 'cudnn', 'gpu')}, sort_keys=True).encode()).hexdigest(),
               gpu=env['gpu'], host=env['host'], job=env['job'],
               code_sha256={f: sha256(os.path.join(here, f)) for f in sorted(os.listdir(here)) if f.endswith('.py')},
               boundary_rule='after the last warmup epoch of both peers and its full-DEV validation; before the first co-train '
                             'adjust_learning_rate / eval_train / data loader / sampling; refiners+queues+IDs first created by train() at co-train epoch 0',
               utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(rec, open(f'{opt.output_dir}/PARENT.json', 'w'), indent=1)
    os.chmod(f'{opt.output_dir}/PARENT.json', 0o444)
    logger.info(f"R30 parent {opt.r30_parent_id} saved {path} sha256 {rec['sha256']} fingerprint {rec['training_state_fingerprint']}")
    return rec


# ----------------------------------------------------------------------------- report-only accumulator

class Acc:
    """GPU-side float64 sums (no host sync per step); flushed into stats at the end of train()."""
    def __init__(self):
        self.v = None

    def add(self, vec):
        self.v = vec if self.v is None else self.v + vec

    def flush(self, stats):
        if self.v is None:
            return
        names = A30.readout_names() + ['main_raw_sum', 'main_scaled_sum', 'steps_main', 'clip_count', 'preclip_norm_sum', 'clip_steps']
        for k, x in zip(names, self.v.tolist()):
            stats['r30_' + k] += x


def _gnorm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return sum(float((x.double() ** 2).sum()) for x in g if x is not None) ** 0.5


# ----------------------------------------------------------------------------- co-training (copy of r27_co_train.train)

def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader, epoch, stats, max_iters=None, probe=None):
    """r27_co_train.train; the only numerical change is loss_1 (R30 marks)."""
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
    acc = Acc(); trace = stats.setdefault('r30_trace', [])                                              # R30 report-only
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
        obs_l = torch.as_tensor(l_obs[np.asarray(ids_l, dtype=np.int64)], device=sims_l.device)          # R30
        loss_1, raw_1 = A30.main_term(opt, net, sims_l, obs_l)                                            # R30 (H0: criterion(sims_l, labels=None))
        img_emb_l2, cap_emb_l2, cap_len_l2 = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)
        loss_2 = criterion(img_emb_l.mm(img_emb_l2.t()), margin=0.3)
        loss_3 = criterion(cap_emb_l.mm(cap_emb_l2.t()), margin=0.3)
        ro = A30.readout(sims_l, obs_l)                                                                   # R30 report-only
        tr = None                                                                                         # R30 report-only
        if timed:                                                                                         # R30 report-only
            g_main, g_intra = _gnorm(loss_1, net.params), _gnorm(loss_2 + loss_3, net.params)            # R30 report-only
            tr = dict(epoch=epoch, step=i, main_raw=float(raw_1.detach()), main_scaled=float(loss_1.detach()), intra=float((loss_2 + loss_3).detach()), g_main=g_main, g_intra=g_intra)  # R30 report-only
            stats['r30_gnorm_main'] += g_main; stats['r30_gnorm_intra'] += g_intra; stats['r30_diag_n'] += 1  # R30 report-only
            for k, v in AL.readout(sims_l, obs_l).items():                                                # R30 report-only
                stats['r30_list_' + k] += v                                                               # R30 report-only
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
                tr.update(g_clean=gc ** 0.5, g_noisy=gn ** 0.5)                                           # R30 report-only
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
        clip_v = torch.zeros(3, dtype=torch.float64, device=sims_l.device)                               # R30 report-only
        if net.grad_clip > 0:
            gpre = clip_grad_norm_(net.params, net.grad_clip)                                            # R30: pre-clip total norm recorded
            clip_v = torch.stack([(gpre > net.grad_clip).double(), gpre.double(), clip_v.new_ones(())])  # R30 report-only
        net.optimizer.step()
        acc.add(torch.cat([ro, torch.stack([raw_1.detach().double(), loss_1.detach().double(), ro.new_ones(())]), clip_v]))  # R30
        if tr is not None:                                                                                # R30 report-only
            tr['preclip_norm'] = float(clip_v[1]); trace.append(tr)                                       # R30 report-only
        if timed:
            stats['time_step_total'] += R27._sync() - ts
            stats['time_n'] += 1
        if i % opt.log_step == 0:
            logger.info(f"Epoch[{epoch}] step {i}/{max_len} loss_l {loss_l.item():.4f} {time.time() - t0:.1f}s")
    acc.flush(stats)                                                                                      # R30 report-only
    stats['train_seconds'] += time.time() - t0


def summarize(stats):
    s = dict(stats)
    out = R27.summarize(stats)
    n = s.get('r30_diag_n', 0)
    for k in list(out):
        if (k.startswith('r30_gnorm_') or k.startswith('r30_list_')) and n:
            out[k + '_mean'] = out.pop(k) / n
    st = s.get('r30_steps_main', 0)
    if st:
        out['r30_main_raw_mean'] = s['r30_main_raw_sum'] / st
        out['r30_main_scaled_mean'] = s['r30_main_scaled_sum'] / st
        out['r30_clip_rate'] = s['r30_clip_count'] / max(s.get('r30_clip_steps', 0), 1)
        out['r30_preclip_norm_mean'] = s['r30_preclip_norm_sum'] / max(s.get('r30_clip_steps', 0), 1)
        for d in ('i2t', 't2i'):
            c = s.get(f'r30_{d}_comparisons', 0)
            out[f'r30_{d}_invalid_frac'] = s[f'r30_{d}_invalid_rows'] / max(s[f'r30_{d}_rows'], 1)
            if c:
                for name in ('margin', 'p', 'Jpair'):
                    out[f'r30_{d}_{name}_mean'] = s[f'r30_{d}_{name}_sum'] / c
                for name in ('pos_beats_hn', 'p_lt01', 'p_gt99'):
                    out[f'r30_{d}_{name}_frac'] = s[f'r30_{d}_{name}'] / c
    return out


# ----------------------------------------------------------------------------- fork / resume identity

def _peek(path):
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def _strict_diff(o, opt):
    return sorted(k for k in STRICT_KEYS if o.get(k) != getattr(opt, k, None))


def check_loadable(path, opt):
    """Identity rule for --resume (raises SystemExit). Modes:
       fork / calibration : the registered immutable parent (sha256, parent id, warmup source, epoch -1 boundary, same cell and
                            scientific config incl. smoke/formal max_iters); never a branch state.
       resume             : only this unit's own state (unit id, parent sha + fingerprint, source, objective, scale,
                            coefficient, budget, cell); never a parent file."""
    st = _peek(path)
    o = st['opt']
    mode = opt.r30_mode
    is_parent = os.path.abspath(path) == os.path.abspath(opt.r30_parent)
    bad = _strict_diff(o, opt)
    if mode in ('fork', 'calibration'):
        if not is_parent:
            raise SystemExit('R30: fork only from the registered parent file')
        d = sha256(path)
        if d != opt.r30_parent_sha:
            raise SystemExit(f'R30: parent sha256 mismatch {d} != {opt.r30_parent_sha}')
        if not (st['epoch'] == -1 and o.get('r30_stage') == 'boundary' and 'i2t_refiner' not in st['model_A']
                and o.get('r30_parent_id') == opt.r30_parent_id and o.get('r30_source_warmup') == opt.r30_source_warmup
                and o.get('r29_arm') == SOURCE_ARM[opt.r30_source_warmup]):
            raise SystemExit('R30: not the registered warmup parent (stage / epoch / id / source)')
        if bad:
            raise SystemExit(f'R30: parent belongs to another cell / configuration: {bad}')
        return mode
    if mode == 'resume':
        if is_parent or o.get('r30_stage') != 'unit':
            raise SystemExit('R30: resume only from this branch state, never from a parent')
        want = dict(r30_unit_id=opt.r30_unit_id, r30_parent_id=opt.r30_parent_id, r30_parent_sha=opt.r30_parent_sha,
                    r30_parent_fingerprint=opt.r30_parent_fingerprint, r30_source_warmup=opt.r30_source_warmup,
                    r30_objective=opt.r30_objective, r30_scale=opt.r30_scale, r30_coef=opt.r30_coef, num_epochs=opt.num_epochs,
                    r30_milestone=opt.r30_milestone)
        wrong = {k: (o.get(k), v) for k, v in want.items() if o.get(k) != v}
        if wrong or bad:
            raise SystemExit(f'R30: cross-branch / foreign state refused {wrong} {bad}')
        if not st['epoch'] + 1 < opt.num_epochs:
            raise SystemExit('R30: branch already completed its budget')
        return mode
    raise SystemExit(f'R30: unknown load mode {mode}')


def fork_report(opt, st, model_A, model_B):
    """Parent file vs live loaded state, component by component; opt differences classified."""
    filefp = ST.fingerprint(st)
    livefp = ST.fingerprint(ST.live_state(R, model_A, model_B, st['epoch'], st['best_rsum'], st['all_loss']))
    cmp = ST.compare(filefp, livefp)
    o, n = st['opt'], vars(opt)
    diff = sorted(k for k in set(o) | set(n) if o.get(k) != n.get(k))
    strict = [k for k in diff if k in STRICT_KEYS]
    other = [k for k in diff if k not in STRICT_KEYS and k not in ALLOWED_FORK_DIFF and not k.startswith('r30_')]
    rep = dict(mode=opt.r30_mode, unit_id=getattr(opt, 'r30_unit_id', None), parent_id=opt.r30_parent_id, parent_sha256=opt.r30_parent_sha,
               parent_fingerprint=filefp['overall'], live_fingerprint=livefp['overall'], registered_fingerprint=opt.r30_parent_fingerprint,
               components_equal=cmp['equal'], differing_components=cmp['differing'] + cmp['missing'],
               opt_differences={k: [repr(o.get(k)), repr(n.get(k))] for k in diff}, strict_differences=strict, unclassified_differences=other,
               continuation=dict(objective=getattr(opt, 'r30_objective', None), scale=getattr(opt, 'r30_scale', None), coef=getattr(opt, 'r30_coef', None)),
               utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    rep['ok'] = (cmp['equal'] and not strict and not other
                 and (opt.r30_parent_fingerprint is None or filefp['overall'] == opt.r30_parent_fingerprint))
    name = 'FORK_REPORT.json' if opt.r30_mode == 'fork' else f'{opt.r30_mode.upper()}_FORK_REPORT.json'
    json.dump(rep, open(os.path.join(opt.output_dir, name), 'w'), indent=1)
    if not rep['ok']:
        raise SystemExit(f'R30: fork state differs from the parent: {rep["differing_components"]} {strict} {other}')
    return rep


def load_full_state(path, opt, model_A, model_B):
    kind = check_loadable(path, opt)
    st = R27.load_full_state(path, opt, model_A, model_B)
    if kind in ('fork', 'calibration'):
        fork_report(opt, st, model_A, model_B)
    logger.info(f'R30 loaded ({kind}) {path} (epoch {st["epoch"]})')
    return st


# ----------------------------------------------------------------------------- per-epoch records, completed-epoch milestone

_ORIG_SAVE = R.save_full_state


def annotate(history, epoch):
    e = history['cotrain'][-1]
    assert e['epoch'] == epoch, ('history / epoch mismatch', e['epoch'], epoch)
    prev = history['cotrain'][-2].get('updates_cumulative', 0) if len(history['cotrain']) > 1 else 0
    e['raw_epoch'] = epoch
    e['completed_cotrain_epochs'] = epoch + 1
    e['updates_epoch'] = int(sum(e['train'][p].get('steps', 0) for p in ('A', 'B')))
    e['updates_cumulative'] = int(prev + e['updates_epoch'])


def metrics_rows(opt, history):
    base = dict(unit_id=opt.r30_unit_id, noise_percent=int(round(opt.noise_ratio * 100)), seed=opt.seed,
                warmup_source=opt.r30_source_warmup, continuation_objective=opt.r30_objective)
    rec = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10')
    w = history['warmup'][-1]['dev']
    rows = [dict(base, completed_cotrain_epochs=0, raw_epoch=None, full_dev_rsum=w['rsum'], updates_cumulative=0, **{k: w[k] for k in rec})]
    for e in history['cotrain']:
        rows.append(dict(base, completed_cotrain_epochs=e['completed_cotrain_epochs'], raw_epoch=e['raw_epoch'], full_dev_rsum=e['dev']['rsum'],
                         updates_cumulative=e['updates_cumulative'], **{k: e['dev'][k] for k in rec}))
    return rows


def write_metrics(opt, history):
    p = os.path.join(opt.output_dir, 'METRICS.jsonl')
    with open(p + '.tmp', 'w') as f:
        for r in metrics_rows(opt, history):
            f.write(json.dumps(r) + '\n')
    os.replace(p + '.tmp', p)


def dev_ranks(opt, models):
    """Per-query full-DEV ranks of the peer-averaged score (native validate readout, recomputed in an extra eval pass)."""
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    captions_dev, images_dev = R.get_dataset(opt.data_path, opt.data_name, "dev")
    val_loader = R.get_loader(captions_dev, images_dev, vocab, "dev", opt.batch_size, opt.workers)
    per_captions = 5
    sims_mean = 0
    for m in models:
        img_embs, cap_embs, cap_lens = R.encode_data(m, val_loader, opt.log_step)
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        sims_mean = sims_mean + R.shard_attn_scores(m, img_embs, cap_embs, cap_lens, opt, shard_size=100)
    sims_mean = sims_mean / len(models)
    r, (ri2t, top1_i2t, _, _) = R.i2t(img_embs.shape[0], sims_mean, per_captions, return_ranks=True)
    ri, (rt2i, top1_t2i, _, _) = R.t2i(img_embs.shape[0], sims_mean, per_captions, return_ranks=True)
    rec = dict(i2t_r1=r[0], i2t_r5=r[1], i2t_r10=r[2], t2i_r1=ri[0], t2i_r5=ri[1], t2i_r10=ri[2])
    rec['rsum'] = float(sum(r[:3]) + sum(ri[:3]))
    return rec, dict(i2t_rank=np.asarray(ri2t), i2t_top1=np.asarray(top1_i2t), t2i_rank=np.asarray(rt2i), t2i_top1=np.asarray(top1_t2i))


def milestone(opt, path, model_A, model_B, epoch, history):
    k = opt.r30_milestone
    out = opt.output_dir
    dst = os.path.join(out, f'state_completed{k}.pt')
    if os.path.exists(dst):
        raise SystemExit(f'R30: {dst} exists; the completed-epoch checkpoint is never overwritten')
    os.link(path, dst)
    os.chmod(dst, 0o444)
    req = lambda x, y: (x['python'] == y['python'] and all(np.array_equal(a, b) for a, b in zip(x['numpy'][1:3], y['numpy'][1:3]))
                        and torch.equal(x['torch'], y['torch']) and all(torch.equal(a, b) for a, b in zip(x['cuda'], y['cuda'])))
    rng0 = R.rng_state()
    rec, ranks = dev_ranks(opt, [model_A, model_B])
    consumed = not req(rng0, R.rng_state())     # every DataLoader iteration draws its base seed from the torch RNG
    R.set_rng_state(rng0)                        # report-only pass: restore the full RNG state so training is unaffected
    rng_same = req(rng0, R.rng_state())
    rp = os.path.join(out, f'RANKS_completed{k}.npz')
    np.savez(rp, **ranks)
    os.chmod(rp, 0o444)
    e = history['cotrain'][-1]
    keys = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
    er = dict(unit_id=opt.r30_unit_id, completed_cotrain_epochs=e['completed_cotrain_epochs'], raw_epoch=e['raw_epoch'],
              updates_cumulative=e['updates_cumulative'], dev=e['dev'], checkpoint=dst, checkpoint_sha256=sha256(dst),
              ranks_file=rp, ranks_sha256=sha256(rp), recomputed=rec, recomputed_equal=all(rec[q] == e['dev'][q] for q in keys),
              rng_consumed_by_rank_pass_before_restore=consumed, rng_unchanged_by_rank_pass=rng_same,
              utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    jp = os.path.join(out, f'E{k}_RECORD.json')
    json.dump(er, open(jp, 'w'), indent=1, default=float)
    os.chmod(jp, 0o444)
    if not rng_same:
        raise SystemExit('R30: the RNG state could not be restored after the rank pass')
    logger.info(f'R30 completed-epoch {k} checkpoint {dst}')


def save_full_state(path, opt, model_A, model_B, epoch, best_rsum, all_loss, history):
    unit = getattr(opt, 'r30_stage', None) == 'unit' and os.path.basename(path) == 'state_last.pt'
    if unit:
        annotate(history, epoch)
    _ORIG_SAVE(path, opt, model_A, model_B, epoch, best_rsum, all_loss, history)
    if unit:
        write_metrics(opt, history)
        if epoch + 1 == opt.r30_milestone:
            milestone(opt, path, model_A, model_B, epoch, history)


def install():
    R27.install()
    R.VSE = V.VSE                     # plain VSE (no extra parameter in any R30 objective)
    R.train = train
    R.summarize = summarize
    R.load_full_state = load_full_state
    R.save_full_state = save_full_state


def main(opt):
    if opt.r30_stage == 'warmup':
        assert opt.r29_arm == SOURCE_ARM[opt.r30_source_warmup] and opt.arm == 'a1_iface' and opt.lam == 0.0
        R29.install()
        warmup_main(opt)
        return
    assert opt.r30_stage == 'unit' and opt.arm == 'a1_iface' and opt.lam == 0.0
    A30.check_term(opt.r30_objective, opt.r30_scale, opt.r30_coef)
    assert opt.resume and opt.r30_mode in ('fork', 'resume'), 'R30 unit forks from its parent or resumes its own state'
    install()
    R.main(opt)
