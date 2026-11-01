"""NRCS-v1 W2/E2/C2 runner on the production GPO adapter (one registered unit per call; resumes the same unit).

  train_gpo.py --unit <unit_id> --data <OUT>/data --ims <dir with train_ims.npy, dev_ims.npy> --runs <OUT>/runs
W2: cold init, cosine hinge + 0.1 intra, 5 epochs, writes parent.pt.   E2: weights+BN of the matching W2 parent,
fresh AdamW + phase counter, 20 epochs.   C2: cold init (same registered init as W2 of that seed), 20 epochs.
Epoch = ceil(N_caption/128) IID with-replacement record batches; per-step cosine LR to 0.1x over the full phase;
FP32 (TF32 off), grad clip 2. Evaluation each epoch (epoch 0 included): SELECT observed J (selection, earliest tie),
AUDIT/DEV clean readouts (development readouts, never used for selection). TEST is never loaded.
"""
import argparse, copy, csv, hashlib, json, math, os, socket, sys, time
import numpy as np
import torch

from native_ratio.objectives import original_j, balanced_logistic, inbatch_pair_weights, marginal_balance
from native_ratio.frozen import evaluate as pkg_evaluate, critic_diagnostics
from native_ratio.models import SimplexRatioCritic
import gpo_adapter as ga

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
MATRIX = f'{W}/package/configs/MATRIX.csv'
PROTOCOL = f'{W}/package/configs/PROTOCOL.json'
PHASE = {'W2': 1, 'E2': 2, 'C2': 3}


def sha_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def source_hashes():
    out = {}
    for d in (HERE, f'{W}/upstream', f'{W}/package/src/native_ratio'):
        for n in sorted(os.listdir(d)):
            if n.endswith('.py'):
                out[os.path.relpath(f'{d}/{n}', W)] = sha_file(f'{d}/{n}')
    return out


def matrix_rows():
    with open(MATRIX, newline='', encoding='utf-8') as f:
        return {r['unit_id']: r for r in csv.DictReader(f)}


def set_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=True)


def gen(seed, noise, phase, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, noise, phase, stream])))


def load_pools(data, ims_dir, noise, device, roles=('fit', 'select', 'audit', 'dev')):
    tr = np.load(f'{ims_dir}/train_ims.npy', mmap_mode='r')
    pools = {}
    for r in roles:
        if r == 'dev':
            pools[r] = ga.RolePool(f'{data}/roles/dev.npz', np.load(f'{ims_dir}/dev_ims.npy', mmap_mode='r'), device)
        elif r == 'audit':
            pools[r] = ga.RolePool(f'{data}/roles/audit.npz', tr, device)
        else:
            pools[r] = ga.RolePool(f'{data}/roles/{r}_n{noise:02d}.npz', tr, device)
    for r in ('fit', 'select'):
        if r in pools:
            assert pools[r].mapping_kind == 'observed_image_index'
    return pools


def ranks_from_scores(s, gt):
    """One-based ranks with retrieval_metrics' tie rule (descending score, increasing candidate index)."""
    i2t = np.zeros(s.shape[0], np.int64) - 1
    for i in range(s.shape[0]):
        pos = np.flatnonzero(gt == i)
        if not len(pos):
            continue
        best = pos[np.argmax(s[i, pos])]
        row = s[i]
        i2t[i] = 1 + np.count_nonzero(row > row[best]) + np.count_nonzero(row[:best] == row[best])
    t2i = np.array([1 + np.count_nonzero(s[:, j] > s[gt[j], j]) + np.count_nonzero(s[:gt[j], j] == s[gt[j], j])
                    for j in range(s.shape[1])], np.int64)
    return i2t[i2t > 0], t2i


class Runner:
    def __init__(self, unit, row, pools, w2i, device, total_epochs=None, steps_per_epoch=None):
        self.unit, self.row = unit, row
        self.stage, self.arm = row['stage'], row['arm']
        self.seed, self.noise = int(row['seed']), int(row['noise_percent'])
        self.bw, self.iw = float(row['balance_weight']), float(row['intra_weight'])
        self.epochs = total_epochs if total_epochs is not None else int(row['epochs'])
        self.pools, self.w2i, self.device = pools, w2i, device
        self.vocab_len, self.mask_id = len(w2i), w2i['<mask>']
        self.B = 128
        self.spe = steps_per_epoch or math.ceil(pools['fit'].n_cap / self.B)
        self.total_steps = self.spe * self.epochs
        self.ratio = self.arm.startswith('ratio')
        ph = PHASE[self.stage]
        self.sampler = gen(self.seed, self.noise, ph, 0)
        self.aug = gen(self.seed, self.noise, ph, 1)
        self.torch_seed = int(np.random.SeedSequence([self.seed, self.noise, ph, 2]).generate_state(1)[0])
        self.p, self.q = inbatch_pair_weights(self.B, device=device)
        self.global_step = 0
        self.batch_digests = []

    # ---------------- model / optimizer
    def init_model(self, parent_state=None):
        torch.manual_seed(self.seed)                       # registered init: same for W2 and C2 of a seed
        self.model = ga.GPOTwoTower(self.arm, self.w2i)
        self.transfer = None
        if parent_state is not None:
            self.transfer = ga.transfer_state(parent_state, self.model)
        self.model.to(self.device)
        torch.manual_seed(self.torch_seed)                 # dropout stream (independent of sampler/augmentation)
        g = self.model.param_groups(2e-4, 1e-3, 1e-4)
        self.opt = torch.optim.AdamW(g, betas=(0.9, 0.999), eps=1e-8)
        self.base_lr = [x['lr'] for x in self.opt.param_groups]
        self.set_lr()

    def lr_factor(self, step):
        T = max(1, self.total_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, T) / T))

    def set_lr(self):
        f = self.lr_factor(self.global_step)
        for g, b in zip(self.opt.param_groups, self.base_lr):
            g['lr'] = b * f

    # ---------------- batches
    def sample_batch(self):
        fit = self.pools['fit']
        cap = self.sampler.integers(0, fit.n_cap, size=self.B)
        img = fit.mapping[cap]                              # observed map only
        if len(self.batch_digests) < 10:
            self.batch_digests.append(hashlib.sha256(cap.astype(np.int64).tobytes()).hexdigest()[:16])
        x, il, gi = ga.image_batch(fit, img, self.aug)
        t, tl, gt = ga.text_batch(fit, cap, self.vocab_len, self.mask_id, self.aug)
        return dict(cap=cap, img=img, x=x, il=il, t=t, tl=tl, guard_img=gi, guard_txt=gt)

    # ---------------- objective
    def losses(self, b):
        m = self.model
        hi = m.encode_image(b['x'], b['il'])
        ht = m.encode_text(b['t'], b['tl'])
        xi, xt = m.heads_in(hi), m.heads_in(ht)
        out = m.critic(xi, xt)
        j = original_j(out['t'], self.p, self.q)
        if self.arm == 'ratio_logistic':
            main = balanced_logistic(out['log_ratio'], self.p, self.q)
        elif self.arm == 'cosine_hinge':
            main = ga.hardest_triplet_2B(out['score'], 0.3)
        else:
            main = -j
        zero = main.new_zeros(())
        bal = marginal_balance(out['u'], out['v'], out['prior'], 'split_cross') if self.ratio else zero
        if self.iw > 0:
            hi2 = m.encode_image(b['x'], b['il'])
            ht2 = m.encode_text(b['t'], b['tl'])
            fit = self.pools['fit']
            intra, li, lt = ga.intra_loss(hi, hi2, fit.image_code[b['img']], ht, ht2, fit.caption_code[b['cap']])
        else:
            intra = li = lt = zero
        loss = main + self.bw * bal + self.iw * intra
        return dict(loss=loss, main=main, bal=bal, intra=intra, intra_img=li, intra_txt=lt, J=j, out=out, xi=xi, xt=xt)

    def grad_diag(self, L):
        """Separate gradient norms (encoder / head) of each term, and of main wrt the pair log-ratio (P vs Q entries)."""
        enc = list(self.model.img_enc.parameters()) + list(self.model.txt_enc.parameters())
        head = list(self.model.critic.parameters())
        rec = {}
        terms = {'main': L['main'], 'balance_weighted': self.bw * L['bal'], 'intra_weighted': self.iw * L['intra']}
        for name, v in terms.items():
            if not v.requires_grad:
                rec[name] = {'encoder': 0.0, 'head': 0.0}
                continue
            gs = torch.autograd.grad(v, enc + head, retain_graph=True, allow_unused=True)
            n = lambda gl: math.sqrt(sum(float((g.double() ** 2).sum()) for g in gl if g is not None))
            rec[name] = {'encoder': n(gs[:len(enc)]), 'head': n(gs[len(enc):])}
        g = torch.autograd.grad(L['main'], L['out']['log_ratio'], retain_graph=True, allow_unused=True)[0]
        if g is not None:
            eye = torch.eye(self.B, dtype=torch.bool, device=g.device)
            rec['dmain_dlogratio_norm_P'] = float(g[eye].norm())
            rec['dmain_dlogratio_norm_Q'] = float(g[~eye].norm())
        out = L['out']
        t = out['t'].detach()
        rec['T_abs_gt_0.99_fraction'] = float((t.abs() > 0.99).double().mean())
        if self.ratio:
            r = out['ratio'].detach()
            rec['r_over_K_gt_0.95_fraction'] = float((r > 0.95 * ga.K).double().mean())
        with torch.no_grad():
            rec['critic'] = critic_diagnostics(self.model.critic, L['xi'].detach(), L['xt'].detach(), {k: v.detach() for k, v in out.items()})
        return rec

    def step(self, diag=False):
        self.model.train()
        self.set_lr()
        b = self.sample_batch()
        self.opt.zero_grad(set_to_none=True)
        L = self.losses(b)
        if not torch.isfinite(L['loss']):
            raise FloatingPointError(f'nonfinite loss at step {self.global_step}')
        d = self.grad_diag(L) if diag else None
        L['loss'].backward()
        params = [p for p in self.model.parameters() if p.grad is not None]
        gn = float(torch.nn.utils.clip_grad_norm_(params, 2.0))
        if not math.isfinite(gn):
            raise FloatingPointError(f'nonfinite gradient at step {self.global_step}')
        t = L['out']['t'].detach()
        eye = torch.eye(self.B, dtype=torch.bool, device=t.device)
        rec = {'step': self.global_step, 'loss': float(L['loss']), 'main': float(L['main']), 'J': float(L['J']),
               'balance_raw': float(L['bal']), 'intra': float(L['intra']), 'intra_img': float(L['intra_img']),
               'intra_txt': float(L['intra_txt']), 'E_P_T': float(t[eye].mean()), 'E_Q_T': float(t[~eye].mean()),
               'E_P_T2': float((t[eye] ** 2).mean()), 'E_Q_T2': float((t[~eye] ** 2).mean()),
               'grad_norm_preclip': gn, 'clipped': gn > 2.0, 'lr_encoder': self.opt.param_groups[0]['lr'],
               'lr_head': self.opt.param_groups[1]['lr'], 'guard_img': b['guard_img'], 'guard_txt': b['guard_txt'],
               'same_image_offdiag_pairs': int((np.equal.outer(b['img'], b['img']) & ~np.eye(self.B, dtype=bool)).sum()),
               'min_img_len': int(b['il'].min()), 'min_txt_len': int(min(b['tl']))}
        if d is not None:
            rec['diag'] = d
        self.opt.step()
        self.global_step += 1
        return rec

    # ---------------- evaluation
    @torch.no_grad()
    def eval_pool(self, role, keep_scores=False):
        pool = self.pools[role]
        hi, ht = ga.encode_pool(self.model, pool, self.vocab_len, self.mask_id)
        m = self.model
        cache = {'x': m.heads_in(hi), 'y': m.heads_in(ht), 'mapping': pool.mapping, 'role': role}
        res = pkg_evaluate(m.critic, cache, self.device, 256)
        if keep_scores:
            a, b = m.critic.retrieval_embeddings(cache['x'], cache['y'])
            return res, (a @ b.T).float().cpu().numpy()
        return res

    def eval_all(self):
        return {r: self.eval_pool(r) for r in ('select', 'audit', 'dev')}

    # ---------------- state
    def rng_state(self):
        return {'sampler': self.sampler.bit_generator.state, 'aug': self.aug.bit_generator.state,
                'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def set_rng_state(self, s):
        self.sampler.bit_generator.state = s['sampler']; self.aug.bit_generator.state = s['aug']
        torch.set_rng_state(s['torch_cpu'])
        if s['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all(s['torch_cuda'])

    def full_state(self):
        return {'model': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                'optimizer': copy.deepcopy(self.opt.state_dict()), 'global_step': self.global_step,
                'rng': self.rng_state(), 'batch_digests': list(self.batch_digests)}

    def load_full_state(self, s):
        self.model.load_state_dict(s['model'])
        self.opt.load_state_dict(s['optimizer'])
        self.global_step = s['global_step']
        self.batch_digests = list(s['batch_digests'])
        self.set_rng_state(s['rng'])
        self.set_lr()


def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, allow_nan=False, default=str)
    os.replace(tmp, path)


def recall_block(ev):
    r = ev['retrieval']
    return {k: r[k] for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')}


def epoch_summary(recs):
    keys = ('loss', 'main', 'J', 'balance_raw', 'intra', 'intra_img', 'intra_txt', 'E_P_T', 'E_Q_T', 'E_P_T2', 'E_Q_T2', 'grad_norm_preclip')
    s = {f'train_mean_{k}': float(np.mean([r[k] for r in recs])) for k in keys}
    s['train_clipped_fraction'] = float(np.mean([r['clipped'] for r in recs]))
    s['train_guard_img'] = int(sum(r['guard_img'] for r in recs)); s['train_guard_txt'] = int(sum(r['guard_txt'] for r in recs))
    s['train_same_image_offdiag_pairs_mean'] = float(np.mean([r['same_image_offdiag_pairs'] for r in recs]))
    s['records_drawn'] = 128 * len(recs)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ims', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    set_precision()
    rows = matrix_rows()
    row = rows[a.unit]
    assert row['stage'] in ('W2', 'E2', 'C2'), row['stage']
    out = f'{a.runs}/{a.unit}'
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f'{out}/DONE.json'):
        print('already DONE', out); return
    t_start = time.time()
    cfg = {'unit': a.unit, 'matrix_row': row, 'protocol_sha256': sha_file(PROTOCOL), 'matrix_sha256': sha_file(MATRIX),
           'data_manifest_sha256': sha_file(f'{a.data}/DATA_MANIFEST.json'), 'source_sha256': source_hashes()}
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    noise = int(row['noise_percent'])
    pools = load_pools(a.data, a.ims, noise, a.device)
    w2i = ga.load_vocab()
    smoke = os.environ.get('NRCS_SMOKE')  # 'epochs,steps_per_epoch' - only allowed under a smoke runs dir
    if smoke:
        assert '/smoke' in os.path.abspath(a.runs), 'NRCS_SMOKE only under a smoke runs dir'
        se, ss = map(int, smoke.split(','))
        R = Runner(a.unit, row, pools, w2i, a.device, total_epochs=se, steps_per_epoch=ss)
        cfg['smoke'] = smoke
        cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    else:
        R = Runner(a.unit, row, pools, w2i, a.device)
    parent_state, parent_info = None, None
    if row['stage'] == 'E2':
        pdir = f"{a.runs}/{row['parent_id']}"
        pdone = json.load(open(f'{pdir}/DONE.json'))
        psha = sha_file(f'{pdir}/parent.pt')
        assert psha == pdone['parent_checkpoint_sha256'], 'parent checkpoint hash mismatch'
        parent_state = torch.load(f'{pdir}/parent.pt', map_location='cpu', weights_only=False)['model']
        parent_info = {'parent_id': row['parent_id'], 'parent_checkpoint_sha256': psha,
                       'parent_gpu': pdone['cost']['gpu_model']}
    R.init_model(parent_state)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    attempt = {'job_id': os.environ.get('SLURM_JOB_ID'), 'host': socket.gethostname(), 'gpu_model': gpu,
               'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    history, best = [], {'J': -float('inf'), 'epoch': None}
    start_epoch = 0
    attempts = []
    if os.path.exists(f'{out}/resume.pt'):
        s = torch.load(f'{out}/resume.pt', map_location='cpu', weights_only=False)
        assert s['unit'] == a.unit and s['cfg_hash'] == cfg_hash, 'resume state belongs to a different unit/config'
        R.load_full_state(s['runner'])
        history, best, start_epoch, attempts = s['history'], s['best'], s['epoch'] + 1, s['attempts']
        attempt['resumed_from_epoch'] = s['epoch']
    attempts.append(attempt)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    if start_epoch == 0:
        ev = R.eval_all()
        history.append({'epoch': 0, 'global_step': 0, 'eval': ev, 'elapsed_s': time.time() - t_start})
        best = {'J': ev['select']['J'], 'epoch': 0}
        if row['stage'] != 'W2':
            atomic_save({'model': R.full_state()['model'], 'arm': R.arm, 'epoch': 0, 'unit': a.unit}, f'{out}/selected.pt')
    steplog = open(f'{out}/steps.jsonl', 'a')
    for ep in range(max(1, start_epoch), R.epochs + 1):
        t_ep = time.time()
        recs = []
        for k in range(R.spe):
            gs = R.global_step
            diag = gs in (0, 1) or gs % 200 == 0
            rec = R.step(diag=diag)
            rec['attempt'] = len(attempts)
            recs.append(rec)
            if diag or gs % 50 == 0:
                steplog.write(json.dumps(rec, default=float) + '\n'); steplog.flush()
        ev = R.eval_all()
        h = {'epoch': ep, 'global_step': R.global_step, 'eval': ev, **epoch_summary(recs),
             'epoch_seconds': time.time() - t_ep, 'elapsed_s': time.time() - t_start,
             'peak_mem_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}
        if ep == 1:
            h['step0'] = recs[0]; h['step1'] = recs[1]
        history.append(h)
        if row['stage'] != 'W2' and ev['select']['J'] > best['J']:
            best = {'J': ev['select']['J'], 'epoch': ep}
            atomic_save({'model': R.full_state()['model'], 'arm': R.arm, 'epoch': ep, 'unit': a.unit}, f'{out}/selected.pt')
        if ep == R.epochs:
            final = R.full_state()
            if row['stage'] == 'W2':
                atomic_save({'model': final['model'], 'optimizer_audit_only': final['optimizer'], 'rng': final['rng'],
                             'global_step': final['global_step'], 'unit': a.unit, 'arm': R.arm, 'cfg': cfg}, f'{out}/parent.pt')
            else:
                atomic_save({'model': final['model'], 'arm': R.arm, 'epoch': ep, 'unit': a.unit}, f'{out}/epoch{ep}.pt')
        atomic_save({'unit': a.unit, 'cfg_hash': cfg_hash, 'runner': R.full_state(), 'history': history, 'best': best,
                     'epoch': ep, 'attempts': attempts}, f'{out}/resume.pt')
        write_json(f'{out}/history.json', history)
        print(f'epoch {ep}/{R.epochs} {time.time()-t_ep:.1f}s selectJ={ev["select"]["J"]:.4f} dev_rsum={ev["dev"]["retrieval"]["rsum"]:.2f}', flush=True)
    steplog.close()
    attempts[-1]['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    attempts[-1]['seconds'] = time.time() - t_start
    attempts[-1]['peak_mem_bytes'] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    done = {'status': 'COMPLETE', 'unit_id': a.unit, 'stage': row['stage'], 'arm': R.arm, 'seed': R.seed, 'noise_percent': noise,
            'config': {'epochs': R.epochs, 'steps_per_epoch': R.spe, 'components': ga.K, 'balance_weight': R.bw,
                       'intra_weight': R.iw, 'batch_size': R.B, 'lr_encoder': 2e-4, 'lr_head': 1e-3, 'weight_decay': 1e-4,
                       'lr_schedule': 'per-step cosine to 0.1x over the full phase', 'grad_clip': 2.0, 'precision': 'FP32 (TF32 off)',
                       'head_input_scale': ga.HEAD_SCALE, 'bn': 'frozen running statistics (all arms)'},
            'cfg_hash': cfg_hash, 'registration': {'protocol_sha256': cfg['protocol_sha256'], 'matrix_sha256': cfg['matrix_sha256'],
            'data_manifest_sha256': cfg['data_manifest_sha256'], **(parent_info or {'parent_id': None})},
            'transfer': R.transfer, 'source_sha256': cfg['source_sha256'], 'batch_record_digest_first10': R.batch_digests,
            'cost': {'gpu_model': gpu, 'attempts': attempts, 'elapsed_seconds_total': float(sum(x.get('seconds', 0) for x in attempts)),
                     'peak_memory_bytes': max(x.get('peak_mem_bytes') or 0 for x in attempts)}}
    if row['stage'] == 'W2':
        done['parent_checkpoint_sha256'] = sha_file(f'{out}/parent.pt')
        done['epoch5'] = {r: (recall_block(history[-1]['eval'][r]) | {'J': history[-1]['eval'][r]['J']}) for r in ('select', 'audit', 'dev')}
    else:
        last = history[-1]
        sel = next(h for h in history if h['epoch'] == best['epoch'])
        done['last'] = {'epoch': last['epoch'], 'audit': last['eval']['audit'], 'dev': last['eval']['dev']}
        done['selected'] = {'epoch': best['epoch'], 'select_J': best['J'], 'tie': 'earliest', 'audit': sel['eval']['audit'], 'dev': sel['eval']['dev']}
        l5 = [h for h in history if h['epoch'] > R.epochs - 5]
        done['last5_dev_mean'] = {k: float(np.mean([recall_block(h['eval']['dev'])[k] for h in l5])) for k in recall_block(last['eval']['dev'])}
        done['epoch20_checkpoint_sha256'] = sha_file(f'{out}/epoch{R.epochs}.pt')
        done['selected_checkpoint_sha256'] = sha_file(f'{out}/selected.pt')
        # per-query DEV ranks at both endpoints
        for tag, path in (('epoch20', f'{out}/epoch{R.epochs}.pt'), ('selected', f'{out}/selected.pt')):
            R.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
            _, s = R.eval_pool('dev', keep_scores=True)
            i2t, t2i = ranks_from_scores(s, pools['dev'].mapping)
            np.savez(f'{out}/dev_ranks_{tag}.npz', i2t=i2t, t2i=t2i, image_ids=pools['dev'].image_ids, caption_ids=pools['dev'].caption_ids)
    write_json(f'{out}/DONE.json', done)
    if row['stage'] != 'W2':   # package tools/aggregate_results.py record (history 'step' = epoch index for E2/C2)
        cfgm = dict(done['config'], noise=noise / 100, noise_percent=noise, parent_id=row['parent_id'] or None)
        hist = [{'step': h['epoch'], 'epoch': h['epoch'], 'select': h['eval']['select'], 'audit_J': h['eval']['audit']['J'],
                 'dev': recall_block(h['eval']['dev']) | {'J': h['eval']['dev']['J']}} for h in history]
        write_json(f'{out}/metrics.json', {'status': 'COMPLETE', 'unit_id': a.unit, 'seed': R.seed, 'arm': R.arm, 'stage': row['stage'],
                   'scope': 'trainable_GPO_development_not_TEST', 'config': cfgm,
                   'selection': {'metric': 'J', 'pool': 'SELECT_observed', 'step': best['epoch'], 'epoch': best['epoch'], 'tie': 'earliest'},
                   'last': done['last'], 'selected': done['selected'], 'last5_dev_mean': done['last5_dev_mean'], 'history': hist,
                   'registration': done['registration'], 'source_sha256': done['source_sha256'], 'cost': done['cost']})
    os.remove(f'{out}/resume.pt')
    print('NRCS_UNIT_DONE', a.unit, flush=True)


if __name__ == '__main__':
    main()
