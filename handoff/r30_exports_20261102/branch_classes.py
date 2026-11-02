#!/usr/bin/env python3
"""R30 read-only export: per-class counts and gradient contributions of the trusted-main (GMM labeled) branch and the
noisy-repair (GMM unlabeled) branch at saved states (review package 2026-11-02, SERVER_PLAN §1 R30, third bullet).
Nothing is trained or written outside out/branch_v1/. Injected-noise truth (dataset _labels / noise map) enters only the
evaluator (class labels and counts), never a score, split, batch choice or gradient.

Per state (R30 warmup parent E0, fixed E10, fixed E40 incl. A1 units) and per peer:
 1) split: the native next-epoch GMM split is recomputed exactly as the training loop would at the start of the next epoch
    (RNG restored from the state, native co_train.eval_train on the native 'warmup' loader, GMM, split_prob(p_threshold));
    peer A trains on pred_B, peer B on pred_A. Checked against the logged split (n_pred_clean / precision / recall) of that
    epoch in the unit's EPOCHS.json when the epoch exists (E10 -> cotrain epoch 10, E0 -> cotrain epoch 0; E40 has none).
    Full-train-set counts per branch: observed-clean / observed-mismatch.
 2) fixed FIT batches drawn from the state's own split (numpy default_rng(20261102); NBATCH labeled and NBATCH unlabeled
    batches of 128 per peer; eval tokenization, all regions, train-mode forward with v3's dropout seeds):
    main branch (labeled batch): the unit's continuation objective, per-row decomposition of the raw main term
      (OBJ.rows of the v3 probe; sum checked against production r30_align.raw_main), classes per direction row:
      observed-clean, observed-mismatch, hard-correct (= observed-clean with >=1 in-batch TRUE negative above the labelled
      positive, margin_core.evaluator_subsets on the same detached scores), easy-correct = clean - hard.
      raw = grad of the class row-sum (encoder params), scaled = c x raw with the unit's formal static coefficient.
    noisy branch (unlabeled batch): native R27.noisy_branches (a1_iface), the per-query rows of the native infonce
      (CE+RCE, both directions, row b attributed to sample b; sum checked against the logged l_nat), raw = scaled (no
      coefficient on this branch). hard-correct uses the in-batch score matrix of the same query embeddings.
    For each class: row count, row-loss sum, ||g_k||, projection on the branch total <g_k,g>/||g|| and on the combined
    pre-clip step G = c g_main + g_intra + g_noisy (<c g_k, G>/||G||), plus the clip factor min(1, 2/||G||).
Parent states have no refiner/queue: the noisy branch is NA there (native: no prototypes at co-train epoch 0).
Usage: branch_classes.py --cond --seed --out-dir --data-path --states LABEL=PATH=UNIT ...
"""
import argparse, copy, csv, glob, hashlib, json, os, sys, time
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch

V3DIR = '/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101/tools'
R30 = '/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101'
OUTR = '/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101'
sys.path.insert(0, V3DIR)
import margin_diag as V1
import margin_diag_v2 as V2
sys.path.insert(1, V1.CODE_R29)
NBATCH, BS, SEED = 32, 128, 20261102
OBJ_OF_R30 = {'H': 'H_hinge', 'L': 'L', 'N': 'N', 'B': 'B', 'PN': 'PN', 'PV': 'PV', 'PB': 'PB'}
CLASSES = ('observed_clean', 'observed_mismatch', 'hard_correct', 'easy_correct')


def flat_dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def flat_norm(a):
    return flat_dot(a, a) ** 0.5


def grads(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [x if x is not None else torch.zeros_like(p) for x, p in zip(g, params)]


def unit_objectives(path, label):
    """(objective, coefficient, unit_id) pairs that train from this state."""
    st = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    so = st['opt'] if isinstance(st['opt'], dict) else vars(st['opt'])
    if so.get('r30_stage') == 'unit':
        return [(OBJ_OF_R30[so['r30_objective']], 1.0 if so['r30_coef'] is None else float(so['r30_coef']), so['r30_unit_id'], so['r30_objective'])], so, st
    pid = so['r30_parent_id']
    out = []
    for f in sorted(glob.glob(f'{OUTR}/runs/*/state_completed10.pt')):
        s2 = torch.load(f, map_location='cpu', mmap=True, weights_only=False)['opt']
        s2 = s2 if isinstance(s2, dict) else vars(s2)
        if s2.get('r30_parent_id') == pid:
            out.append((OBJ_OF_R30[s2['r30_objective']], 1.0 if s2['r30_coef'] is None else float(s2['r30_coef']), s2['r30_unit_id'], s2['r30_objective']))
    return sorted(set(out)), so, st


def logged_split(unit_ids, epoch):
    out = {}
    for u in unit_ids:
        p = f'{OUTR}/runs/{u}/EPOCHS.json'
        if not os.path.isfile(p):
            continue
        e = [x for x in json.load(open(p))['cotrain'] if x['epoch'] == epoch]
        if e:
            out[u] = {'split_A': e[0]['split_A'], 'split_B': e[0]['split_B']}
    return out


def make_batch(idx, caps, ims, noise, true_all, vocab, process_caption):
    obs = noise[idx]
    toks = [process_caption(vocab, caps[c], False) for c in idx]
    lengths = [len(t) for t in toks]
    text = torch.zeros(len(idx), max(lengths)).long()
    for i, t in enumerate(toks):
        text[i, :len(t)] = t
    img = torch.from_numpy(np.ascontiguousarray(ims[obs], dtype=np.float32))
    return dict(img=img, il=torch.full((len(idx),), float(img.shape[1])), text=text, lengths=lengths, obs=torch.as_tensor(obs),
                true=torch.as_tensor(true_all[idx]), gidx=torch.as_tensor(idx, dtype=torch.long), cap_idx=idx)


def class_masks(S, obs, true, MC):
    ev = MC.evaluator_subsets(S, obs, true)
    mis = obs.ne(true)
    return {d: {'observed_clean': ~mis, 'observed_mismatch': mis, 'hard_correct': ev[d]['correct_hard'], 'easy_correct': ev[d]['correct_easy']}
            for d in ('i2t', 't2i')}


class Agg:
    def __init__(self):
        self.d = defaultdict(lambda: defaultdict(float))

    def add(self, key, **kv):
        for k, v in kv.items():
            if v is not None:
                self.d[key][k] += float(v)

    def out(self):
        return {'|'.join(map(str, k)): dict(v) for k, v in self.d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True); ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--states', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--data-path', required=True)
    a = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ref = V1.load_reference()
    import r29_align as AL
    import vse
    import r25_co_train as R
    import r27_co_train as R27
    import r27_interface as I
    import co_train as N
    from vocab import deserialize_vocab
    from data import get_dataset, get_loader, process_caption
    import margin_core as MC
    import r30_pair_adapter as AD
    pr, psrc, pinfo = V1.pair_source(ref)
    OBJ = MC.Objectives(AL, pr, psrc)
    vocab = deserialize_vocab(os.path.join(V1.VOCAB, 'f30k_precomp_vocab.json')); vocab.add_word('<mask>')
    caps, ims = get_dataset(a.data_path, 'f30k_precomp', 'train')
    I.Ctx.set(caps, ims, vocab)
    true_all = np.arange(len(caps)) // 5
    env = {'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename, 'job': os.environ.get('SLURM_JOB_ID'), 'torch': torch.__version__}
    code = {'branch_classes.py': V1.sha(os.path.abspath(__file__)), 'co_train.py': V1.sha(os.path.join(V1.CODE_R29, 'co_train.py')),
            'r27_co_train.py': V1.sha(os.path.join(V1.CODE_R29, 'r27_co_train.py')), 'r30_align(production)': AD.SOURCE,
            'margin_core.py': V1.sha(os.path.join(V3DIR, 'margin_core.py'))}
    os.makedirs(a.out_dir, exist_ok=True)
    A = B = None
    gpu_imgs = False
    for item in a.states:
        label, path, unit_id = (item.split('=') + [None])[:3]
        dest = os.path.join(a.out_dir, f'{a.cond}_s{a.seed}__{label}.json')
        if os.path.isfile(dest):
            print('skip existing', dest, flush=True); continue
        t0 = time.time()
        objs, so, st = unit_objectives(path, label)
        opt = SimpleNamespace(**so)
        assert abs(float(opt.noise_ratio) - (0.0 if a.cond == 'f00' else 0.6)) < 1e-9 and int(opt.seed) == a.seed
        if A is None:
            A = vse.VSE(opt, vocab.word2idx); B = vse.VSE(opt, vocab.word2idx)

        def refs_needed():
            R.ensure_refiners(opt, A, B)
            for m in (A, B):
                I.ensure_ids(m.memory_queue)
        has = {p: V2.load_peer(m, st['model_' + p], R, I, opt, refs_needed) for p, m in (('A', A), ('B', B))}
        if any(has.values()) and not gpu_imgs:
            I.Ctx.gpu_images(); gpu_imgs = True
        raw_epoch = int(st['epoch'])
        next_epoch = raw_epoch + 1
        lr, _ = V2.lr_next(opt, raw_epoch)
        # ---------------- 1) native next-epoch split (RNG of the state restored right before the native loop)
        noise_file = opt.noise_file
        noise = np.load(noise_file).astype(np.int64) if a.cond == 'f60' else np.arange(len(caps)) // 5
        loader, data_size, clean_labels = get_loader(caps, ims, vocab, 'warmup', opt.batch_size, opt.workers, opt.noise_ratio, noise_file)
        clean = np.asarray(clean_labels).astype(bool)
        assert np.array_equal(clean, noise == true_all)
        for m in (A, B):
            m.val_start()
        R.set_rng_state(st['rng'])
        ts = time.time()
        prob_A, prob_B, _ = N.eval_train(opt, A, B, loader, data_size, [[], []], clean_labels, next_epoch)
        pred_A = N.split_prob(prob_A, opt.p_threshold); pred_B = N.split_prob(prob_B, opt.p_threshold)
        split_seconds = time.time() - ts
        del loader
        sq = lambda p: dict(n_pred_clean=int(p.sum()), precision=float((p & clean).sum() / max(p.sum(), 1)), recall=float((p & clean).sum() / max(clean.sum(), 1)))
        rec_split = {'split_A': sq(pred_A), 'split_B': sq(pred_B)}
        logged = logged_split([u for _, _, u, _ in objs], next_epoch)
        split_check = {u: {k: {f: [rec_split[k][f], v[k][f]] for f in ('n_pred_clean', 'precision', 'recall')} for k in ('split_A', 'split_B')}
                       for u, v in logged.items()}
        exact = all(rec_split[k]['n_pred_clean'] == v[k]['n_pred_clean'] and abs(rec_split[k]['precision'] - v[k]['precision']) < 1e-12
                    for v in logged.values() for k in ('split_A', 'split_B')) if logged else None
        full_counts = {}
        for peer, pred in (('A', pred_B), ('B', pred_A)):            # peer A trains on pred_B (native co-teaching)
            full_counts[peer] = {'split_from_peer': 'B' if peer == 'A' else 'A', 'main_n': int(pred.sum()), 'main_observed_clean': int((pred & clean).sum()),
                                 'main_observed_mismatch': int((pred & ~clean).sum()), 'noisy_n': int((~pred).sum()),
                                 'noisy_observed_clean': int((~pred & clean).sum()), 'noisy_observed_mismatch': int((~pred & ~clean).sum())}
        # ---------------- 2) fixed batches from the state's own split
        agg = Agg()
        checks = {'max_abs_rowsum_minus_production_raw_main': 0.0, 'max_abs_noisy_rowsum_minus_lnat': 0.0, 'noisy_no_proto_events': 0, 'nonfinite': 0}
        rng = np.random.default_rng(SEED)
        for peer, net, net2, pred in (('A', A, B, pred_B), ('B', B, A, pred_A)):
            params = net.params
            sd_backup = {'img': copy.deepcopy(net.img_enc.state_dict()), 'txt': copy.deepcopy(net.txt_enc.state_dict())}
            lab, unl = np.flatnonzero(pred), np.flatnonzero(~pred)
            Lb = [rng.choice(lab, BS, replace=False) for _ in range(NBATCH)]
            Ub = [rng.choice(unl, BS, replace=False) for _ in range(NBATCH)] if len(unl) >= BS else []
            nopt = copy.copy(opt); nopt.r27_epoch = next_epoch
            nopt.arm, nopt.lam, nopt.i2t, nopt.t2i, nopt.k = 'a1_iface', 0.0, True, True, 5
            for bi in range(NBATCH):
                # ---- main branch, labeled batch
                bl = make_batch(Lb[bi], caps, ims, noise, true_all, vocab, process_caption)
                obs, tru = bl['obs'].cuda(), bl['true'].cuda()
                net.train_start(); net2.val_start()
                torch.manual_seed(V2.SEED_TRAIN + bi)
                ie, ce, cl = net.forward_emb(bl['img'], bl['text'], bl['lengths'], imgae_lengths=bl['il'])
                sims = net.forward_sim(ie, ce, cl)
                ie2, ce2, _ = net.forward_emb(bl['img'], bl['text'], bl['lengths'], imgae_lengths=bl['il'])
                intra = net.criterion(ie.mm(ie2.t()), margin=0.3) + net.criterion(ce.mm(ce2.t()), margin=0.3)
                g_intra = grads(intra, params)
                cm = class_masks(sims.detach(), obs, tru, MC)
                main_parts = {}
                for o, c, u, o30 in objs:
                    r1, r2 = OBJ.rows(o, sims, obs)
                    prod = AD.R30.raw_main(o30, net, sims, obs)
                    checks['max_abs_rowsum_minus_production_raw_main'] = max(checks['max_abs_rowsum_minus_production_raw_main'], abs(float((r1.sum() + r2.sum()) - prod)))
                    g_tot = grads(r1.sum() + r2.sum(), params)
                    gk = {}
                    for k in CLASSES:
                        m1, m2 = cm['i2t'][k], cm['t2i'][k]
                        lk = (r1 * m1.to(r1.dtype)).sum() + (r2 * m2.to(r2.dtype)).sum()
                        gk[k] = (grads(lk, params) if bool(m1.any() or m2.any()) else [torch.zeros_like(p) for p in params],
                                 int(m1.sum() + m2.sum()), float(lk.detach()))
                    main_parts[(o, c, u)] = (g_tot, gk, float((r1.sum() + r2.sum()).detach()))
                del ie, ce, ie2, ce2, sims, intra
                # ---- noisy branch, unlabeled batch
                g_noisy, nk = None, None
                if has[peer] and Ub:
                    bu = make_batch(Ub[bi], caps, ims, noise, true_all, vocab, process_caption)
                    obsu, truu = bu['obs'].cuda(), bu['true'].cuda()
                    for p in list(params) + [p for r_ in (net.i2t_refiner_node, net.t2i_refiner_node) for p in r_.parameters()]:
                        p.grad = None
                    torch.manual_seed(V2.SEED_TRAIN + 100000 + bi)
                    iu, cu, _, mi, mt = I.forward_emb_ctx(net, bu['img'], bu['text'], bu['lengths'], imgae_lengths=bu['il'])
                    Su = iu.detach() @ cu.detach().t()          # = VSE.forward_sim (dot product) of the same query embeddings
                    cmu = class_masks(Su, obsu, truu, MC)
                    nk = {k: [None, 0, 0.0] for k in CLASSES}
                    captured = {}

                    def probe(branch, d):
                        X, proto = d['X'], d['proto']
                        tau, eps = 0.05, 1e-7
                        sc = (X.mm(proto.t()) / tau).exp()
                        i2t = sc / sc.sum(1, keepdim=True); t2i = sc.t() / sc.t().sum(1, keepdim=True)
                        mask = torch.eye(sc.shape[0], device=sc.device) + eps
                        rows = 0.5 * (-(mask.log() * i2t).sum(1) - (mask.log() * t2i).sum(1) - i2t.diag().log() - t2i.diag().log()) / sc.shape[0]
                        checks['max_abs_noisy_rowsum_minus_lnat'] = max(checks['max_abs_noisy_rowsum_minus_lnat'], abs(float(rows.sum() - d['l_nat'])))
                        w = 0.5                                   # native: loss = 0.5 * l_nat (lam 0 for a1_iface)
                        captured[branch] = float(d['l_nat'].detach())
                        for k in CLASSES:
                            m = cmu[branch][k].to(rows.dtype)
                            lk = w * (rows * m).sum()
                            g = grads(lk, params) if bool(cmu[branch][k].any()) else [torch.zeros_like(p) for p in params]
                            nk[k][0] = g if nk[k][0] is None else [x + y for x, y in zip(nk[k][0], g)]
                            nk[k][1] += int(cmu[branch][k].sum()); nk[k][2] += float(lk.detach())
                    nst = defaultdict(float)
                    R27.noisy_branches(nopt, net, net2, bu['img'], bu['il'], bu['text'], bu['lengths'], iu, cu, mi, mt, nst, (bu['gidx'].cuda(), obsu), False, probe=probe)
                    checks['noisy_no_proto_events'] += int(nst.get('i2t_no_proto', 0) + nst.get('t2i_no_proto', 0))
                    g_noisy = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                    for p in list(params) + [p for r_ in (net.i2t_refiner_node, net.t2i_refiner_node) for p in r_.parameters()]:
                        p.grad = None
                    del iu, cu
                    if not captured:
                        nk = None
                net.img_enc.load_state_dict(sd_backup['img']); net.txt_enc.load_state_dict(sd_backup['txt'])
                net.val_start()
                gn_norm = flat_norm(g_noisy) if g_noisy is not None else None
                for (o, c, u), (g_tot, gk, lsum) in main_parts.items():
                    G = [c * x + y + (z if g_noisy is not None else 0) for x, y, z in zip(g_tot, g_intra, g_noisy if g_noisy is not None else g_intra)]
                    Gn = flat_norm(G); gt = flat_norm(g_tot)
                    clipf = min(1.0, V2.CLIP / (Gn + 1e-6))
                    key = (peer, o, c, u)
                    agg.add(key + ('main', 'TOTAL'), n_batches=1, rows=2 * BS, loss_raw=lsum, loss_scaled=c * lsum, gnorm_raw=gt, gnorm_scaled=c * gt,
                            proj_on_G=c * flat_dot(g_tot, G) / Gn if Gn else None, G_norm=Gn, clip_factor=clipf,
                            g_intra_norm=flat_norm(g_intra), g_noisy_norm=gn_norm)
                    for k, (g, n, l) in gk.items():
                        gkn = flat_norm(g)
                        agg.add(key + ('main', k), rows=n, loss_raw=l, loss_scaled=c * l, gnorm_raw=gkn, gnorm_scaled=c * gkn,
                                proj_on_branch_raw=flat_dot(g, g_tot) / gt if gt else None, proj_on_branch_scaled=c * flat_dot(g, g_tot) / gt if gt else None,
                                proj_on_G=c * flat_dot(g, G) / Gn if Gn else None, proj_on_G_postclip=clipf * c * flat_dot(g, G) / Gn if Gn else None)
                    if nk is not None:
                        gnt = gn_norm
                        agg.add(key + ('noisy', 'TOTAL'), n_batches=1, gnorm_raw=gnt, gnorm_scaled=gnt,
                                proj_on_G=flat_dot(g_noisy, G) / Gn if Gn else None)
                        for k, (g, n, l) in nk.items():
                            if g is None:
                                continue
                            gkn = flat_norm(g)
                            agg.add(key + ('noisy', k), rows=n, loss_raw=l, loss_scaled=l, gnorm_raw=gkn, gnorm_scaled=gkn,
                                    proj_on_branch_raw=flat_dot(g, g_noisy) / gnt if gnt else None, proj_on_branch_scaled=flat_dot(g, g_noisy) / gnt if gnt else None,
                                    proj_on_G=flat_dot(g, G) / Gn if Gn else None, proj_on_G_postclip=clipf * flat_dot(g, G) / Gn if Gn else None)
                    # main-branch per-class counts of this batch (both directions = 2 rows per sample)
                del main_parts, g_intra, g_noisy
            net.img_enc.load_state_dict(sd_backup['img']); net.txt_enc.load_state_dict(sd_backup['txt'])
        del st
        out = {'meta': {'unit_id': unit_id, 'label': label, 'path': path, 'cond': a.cond, 'seed': a.seed, 'raw_epoch': raw_epoch,
                        'completed_cotrain_epochs': raw_epoch + 1, 'split_epoch': next_epoch, 'lr_next': lr,
                        'objectives': [{'objective': o, 'coefficient': c, 'unit_id': u} for o, c, u, _ in objs],
                        'noise_file': noise_file, 'noise_sha256': V1.sha(noise_file) if os.path.isfile(noise_file) else None,
                        'p_threshold': opt.p_threshold, 'workers': opt.workers, 'noisy_branch_available': has,
                        'nbatch_per_branch_per_peer': NBATCH, 'batch_size': BS, 'batch_seed': SEED, 'split_seconds': split_seconds,
                        'seconds': time.time() - t0, 'env': env, 'code_sha256': code, 'pair_source': psrc},
               'split_recomputed': rec_split, 'split_logged_vs_recomputed': split_check, 'split_exact_match_logged': exact,
               'full_train_counts': full_counts, 'checks': checks, 'batches': agg.out()}
        json.dump(out, open(dest + '.tmp', 'w'), indent=1, default=float); os.replace(dest + '.tmp', dest)
        print('done', label, json.dumps(rec_split), 'exact', exact, json.dumps(checks), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
