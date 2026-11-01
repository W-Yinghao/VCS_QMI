"""R30 v1.1 static main-term calibration (docs/SERVER_PLAN_CN.md §7). One run per matching H parent (wp_f00_s1001_H_e5,
wp_f60_s1001_H_e5), in its own process on a copy of the parent (the parent file is read-only; its sha256 and training-state
fingerprint are verified unchanged after the run). Stage-path N/L/B use coefficient 1 and are not calibrated.

From the parent (loaded through the R30 identity rule, mode calibration = fork checks + FORK report), exactly the native co-train epoch-0 prelude of r25_co_train.main:
adjust_learning_rate(epoch 0) for both peers, eval_train (native GMM; the injected-noise truth argument is passed as None --
eval_train does not use it), split_prob, avg_clean_prob. Then for peer A (labeled/unlabeled loaders from pred_B/prob_B, net2 = B)
and peer B (from pred_A/prob_A, net2 = A), in the order of r30_co_train.train(): ensure_refiners, ensure_ids, iter(unlabeled),
iter(labeled); the first N_BATCHES labeled minibatches of size >= 2 are taken (for peer A these are exactly the first labeled
batches of co-train epoch 0 of every unit; peer B's are fixed by this procedure). Per batch ONE train-mode forward
(net.forward_emb + forward_sim, the forward that feeds loss_1 in training), shared by all terms (same batch, same dropout
draw, unupdated model); for H (native hinge), L (R29 listwise VCS = Ls raw), PN, PV, PB the encoder gradient of the main
term alone is taken with torch.autograd.grad (no intra/noisy term, .grad untouched, no optimizer.step). Pooled over both
peers' batches:  c_a = sqrt(sum_b ||g_H,b||^2) / sqrt(sum_b ||g_a,b||^2)  for a in Ls (raw L), PN, PV, PB; L (stage/L0) = 1 (registered).
Zero / non-finite numerator or denominator = TECHNICAL_ANOMALY, no coefficient. No DEV is loaded, no truth is read.
Usage: r30_calibrate.py <r25 args> --r30_parent P --r30_parent_sha S --r30_parent_id ID [--r30_parent_fingerprint F] --n_batches 64
       --out JSON [--r30_require_gpu NAME]
"""
import os, sys, json, math, time, hashlib, argparse
import numpy as np
import torch
import r25_run
import r25_co_train as R
import r27_interface as I
import r30_align as A30
import r30_co_train as R30
import r30_state as ST
from vocab import deserialize_vocab

TERMS = ('H', 'L', 'PN', 'PV', 'PB')
TERM_KEY = {'H': 'H', 'L': 'Ls', 'PN': 'PN', 'PV': 'PV', 'PB': 'PB'}   # raw term -> coefficient key


def params_digest(models):
    h = hashlib.sha256()
    for m in models:
        for p in m.params:
            h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def coefficients(sums):
    out = {'L0': dict(c=1.0, status='REGISTERED', note='L (L0 and stage paths) coefficient fixed at 1 by design; ratio for the record only',
                      ratio_if_calibrated=(math.sqrt(sums['H']) / math.sqrt(sums['L'])) if sums['L'] > 0 else None)}
    num = math.sqrt(sums['H']) if sums['H'] >= 0 else float('nan')
    for t in ('L', 'PN', 'PV', 'PB'):
        den = math.sqrt(sums[t]) if sums[t] >= 0 else float('nan')
        ok = math.isfinite(num) and math.isfinite(den) and num > 0 and den > 0
        out[TERM_KEY[t]] = dict(c=(num / den) if ok else None, numerator=num, denominator=den, raw_term=t,
                                status='OK' if ok else 'TECHNICAL_ANOMALY')
    return out


def epoch0_prelude(opt, model_A, model_B, noisy_trainloader, data_size, all_loss, epoch):
    """The native co-train epoch prelude of r25_co_train.main (truth argument of eval_train passed as None)."""
    R.adjust_learning_rate(opt, model_A.optimizer, epoch)
    R.adjust_learning_rate(opt, model_B.optimizer, epoch)
    prob_A, prob_B, all_loss = R.eval_train(opt, model_A, model_B, noisy_trainloader, data_size, all_loss, None, epoch)
    pred_A = R.split_prob(prob_A, opt.p_threshold); pred_B = R.split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((model_A, prob_A, pred_A), (model_B, prob_B, pred_B)):
        cp = prob[pred]
        m.avg_clean_prob = cp.mean() if len(cp) > 0 else 0.0
    return pred_A, prob_A, pred_B, prob_B


def batch_record(net, sims, obs):
    """Main-term-only encoder gradient squared norms of every calibration term on ONE shared forward (.grad untouched)."""
    raw, sq = {}, {}
    for t in TERMS:
        v = A30.raw_main(t, net, sims, obs)
        g = torch.autograd.grad(v, net.params, retain_graph=True, allow_unused=True)
        sq[t] = sum(float((x.double() ** 2).sum()) for x in g if x is not None)
        raw[t] = float(v.detach())
    return raw, sq


def peer_batches(opt, net, net2, lt, ut, n_batches, name):
    """Order of r30_co_train.train(): ensure_refiners, ensure_ids, iter(unlabeled), iter(labeled); first n labeled batches (B >= 2),
    one train-mode forward each (the forward that feeds loss_1)."""
    R.ensure_refiners(opt, net, net2)
    I.ensure_ids(net.memory_queue); I.ensure_ids(net2.memory_queue)
    net.train_start(); net2.val_start()
    have_u = len(ut.dataset) > 0
    _u = iter(ut) if have_u else None          # iterator creation order of train()
    it = iter(lt)
    l_obs = np.asarray(lt.dataset.t2i_index, dtype=np.int64)
    info = dict(n_labeled=len(lt.dataset), n_unlabeled=len(ut.dataset), avg_clean_prob=float(net.avg_clean_prob), labeled_batches_per_epoch=len(lt))
    recs, skipped, b = [], 0, 0
    while b < n_batches:
        images, ilen, text, lengths, ids = next(it)[:5]
        if images.size(0) < 2:
            skipped += 1
            continue
        img, cap, cl = net.forward_emb(images, text, lengths, imgae_lengths=ilen)
        sims = net.forward_sim(img, cap, cl)
        obs = torch.as_tensor(l_obs[np.asarray(ids, dtype=np.int64)], device=sims.device)
        raw, sq = batch_record(net, sims, obs)
        recs.append(dict(peer=name, b=b, n=int(images.size(0)), ids=[int(x) for x in ids],
                         ids_sha256=hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(), raw=raw, sqnorm=sq))
        del img, cap, sims
        b += 1
    return recs, skipped, info


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--r30_parent', required=True)
    p.add_argument('--r30_parent_sha', required=True)
    p.add_argument('--r30_parent_id', required=True)
    p.add_argument('--r30_parent_fingerprint', default=None)
    p.add_argument('--n_batches', type=int, default=64)
    p.add_argument('--out', required=True)
    p.add_argument('--r30_require_gpu', default='')
    p.add_argument('--protocol_sha256', default=None)
    a, rest = p.parse_known_args()
    opt = r25_run.parse(rest)
    assert opt.data_name == 'f30k_precomp' and not opt.resume
    opt.arm, opt.lam, opt.r29_arm = 'a1_iface', 0.0, None
    opt.r30_stage, opt.r30_mode, opt.r30_source_warmup, opt.r30_tau = 'calibration', 'calibration', 'H', A30.TAU
    opt.r30_objective, opt.r30_scale, opt.r30_coef, opt.r30_unit_id = None, None, None, None
    opt.r30_parent, opt.r30_parent_sha, opt.r30_parent_id = os.path.abspath(a.r30_parent), a.r30_parent_sha, a.r30_parent_id
    opt.r30_parent_fingerprint = a.r30_parent_fingerprint
    os.makedirs(opt.output_dir, exist_ok=True)
    fp_before = ST.fingerprint(torch.load(opt.r30_parent, map_location='cpu', weights_only=False))['overall']
    r25_run.seed_all(opt.seed)
    gpu = torch.cuda.get_device_name(0)
    if a.r30_require_gpu and gpu != a.r30_require_gpu:
        raise SystemExit(f'R30: GPU {gpu} != condition block GPU {a.r30_require_gpu}')
    R30.install()
    t0 = time.time()
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    captions_train, images_train = R.get_dataset(opt.data_path, opt.data_name, "train")       # FIT only; no DEV loaded
    noisy_trainloader, data_size, _truth = R.get_loader(captions_train, images_train, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
    del _truth                                                                                 # injected-noise truth: never used
    model_A = R.VSE(opt, vocab.word2idx)
    model_B = R.VSE(opt, vocab.word2idx)
    model_B.load_state_dict(model_A.state_dict())
    st = R.load_full_state(opt.r30_parent, opt, model_A, model_B)                              # parent identity rule (sha256) + fork report
    assert st['epoch'] == -1
    epoch, all_loss = st['epoch'] + 1, st['all_loss']
    pred_A, prob_A, pred_B, prob_B = epoch0_prelude(opt, model_A, model_B, noisy_trainloader, data_size, all_loss, epoch)
    d0 = params_digest((model_A, model_B))
    grads_none0 = all(q.grad is None for m in (model_A, model_B) for q in m.params)
    batches, split, skipped = [], {}, 0
    for name, net, net2, pred, prob in (('A', model_A, model_B, pred_B, prob_B), ('B', model_B, model_A, pred_A, prob_A)):
        lt, ut = R.get_loader(captions_train, images_train, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred, prob=prob)
        recs, sk, info = peer_batches(opt, net, net2, lt, ut, a.n_batches, name)
        batches += recs; skipped += sk; split[name] = info
    sums = {t: sum(r['sqnorm'][t] for r in batches) for t in TERMS}
    d1 = params_digest((model_A, model_B))
    grads_none1 = all(q.grad is None for m in (model_A, model_B) for q in m.params)
    co = coefficients(sums)
    sha_after = R30.sha256(opt.r30_parent)
    fp_after = ST.fingerprint(torch.load(opt.r30_parent, map_location='cpu', weights_only=False))['overall']
    parent_unchanged = sha_after == opt.r30_parent_sha and fp_after == fp_before and (a.r30_parent_fingerprint in (None, fp_before))
    here = os.path.dirname(os.path.abspath(__file__))
    out = dict(stage='R30 static main-term calibration', rule=__doc__.split('Usage:')[0].strip(),
               cell=dict(seed=opt.seed, noise_ratio=opt.noise_ratio, noise_file=opt.noise_file),
               n_batches_per_peer=a.n_batches, n_gradient_pairs=len(batches), skipped_lt2=skipped,
               parent_id=opt.r30_parent_id, parent=opt.r30_parent, parent_sha256=opt.r30_parent_sha, parent_sha256_after=sha_after,
               parent_fingerprint_before=fp_before, parent_fingerprint_after=fp_after, parent_unchanged=parent_unchanged,
               protocol_frozen_sha256=a.protocol_sha256,
               code_sha256={f: R30.sha256(os.path.join(here, f)) for f in sorted(os.listdir(here)) if f.endswith('.py')},
               gpu=gpu, host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__,
               split_training_visible=split, sum_sqnorm=sums, coefficients=co,
               params_unchanged=(d0 == d1), grads_none_before=grads_none0, grads_none_after=grads_none1,
               all_ok=(d0 == d1 and grads_none0 and grads_none1 and parent_unchanged and all(co[k]['status'] == 'OK' for k in A30.CALIBRATED)),
               seconds=time.time() - t0, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), batches=batches)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('coefficients', 'sum_sqnorm', 'params_unchanged', 'all_ok', 'seconds')}, indent=1), flush=True)
    sys.exit(0 if out['all_ok'] else 3)


if __name__ == '__main__':
    main()
