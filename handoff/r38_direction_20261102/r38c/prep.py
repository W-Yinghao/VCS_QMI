"""R38 prep (byte copy of R37C prep.py; R38 edit: --cond selects one of the new R31 parents cc_s1002/cc_s1003 and writes its own
prep dir OUT/prep/<cond> (same rules, same lock seeds -> locks identical to R37C); s1001 keeps the R37C prep, read-only).
R37C prep (port of R37/R35 prep.py to CC152K; same rules, rebuilt for this dataset). Writes OUT/prep/*:
 locks_records.npz    record locks rebuilt for the 150000 CC152K records with the R35 rule and the SAME seeds
                      (R35 locks index 145000 Flickr records and cannot be carried over): cal_p/ev_p 5000 observed records,
                      cal_q/ev_q 50000 independent record pairs (a != b), lcal/lev list records (1024 each) + 32-slot perms.
 calibration_init[cc] common positive affine (a0>0, b0): balanced logistic on cal_p (P: observed record pairs = image k,
                      caption k) vs cal_q (Q: image of record a, caption of record b) under the parent's native peer-mean score.
 candidates_cc.npz    frozen-H FIT candidate tables (peer-mean native score over the full 150000 x 150000 pool, fp32 top-32).
 trust_cc.npz         B-TRUST: native IN2R eval_train (both peers, native 'warmup' loader with noise_ratio 0, GMM; seeds 1001
                      as in R35) once; subset = mean(prob_A, prob_B) >= 0.5. The native clean labels (all ones for CC152K, no
                      truth exists) are discarded unread.
 Tokens are not built: G units use only the frozen global embeddings (no local encoder).
Usage: python -B prep.py --data-path <dir containing cc152k_precomp>"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONDS, PEERS, OUT, CODE_NATIVE, VOCAB, N_TRAIN_CAP, N_TRAIN_IMG, install_guards,
                    guard_record, sha256, write_json, read_json, utc, assert_no_oracle_loaded)
import evals as EV
from data35 import load_cond

N_CAL_P, N_CAL_Q, N_LIST = 5000, 50000, 1024


def record_locks(n_rec=N_TRAIN_CAP):
    """R35 prep.record_locks verbatim, parametrised by the number of records (same seeds 20261110..20261113)."""
    rng = np.random.default_rng(20261110)
    perm = rng.permutation(n_rec)
    cal_p, ev_p = np.sort(perm[:N_CAL_P]), np.sort(perm[N_CAL_P:2 * N_CAL_P])
    def qpairs(r):
        a = r.integers(0, n_rec, N_CAL_Q); b = r.integers(0, n_rec, N_CAL_Q)
        keep = a != b
        while not keep.all():
            b[~keep] = r.integers(0, n_rec, int((~keep).sum())); keep = a != b
        return np.stack([a, b], 1)
    cal_q = qpairs(np.random.default_rng(20261111)); ev_q = qpairs(np.random.default_rng(20261112))
    r = np.random.default_rng(20261113)
    lp = r.permutation(n_rec)[:4 * N_LIST]
    L = dict(cal_p=cal_p, cal_q=cal_q, ev_p=ev_p, ev_q=ev_q,
             lcal_i=lp[:N_LIST], lcal_t=lp[N_LIST:2 * N_LIST], lev_i=lp[2 * N_LIST:3 * N_LIST], lev_t=lp[3 * N_LIST:])
    for k in ('lcal_i', 'lcal_t', 'lev_i', 'lev_t'):
        L[k + '_perm'] = np.stack([r.permutation(32) for _ in range(N_LIST)])
    return L


@torch.no_grad()
def native_pair_scores(d, img_idx, cap_idx):
    """per-peer native score of explicit pairs (fp32 elementwise dot), peer mean in fp64."""
    out = 0
    for p in PEERS:
        s = (d[('fit', 'img', p)][torch.as_tensor(img_idx, device='cuda')] * d[('fit', 'cap', p)][torch.as_tensor(cap_idx, device='cuda')]).sum(-1)
        out = out + s.double().cpu().numpy()
    return out / 2


@torch.no_grad()
def candidate_tables(d, K=32):
    xs = [d[('fit', 'img', p)] for p in PEERS]; ys = [d[('fit', 'cap', p)] for p in PEERS]
    i2t = np.zeros((N_TRAIN_IMG, K), np.int32); t2i = np.zeros((N_TRAIN_CAP, K), np.int32)
    for i in range(0, N_TRAIN_IMG, 1000):
        sc = sum(x[i:i + 1000] @ y.T for x, y in zip(xs, ys)) / 2
        i2t[i:i + 1000] = sc.topk(K, dim=1).indices.int().cpu().numpy()
    for j in range(0, N_TRAIN_CAP, 4000):
        sc = sum(y[j:j + 4000] @ x.T for x, y in zip(xs, ys)) / 2
        t2i[j:j + 4000] = sc.topk(K, dim=1).indices.int().cpu().numpy()
    return i2t, t2i


def trust_cond(cond, data_path, workers=0):
    """native IN2R eval_train on the frozen two peers of the parent (one run; R35 trust rule)."""
    sys.path.insert(1, str(CODE_NATIVE))
    import co_train as N
    import vse as V
    import data as D
    from vocab import deserialize_vocab
    c = CONDS[cond]; hdir = Path(c['parent_dir'])
    ck = torch.load(hdir / 'model_best.pth.tar', map_location='cpu', weights_only=False)
    opt = ck['opt']; opt.data_path = data_path; opt.vocab_path = str(VOCAB); opt.workers = workers
    assert float(opt.noise_ratio) == 0 and opt.data_name == 'cc152k_precomp'
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    models = []
    for key in ('model_A', 'model_B'):
        m = V.VSE(opt, vocab.word2idx); m.load_state_dict(ck[key]); models.append(m)
    caps, ims = D.get_dataset(opt.data_path, opt.data_name, 'train')
    random.seed(1001); np.random.seed(1001); torch.manual_seed(1001)
    loader, data_size, _clean_labels_discarded = D.get_loader(caps, ims, vocab, 'warmup', opt.batch_size, workers, opt.noise_ratio, opt.noise_file)
    del _clean_labels_discarded
    assert data_size == N_TRAIN_CAP and np.array_equal(np.asarray(loader.dataset.t2i_index), np.arange(N_TRAIN_CAP))
    t0 = time.time()
    prob_A, prob_B, _ = N.eval_train(opt, models[0], models[1], loader, data_size, [[], []], None, 0)
    prob_A = np.asarray(prob_A, np.float64); prob_B = np.asarray(prob_B, np.float64)
    mean = (prob_A + prob_B) / 2
    subset = np.nonzero(mean >= 0.5)[0].astype(np.int64)
    info = dict(n_records=int(data_size), n_subset=int(len(subset)), retention=float(len(subset) / data_size),
                n_images_in_subset=int(len(np.unique(subset))), captions_per_image_in_subset=1,
                prob_A_mean=float(prob_A.mean()), prob_B_mean=float(prob_B.mean()), agreement_A_B_ge_half=float(((prob_A >= .5) == (prob_B >= .5)).mean()),
                seconds=time.time() - t0, seeds=dict(python=1001, numpy=1001, torch=1001), loader="native get_loader('warmup', noise_ratio 0): shuffle, region/caption augmentation, workers=%d" % workers,
                rule='native co_train.eval_train (eval_loss, min-max normalised, GaussianMixture(2,max_iter=10,tol=1e-2,reg_covar=5e-4), prob of low-mean component); subset = mean(prob_A,prob_B) >= 0.5; computed once, never updated; record k = (image k, caption k)')
    return dict(prob_A=prob_A, prob_B=prob_B, subset=subset), info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-path', required=True); ap.add_argument('--cond', required=True, choices=('cc_s1002', 'cc_s1003'))
    a = ap.parse_args()
    install_guards(role='gate'); assert_no_oracle_loaded()
    cond = a.cond
    PREP = Path(CONDS[cond]['prep_dir']); assert str(PREP).startswith(str(OUT)), PREP
    t0 = time.time(); PREP.mkdir(parents=True, exist_ok=True)
    assert not (PREP / 'PREP_MANIFEST.json').exists(), 'R38 prep already built'
    files, man = {}, dict(line='R38 CC152K prep of parent %s (R37C prep rules unchanged; development experiment, no TEST read)' % cond, cond=cond, parent_unit=CONDS[cond]['h_unit'], created_utc=utc())

    def save_npz(name, **arrs):
        np.savez(PREP / name, **arrs); files[name] = dict(sha256=sha256(PREP / name), keys=sorted(arrs))
    L = record_locks()
    save_npz('locks_records.npz', **L)
    man['locks'] = dict(rule='R35 prep.record_locks with n_records=150000 (seeds 20261110/1/2/3); Q pairs a != b (distinct records = distinct images in CC152K)',
                        sizes={k: list(v.shape) for k, v in L.items()})
    man['calibration_init'] = {}; man['candidates'] = {}; man['trust'] = {}; man['seconds'] = {}
    tc = time.time()
    d = load_cond(cond, torch.device('cuda'), need_prep=False)
    obs = d['obs_np']
    sp = native_pair_scores(d, obs[L['cal_p']], L['cal_p']); sq = native_pair_scores(d, obs[L['cal_q'][:, 0]], L['cal_q'][:, 1])
    g = np.concatenate([sp, sq]); wp = np.r_[np.full(len(sp), 1 / len(sp)), np.zeros(len(sq))]; wq = np.r_[np.zeros(len(sp)), np.full(len(sq), 1 / len(sq))]
    fit = EV.fit_affine(g, wp, wq, 'balanced_logistic')
    assert fit['a'] > 0 and fit['finite']
    nat = d['native_dev']
    man['calibration_init'][cond] = dict(a0=fit['a'], b0=fit['b'], balanced_logistic_value=fit['value'],
                                         fit_set='FIT calibration: R37C locks cal_p (5000 observed records, P) + cal_q (50000 independent record pairs, Q), native identity relation',
                                         score='native peer-mean s=(sA+sB)/2, sA/sB fp32 dot products of the raw native unit vectors (R37C cache)',
                                         J_raw_fit=EV.j_weighted(g, wp, wq), J_affine_fit=EV.j_weighted(fit['a'] * g + fit['b'], wp, wq),
                                         dev_rank_invariance=EV.rank_invariance(nat, fit['a'], fit['b']))
    t1 = time.time(); i2t, t2i = candidate_tables(d)
    save_npz(f'candidates_{cond}.npz', i2t_top32=i2t, t2i_top32=t2i)
    man['candidates'][cond] = dict(seconds=time.time() - t1, rule='peer-mean native fp32 score (xA.yA + xB.yB)/2 over the full FIT pool (150000 x 150000), torch.topk(32)',
                                   i2t_frac_images_with_an_observed_caption_in_top32=float((obs[i2t.astype(np.int64)] == np.arange(N_TRAIN_IMG)[:, None]).any(1).mean()),
                                   t2i_frac_captions_observed_image_in_top32=float((t2i.astype(np.int64) == obs[:, None]).any(1).mean()),
                                   i2t_frac_observed_partner_at_rank1=float((i2t[:, 0].astype(np.int64) == np.arange(N_TRAIN_IMG)).mean()),
                                   t2i_frac_observed_partner_at_rank1=float((t2i[:, 0].astype(np.int64) == obs).mean()))
    del d; torch.cuda.empty_cache()
    T, info = trust_cond(cond, a.data_path); save_npz(CONDS[cond]['trust_file'], **T); man['trust'][cond] = info
    man['seconds'][cond] = time.time() - tc
    man['files'] = files; man['test_guard'] = guard_record(); man['elapsed_seconds'] = time.time() - t0
    man['env'] = dict(gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__)
    write_json(PREP / 'PREP_MANIFEST.json', man)
    print('R38_PREP_DONE', cond, {k: v for k, v in man.items() if k in ('calibration_init', 'candidates', 'trust')})


if __name__ == '__main__':
    main()
