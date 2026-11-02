"""R37 prep of the NEW bound parents (copy of R35 prep.py; the same rules, parametrised by parent). Writes OUT/prep/*:
 calibration_init[c]  common positive affine (a0>0, b0) of parent c: balanced logistic on the R35 FIT calibration
                      record set (R35 locks_records.npz, read-only, same records for every parent) applied to the
                      parent's native peer-mean score; observed map of the parent.
 candidates_<c>.npz   frozen-H FIT candidate tables of parent c (peer-mean native score, fp32 topk 32).
 trust_<c>.npz        B-TRUST of parent c: native IN2R eval_train (both peers, 'warmup' loader, GMM; seeds fixed 1001 as in R35)
                      once; subset = mean(prob_A, prob_B) >= 0.5. Clean labels built inside the native loader are discarded unread.
 Record locks and tokens are NOT rebuilt: R35 prep/locks_records.npz and tokens.npz are shared (hash-bound).
Usage: python -B prep.py --data-path <staged dir containing f30k_precomp> --conds n60s1002,n60s1003"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONDS, PEERS, R35_PREP, R37_PREP, CODE_R30, VOCAB, R30_RUNS, N_TRAIN_CAP, N_TRAIN_IMG, N_DEV_CAP, install_guards,
                    guard_record, sha256, write_json, read_json, utc, assert_no_oracle_loaded)
import evals as EV
from data35 import load_cond

N_CAL_P, N_CAL_Q, N_LIST = 5000, 50000, 1024


def record_locks():
    rng = np.random.default_rng(20261110)
    perm = rng.permutation(N_TRAIN_CAP)
    cal_p, ev_p = np.sort(perm[:N_CAL_P]), np.sort(perm[N_CAL_P:2 * N_CAL_P])
    def qpairs(r):
        a = r.integers(0, N_TRAIN_CAP, N_CAL_Q); b = r.integers(0, N_TRAIN_CAP, N_CAL_Q)
        keep = a != b
        while not keep.all():
            b[~keep] = r.integers(0, N_TRAIN_CAP, int((~keep).sum())); keep = a != b
        return np.stack([a, b], 1)
    cal_q = qpairs(np.random.default_rng(20261111)); ev_q = qpairs(np.random.default_rng(20261112))
    r = np.random.default_rng(20261113)
    lp = r.permutation(N_TRAIN_CAP)[:4 * N_LIST]
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


def tokens(data_path):
    sys.path.insert(1, str(CODE_R30))
    import data as D
    from vocab import deserialize_vocab
    vocab = deserialize_vocab(os.path.join(str(VOCAB), 'f30k_precomp_vocab.json')); vocab.add_word('<mask>')
    out = {}
    for split in ('train', 'dev'):
        caps = [l.strip() for l in open(os.path.join(data_path, 'f30k_precomp', f'{split}_caps.txt'))]
        toks = [D.process_caption(vocab, c, False).long().numpy() for c in caps]
        L = max(len(t) for t in toks)
        arr = np.zeros((len(toks), L), np.int64)
        for k, t in enumerate(toks):
            arr[k, :len(t)] = t
        out[f'{split}_tok'] = arr; out[f'{split}_len'] = np.array([len(t) for t in toks], np.int64)
    assert len(out['train_len']) == N_TRAIN_CAP and len(out['dev_len']) == N_DEV_CAP
    return out, len(vocab)


def trust_cond(cond, data_path, workers=0):
    """native IN2R eval_train on the frozen two peers of parent `cond` (one run; R35 trust_n60 rule)."""
    sys.path.insert(1, str(CODE_R30))
    import co_train as N
    import vse as V
    import data as D
    from vocab import deserialize_vocab
    c = CONDS[cond]; hdir = R30_RUNS / c['h_unit']
    ck = torch.load(hdir / 'model_best.pth.tar', map_location='cpu', weights_only=False)
    opt = ck['opt']; opt.data_path = data_path; opt.vocab_path = str(VOCAB); opt.workers = workers
    assert opt.noise_file == c['noise_file'] and float(opt.noise_ratio) == 0.6
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    models = []
    for key in ('model_A', 'model_B'):
        m = V.VSE(opt, vocab.word2idx); m.load_state_dict(ck[key]); models.append(m)
    caps, ims = D.get_dataset(opt.data_path, opt.data_name, 'train')
    random.seed(1001); np.random.seed(1001); torch.manual_seed(1001)
    loader, data_size, _clean_labels_discarded = D.get_loader(caps, ims, vocab, 'warmup', opt.batch_size, workers, opt.noise_ratio, opt.noise_file)
    del _clean_labels_discarded
    t0 = time.time()
    prob_A, prob_B, _ = N.eval_train(opt, models[0], models[1], loader, data_size, [[], []], None, 0)
    prob_A = np.asarray(prob_A, np.float64); prob_B = np.asarray(prob_B, np.float64)
    mean = (prob_A + prob_B) / 2
    subset = np.nonzero(mean >= 0.5)[0].astype(np.int64)
    obs = np.load(c['noise_file']).astype(np.int64)
    info = dict(n_records=int(data_size), n_subset=int(len(subset)), retention=float(len(subset) / data_size),
                n_images_in_subset=int(len(np.unique(obs[subset]))),
                captions_per_image_in_subset=dict(zip(*[x.tolist() for x in np.unique(np.bincount(obs[subset], minlength=N_TRAIN_IMG), return_counts=True)])),
                prob_A_mean=float(prob_A.mean()), prob_B_mean=float(prob_B.mean()), agreement_A_B_ge_half=float(((prob_A >= .5) == (prob_B >= .5)).mean()),
                seconds=time.time() - t0, seeds=dict(python=1001, numpy=1001, torch=1001), loader="native get_loader('warmup'): shuffle, region/caption augmentation, workers=%d" % workers,
                rule='native co_train.eval_train (eval_loss, min-max normalised, GaussianMixture(2,max_iter=10,tol=1e-2,reg_covar=5e-4), prob of low-mean component); subset = mean(prob_A,prob_B) >= 0.5; computed once, never updated')
    return dict(prob_A=prob_A, prob_B=prob_B, subset=subset), info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-path', required=True); ap.add_argument('--conds', required=True)
    a = ap.parse_args()
    install_guards(role='gate'); assert_no_oracle_loaded()
    conds = a.conds.split(',')
    assert all(CONDS[c]['prep_dir'] == str(R37_PREP) for c in conds), 'R37 prep only builds the new parents'
    t0 = time.time(); R37_PREP.mkdir(parents=True, exist_ok=True)
    assert not (R37_PREP / 'PREP_MANIFEST.json').exists(), 'R37 prep already built'
    r35m = read_json(R35_PREP / 'PREP_MANIFEST.json')
    for name in ('locks_records.npz', 'tokens.npz'):
        assert sha256(R35_PREP / name) == r35m['files'][name]['sha256']
    files, man = {}, dict(line='R37 CONFIRM prep (R35 prep rules, new parents)', created_utc=utc(),
                          r35_prep_manifest_sha256=sha256(R35_PREP / 'PREP_MANIFEST.json'),
                          shared_from_r35={k: r35m['files'][k] for k in ('locks_records.npz', 'tokens.npz')})

    def save_npz(name, **arrs):
        np.savez(R37_PREP / name, **arrs); files[name] = dict(sha256=sha256(R37_PREP / name), keys=sorted(arrs))
    L = np.load(R35_PREP / 'locks_records.npz'); L = {k: L[k] for k in L.files}
    man['calibration_init'] = {}; man['candidates'] = {}; man['trust'] = {}; man['seconds'] = {}
    for cond in conds:
        tc = time.time()
        d = load_cond(cond, torch.device('cuda'), need_prep=False)
        obs = d['obs_np']
        sp = native_pair_scores(d, obs[L['cal_p']], L['cal_p']); sq = native_pair_scores(d, obs[L['cal_q'][:, 0]], L['cal_q'][:, 1])
        g = np.concatenate([sp, sq]); wp = np.r_[np.full(len(sp), 1 / len(sp)), np.zeros(len(sq))]; wq = np.r_[np.zeros(len(sp)), np.full(len(sq), 1 / len(sq))]
        fit = EV.fit_affine(g, wp, wq, 'balanced_logistic')
        assert fit['a'] > 0 and fit['finite']
        nat = d['native_dev']
        man['calibration_init'][cond] = dict(a0=fit['a'], b0=fit['b'], balanced_logistic_value=fit['value'],
                                             fit_set='FIT calibration: R35 locks cal_p (5000 observed records, P) + cal_q (50000 independent record pairs, Q), observed map of the parent',
                                             score='native peer-mean s=(sA+sB)/2, sA/sB fp32 dot products of the raw native unit vectors (R37 cache)',
                                             J_raw_fit=EV.j_weighted(g, wp, wq), J_affine_fit=EV.j_weighted(fit['a'] * g + fit['b'], wp, wq),
                                             dev_rank_invariance=EV.rank_invariance(nat, fit['a'], fit['b']))
        t1 = time.time(); i2t, t2i = candidate_tables(d)
        save_npz(f'candidates_{cond}.npz', i2t_top32=i2t, t2i_top32=t2i)
        man['candidates'][cond] = dict(seconds=time.time() - t1, rule='peer-mean native fp32 score (xA.yA + xB.yB)/2 over the full FIT pool, torch.topk(32)',
                                       i2t_frac_images_with_an_observed_caption_in_top32=float((obs[i2t.astype(np.int64)] == np.arange(N_TRAIN_IMG)[:, None]).any(1).mean()),
                                       t2i_frac_captions_observed_image_in_top32=float((t2i.astype(np.int64) == obs[:, None]).any(1).mean()))
        del d; torch.cuda.empty_cache()
        T, info = trust_cond(cond, a.data_path); save_npz(CONDS[cond]['trust_file'], **T); man['trust'][cond] = info
        man['seconds'][cond] = time.time() - tc
    man['files'] = files; man['test_guard'] = guard_record(); man['elapsed_seconds'] = time.time() - t0
    man['env'] = dict(gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__)
    write_json(R37_PREP / 'PREP_MANIFEST.json', man)
    print('R37_PREP_DONE', {k: v for k, v in man.items() if k in ('calibration_init', 'candidates', 'trust')})


if __name__ == '__main__':
    main()
