"""R39 COMPLETE prep + non-training readouts (package docs/SERVER_PLAN_CN.md 2.1-2.3), one dataset per call (--tree r39f | r39c).
For each of the two fixed teacher parents (same series, s1002/s1003; their frozen native caches, read-only) and their OWN peer-mean
native score S = (X_A Y_A^T + X_B Y_B^T)/2 (fp32 matmul, as the R35/R37C candidate tables):
  i2t: every FIT image q -> full-FIT-caption top-1 c(q); reverse rank of q among ALL FIT images for caption c(q) (1 + #strictly higher)
  t2i: every FIT caption q -> full-FIT-image top-1 i(q); reverse rank of q among ALL FIT captions for image i(q)
(each rank is computed inside the same matmul as the compared score). Acceptance (package reference rule, no truth): target N row
all-null (observed partner absent from the target H's natural top-32), both teachers agree, agreed ID inside the target top-32,
both reverse ranks <= 5. Writes OUT/prep/<ds>/completion_<ds>.npz + COMPLETE_PREP_MANIFEST.json, COVERAGE_<ds>.json (coverage
tables, label masses, repeats under the actual N sampler stream) and TEACHER_ONLY_RERANK_<ds>.json (DEV: mean of the two teachers'
peer-mean native DEV scores reranks the target H top-100, rest in H order; no DEV label used for any choice).
Usage: python -I -B tools/prep_complete.py --tree r39f|r39c"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tree', required=True, choices=('r39f', 'r39c')); a = ap.parse_args()
    sys.path.insert(0, str(W / a.tree))
    import torch
    import common as C
    from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, install_guards, guard_record, sha256, write_json, read_json, utc
    from data35 import load_cond
    import data36 as D36
    import evals as EV
    from r39_targets import complete_missing
    install_guards(role='gate')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    t_all = time.time(); dev = torch.device('cuda')
    pdir = C.R39_PREP / C.DS_KEY; pdir.mkdir(parents=True, exist_ok=True)
    tgt = load_cond(C.TARGET_COND, dev)
    obs = tgt['obs_np']; ci = tgt['cand_i2t'].astype(np.int64); ct = tgt['cand_t2i'].astype(np.int64)
    res = {}; tinfo = {}
    for k, tc in enumerate(C.TEACHER_CONDS, 1):
        t0 = time.time()
        T = load_cond(tc, dev, need_prep=False)
        assert T['manifest']['parent']['unit'] != tgt['manifest']['parent']['unit']
        assert np.array_equal(T['obs_np'], obs), 'teacher trained on a different observed map'
        XA, XB, YA, YB = T[('fit', 'img', 'A')], T[('fit', 'img', 'B')], T[('fit', 'cap', 'A')], T[('fit', 'cap', 'B')]
        R = 1000
        i2t_top = torch.empty(N_TRAIN_IMG, dtype=torch.long, device=dev)
        for i in range(0, N_TRAIN_IMG, R):          # pass A: image rows
            S = (XA[i:i + R] @ YA.T + XB[i:i + R] @ YB.T) / 2
            i2t_top[i:i + R] = S.argmax(1)
        t2i_top = torch.empty(N_TRAIN_CAP, dtype=torch.long, device=dev); i2t_rr = torch.zeros(N_TRAIN_IMG, dtype=torch.long, device=dev)
        for c in range(0, N_TRAIN_CAP, R):          # pass B: caption rows (S^T) -> t2i top-1 + i2t reverse ranks
            ST = (YA[c:c + R] @ XA.T + YB[c:c + R] @ XB.T) / 2
            t2i_top[c:c + R] = ST.argmax(1)
            sel = torch.nonzero((i2t_top >= c) & (i2t_top < c + R)).squeeze(1)
            for s0 in range(0, len(sel), 2048):
                ss = sel[s0:s0 + 2048]; rows = ST[i2t_top[ss] - c]
                v = rows.gather(1, ss[:, None])
                i2t_rr[ss] = (rows > v).sum(1) + 1
        t2i_rr = torch.zeros(N_TRAIN_CAP, dtype=torch.long, device=dev)
        for i in range(0, N_TRAIN_IMG, R):          # pass C: image rows -> t2i reverse ranks
            S = (XA[i:i + R] @ YA.T + XB[i:i + R] @ YB.T) / 2
            sel = torch.nonzero((t2i_top >= i) & (t2i_top < i + R)).squeeze(1)
            for s0 in range(0, len(sel), 2048):
                ss = sel[s0:s0 + 2048]; rows = S[t2i_top[ss] - i]
                v = rows.gather(1, ss[:, None])
                t2i_rr[ss] = (rows > v).sum(1) + 1
        res[k] = dict(i2t_top1=i2t_top.cpu().numpy(), i2t_rr=i2t_rr.cpu().numpy(), t2i_top1=t2i_top.cpu().numpy(), t2i_rr=t2i_rr.cpu().numpy())
        assert (res[k]['i2t_rr'] >= 1).all() and (res[k]['t2i_rr'] >= 1).all()
        tinfo[f't{k}'] = dict(cond=tc, unit=T['manifest']['parent']['unit'], model_best_sha256=T['manifest']['parent']['model_best_sha256'],
                              cache_manifest_sha256=T['manifest_sha256'], cache_dir=C.CONDS[tc]['cache_dir'], native_dev_rsum=T['manifest']['native_dev']['rsum'],
                              seconds=time.time() - t0, gpu=torch.cuda.get_device_name(0))
        tinfo[f't{k}']['_dev_native'] = T['native_dev']
        del T, XA, XB, YA, YB
        torch.cuda.empty_cache()
    # ---------------------------------------------------------------- acceptance per query (target lists = natural top-32 + null)
    img_ids = np.arange(N_TRAIN_IMG); cap_ids = np.arange(N_TRAIN_CAP)
    i2t_null = ~(obs[ci] == img_ids[:, None]).any(1)                  # i2t query image q: no top-32 caption observed with q
    t2i_null = ~(ct == obs[:, None]).any(1)                            # t2i query caption q: observed image not in top-32
    def acc(null, cand, pa, pb, r1, r2):
        agree = pa == pb; inl = (cand == pa[:, None]).any(1); rrok = (r1 <= 5) & (r2 <= 5)
        return null & agree & inl & rrok, dict(agree=agree, in_list=inl, rr_ok=rrok, rr1_ok=r1 <= 5, rr2_ok=r2 <= 5)
    i2t_acc, fi = acc(i2t_null, ci, res[1]['i2t_top1'], res[2]['i2t_top1'], res[1]['i2t_rr'], res[2]['i2t_rr'])
    t2i_acc, ft = acc(t2i_null, ct, res[1]['t2i_top1'], res[2]['t2i_top1'], res[1]['t2i_rr'], res[2]['t2i_rr'])
    # cross-check vs the package reference applied to full per-query N rows (null slot last, unpermuted)
    for dn, null, cand, n, k1 in (('i2t', i2t_null, ci, N_TRAIN_IMG, 'i2t'), ('t2i', t2i_null, ct, N_TRAIN_CAP, 't2i')):
        ids = np.concatenate([cand, np.full((n, 1), -1)], 1)
        if dn == 'i2t':
            m = obs[cand] == img_ids[:, None]
        else:
            m = cand == obs[:, None]
        A = np.zeros(ids.shape); A[:, :32] = m / np.maximum(m.sum(1, keepdims=True), 1); A[:, 32] = (m.sum(1) == 0)
        out, accr = complete_missing(A, ids, res[1][f'{k1}_top1'], res[2][f'{k1}_top1'], np.stack([res[1][f'{k1}_rr'], res[2][f'{k1}_rr']], 1).astype(float))
        assert np.array_equal(accr, i2t_acc if dn == 'i2t' else t2i_acc), f'{dn}: vectorised acceptance != package reference'
    z = dict(t1_i2t_top1=res[1]['i2t_top1'], t1_i2t_rr=res[1]['i2t_rr'], t2_i2t_top1=res[2]['i2t_top1'], t2_i2t_rr=res[2]['i2t_rr'],
             t1_t2i_top1=res[1]['t2i_top1'], t1_t2i_rr=res[1]['t2i_rr'], t2_t2i_top1=res[2]['t2i_top1'], t2_t2i_rr=res[2]['t2i_rr'],
             i2t_null=i2t_null, t2i_null=t2i_null, i2t_accept=i2t_acc, t2i_accept=t2i_acc,
             i2t_agreed=np.where(i2t_acc, res[1]['i2t_top1'], -1), t2i_agreed=np.where(t2i_acc, res[1]['t2i_top1'], -1))
    f = pdir / f'completion_{C.DS_KEY}.npz'; np.savez(f, **z)
    # ---------------------------------------------------------------- coverage tables
    wi = np.bincount(obs, minlength=N_TRAIN_IMG).astype(np.float64)   # i2t query distribution of N: image obs[rec], rec uniform
    def cov(null, acc_, fl, w):
        w = w / w.sum(); out = {}
        for nm, ww in (('per_query_id', np.full(len(null), 1.0 / len(null))), ('training_record_weighted', w)):
            nr = float(ww[null].sum())
            out[nm] = dict(missing_row_fraction=nr,
                           teacher_agree_given_missing=float(ww[null & fl['agree']].sum() / nr) if nr else float('nan'),
                           agreed_in_target_top32_given_missing=float(ww[null & fl['agree'] & fl['in_list']].sum() / nr) if nr else float('nan'),
                           reciprocal_pass_given_agreed_in_list=float(ww[null & fl['agree'] & fl['in_list'] & fl['rr_ok']].sum() / max(ww[null & fl['agree'] & fl['in_list']].sum(), 1e-300)),
                           reciprocal_t1_pass_given_agreed_in_list=float(ww[null & fl['agree'] & fl['in_list'] & fl['rr1_ok']].sum() / max(ww[null & fl['agree'] & fl['in_list']].sum(), 1e-300)),
                           reciprocal_t2_pass_given_agreed_in_list=float(ww[null & fl['agree'] & fl['in_list'] & fl['rr2_ok']].sum() / max(ww[null & fl['agree'] & fl['in_list']].sum(), 1e-300)),
                           completed_given_missing=float(ww[acc_].sum() / nr) if nr else float('nan'), completed_fraction_of_all_rows=float(ww[acc_].sum()),
                           teacher_agree_all_rows=float(ww[fl['agree']].sum()),
                           label_mass_null_N=nr, label_mass_null_COMPLETE=nr - 0.5 * float(ww[acc_].sum()),
                           label_mass_real_N=1 - nr, label_mass_real_COMPLETE=1 - nr + 0.5 * float(ww[acc_].sum()))
        out['n_queries'] = int(len(null)); out['n_missing'] = int(null.sum()); out['n_completed'] = int(acc_.sum())
        return out
    coverage = dict(i2t=cov(i2t_null, i2t_acc, fi, wi), t2i=cov(t2i_null, t2i_acc, ft, np.ones(N_TRAIN_CAP)))
    # repeats under the actual N sampler stream (seed [1001, 3], 8000 updates x 16 queries per direction; draw order of data36.draw_main)
    rng = np.random.default_rng([1001, 3]); cnt_i = np.zeros(N_TRAIN_IMG, np.int64); cnt_t = np.zeros(N_TRAIN_CAP, np.int64)
    for _ in range(8000):
        rec_i, rec_t, _, _ = D36.draw_main(rng, 16, D36.K_NULL)
        np.add.at(cnt_i, obs[rec_i], 1); np.add.at(cnt_t, rec_t, 1)
    for dn, cn, acc_, null in (('i2t', cnt_i, i2t_acc, i2t_null), ('t2i', cnt_t, t2i_acc, t2i_null)):
        coverage[dn]['sampler_stream_8000'] = dict(rows_drawn=int(cn.sum()), missing_rows_drawn=int(cn[null].sum()), completed_rows_drawn=int(cn[acc_].sum()),
                                                   unique_completed_queries_drawn=int(((cn > 0) & acc_).sum()),
                                                   completed_draws_per_unique_completed_query_mean=float(cn[acc_ & (cn > 0)].mean()) if (acc_ & (cn > 0)).any() else 0.0,
                                                   completed_draws_per_unique_completed_query_max=int(cn[acc_].max()) if acc_.any() else 0)
    coverage['note'] = 'observed-absence rates (no truth); completion = fixed teacher cross-check, 0.5 real / 0.5 null; i2t weight = observed-caption count of the image (N query distribution)'
    write_json(pdir / f'COVERAGE_{C.DS_KEY}.json', coverage)
    # ---------------------------------------------------------------- teacher-only DEV rerank (target H top-100, rest in H order)
    tab = tgt['dev_tab']; tv = .5 * (tinfo['t1']['_dev_native'] + tinfo['t2']['_dev_native'])
    gi = np.take_along_axis(tv, tab['i_top'], 1); gt = tv[tab['t_top'], np.arange(tv.shape[1])[:, None]]
    m, tb, cv, rk = EV.composite_metrics(tgt['native_dev'], tab, gi, gt)
    nat, _, nrk = EV.full_metrics(tgt['native_dev'])
    tr = EV.transitions(rk, nrk, rk, nrk, tab)
    rer = dict(dev_retrieval=m, native_H=nat, diff_vs_H={k: m[k] - nat[k] for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')},
               transitions_vs_H=tr, tie_bounded=tb, coverage=cv,
               teachers_full_dev_native_rsum={k: tinfo[k]['native_dev_rsum'] for k in ('t1', 't2')},
               rule='score = mean over the two teachers of their peer-mean native DEV score; rerank the target H DEV top-100 per query, rest in H order; no DEV-label choice')
    np.savez(pdir / f'teacher_only_dev_ranks_{C.DS_KEY}.npz', **rk)
    write_json(pdir / f'TEACHER_ONLY_RERANK_{C.DS_KEY}.json', rer)
    for k in tinfo:
        tinfo[k].pop('_dev_native')
    man = dict(line='R39 COMPLETE prep (package VCS_R38_Review_R39_Design_20261102 section 2)', created_utc=utc(), ds=C.DS_KEY,
               target=dict(cond=C.TARGET_COND, unit=tgt['manifest']['parent']['unit'], model_best_sha256=tgt['manifest']['parent']['model_best_sha256'],
                           cache_manifest_sha256=tgt['manifest_sha256'], candidate_table_sha256=tgt['cand_sha256'], prep_manifest_sha256=tgt['prep_manifest_sha256']),
               teachers=tinfo, rule=dict(real_mass=0.5, reciprocal_max=5, agreement='both teachers same full-gallery top-1', no_insertion='agreed ID must be in the target natural top-32',
                                          rows='only rows whose N label is all-null', reference='r39_targets.complete_missing (byte copy of package reference/targets.py)',
                                          reference_sha256=sha256(W / a.tree / 'r39_targets.py'), score='teacher peer-mean native fp32 dot product (no TF32)'),
               files={p.name: dict(sha256=sha256(p)) for p in (f, pdir / f'COVERAGE_{C.DS_KEY}.json', pdir / f'TEACHER_ONLY_RERANK_{C.DS_KEY}.json', pdir / f'teacher_only_dev_ranks_{C.DS_KEY}.npz')},
               counts=dict(i2t_missing=int(i2t_null.sum()), i2t_completed=int(i2t_acc.sum()), t2i_missing=int(t2i_null.sum()), t2i_completed=int(t2i_acc.sum())),
               empty_acceptance=bool(i2t_acc.sum() + t2i_acc.sum() == 0), test_guard=guard_record(), elapsed_seconds=time.time() - t_all,
               env=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__))
    write_json(pdir / 'COMPLETE_PREP_MANIFEST.json', man)
    print('R39_COMPLETE_PREP_DONE', json.dumps(man['counts']), 'teacher_only_rsum', m['rsum'], 'H', nat['rsum'])


if __name__ == '__main__':
    main()
