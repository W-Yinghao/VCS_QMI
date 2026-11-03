"""diag6 item 2 (CPU, evaluator process; saved DEV rank exports only, no recomputation, no TEST file opened).
Flickr60 R30-H parents s1001/s1002/s1003: every critic unit on the parent (R36/R37 shared-constant N/T references, R37 B-block factor
units on s1001, R38 units) and the native H. Per unit and direction: R@1/5/10, first-place rescues / losses vs H (also R@5/R@10), and for
t2i the image grouping (5 captions of one DEV image kept together: per-image count changes). Pairwise (all unit pairs within a parent,
incl. H): R@1 both-correct / only-a / only-b / both-wrong, both-wrong with the same wrong top-1 item, Jaccard of the error sets, t2i
per-image better/worse counts, image-cluster bootstrap 95% CI of the R@1 difference (resample DEV images; i2t query = image, t2i = its 5
captions; B = 2000, seed 20261102). Family summary: H errors rescued by >= 1 method, errors no method fixes, unique rescues per method.
Writes W6/ERR_*.csv and OUT6/errors/ERR_SUMMARY.json."""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import units6 as U6
sys.path.insert(0, str(U6.TREES['r38f']))
from common import install_guards, guard_record
install_guards(role='audit')
import numpy as np

B, SEED = 2000, 20261102


def label(u):
    return f"{u['line']}:{u['route']}-{u['architecture'][:3]}-{u['schedule'][:5]}-{u['objective']}"


def load(path):
    z = np.load(path)
    return {k: z[k] for k in ('ranks_i', 'ranks_t', 'top1_i', 'top1_t')}


rows_u, rows_p, rows_f = [], [], []
for pk in U6.FLICKR:
    P = U6.PARENTS[pk]; n_img, per = P['n_img'], P['per']
    M = {'H': dict(rk=load(P['native']), u=dict(unit='native_H', line='H', route='H', architecture='-', schedule='-', objective='-', source='parent'))}
    for u in U6.unit_index():
        if u['parent'] == pk:
            M[u['unit']] = dict(rk=load(Path(u['dir']) / 'dev_ranks_step8000.npz'), u=u)
    H = M['H']['rk']
    rng = np.random.default_rng(SEED); idx = rng.integers(0, n_img, (B, n_img))
    for name, m in M.items():
        rk, u = m['rk'], m['u']
        for dn, key, n in (('i2t', 'ranks_i', n_img), ('t2i', 'ranks_t', n_img * per)):
            r, h = rk[key], H[key]
            row = dict(parent=pk, unit=u['unit'], label='H' if name == 'H' else label(u), line=u['line'], route=u['route'], architecture=u['architecture'],
                       schedule=u['schedule'], objective=u['objective'], direction=dn, n_queries=n)
            for K in (1, 5, 10):
                c, ch = r < K, h < K
                row[f'r{K}'] = 100 * c.mean(); row[f'd_r{K}_vs_H'] = 100 * (c.mean() - ch.mean())
                row[f'rescued_r{K}'] = int((c & ~ch).sum()); row[f'lost_r{K}'] = int((~c & ch).sum())
            if dn == 't2i':
                cm = (r < 1).reshape(n_img, per).sum(1); chm = (h < 1).reshape(n_img, per).sum(1)
                row.update(img_more_r1_than_H=int((cm > chm).sum()), img_fewer_r1_than_H=int((cm < chm).sum()), img_with_rescue=int((((r < 1) & ~(h < 1)).reshape(n_img, per)).any(1).sum()),
                           img_with_loss=int(((~(r < 1) & (h < 1)).reshape(n_img, per)).any(1).sum()))
            rows_u.append(row)
    names = list(M)
    for a, b in combinations(names, 2):
        ra, rb = M[a]['rk'], M[b]['rk']
        for dn, key, tkey in (('i2t', 'ranks_i', 'top1_i'), ('t2i', 'ranks_t', 'top1_t')):
            ca, cb = ra[key] < 1, rb[key] < 1
            bw = ~ca & ~cb
            union = int((~ca | ~cb).sum())
            row = dict(parent=pk, a=a, b=b, label_a='H' if a == 'H' else label(M[a]['u']), label_b='H' if b == 'H' else label(M[b]['u']), direction=dn,
                       both_correct=int((ca & cb).sum()), only_a_correct=int((ca & ~cb).sum()), only_b_correct=int((~ca & cb).sum()), both_wrong=int(bw.sum()),
                       both_wrong_same_top1=int((bw & (ra[tkey] == rb[tkey])).sum()), jaccard_errors=float(bw.sum() / union) if union else float('nan'))
            row['net_r1_a_minus_b'] = row['only_a_correct'] - row['only_b_correct']
            ga = ca.reshape(n_img, -1).sum(1).astype(float); gb = cb.reshape(n_img, -1).sum(1).astype(float)
            dimg = ga - gb; nq = ca.size
            bs = dimg[idx].sum(1) / nq * 100
            row.update(d_r1_pp=100 * dimg.sum() / nq, d_r1_ci95_lo=float(np.quantile(bs, .025)), d_r1_ci95_hi=float(np.quantile(bs, .975)))
            if dn == 't2i':
                row.update(img_a_better=int((dimg > 0).sum()), img_b_better=int((dimg < 0).sum()), img_equal=int((dimg == 0).sum()))
            rows_p.append(row)
    fams = {'all_units': [n for n in names if n != 'H'],
            'shared_constant_N_T': [n for n in names if n != 'H' and M[n]['u']['architecture'] == 'shared' and M[n]['u']['schedule'] == 'constant' and M[n]['u']['route'] in ('N', 'T')],
            'N_only': [n for n in names if n != 'H' and M[n]['u']['route'] == 'N']}
    for fam, mem in fams.items():
        for dn, key in (('i2t', 'ranks_i'), ('t2i', 'ranks_t')):
            ch = H[key] < 1
            C = np.stack([M[n]['rk'][key] < 1 for n in mem])
            any_c = C.any(0); all_c = C.all(0)
            row = dict(parent=pk, family=fam, n_methods=len(mem), direction=dn, n_queries=int(ch.size), H_errors=int((~ch).sum()),
                       H_errors_rescued_by_any=int((~ch & any_c).sum()), H_errors_rescued_by_all=int((~ch & all_c).sum()),
                       H_errors_no_method_fixes=int((~ch & ~any_c).sum()), H_correct_lost_by_any=int((ch & ~C.all(0)).sum()), H_correct_lost_by_all=int((ch & ~any_c).sum()),
                       oracle_union_r1=100 * any_c.mean(), best_single_r1=100 * C.mean(1).max(), H_r1=100 * ch.mean(),
                       mean_pairwise_jaccard_errors=float(np.mean([((~C[i]) & (~C[j])).sum() / max(1, ((~C[i]) | (~C[j])).sum()) for i, j in combinations(range(len(mem)), 2)])) if len(mem) > 1 else float('nan'))
            if fam == 'shared_constant_N_T':
                uniq = {}
                for i, n in enumerate(mem):
                    others = np.delete(C, i, 0).any(0)
                    uniq[label(M[n]['u'])] = int((~ch & C[i] & ~others).sum())
                row['unique_rescues_per_method'] = ' '.join(f'{k}={v}' for k, v in uniq.items())
            rows_f.append(row)
U6.wcsv(U6.W6 / 'ERR_DIRECTIONAL_RECALL.csv', rows_u)
U6.wcsv(U6.W6 / 'ERR_PAIRWISE.csv', rows_p)
U6.wcsv(U6.W6 / 'ERR_FAMILY_SUMMARY.csv', rows_f)
U6.write_json(U6.OUT6 / 'errors' / 'ERR_SUMMARY.json', dict(n_unit_rows=len(rows_u), n_pair_rows=len(rows_p), n_family_rows=len(rows_f), test_guard=guard_record(),
                                                          source='dev_ranks_step8000.npz of each unit (saved at training time) + R38 summary native_ranks_<cond>.npz'))
print('ERR_DONE', len(rows_u), len(rows_p), len(rows_f))
