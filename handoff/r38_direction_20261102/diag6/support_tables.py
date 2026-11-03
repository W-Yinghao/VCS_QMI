"""diag6 items 1 + 5 (CPU, evaluator process; observed relation only, CC152K has no truth; no TEST file opened).
Inputs: OUT6/support/ranks_<parent>.npz (support_ranks.py: native-H rank of the observed partner over the full pool, frozen top-32
membership, frozen native GMM trust subset) and the frozen candidate tables of each parent's prep.
Item 1 (per CC parent): trust x {in native top-32, not} cross-table per direction (real intersection counts), observed-partner rank bins
by trust, quantiles; record-level joint presence. A retention rate is a retention rate, never called precision.
Item 5 (pairs of CC parents; R37S R30-style parent vs each R31 IN2R-default parent, and the R31 seed pairs as the within-recipe
reference): trust-set agreement, presence agreement, top-32 set overlap, top-1 agreement, Spearman of observed-partner ranks.
Writes W6/CC_SUPPORT_*.csv and OUT6/support/SUPPORT_SUMMARY.json."""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import units6 as U6
sys.path.insert(0, str(U6.TREES['r38c']))
from common import install_guards, guard_record
install_guards(role='audit')
import numpy as np

OV = U6.OUTV
CAND = {'cc_R31_s1001': OV / 'r37c_cc152k_20261102/prep/candidates_cc.npz', 'cc_R31_s1002': OV / 'r38_direction_20261102/prep/cc_s1002/candidates_cc_s1002.npz',
        'cc_R31_s1003': OV / 'r38_direction_20261102/prep/cc_s1003/candidates_cc_s1003.npz', 'cc_R37S_s1001': OV / 'r37s_cc152k_r30h_20261102/critic/prep/cc_r37s/candidates_cc_r37s.npz'}
BINS = [(0, 1, '1'), (1, 5, '2-5'), (5, 10, '6-10'), (10, 32, '11-32'), (32, 100, '33-100'), (100, 1000, '101-1000'), (1000, 10 ** 9, '>1000')]
N = 150000
R, T, C = {}, {}, {}
for pk in U6.CC_PARENTS:
    z = np.load(U6.OUT6 / 'support' / f'ranks_{pk}.npz')
    R[pk] = dict(i2t=z['rank_i2t'].astype(np.int64), t2i=z['rank_t2i'].astype(np.int64), in_i2t=z['in_top32_i2t'], in_t2i=z['in_top32_t2i'])
    m = np.zeros(N, bool); m[z['trust']] = True; T[pk] = m
    c = np.load(CAND[pk]); C[pk] = dict(i2t=c['i2t_top32'].astype(np.int64), t2i=c['t2i_top32'].astype(np.int64))

cross, bins, quant, joint = [], [], [], []
for pk in U6.CC_PARENTS:
    tr = T[pk]
    for dn in ('i2t', 't2i'):
        inn = R[pk][f'in_{dn}']; r = R[pk][dn]
        cross.append(dict(parent=pk, direction=dn, n_records=N, trust_n=int(tr.sum()), trust_retention=float(tr.mean()),
                          trust_and_in_top32=int((tr & inn).sum()), trust_and_not_in_top32=int((tr & ~inn).sum()),
                          notrust_and_in_top32=int((~tr & inn).sum()), notrust_and_not_in_top32=int((~tr & ~inn).sum()),
                          frac_of_trusted_not_in_top32=float((tr & ~inn).sum() / tr.sum()), frac_of_untrusted_not_in_top32=float((~tr & ~inn).sum() / (~tr).sum()),
                          frac_of_in_top32_trusted=float((tr & inn).sum() / inn.sum()), frac_of_not_in_top32_trusted=float((tr & ~inn).sum() / (~inn).sum()),
                          frac_all_in_top32=float(inn.mean()), frac_observed_at_rank1=float((r == 0).mean())))
        for sel, nm in ((tr, 'trusted'), (~tr, 'untrusted'), (np.ones(N, bool), 'all')):
            rs = r[sel]
            row = dict(parent=pk, direction=dn, subset=nm, n=int(sel.sum()))
            for lo, hi, lab in BINS:
                row[f'n_rank_{lab}'] = int(((rs >= lo) & (rs < hi)).sum())
            for lo, hi, lab in BINS:
                row[f'pct_rank_{lab}'] = 100 * float(((rs >= lo) & (rs < hi)).mean())
            bins.append(row)
            quant.append(dict(parent=pk, direction=dn, subset=nm, n=int(sel.sum()), rank1based_median=float(np.median(rs) + 1), rank1based_p75=float(np.quantile(rs, .75) + 1),
                              rank1based_p90=float(np.quantile(rs, .9) + 1), rank1based_p99=float(np.quantile(rs, .99) + 1), rank1based_max=int(rs.max() + 1),
                              mean_log10_rank1based=float(np.log10(rs + 1).mean())))
    ii, it = R[pk]['in_i2t'], R[pk]['in_t2i']
    for sel, nm in ((tr, 'trusted'), (~tr, 'untrusted'), (np.ones(N, bool), 'all')):
        joint.append(dict(parent=pk, subset=nm, n=int(sel.sum()), both_dirs_in_top32=int((sel & ii & it).sum()), only_i2t_in=int((sel & ii & ~it).sum()),
                          only_t2i_in=int((sel & ~ii & it).sum()), neither_in=int((sel & ~ii & ~it).sum())))


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(np.float64); rb = np.argsort(np.argsort(b, kind='stable'), kind='stable').astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def overlap(A, Bm):
    s = np.sort(np.concatenate([A, Bm], 1), 1)
    return (s[:, 1:] == s[:, :-1]).sum(1)   # tables hold 32 distinct ids per row


pairs = []
for a, b in combinations(U6.CC_PARENTS, 2):
    ta, tb = T[a], T[b]
    row = dict(parent_a=a, parent_b=b, kind='R37S vs R31' if 'R37S' in a + b else 'R31 seed vs R31 seed (within recipe)',
               trust_n_a=int(ta.sum()), trust_n_b=int(tb.sum()), trust_retention_a=float(ta.mean()), trust_retention_b=float(tb.mean()),
               trust_retention_diff_a_minus_b=float(ta.mean() - tb.mean()), trust_both=int((ta & tb).sum()), trust_only_a=int((ta & ~tb).sum()),
               trust_only_b=int((~ta & tb).sum()), trust_neither=int((~ta & ~tb).sum()), trust_jaccard=float((ta & tb).sum() / (ta | tb).sum()),
               trust_agreement=float((ta == tb).mean()))
    for dn in ('i2t', 't2i'):
        ia, ib = R[a][f'in_{dn}'], R[b][f'in_{dn}']
        ov = overlap(C[a][dn], C[b][dn])
        row.update({f'{dn}_in_top32_a': float(ia.mean()), f'{dn}_in_top32_b': float(ib.mean()), f'{dn}_in_top32_diff_a_minus_b': float(ia.mean() - ib.mean()),
                    f'{dn}_present_both': int((ia & ib).sum()), f'{dn}_present_only_a': int((ia & ~ib).sum()), f'{dn}_present_only_b': int((~ia & ib).sum()),
                    f'{dn}_present_neither': int((~ia & ~ib).sum()),
                    f'{dn}_top32_mean_overlap_frac': float(ov.mean() / 32), f'{dn}_top32_identical_set_frac': float((ov == 32).mean()),
                    f'{dn}_top1_agree_frac': float((C[a][dn][:, 0] == C[b][dn][:, 0]).mean()),
                    f'{dn}_observed_rank_spearman': spearman(R[a][dn], R[b][dn]),
                    f'{dn}_trusted_in_both_and_present_in_a': int((ta & tb & ia).sum()), f'{dn}_trusted_in_both_and_present_in_b': int((ta & tb & ib).sum())})
    pairs.append(row)
U6.wcsv(U6.W6 / 'CC_SUPPORT_CROSSTAB6.csv', cross)
U6.wcsv(U6.W6 / 'CC_SUPPORT_RANK_BINS.csv', bins)
U6.wcsv(U6.W6 / 'CC_SUPPORT_RANK_QUANTILES.csv', quant)
U6.wcsv(U6.W6 / 'CC_SUPPORT_JOINT.csv', joint)
U6.wcsv(U6.W6 / 'CC_PARENT_PAIRS.csv', pairs)
cons = {pk: U6.read_json(U6.OUT6 / 'support' / f'ranks_{pk}.json')['table_consistency'] for pk in U6.CC_PARENTS}
U6.write_json(U6.OUT6 / 'support' / 'SUPPORT_SUMMARY.json', dict(table_consistency=cons, test_guard=guard_record(), candidate_tables={k: str(v) for k, v in CAND.items()}))
print('SUPPORT_TABLES_DONE', cons)
