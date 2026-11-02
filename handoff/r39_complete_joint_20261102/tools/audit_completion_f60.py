"""R39 isolated evaluator (package docs/SERVER_PLAN_CN.md 2.3): Flickr60 completion precision against the pre-injection pairing
(caption k <-> image k//5), coverage of target-H-wrong / -correct queries, and the observed-vs-true list-presence intersection
(review H1). Evaluator role: reads the frozen completion table, the target candidate tables, the observed map and the truth;
writes only OUT/audit/ (refused by trainer/oracle processes). Nothing here feeds back into any rule, threshold or unit.
Usage: python -I -B tools/audit_completion_f60.py"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r39f'))
import common as C
from common import N_TRAIN_IMG, N_TRAIN_CAP, install_guards, guard_record, sha256, write_json, read_json, utc


def main():
    install_guards(role='audit')
    pdir = C.R39_PREP / 'f60'
    man = read_json(pdir / 'COMPLETE_PREP_MANIFEST.json')
    f = pdir / 'completion_f60.npz'; assert sha256(f) == man['files'][f.name]['sha256']
    z = np.load(f)
    cand = np.load(Path(C.CONDS['n60']['prep_dir']) / 'candidates_n60.npz')
    ci, ct = cand['i2t_top32'].astype(np.int64), cand['t2i_top32'].astype(np.int64)
    obs = np.load(C.CONDS['n60']['noise_file']).astype(np.int64)
    truth = np.arange(N_TRAIN_CAP) // 5                                   # evaluator-only truth
    wi = np.bincount(obs, minlength=N_TRAIN_IMG).astype(np.float64); wi /= wi.sum(); wt = np.full(N_TRAIN_CAP, 1.0 / N_TRAIN_CAP)
    q_img = np.arange(N_TRAIN_IMG); q_cap = np.arange(N_TRAIN_CAP)
    out = {}
    for dn in ('i2t', 't2i'):
        if dn == 'i2t':
            cand_, w = ci, wi
            true_in = (cand_ // 5 == q_img[:, None]).any(1); obs_in = (obs[cand_] == q_img[:, None]).any(1)
            h_correct = cand_[:, 0] // 5 == q_img                       # target H full-gallery top-1 (table is sorted) is a true caption
            agreed = z['i2t_agreed']; acc = z['i2t_accept'].astype(bool)
            agreed_true = np.where(acc, (agreed // 5) == q_img, False)
            agreed_is_htop1 = np.where(acc, agreed == cand_[:, 0], False)
            t1_true = (z['t1_i2t_top1'] // 5) == q_img; t2_true = (z['t2_i2t_top1'] // 5) == q_img
        else:
            cand_, w = ct, wt
            true_in = (cand_ == truth[:, None]).any(1); obs_in = (cand_ == obs[:, None]).any(1)
            h_correct = cand_[:, 0] == truth
            agreed = z['t2i_agreed']; acc = z['t2i_accept'].astype(bool)
            agreed_true = np.where(acc, agreed == truth, False)
            agreed_is_htop1 = np.where(acc, agreed == cand_[:, 0], False)
            t1_true = z['t1_t2i_top1'] == truth; t2_true = z['t2_t2i_top1'] == truth
        null = z[f'{dn}_null'].astype(bool)
        assert np.array_equal(null, ~obs_in)
        ws = lambda m: float(w[m].sum())
        r = {}
        for nm, ww in (('per_query_id', None), ('training_record_weighted', w)):
            s = (lambda m: float(m.mean())) if ww is None else ws
            r[nm] = dict(observed_partner_in_top32=s(obs_in), true_partner_in_top32=s(true_in), both=s(obs_in & true_in),
                         true_in_top32_given_missing=s(null & true_in) / max(s(null), 1e-300),
                         completed_fraction=s(acc), completion_precision=s(acc & agreed_true) / max(s(acc), 1e-300),
                         completed_given_missing_and_true_in_list=s(acc & true_in & null) / max(s(null & true_in), 1e-300),
                         completed_correct_given_missing_and_true_in_list=s(acc & agreed_true) / max(s(null & true_in), 1e-300),
                         H_top1_correct=s(h_correct),
                         completed_given_H_wrong=s(acc & ~h_correct) / max(s(~h_correct), 1e-300), completed_given_H_correct=s(acc & h_correct) / max(s(h_correct), 1e-300),
                         completion_precision_on_H_wrong=s(acc & agreed_true & ~h_correct) / max(s(acc & ~h_correct), 1e-300),
                         completion_precision_on_H_correct=s(acc & agreed_true & h_correct) / max(s(acc & h_correct), 1e-300),
                         completed_equals_H_top1=s(acc & agreed_is_htop1) / max(s(acc), 1e-300),
                         teacher1_top1_true=s(t1_true), teacher2_top1_true=s(t2_true))
        r['counts'] = dict(n=int(len(null)), missing=int(null.sum()), completed=int(acc.sum()), completed_correct=int((acc & agreed_true).sum()),
                           missing_with_true_in_list=int((null & true_in).sum()))
        out[dn] = r
    write_json(C.OUT / 'audit' / 'COMPLETION_AUDIT_F60.json', dict(line='R39 isolated evaluator: Flickr60 completion audit (truth = pre-injection pairing k//5)', utc=utc(),
               completion_table_sha256=man['files'][f.name]['sha256'], rule_frozen_before_audit=True, results=out, test_guard=guard_record(),
               note='evaluator only; no rule/threshold/unit uses this file'))
    print('R39_AUDIT_DONE', {dn: out[dn]['counts'] for dn in out})


if __name__ == '__main__':
    main()
