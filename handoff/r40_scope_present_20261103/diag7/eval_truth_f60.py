"""R40 section 7 isolated Flickr truth evaluator (separate process; reads only the arrays exported by diag7/readouts.py for the Flickr60
tag and the Flickr30K precomp pairing truth: FIT caption record k belongs to image k//5). Never run for CC152K (real noise, no truth).
Outputs go to OUT/diag7/oracle (refused by every trainer-role process) plus a numbers-only copy in handoff diag7/.
  item 2 split: FIT feature change of captions by observed-correct vs observed-wrong pairing (and by present / missing x correct / wrong).
  item 3 split: on the common states' fixed FIT batches, per objective, dL/dg sign and one-step Delta g on (i) the true partner slot of
          observed-missing (null-labelled) rows whose list holds the true partner, (ii) real slots of those rows that are not the true partner,
          (iii) the observed-positive slot of present rows split into observed-correct / observed-wrong.
Usage: python -I -B diag7/eval_truth_f60.py <tag>"""
import glob, json, re, sys
from pathlib import Path
import numpy as np

TAG = sys.argv[1]
D = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/diag7') / TAG
OD = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/diag7/oracle'); OD.mkdir(parents=True, exist_ok=True)
W = Path(__file__).resolve().parents[1]
NOISE = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.6.npy'
N_CAP = 145000
truth = np.arange(N_CAP) // 5
obs = np.load(NOISE).astype(np.int64)
assert obs.shape == (N_CAP,)
cap_correct = obs == truth


def item2():
    out = {}
    for f in sorted(D.glob('item2_*.npz')):
        z = np.load(f); ids = z['cap_ids']; c = cap_correct[ids]
        r = {}
        for p in ('A', 'B'):
            cs = z[f'cos_cap_{p}']
            r[p] = dict(caption_cos_observed_correct=float(cs[c].mean()), caption_cos_observed_wrong=float(cs[~c].mean()), n_correct=int(c.sum()), n_wrong=int((~c).sum()))
        out[f.stem[len('item2_'):]] = r
    return out


def item3():
    out = {}
    for f in sorted(D.glob('item3_*.npz')):
        z = np.load(f); nb = len({k.split('_')[0] for k in z.files})
        acc = {}
        for b in range(nb):
            g = lambda k: z[f'b{b}_{k}']
            qi, caps, ai, qc, imgs, at = g('qi'), g('caps'), g('ai'), g('qc'), g('imgs'), g('at')
            # i2t: query image qi, caption slots; true slot = caption with truth == qi; t2i: query caption qc, image slots; true slot = truth[qc]
            real_i = caps >= 0; real_t = imgs >= 0
            true_i = real_i & (truth[np.where(real_i, caps, 0)] == qi[:, None]); true_t = real_t & (imgs == truth[qc][:, None])
            null_i = ai[~real_i].reshape(-1) == 1.0; null_t = at[~real_t].reshape(-1) == 1.0
            pos_i = (ai > 0) & real_i; pos_t = (at > 0) & real_t
            cls = {
                'i2t_missing_row_true_slot': true_i & null_i[:, None], 'i2t_missing_row_other_real': real_i & ~true_i & null_i[:, None],
                'i2t_present_pos_slot_correct': pos_i & true_i, 'i2t_present_pos_slot_wrong': pos_i & ~true_i,
                't2i_missing_row_true_slot': true_t & null_t[:, None], 't2i_missing_row_other_real': real_t & ~true_t & null_t[:, None],
                't2i_present_pos_slot_correct': pos_t & true_t, 't2i_present_pos_slot_wrong': pos_t & ~true_t}
            for o in ('J', 'BCE', 'NCE'):
                for name, m in cls.items():
                    dn = name[0]
                    dl = g(f'dLdg_{dn}_{o}')[m]; dg = g(f'dg_{dn}_{o}')[m]
                    a = acc.setdefault((o, name), dict(n=0, pushed_down_grad=0, dg_neg=0, dg_sum=0.0))
                    a['n'] += int(m.sum()); a['pushed_down_grad'] += int((dl > 0).sum()); a['dg_neg'] += int((dg < 0).sum()); a['dg_sum'] += float(dg.sum())
        res = {}
        for (o, name), a in acc.items():
            res.setdefault(o, {})[name] = dict(n_slots=a['n'], frac_dLdg_pos_pushed_down=a['pushed_down_grad'] / a['n'] if a['n'] else None,
                                               frac_one_step_dg_negative=a['dg_neg'] / a['n'] if a['n'] else None, mean_one_step_dg=a['dg_sum'] / a['n'] if a['n'] else None)
        out[f.stem[len('item3_'):]] = res
    return out


def main():
    rec = dict(tag=TAG, truth_rule='FIT caption record k <-> image k//5 (pre-injection pairing)', observed_map=NOISE, observed_wrong_fraction=float((~cap_correct).mean()),
               item2_caption_feature_change_by_truth=item2(), item3_slot_effects_by_truth=item3(),
               isolation='separate evaluator process; inputs = diag7 exports + noise file; outputs only OUT/diag7/oracle and a numbers-only copy')
    (OD / f'TRUTH_{TAG}.json').write_text(json.dumps(rec, indent=1))
    (W / 'diag7' / f'TRUTH_{TAG}.json').write_text(json.dumps(rec, indent=1))
    print('R40_DIAG7_TRUTH_DONE', TAG)


if __name__ == '__main__':
    main()
