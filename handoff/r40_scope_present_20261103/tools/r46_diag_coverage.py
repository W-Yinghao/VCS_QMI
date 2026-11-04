"""R46 zero-training readouts (no training, no TEST). Evaluator-only process (role 'audit'; writes OUT/r46_parents/truth_eval, a directory every
trainer refuses to open). Inputs: observed maps, candidate tables (parent H tables, R46-A pool tables, R46-B round-2 tables), record pools.
Per condition / table, over the training record pool:
  observed-partner-in-top32 share (i2t: query = observed image of the record; t2i: query = the caption), mean observed-positive real slots per i2t
  row, unique queries (images / captions) in the pool;
  Flickr only (evaluator-only truth = the clean caption -> image index k // 5): share of records whose TRUE partner is in the top-32, share of
  observed-positive slots that are true, share of records whose observed pair is correct.
Writes R46_COVERAGE.json / R46_COVERAGE.csv (ROOT) and the same under truth_eval. Usage: python -I -B tools/r46_diag_coverage.py"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
sys.path.insert(0, str(W / 'r46o'))
import common as CO
CO.install_guards(role='audit')
COND = {  # name -> (observed map, candidate npz, n_img, pool npz or None, flickr)
    'COCO80_full': (CO.CONDS['c80in_s1001']['noise_file'], Path(CO.CONDS['c80in_s1001']['prep_dir']) / 'candidates_c80in_s1001.npz', 113287, None, False),
    'COCO80_p58k': (CO.CONDS['c80in_s1001']['noise_file'], OUT / 'r46_parents/prep/c80in_s1001_p58k/candidates_c80in_s1001_p58k.npz', 113287, OUT / 'r46_parents/prep/c80in_s1001_p58k/pool_c80in_s1001_p58k.npz', False),
    'COCO80_p29k': (CO.CONDS['c80in_s1001']['noise_file'], OUT / 'r46_parents/prep/c80in_s1001_p29k/candidates_c80in_s1001_p29k.npz', 113287, OUT / 'r46_parents/prep/c80in_s1001_p29k/pool_c80in_s1001_p29k.npz', False),
    'Flickr80_f80_s1001': ('/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/noise/f30k_precomp_0.8.npy', OUT / 'r41_parents/prep/candidates_f80in.npz', 29000, None, True),
    'Flickr60_f60_s1001': ('/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.6.npy', Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/candidates_n60.npz'), 29000, None, True),
}
R2 = {'Flickr60_f60_s1001': 'n60', 'Flickr80_f80_s1001': 'f80in', 'COCO80_full': 'c80in_s1001'}


def stats(obs, i2t, t2i, R, flickr):
    i2t = i2t.astype(np.int64); t2i = t2i.astype(np.int64); q = obs[R]
    pos_i = obs[i2t[q]] == q[:, None]; pos_t = t2i[R] == q[:, None]
    s = dict(n_records=int(len(R)), unique_query_images=int(len(np.unique(q))), unique_query_captions=int(len(np.unique(R))),
             i2t_partner_in_top32=float(pos_i.any(1).mean()), t2i_partner_in_top32=float(pos_t.any(1).mean()), i2t_mean_observed_positive_slots=float(pos_i.sum(1).mean()))
    if flickr:   # evaluator-only truth: caption k belongs to image k // 5
        true_img = R // 5
        s['truth_observed_pair_correct'] = float((q == true_img).mean())
        s['truth_t2i_true_image_in_top32'] = float((t2i[R] == true_img[:, None]).any(1).mean())
        s['truth_i2t_true_caption_of_query_in_top32'] = float(((i2t[q] // 5) == q[:, None]).any(1).mean())
        ok_i = (i2t[q] // 5) == q[:, None]
        s['truth_share_of_i2t_observed_positive_slots_that_are_true'] = float((pos_i & ok_i).sum() / max(pos_i.sum(), 1))
        s['truth_share_of_t2i_observed_positive_slots_that_are_true'] = float((pos_t & (t2i[R] == true_img[:, None])).sum() / max(pos_t.sum(), 1))
    return s


def main():
    rows, out = [], dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), conditions={}, guard=CO.guard_record())
    for name, (nf, cf, n_img, pf, flickr) in COND.items():
        obs = np.load(nf).astype(np.int64); Z = np.load(cf); R = np.load(pf)['records'].astype(np.int64) if pf else np.arange(len(obs))
        tabs = {'parent_H': (Z['i2t_top32'], Z['t2i_top32'])}
        if name in R2:
            for o in ('J', 'BCE', 'NCE'):
                z = np.load(OUT / 'r46_parents/r2cand' / f'{R2[name]}_{o}.npz'); tabs[f'round2_{o}'] = (z['i2t_top32'], z['t2i_top32'])
        out['conditions'][name] = {}
        for tn, (a, b) in tabs.items():
            s = stats(obs, a, b, R, flickr); out['conditions'][name][tn] = s; rows.append(dict(condition=name, table=tn, **s))
            print(name, tn, json.dumps(s), flush=True)
    td = OUT / 'r46_parents' / 'truth_eval'; td.mkdir(parents=True, exist_ok=True)
    for base in (W, td):
        (base / 'R46_COVERAGE.json').write_text(json.dumps(out, indent=1))
        with open(base / 'R46_COVERAGE.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
