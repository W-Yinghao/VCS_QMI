"""R40 section 7 item 4 (zero training, CPU): per-modality exposure of the 8000-update N sampler stream ([1001, 3], 16 queries/direction/update)
on the two R39 parents. Replays the production sampler (data36.sample, the stream every R38/R39/R40 unit of the parent uses) and counts, per
modality and role (query / real candidate slot / observed-positive slot), unique IDs, exposure count distribution and cumulative coverage at the
checkpoint steps; also the PRESENT-valid subset. Observed relation only (no truth). Flickr (5 captions per image, one-to-many) and CC152K
(one-to-one) differ in structure: steps / epochs are not an equal per-image supervision budget. Output: OUT/diag7/exposure_<ds>.json + CSV.
Usage: python -I -B diag7/exposure.py r40f|r40c"""
import csv, json, sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
t = sys.argv[1]
sys.path.insert(0, str(W / t))
import common as C
C.install_guards('trainer')
import torch
from data35 import load_cond
import data36 as D36

CK = (250, 500, 1000, 2000, 4000, 8000)


def summ(cnt, n_total):
    nz = cnt[cnt > 0]
    return dict(unique=int((cnt > 0).sum()), frac_of_pool=float((cnt > 0).mean()), pool=int(n_total), exposures=int(cnt.sum()),
                mean_per_seen=float(nz.mean()) if len(nz) else 0.0, p50_per_seen=float(np.median(nz)) if len(nz) else 0.0,
                p95_per_seen=float(np.quantile(nz, .95)) if len(nz) else 0.0, max=int(cnt.max()))


def main():
    d = load_cond(C.TARGET_COND, torch.device('cpu'))
    rng = np.random.default_rng([1001, 3])
    nI, nT = C.N_TRAIN_IMG, C.N_TRAIN_CAP
    cnt = {k: np.zeros(n, np.int64) for k, n in (('img_query', nI), ('cap_query', nT), ('img_candidate', nI), ('cap_candidate', nT),
                                                   ('img_pos_slot', nI), ('cap_pos_slot', nT), ('img_query_present', nI), ('cap_query_present', nT),
                                                   ('img_any', nI), ('cap_any', nT))}
    curve = []
    for step in range(8000):
        lists, idx = D36.sample('OBS_MISSING_NULL', rng, None, d, 16, None)
        (qi, caps, ai), (qc, imgs, at) = lists
        mi, mt = D36.valid_rows(lists)
        np.add.at(cnt['img_query'], qi, 1); np.add.at(cnt['cap_query'], qc, 1)
        np.add.at(cnt['img_query_present'], qi[mi], 1); np.add.at(cnt['cap_query_present'], qc[mt], 1)
        rc = caps[caps != D36.NULL_ID]; ri = imgs[imgs != D36.NULL_ID]
        np.add.at(cnt['cap_candidate'], rc, 1); np.add.at(cnt['img_candidate'], ri, 1)
        np.add.at(cnt['cap_pos_slot'], caps[(ai > 0) & (caps != D36.NULL_ID)], 1); np.add.at(cnt['img_pos_slot'], imgs[(at > 0) & (imgs != D36.NULL_ID)], 1)
        np.add.at(cnt['img_any'], np.unique(np.concatenate([qi, ri])), 1); np.add.at(cnt['cap_any'], np.unique(np.concatenate([qc, rc])), 1)
        if step + 1 in CK:
            curve.append(dict(step=step + 1, **{f'{k}_unique': int((v > 0).sum()) for k, v in cnt.items()}))
    out = dict(dataset=C.DS_KEY, parent=C.TARGET_COND, stream='[1001, 3]', updates=8000, queries_per_direction=16, n_train_img=nI, n_train_cap=nT,
               per_cap=C.PER_CAP, summary={k: summ(v, len(v)) for k, v in cnt.items()}, curve=curve,
               note='observed relation only; image "any" = query or real candidate in an update (per-update unique)')
    od = C.OUT / 'diag7'; od.mkdir(parents=True, exist_ok=True)
    (od / f'exposure_{C.DS_KEY}.json').write_text(json.dumps(out, indent=1))
    with open(W / 'diag7' / f'EXPOSURE_{C.DS_KEY}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'role', 'unique', 'pool', 'frac_of_pool', 'exposures', 'mean_per_seen', 'p50_per_seen', 'p95_per_seen', 'max'])
        for k, s in out['summary'].items():
            w.writerow([C.DS_KEY, k, s['unique'], s['pool'], f"{s['frac_of_pool']:.6f}", s['exposures'], f"{s['mean_per_seen']:.4f}", s['p50_per_seen'], s['p95_per_seen'], s['max']])
    print('R40_DIAG7_EXPOSURE', C.DS_KEY, json.dumps({k: v['unique'] for k, v in out['summary'].items()}))


if __name__ == '__main__':
    main()
