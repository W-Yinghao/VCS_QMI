#!/usr/bin/env python3
"""EVALUATOR-ONLY (separate process after d0.py): split the saved P-pair logits of D0 by the isolated injected-noise identity
(clean = observed image equals the clean-condition observed image of the same caption). Flickr/COCO only; CC152K_RECOVERED
has no injected-noise truth -> NA. Nothing here feeds any calibration, training or selection."""
import argparse
import numpy as np
from nr1.io import FeatureBank, load_observed, write_json
from nr3.paths import OUT, cond_paths
from nr3.d0 import strata


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); c = ap.parse_args().cond
    P = cond_paths(c); od = OUT/'d0'/c
    if P['clean_observed'] is None:
        write_json(od/'D0_TRUTH_EVALONLY.json', dict(condition=c, evaluator_only=True, status='NA', reason='real web noise; no injected-noise truth')); return
    bank = FeatureBank(P['fit'], 'fit', verify_hashes=False); im, tx = load_observed(P['observed'], bank); imc, txc = load_observed(P['clean_observed'], bank)
    clean_img = np.full(len(bank.text_ids), -1); clean_img[txc] = imc
    z = np.load(od/'D0_PAIR_G.npz'); res = dict(condition=c, evaluator_only=True, readB={}, step0={})
    for d in ('i2t', 't2i'):
        rb = z[f'READB_{d}_P_records']; cl = clean_img[tx[rb]] == im[rb]; hard = z[f'READB_{d}_P_hard']
        for o in ('J', 'BCE', 'NCE'):
            g = z[f'READB_{o}_{d}_P_g']
            res['readB'].setdefault(o, {})[d] = dict(clean=strata(g[cl], hard[cl], 'P'), mismatched=strata(g[~cl], hard[~cl], 'P'), clean_fraction=float(cl.mean()))
        r0 = z[f'step0_{d}_P_records']; c0 = clean_img[tx[r0]] == im[r0]; h0 = z[f'step0_{d}_P_hard']
        for tag in ('NR2_init', 'NR3A_init'):
            g = z[f'step0_{tag}_{d}_P_g']
            res['step0'].setdefault(tag, {})[d] = dict(clean=strata(g[c0], h0[c0], 'P'), mismatched=strata(g[~c0], h0[~c0], 'P'), n_clean=int(c0.sum()), n_mismatched=int((~c0).sum()))
    write_json(od/'D0_TRUTH_EVALONLY.json', res); print('truth strata done', c)


if __name__ == '__main__':
    main()
