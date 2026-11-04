#!/usr/bin/env python3
"""NR3 training-side FIT readout (no truth) for every saved checkpoint of a unit:
  HREF-J and HREF-BCE on the condition's fixed READ-B pool (identical Q_h for all arms of a condition; g = s - log w),
  plus the NR2 common product-Q J (nr2 site/fit_readout.py pool, seed 2026110451) for continuity with NR2 tables."""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, torch
from nr1.io import FeatureBank, load_observed, sha256, write_json
from nr1.models import DirectScorer
from nr2.reference import Marginals
from nr3 import pools as PL
from nr3.paths import OUT, cond_paths
from nr3.ref_init import pooled_j

spec = importlib.util.spec_from_file_location('fr', '/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code/site/fit_readout.py')
FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--cond', required=True); a = ap.parse_args()
    run = Path(a.run); out = run/'NR3_READOUT.json'
    if out.exists(): print('exists'); return
    P = cond_paths(a.cond); pool = PL.load(OUT/'prep'/a.cond/'POOL_READ_B.npz')
    bank = FeatureBank(P['fit'], 'fit'); im, tx = load_observed(P['observed'], bank)
    marg = Marginals(im, tx, len(bank.image_ids), len(bank.text_ids)); tab = np.load(P['tables']); top = dict(i2t=tab['i2t'], t2i=tab['t2i'])
    p, qi, qt, _ = FR.pools(im, tx, top, marg)
    X = torch.from_numpy(np.array(bank.x, copy=True)).cuda(); Y = torch.from_numpy(np.array(bank.y, copy=True)).cuda()
    res = dict(run=str(run), READ_B_sha256=sha256(OUT/'prep'/a.cond/'POOL_READ_B.npz'), truth_read=False, steps=[])
    for ck_path in sorted(run.glob('step_*.pt')):
        ck = torch.load(ck_path, map_location='cuda', weights_only=False)
        m = DirectScorer(**ck['binding']['model_config']).cuda(); m.load_state_dict(ck['model']); m.eval()
        fP = {d: PL.model_scores(m, X, Y, d, pool[d]['P_query'], pool[d]['P_cand']) for d in pool}
        fQ = {d: PL.model_scores(m, X, Y, d, pool[d]['Q_query'], pool[d]['Q_cand']) for d in pool}
        st = PL.stack(pool, fP, fQ)
        gp, gq = st[0] - st[2], st[1] - st[3]
        sp, sq = FR.scores(m, X, Y, im[p], tx[p]), FR.scores(m, X, Y, im[qi], tx[qt])
        res['steps'].append(dict(step=int(ck['step']), stage_step=int(ck.get('stage_step', ck['step'])), READ_B_HREF_J=pooled_j(*st),
                                 READ_B_HREF_BCE=float(np.logaddexp(0, -gp).mean() + np.logaddexp(0, gq).mean()),
                                 READ_B_P_mean_g=float(gp.mean()), READ_B_Q_mean_g=float(gq.mean()),
                                 READ_B_sat_frac=float((np.abs(np.tanh(np.concatenate([gp.ravel(), gq.ravel()]) / 2)) > .99).mean()),
                                 productQ_J=FR.jv(np.tanh(sp/2), np.tanh(sq/2))))
    write_json(out, res); print('done', out)


if __name__ == '__main__':
    main()
