#!/usr/bin/env python3
"""Training-side FIT readout (no truth): J on a COMMON independent product-Q pool and on the fixed-proposal pools,
for every saved checkpoint of a unit (NR1 or NR2 checkpoint format). Pool seeds are fixed per condition and shared by
every unit of that condition. Scores s are the deployed scorer; HREF-type J uses g=s-log w (labelled as such)."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import FeatureBank, load_observed, read_json, sha256, write_json
from nr1.models import DirectScorer
from nr2.reference import Marginals, log_w

STEPS = [0, 250, 500, 1000, 2000, 4000, 8000]; NP = 32768; ALPHA = .5


def jv(tp, tq, wq=None):
    wq = np.ones_like(tq) if wq is None else wq
    return float((tp - tp**2/2).mean() - (wq*(tq + tq**2/2)).mean())


def bce(gp, gq, wq=None):
    wq = np.ones_like(gq) if wq is None else wq
    return float(np.logaddexp(0, -gp).mean() + (wq*np.logaddexp(0, gq)).mean())


def pools(im, tx, top, marg):
    g = np.random.default_rng(2026110451)
    p, qi, qt = g.integers(len(im), size=NP), g.integers(len(im), size=NP), g.integers(len(im), size=NP)
    prop = {}
    for d, seed in (('i2t', 2026110452), ('t2i', 2026110453)):
        h = np.random.default_rng(seed)
        u, m, k = h.random(NP), h.integers(len(im), size=NP), h.integers(top[d].shape[1], size=NP)
        if d == 'i2t':
            q = im[p]; y = np.where(u < ALPHA, tx[m], top[d][q, k]); pc = marg.p_txt; pos = tx[p]
            prop[d] = dict(img=q, txt=y, lw=log_w(q, y, top[d], pc, ALPHA), lw_pos=log_w(q, pos, top[d], pc, ALPHA),
                           hard=(top[d][q] == y[:, None]).any(1))
        else:
            q = tx[p]; y = np.where(u < ALPHA, im[m], top[d][q, k]); pc = marg.p_img; pos = im[p]
            prop[d] = dict(img=y, txt=q, lw=log_w(q, y, top[d], pc, ALPHA), lw_pos=log_w(q, pos, top[d], pc, ALPHA),
                           hard=(top[d][q] == y[:, None]).any(1))
    return p, qi, qt, prop


@torch.no_grad()
def scores(m, X, Y, ii, tt, chunk=8192):
    return np.concatenate([m.pairs(X[torch.as_tensor(ii[s:s+chunk], device=X.device)],
                                   Y[torch.as_tensor(tt[s:s+chunk], device=X.device)]).double().cpu().numpy()
                           for s in range(0, len(ii), chunk)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--fit-cache', required=True); ap.add_argument('--observed', required=True); ap.add_argument('--tables', required=True)
    a = ap.parse_args(); run = Path(a.run); out = Path(a.out)
    if out.exists(): print('exists', out); return
    bank = FeatureBank(a.fit_cache, 'fit'); im, tx = load_observed(a.observed, bank)
    marg = Marginals(im, tx, len(bank.image_ids), len(bank.text_ids)); tab = np.load(a.tables); top = dict(i2t=tab['i2t'], t2i=tab['t2i'])
    p, qi, qt, prop = pools(im, tx, top, marg)
    X = torch.from_numpy(np.array(bank.x, copy=True)).cuda(); Y = torch.from_numpy(np.array(bank.y, copy=True)).cuda()
    res = dict(run=str(run), observed_sha256=sha256(a.observed), tables_sha256=sha256(a.tables), pool_size=NP,
               pool_seeds=[2026110451, 2026110452, 2026110453], alpha=ALPHA, truth_read=False,
               proposal_hard_frac={d: float(prop[d]['hard'].mean()) for d in prop}, steps=[])
    for s in STEPS:
        ck = torch.load(run/f'step_{s:06d}.pt', map_location='cuda', weights_only=False)
        m = DirectScorer(**ck['binding']['model_config']).cuda(); m.load_state_dict(ck['model']); m.eval()
        sp, sq = scores(m, X, Y, im[p], tx[p]), scores(m, X, Y, im[qi], tx[qt])
        tp, tq = np.tanh(sp/2), np.tanh(sq/2)
        row = dict(step=s, J_productQ=jv(tp, tq), BCE_productQ=bce(sp, sq),
                   P_mean_s=float(sp.mean()), Q0_mean_s=float(sq.mean()))
        for d, pr in prop.items():
            sv = scores(m, X, Y, pr['img'], pr['txt'])
            w_inv = np.exp(-pr['lw'])
            row[f'J_IW_productQ_{d}'] = jv(tp, np.tanh(sv/2), w_inv)
            gp, gq = sp - pr['lw_pos'], sv - pr['lw']
            row[f'J_proposalQ_g_{d}'] = jv(np.tanh(gp/2), np.tanh(gq/2))   # J on g=s-log w under Q_d (HREF-type)
            row[f'J_proposalQ_s_{d}'] = jv(tp, np.tanh(sv/2))              # uncorrected s under Q_d (descriptive)
        res['steps'].append(row)
    write_json(out, res); print('done', out)


if __name__ == '__main__':
    main()
