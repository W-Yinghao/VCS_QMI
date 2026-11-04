#!/usr/bin/env python3
"""EVALUATOR-ONLY FIT stratification (separate process, after training): the isolated injected-noise identity
(clean flag = observed image equals the Flickr00 observed image of the same caption) splits the common P pool of
fit_readout into correct / mismatched observed pairs. Nothing here feeds training or selection."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import FeatureBank, load_observed, write_json
from nr1.models import DirectScorer
from scipy.stats import rankdata
NP = 32768


def auc(a, b):
    r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a)*(len(a)+1)/2)/(len(a)*len(b)))


def main():
    ap = argparse.ArgumentParser()
    for k in ('run', 'out', 'fit_cache', 'observed', 'clean_observed'): ap.add_argument('--'+k.replace('_', '-'), required=True)
    a = ap.parse_args(); run = Path(a.run); out = Path(a.out)
    if out.exists(): print('exists', out); return
    bank = FeatureBank(a.fit_cache, 'fit'); im, tx = load_observed(a.observed, bank); imc, txc = load_observed(a.clean_observed, bank)
    clean_img = np.full(len(bank.text_ids), -1); clean_img[txc] = imc
    g = np.random.default_rng(2026110451); p, qi, qt = g.integers(len(im), size=NP), g.integers(len(im), size=NP), g.integers(len(im), size=NP)
    cl = clean_img[tx[p]] == im[p]
    X = torch.from_numpy(np.array(bank.x, copy=True)).cuda(); Y = torch.from_numpy(np.array(bank.y, copy=True)).cuda()
    res = dict(run=str(run), evaluator_only=True, n_clean=int(cl.sum()), n_mismatched=int((~cl).sum()), steps=[])
    for s in (0, 8000):
        ck = torch.load(run/f'step_{s:06d}.pt', map_location='cuda', weights_only=False)
        m = DirectScorer(**ck['binding']['model_config']).cuda(); m.load_state_dict(ck['model']); m.eval()
        with torch.no_grad():
            f = lambda i, t: m.pairs(X[torch.as_tensor(i, device='cuda')], Y[torch.as_tensor(t, device='cuda')]).double().cpu().numpy()
            sp, sq = f(im[p], tx[p]), f(im[qi], tx[qt])
        row = dict(step=s, mean_s_clean=float(sp[cl].mean()), mean_s_Q0=float(sq.mean()))
        if (~cl).any():
            row.update(mean_s_mismatched=float(sp[~cl].mean()), auc_clean_vs_mismatched=auc(sp[cl], sp[~cl]),
                       auc_mismatched_vs_Q0=auc(sp[~cl], sq))
        row['auc_clean_vs_Q0'] = auc(sp[cl], sq)
        res['steps'].append(row)
    write_json(out, res); print('done', out)


if __name__ == '__main__':
    main()
