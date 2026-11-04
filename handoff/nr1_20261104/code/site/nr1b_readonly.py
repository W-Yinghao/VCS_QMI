#!/usr/bin/env python3
"""NR1B read-only diagnostics (no training, DEV only, TEST never read).
 (2) GL local ablation: zero local_readout at step 8000, rerank SAME base top-100 (separate nr1.evaluate process,
     float64, scores hashed before truth), compare per query vs full GL and vs G (same condition/objective/seed).
 (3) DEV instance errors (evaluator-only ranks): 4 categories per direction + FIT post-hoc stratification of
     scores by isolated injected-noise truth (Flickr60/80; clean flag = observed image == Flickr00 identity image).
 (4) Training signal: TRAIN.jsonl summaries + independent FIT samples (seed 2026110401 = A, 2026110402 = B):
     score quantiles on P/Q, |T|>.99, raw J, and a common-calibration J (positive affine refit on A by the package
     fit_affine, J read on B) so J/BCE/NCE are on one scale.
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import FeatureBank, load_observed, write_json, read_json
from nr1.models import DirectScorer
from nr1.calibrate import fit_affine

O = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'); OB = O/'nr1b'
STEPS = [0, 250, 500, 1000, 2000, 4000, 8000]; NS = 32768


def ranks(d):
    z = np.load(d/'RANKS_EVAL_ONLY.npz'); return dict(i2t=(z['base_i_ranks'], z['i_ranks']), t2i=(z['base_t_ranks'], z['t_ranks']))


def categories(b, r):
    return dict(base_correct_head_correct=int(((b == 1) & (r == 1)).sum()), base_correct_head_lost=int(((b == 1) & (r > 1)).sum()),
                base_wrong_head_rescued=int(((b > 1) & (r == 1)).sum()),
                in_top100_still_failed=int(((b > 1) & (b <= 100) & (r > 1)).sum()), coverage_failure=int((b > 100).sum()), n=int(len(b)))


def load_model(ck_path, dev='cuda'):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    m = DirectScorer(**ck['binding']['model_config']).to(dev); m.load_state_dict(ck['model']); m.eval(); return m, ck


@torch.no_grad()
def pair_scores(m, bank, ii, tt, local, chunk=4096):
    out = []
    for s in range(0, len(ii), chunk):
        d = bank.tensors(ii[s:s+chunk], tt[s:s+chunk], 'cuda', local=local); out.append(m.pairs(**d).double().cpu().numpy())
    return np.concatenate(out)


def jval(gp, gq):
    tp, tq = np.tanh(gp/2), np.tanh(gq/2)
    return float((tp - tp**2/2).mean() - (tq + tq**2/2).mean())


def auc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg])); return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', required=True); a = ap.parse_args()
    py = sys.executable; OB.mkdir(parents=True, exist_ok=True)
    units = [Path(p) for p in a.runs]
    bank = FeatureBank(O/'cache/fit', 'fit')
    clean_img = load_observed(O/'data/observed_Flickr00.tsv', bank)[0]
    for run in units:
        u = run.name; cfg = read_json(run/'BINDING.json')['config']; cond, arch = cfg['condition'], cfg['architecture']
        res_p = OB/'readonly'/f'{u}.json'
        if res_p.exists(): print('skip', u); continue
        res = dict(unit_id=u, condition=cond, architecture=arch, objective=cfg['objective'], seed=cfg['seed'])
        # (4a) training log
        log = [json.loads(l) for l in open(run/'TRAIN.jsonl')]
        res['train_log'] = [dict(step=r['step'], loss=r['loss'], J_batch=r['J'], grad_norm=r['grad_norm'], mean_abs_T=r['mean_abs_T'],
                                  saturated_fraction=r['saturated_fraction']) for r in log if r['step'] in STEPS or r['step'] == 1]
        gn = np.array([r['grad_norm'] for r in log]); res['grad_norm_summary'] = dict(mean=float(gn.mean()), max=float(gn.max()), last=float(gn[-1]),
                                                                                      frac_logged_steps_clipped=float((gn > cfg['grad_clip']).mean()))
        # (4b)+(3b) independent FIT samples
        im, tx = load_observed(O/f'data/observed_{cond}.tsv', bank); clean = im == clean_img
        rs = {}
        for tag, seed in [('A', 2026110401), ('B', 2026110402)]:
            g = np.random.default_rng(seed); p, qi, qt = g.integers(len(im), size=NS), g.integers(len(im), size=NS), g.integers(len(im), size=NS)
            rs[tag] = (p, qi, qt)
        res['fit_read'] = []
        for s in STEPS:
            m, _ = load_model(run/f'step_{s:06d}.pt')
            sc = {}
            for tag, (p, qi, qt) in rs.items():
                sc[tag] = (pair_scores(m, bank, im[p], tx[p], arch == 'GL'), pair_scores(m, bank, im[qi], tx[qt], arch == 'GL'))
            gpB, gqB = sc['B']
            cal = fit_affine(*sc['A'])
            row = dict(step=s, raw_J_B=jval(gpB, gqB), common_cal=dict(a=cal['a0'], b=cal['b0']),
                       common_cal_J_B=jval(cal['a0']*gpB+cal['b0'], cal['a0']*gqB+cal['b0']),
                       P_quantiles={q: float(np.quantile(gpB, q/100)) for q in (1, 5, 25, 50, 75, 95, 99)},
                       Q_quantiles={q: float(np.quantile(gqB, q/100)) for q in (1, 5, 25, 50, 75, 95, 99)},
                       P_sat=float((np.abs(np.tanh(gpB/2)) > .99).mean()), Q_sat=float((np.abs(np.tanh(gqB/2)) > .99).mean()),
                       PQ_auc=auc(gpB, gqB))
            if cond != 'Flickr00':  # evaluator-only post-hoc stratification by isolated injected-noise truth
                cB = clean[rs['B'][0]]
                row['fit_noise_strata'] = dict(n_clean=int(cB.sum()), n_mismatched=int((~cB).sum()), mean_g_clean=float(gpB[cB].mean()),
                                               mean_g_mismatched=float(gpB[~cB].mean()), auc_clean_vs_mismatched=auc(gpB[cB], gpB[~cB]),
                                               auc_mismatched_vs_Q=auc(gpB[~cB], gqB))
            res['fit_read'].append(row)
        # (3a) DEV categories at step 8000 (and step 0 = base)
        rk = ranks(run/'eval_008000')
        res['dev_categories_step8000'] = {d: categories(*v) for d, v in rk.items()}
        # (2) GL local ablation
        if arch == 'GL':
            ab = OB/'ablation'/u; ab.mkdir(parents=True, exist_ok=True); ckp = ab/'step_008000_local_readout_zeroed.pt'
            if not ckp.exists():
                ck = torch.load(run/'step_008000.pt', map_location='cpu', weights_only=False)
                ck['model']['local_readout.weight'] = torch.zeros_like(ck['model']['local_readout.weight'])
                ck['ablation'] = 'local_readout.weight zeroed (read-only diagnostic, not a training control)'; ck.pop('optimizer', None)
                torch.save(ck, ckp)
            if not (ab/'eval'/'METRICS.json').exists():
                subprocess.run([py, '-B', '-m', 'nr1.evaluate', '--cache', str(O/'cache/dev'), '--truth', str(O/'evaluator_only/dev_positives.json'),
                                '--checkpoint', str(ckp), '--out', str(ab/'eval'), '--device', 'cuda', '--score-dtype', 'float64'], check=True)
            mab = read_json(ab/'eval'/'METRICS.json'); rab = ranks(ab/'eval')
            gname = u.replace('_GL_', '_G_')
            gdir = (run.parent/gname/'eval_008000')
            rg = ranks(gdir) if gdir.exists() else None
            cmp = {}
            for d in ('i2t', 't2i'):
                full, abl = rk[d][1], rab[d][1]
                cmp[d] = dict(full_R1_ablated_not=int(((full == 1) & (abl > 1)).sum()), ablated_R1_full_not=int(((abl == 1) & (full > 1)).sum()),
                              ablated_vs_base=categories(rab[d][0], abl))
                if rg is not None:
                    gr = rg[d][1]
                    cmp[d].update(GL_R1_G_not=int(((full == 1) & (gr > 1)).sum()), G_R1_GL_not=int(((gr == 1) & (full > 1)).sum()),
                                  ablated_R1_G_not=int(((abl == 1) & (gr > 1)).sum()), G_R1_ablated_not=int(((gr == 1) & (abl > 1)).sum()))
            res['local_ablation'] = dict(metrics=mab['metrics'], delta_rSum_vs_base=mab['delta_rSum'],
                                         full_GL_rSum=read_json(run/'eval_008000/METRICS.json')['metrics']['rSum'],
                                         G_rSum=read_json(gdir/'METRICS.json')['metrics']['rSum'] if gdir.exists() else None,
                                         readout_weight=torch.load(run/'step_008000.pt', map_location='cpu', weights_only=False)['model']['local_readout.weight'].tolist(),
                                         per_query=cmp)
        res_p.parent.mkdir(parents=True, exist_ok=True); write_json(res_p, res); print('done', u, flush=True)


if __name__ == '__main__':
    main()
