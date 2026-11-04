#!/usr/bin/env python3
"""NR3 D0 read-only checks for one condition (FIT pools only; DEV numbers are read from existing evaluator METRICS files;
no truth). For the registered NR2-HREF seed-1001 step-8000 J/BCE/NCE checkpoints:
  raw HREF-J / BCE on READ-B (g = s - log w), positive-affine calibration fitted on CAL-A (start (1,0)) and its HREF-J on
  READ-B, strata (direction x P/Q x hard/non-hard): g quantiles, |T|>.99, 1-T^2 quantiles, mean dL_J/dg and dL_BCE/dg;
  existing product-Q J (FIT_READOUT, separate product pool) and DEV six recalls.
Step 0: NR2 init (NR1 a0,b0) vs NR3-A init (HCAL a,b) on the identical first training batch (ReferenceSampler seed 1001):
  actual g = s - log w strata, d(-J)/dg = (1-T^2) dBCE/dg check on the same logits, parameter gradients of the J/BCE/NCE step
  losses (norm, clip trigger, cosines). Per-pair g of P pairs are saved for the isolated evaluator-only truth strata."""
import argparse, json
import numpy as np, torch
from nr1.io import FeatureBank, load_observed, read_json, write_json, sha256
from nr1.models import DirectScorer
from nr2.reference import Marginals, ReferenceSampler, pair_scores, step_loss
from nr2.proposal_reference import sampled_rows
from nr3 import pools as PL
from nr3.paths import OUT, cond_paths
from nr3.ref_init import fit_reference_affine, pooled_j

Q = (1, 5, 25, 50, 75, 95, 99)


def strata(g, hard, kind):
    g = np.asarray(g, np.float64); t = np.tanh(g / 2); w = 1 - t * t
    dj = -(1 - t) * w / 2 if kind == 'P' else (1 + t) * w / 2
    db = -1 / (1 + np.exp(g)) if kind == 'P' else 1 / (1 + np.exp(-g))
    out = {}
    for name, m in (('all', np.ones_like(hard, bool)), ('hard', hard), ('nonhard', ~hard)):
        if m.sum() == 0: out[name] = dict(n=0); continue
        out[name] = dict(n=int(m.sum()), g_q={q: float(np.percentile(g[m], q)) for q in Q}, sat_T_gt_099=float((np.abs(t[m]) > .99).mean()),
                         one_minus_T2_q={q: float(np.percentile(w[m], q)) for q in Q}, mean_dLJ_dg=float(dj[m].mean()), mean_dLBCE_dg=float(db[m].mean()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); c = ap.parse_args().cond
    P = cond_paths(c); od = OUT/'d0'/c; od.mkdir(parents=True, exist_ok=True)
    pp = OUT/'prep'/c; pools = {nm: PL.load(pp/f'POOL_{nm}.npz') for nm in ('CAL_A', 'READ_B')}
    bank = FeatureBank(P['fit'], 'fit'); im, tx = load_observed(P['observed'], bank)
    marg = Marginals(im, tx, len(bank.image_ids), len(bank.text_ids)); tab = np.load(P['tables']); top = dict(i2t=tab['i2t'], t2i=tab['t2i'])
    X = torch.from_numpy(np.array(bank.x, copy=True)).cuda(); Y = torch.from_numpy(np.array(bank.y, copy=True)).cuda()
    res = dict(condition=c, pools_sha256={nm: sha256(pp/f'POOL_{nm}.npz') for nm in pools}, truth_read=False, checkpoints={}); dump = {}
    for o in ('J', 'BCE', 'NCE'):
        par = P['parent'](o); ck = torch.load(par/'step_008000.pt', map_location='cuda', weights_only=False)
        m = DirectScorer(**ck['binding']['model_config']).cuda(); m.load_state_dict(ck['model']); m.eval()
        s = {nm: ({d: PL.model_scores(m, X, Y, d, pools[nm][d]['P_query'], pools[nm][d]['P_cand']) for d in pools[nm]},
                  {d: PL.model_scores(m, X, Y, d, pools[nm][d]['Q_query'], pools[nm][d]['Q_cand']) for d in pools[nm]}) for nm in pools}
        stA, stB = PL.stack(pools['CAL_A'], *s['CAL_A']), PL.stack(pools['READ_B'], *s['READ_B'])
        fit = fit_reference_affine(*stA, initial=(1., 0.), maxiter=500)
        gp = {d: s['READ_B'][0][d] - pools['READ_B'][d]['P_logw'] for d in ('i2t', 't2i')}
        gq = {d: s['READ_B'][1][d] - pools['READ_B'][d]['Q_logw'] for d in ('i2t', 't2i')}
        bce_raw = float(np.mean([np.logaddexp(0, -gp[d]).mean() + np.logaddexp(0, gq[d]).mean() for d in gp]))
        ev = read_json(par/'eval_008000/METRICS.json'); fr = read_json(par/'FIT_READOUT.json'); fr8 = {x['step']: x for x in fr['steps']}[8000]
        res['checkpoints'][o] = dict(unit=par.name, checkpoint_sha256=sha256(par/'step_008000.pt'),
                                     READ_B_HREF_J_raw=pooled_j(*stB), READ_B_HREF_J_calibrated=pooled_j(*stB, fit.a, fit.b), READ_B_HREF_BCE_raw=bce_raw,
                                     CAL_A_HREF_J_raw=pooled_j(*stA), CAL_A_HREF_J_calibrated=pooled_j(*stA, fit.a, fit.b), calibration_fit=fit.json(),
                                     productQ_J_FIT_READOUT=fr8['J_productQ'], productQ_pool='fit_readout common product pool (seed 2026110451), not READ-B',
                                     DEV=ev['metrics'], DEV_rescued_lost={d: [ev[d]['rescued'], ev[d]['lost']] for d in ('i2t', 't2i')},
                                     strata={d: dict(P=strata(gp[d], pools['READ_B'][d]['P_hard'], 'P'), Q=strata(gq[d], pools['READ_B'][d]['Q_hard'], 'Q')) for d in gp})
        for d in gp: dump[f'READB_{o}_{d}_P_g'] = gp[d]
    for d in ('i2t', 't2i'): dump[f'READB_{d}_P_records'] = pools['READ_B'][d]['P_records']; dump[f'READB_{d}_P_hard'] = pools['READ_B'][d]['P_hard']
    # ---- step 0: NR2 init vs NR3-A init on the identical first batch
    cal1 = read_json(P['calibration']); hcal = read_json(pp/'calibration_HCAL.json')
    batch = ReferenceSampler(marg, top['i2t'], top['t2i'], .5, 1001).draw()
    res['step0'] = {}; grads = {}
    for tag, (a0, b0) in (('NR2_init', (cal1['a0'], cal1['b0'])), ('NR3A_init', (hcal['a0'], hcal['b0']))):
        torch.manual_seed(1001)
        m = DirectScorer(**bank.dims, architecture='G', a0=a0, b0=b0, seed=1001).cuda()
        r = dict(a0=a0, b0=b0, strata={}, identity={}, param_grads={})
        with torch.no_grad(): sc = pair_scores(m, X, Y, batch)
        for d, b in batch.items():
            g = sc[d].double().cpu().numpy() - b['log_w']
            r['strata'][d] = dict(P=strata(g[:, 0], b['hard'][:, 0], 'P'), Q=strata(g[:, 1:].reshape(-1), b['hard'][:, 1:].reshape(-1), 'Q'))
            dump[f'step0_{tag}_{d}_P_g'] = g[:, 0]; dump[f'step0_{d}_P_records'] = b['records']; dump[f'step0_{d}_P_hard'] = b['hard'][:, 0]
            sd = sc[d].double().clone().requires_grad_(True); lw = torch.as_tensor(b['log_w'], device=sd.device)
            gj = torch.autograd.grad(sampled_rows(sd, lw, 'J', 'HREF').sum(), sd)[0]
            gb = torch.autograd.grad(sampled_rows(sd, lw, 'BCE', 'HREF').sum(), sd)[0]
            t = torch.tanh((sd - lw) / 2).detach()
            r['identity'][d] = dict(max_abs_diff=float((gj - (1 - t * t) * gb).abs().max()), max_abs_grad=float(gj.abs().max()))
        for o in ('J', 'BCE', 'NCE'):
            m.zero_grad(set_to_none=True); loss = step_loss(pair_scores(m, X, Y, batch), batch, o, 'HREF'); loss.backward()
            v = torch.cat([p.grad.reshape(-1) for p in m.parameters() if p.grad is not None]).double()
            grads[(tag, o)] = v.cpu()
            r['param_grads'][o] = dict(loss=float(loss), grad_norm=float(v.norm()), clip_triggered=bool(float(v.norm()) > 2.0),
                                       grad_norm_log_scale_bias=[float(m.log_scale.grad), float(m.bias.grad)])
        cos = lambda u, w: float((u @ w) / (u.norm() * w.norm() + 1e-300))
        r['param_grad_cosines'] = dict(J_BCE=cos(grads[(tag, 'J')], grads[(tag, 'BCE')]), J_NCE=cos(grads[(tag, 'J')], grads[(tag, 'NCE')]),
                                       BCE_NCE=cos(grads[(tag, 'BCE')], grads[(tag, 'NCE')]))
        res['step0'][tag] = r
    res['step0']['batch'] = 'first training batch of ReferenceSampler(alpha .5, seed 1001) = step 1 of NR2-HREF and NR3-A seed-1001 units'
    np.savez(od/'D0_PAIR_G.npz', **dump); write_json(od/'D0.json', res); print('D0 done', c)


if __name__ == '__main__':
    main()
