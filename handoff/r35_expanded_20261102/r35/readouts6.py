"""R35 plan §6.1-6.3 zero-training readouts (CPU; reads R34 outputs read-only, R35 prep locks). Writes
ROOT/READOUTS6.json and ROOT/CALIBRATION_R34.csv.
 6.1 common FIT-only positive affine calibration of every R34 step-2000 peer-mean g (balanced logistic and max-J),
     DEV raw vs calibrated J (record U-statistic, peer-mean score), rank invariance on DEV, R34 T saturation.
 6.2 structure usability: native H raw / FIT-standardised cosine / R34 step-0 heads / R35 A step-0 (= native).
 6.3 error location of R34 units: rescued/lost, wrong top-1 inside H top-10/100, hubs, hard/random pair wins.
TEST is never opened. R34 critics are loaded with R34's own critics.py (read-only import by path)."""
from __future__ import annotations
import csv, importlib.util, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, PREP, R34_ROOT, R34_OUT, PEERS, CONDS, N_DEV_CAP, install_guards, guard_record, assert_no_oracle_loaded, sha256, write_json, read_json, utc
import evals as EV


def r34_critics():
    spec = importlib.util.spec_from_file_location('r34_critics', str(R34_ROOT / 'r34' / 'critics.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@torch.no_grad()
def pair_g(model, x, y, chunk=8192):
    out = []
    for i in range(0, len(x), chunk):
        xx, yy = x[i:i + chunk], y[i:i + chunk]
        if hasattr(model, 'first'):
            ax, ay = model.first(xx, yy); out.append(model.head(ax + ay))
        else:
            u, v = model.embed(xx, yy); a, b = model.ab(); out.append(a * (u * v).sum(-1) + b)
    return torch.cat(out).double().numpy()


def main():
    install_guards(); assert_no_oracle_loaded(); torch.set_num_threads(4)
    C = r34_critics()
    L = np.load(PREP / 'locks_records.npz')
    res = dict(created_utc=utc(), r34_critics_sha256=sha256(R34_ROOT / 'r34' / 'critics.py'), units={}, usability={}, error_location={})
    rows = []
    for cond in CONDS:
        cdir = R34_OUT / 'cache' / cond
        obs = np.load(CONDS[cond]['noise_file']).astype(np.int64)
        st = {p: np.load(cdir / f'standardization_peer{p}.npz') for p in PEERS}
        fit = {p: ((np.load(cdir / f'fit_img_peer{p}.npy') - st[p]['image_mean']) / st[p]['image_std'],
                   (np.load(cdir / f'fit_cap_peer{p}.npy') - st[p]['text_mean']) / st[p]['text_std']) for p in PEERS}
        devf = {p: ((np.load(cdir / f'dev_img_peer{p}.npy') - st[p]['image_mean']) / st[p]['image_std'],
                    (np.load(cdir / f'dev_cap_peer{p}.npy') - st[p]['text_mean']) / st[p]['text_std']) for p in PEERS}
        nat = np.load(cdir / 'native_dev_sims.npy'); tab = EV.dev_tables(nat)
        dev_obs = np.load(read_json(R34_OUT / 'locks' / 'LOCKS.json')['dev_observed_maps'][cond]['path']).astype(np.int64)
        dev_clean = np.arange(N_DEV_CAP) // 5
        nat_m, _, nat_rk = EV.full_metrics(nat)
        # 6.2 usability
        def cosm(x, y):
            x = x / np.linalg.norm(x, axis=1, keepdims=True); y = y / np.linalg.norm(y, axis=1, keepdims=True); return x @ y.T
        std_cos = sum(cosm(*devf[p]).astype(np.float64) for p in PEERS) / 2
        us = dict(native_H_raw_rsum=nat_m['rsum'], fit_standardised_cosine_rsum=EV.full_metrics(std_cos)[0]['rsum'],
                  R35_A_step0='equals native by construction; measured per unit in R35 eval_step0.json')
        # 6.1 / 6.3 per R34 unit
        for u in read_json(R34_ROOT / 'configs' / 'UNITS.json'):
            if f'_{cond}_' not in u:
                continue
            rd = R34_OUT / 'runs' / u; R = read_json(rd / 'RESULT.json')
            us[f'{u}_step0_rsum'] = R['curves'][0]['dev_retrieval']['rsum']
            ck = torch.load(rd / 'critic_step2000.pt', map_location='cpu', weights_only=False)
            model = C.build(R['config']['structure']); model.load_state_dict(ck['state_dict']); model.eval()
            def fit_set(pk, qk):
                ip, cp = obs[L[pk]], L[pk]; iq, cq = obs[L[qk][:, 0]], L[qk][:, 1]
                gp = np.mean([pair_g(model, torch.from_numpy(fit[p][0][ip]), torch.from_numpy(fit[p][1][cp])) for p in PEERS], 0)
                gq = np.mean([pair_g(model, torch.from_numpy(fit[p][0][iq]), torch.from_numpy(fit[p][1][cq])) for p in PEERS], 0)
                g = np.r_[gp, gq]; wp = np.r_[np.full(len(gp), 1 / len(gp)), np.zeros(len(gq))]; wq = np.r_[np.zeros(len(gp)), np.full(len(gq), 1 / len(gq))]
                return g, wp, wq
            gc, wpc, wqc = fit_set('cal_p', 'cal_q'); ge, wpe, wqe = fit_set('ev_p', 'ev_q')
            s = np.load(rd / 'dev_scores_step2000.npy'); st_ = torch.from_numpy(s)
            rec = dict(raw=dict(dev_observed_J=EV.j_from_matrix(st_, dev_obs)['J'], dev_clean_J=EV.j_from_matrix(st_, dev_clean)['J'],
                                fit_eval_J=EV.j_weighted(ge, wpe, wqe)),
                       T_saturation_dev_R34=R['T_saturation_dev'], rsum=R['dev_retrieval_step2000']['rsum'])
            for crit in ('balanced_logistic', 'max_J'):
                f = EV.fit_affine(gc, wpc, wqc, crit); z = torch.from_numpy(f['a'] * s + f['b'])
                rec[crit] = dict(a=f['a'], b=f['b'], dev_observed_J=EV.j_from_matrix(z, dev_obs)['J'], dev_clean_J=EV.j_from_matrix(z, dev_clean)['J'],
                                 fit_eval_J=EV.j_weighted(f['a'] * ge + f['b'], wpe, wqe), dev_rank_invariant=EV.rank_invariance(s, f['a'], f['b']))
                rows.append(dict(source='R34', unit=u, cond=cond, criterion=crit, rsum=rec['rsum'], a=f['a'], b=f['b'],
                                 raw_dev_clean_J=rec['raw']['dev_clean_J'], affine_dev_clean_J=rec[crit]['dev_clean_J'],
                                 raw_dev_observed_J=rec['raw']['dev_observed_J'], affine_dev_observed_J=rec[crit]['dev_observed_J'],
                                 raw_fit_eval_J=rec['raw']['fit_eval_J'], affine_fit_eval_J=rec[crit]['fit_eval_J'],
                                 rank_invariant=rec[crit]['dev_rank_invariant']))
            res['units'][u] = rec
            m, _, rk = EV.full_metrics(s)
            gt = np.arange(N_DEV_CAP) // 5
            pos_t = s[gt, np.arange(N_DEV_CAP)]; pos_i = s[np.arange(len(s))[:, None], np.arange(len(s))[:, None] * 5 + np.arange(5)].max(1)
            res['error_location'][u] = dict(transitions=EV.transitions(rk, nat_rk, rk, nat_rk, tab),
                                            pair_win_rates=EV.pair_win_rates(pos_t, np.take_along_axis(s.T, tab['hard_t'], 1), np.take_along_axis(s.T, tab['rand_t'], 1),
                                                                             pos_i, np.take_along_axis(s, tab['hard_i'], 1), np.take_along_axis(s, tab['rand_i'], 1)))
        res['usability'][cond] = us
        print(cond, us, flush=True)
    res['test_guard'] = guard_record()
    write_json(ROOT / 'READOUTS6.json', res)
    with open(ROOT / 'CALIBRATION_R34.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('R35_READOUTS6_DONE', len(rows))


if __name__ == '__main__':
    main()
