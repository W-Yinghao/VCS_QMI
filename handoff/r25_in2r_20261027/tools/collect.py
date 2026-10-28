"""Collect R25 IN2R/SPS results -> RESULTS_PER_RUN.json, CRITIC_AND_SCALE.json, PAIRED_COMPARISONS.json (pilot).

Primary endpoint per run: best full-DEV rSum over co-training epochs 0..39 (native rule: first strictly greater wins;
epoch 0 is the shared prefix epoch). Also last epoch and mean of the last 5 epochs.
Critic selection (pre-registered, protocol.json critics.selection): VCS critic = argmax best full-DEV rSum of V0 vs V1
(Flickr40%, seed1001); matched BCE uses the same link; the other BCE link is retained and flagged if stronger.
"""
import glob, json, os, re, sys
import numpy as np

ROOT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027'
PKG = '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027'
KEYS = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')


def run_record(g, arm):
    d = f'{ROOT}/groups/{g}/{arm}'
    if not os.path.exists(f'{d}/DONE.json'):
        return None
    h = json.load(open(f'{d}/EPOCHS.json'))
    co = sorted(h['cotrain'], key=lambda e: e['epoch'])
    assert [e['epoch'] for e in co] == list(range(len(co))), 'missing epochs'
    best, br = None, -1
    for e in co:
        if e['dev']['rsum'] > br:
            br, best = e['dev']['rsum'], e
    runs = sorted(glob.glob(f'{d}/RUN_*.json')) + sorted(glob.glob(f'{ROOT}/groups/{g}/prefix/RUN_*.json'))
    envs = [json.load(open(r))['env'] for r in runs]
    skips = {p: sum(e['train'][p].get('skip_small_or_empty', 0) for e in co if p in e['train']) for p in ('A', 'B')}
    noisy_n = {p: sum(e['train'][p].get('i2t_n', 0) + e['train'][p].get('t2i_n', 0) for e in co if p in e['train']) for p in ('A', 'B')}
    # labeled split for peer A = pred_B (split_B), for peer B = pred_A; size % 128 == 1 -> native 1-sample batch
    b1 = [(e['epoch'], p) for e in co for p, sp in (('A', 'split_B'), ('B', 'split_A')) if e[sp]['n_pred_clean'] % 128 == 1]
    skipped = sum(e['train'][p].get('skip_labeled_lt2', 0) for e in co for p in ('A', 'B') if p in e['train'])
    return {
        'labeled_size1_events': b1, 'labeled_size1_skipped': skipped,
        'group': g, 'arm': arm, 'epochs': len(co), 'best_epoch': best['epoch'], 'best': {k: best['dev'][k] for k in KEYS},
        'last': {k: co[-1]['dev'][k] for k in KEYS}, 'mean_last5_rsum': float(np.mean([e['dev']['rsum'] for e in co[-5:]])),
        'peer_rsum_at_best': [best['dev']['peer0_rsum'], best['dev']['peer1_rsum']],
        'dev_size': [best['dev']['n_images'], best['dev']['n_captions']], 'inference_models': 2,
        'seconds_prefix_shared': float(sum(w['seconds'] for w in h['warmup']) + co[0]['seconds']),
        'seconds_arm': float(sum(e['seconds'] for e in co[1:])),
        'peak_mem_gb': max([e.get('peak_mem_gb', 0) for e in co]),
        'gpus': sorted({(x['gpu'], x['host'], x['job']) for x in envs}), 'noisy_updates': noisy_n, 'noisy_skips': skips,
        'curve_rsum': [e['dev']['rsum'] for e in co],
    }


def main():
    res = {}
    for d in sorted(glob.glob(f'{ROOT}/groups/*/*/DONE.json')):
        g, arm = d.split('/')[-3], d.split('/')[-2]
        if arm == 'prefix':
            continue
        r = run_record(g, arm)
        if r:
            res[f'{arm}_{g}'] = r
    sps = f'{ROOT}/runs/S0_sps_f40_s1001/FULLDEV.json'
    if os.path.exists(sps):
        f = json.load(open(sps))
        res['S0_f40_s1001'] = {'group': 'f40_s1001', 'arm': 'S0_SPS_native', 'best_epoch': f['best']['epoch'],
                               'best': {k: f['best'][k] for k in KEYS}, 'last': {k: f['rows'][-1][k] for k in KEYS},
                               'mean_last5_rsum': float(np.mean([r['rsum'] for r in f['rows'][-5:]])), 'epochs': len(f['rows']),
                               'dev_size': [f['best']['n_images'], f['best']['n_captions']], 'inference_models': 2,
                               'note': 'selected offline on full DEV over saved co-training epochs; SPS internal choices used its native 200-image DEV subset'}
    json.dump({'status_labels': 'all values are reproduced (this server), DEV only, no TEST', 'runs': res},
              open(f'{PKG}/RESULTS_PER_RUN.json', 'w'), indent=1)
    cal = {g.split('/')[-3]: {k: v for k, v in json.load(open(g)).items() if k != 'rows'} for g in sorted(glob.glob(f'{ROOT}/groups/*/calib/CALIBRATION.json'))}
    sel = None
    p = {a: res.get(f'{a}_f40_s1001') for a in ('H0', 'V0', 'V1', 'B0', 'B1')}
    if all(p.values()):
        v = 'c0' if p['V0']['best']['rsum'] >= p['V1']['best']['rsum'] else 'c1'
        matched = 'B0' if v == 'c0' else 'B1'
        other = 'B1' if v == 'c0' else 'B0'
        sel = {'rule': 'VCS link = argmax best full-DEV rSum (V0 vs V1, Flickr40% seed1001; tie -> C0 default); matched BCE uses the same link',
               'V0_best': p['V0']['best']['rsum'], 'V1_best': p['V1']['best']['rsum'], 'selected_link': v, 'matched_bce': matched,
               'other_bce': other, 'other_bce_stronger': p[other]['best']['rsum'] > p[matched]['best']['rsum'],
               'note': 'if other_bce_stronger, it is kept and must be added as a control in the confirmation stages'}
        va = 'V0' if v == 'c0' else 'V1'
        pc = {f'{a}-H0': p[a]['best']['rsum'] - p['H0']['best']['rsum'] for a in ('V0', 'V1', 'B0', 'B1')}
        pc[f'{va}-{matched}'] = p[va]['best']['rsum'] - p[matched]['best']['rsum']
        pc['R1_i2t'] = {a: p[a]['best']['i2t_r1'] for a in p}; pc['R1_t2i'] = {a: p[a]['best']['t2i_r1'] for a in p}
        json.dump({'pilot_f40_s1001': pc, 'note': 'single seed pilot; formal paired comparisons over 3 seeds x 2 noise rates come with P2'},
                  open(f'{PKG}/PAIRED_COMPARISONS.json', 'w'), indent=1)
    json.dump({'calibration_per_group': cal, 'critic_selection': sel}, open(f'{PKG}/CRITIC_AND_SCALE.json', 'w'), indent=1)
    for k, r in res.items():
        print(k, r['best_epoch'], {a: round(b, 2) for a, b in r['best'].items()}, 'last', round(r['last']['rsum'], 2), 'last5', round(r['mean_last5_rsum'], 2))
    print('selection', sel)


if __name__ == '__main__':
    main()
