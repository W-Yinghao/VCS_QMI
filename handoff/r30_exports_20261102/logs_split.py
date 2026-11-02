#!/usr/bin/env python3
"""R30 per-epoch branch counts from the existing training logs (EPOCHS.json split_A/split_B, written post hoc by the native
loop; truth never fed to training). Peer A trains on pred_B (labeled = trusted-main branch, the rest = noisy-repair branch),
peer B on pred_A. Counts: main_observed_clean = round(precision * n_pred_clean); total observed-clean from the noise map.
Also the logged per-epoch main-term readouts of the unit (r30_* train stats: raw/scaled main term, preclip norm, clip rate,
g_main / g_intra / g_clean / g_noisy means). Writes out/LOGGED_SPLIT_COUNTS.csv. Usage: python3 -B logs_split.py"""
import csv, glob, json, os
import numpy as np

OUTR = '/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101'
NOISE = {'0': None, '60': '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.6.npy'}
HERE = os.path.dirname(os.path.abspath(__file__))
N = 145000
tot_clean = {'0': N}
z = np.load(NOISE['60']); tot_clean['60'] = int((z == np.arange(N) // 5).sum())
rows = []
for p in sorted(glob.glob(f'{OUTR}/runs/*/EPOCHS.json')):
    u = os.path.basename(os.path.dirname(p))
    rj = sorted(glob.glob(f'{os.path.dirname(p)}/RUN_*.json'))
    noise = '0' if '_f00_' in u else '60'
    h = json.load(open(p))
    for e in h['cotrain']:
        for peer, sk in (('A', 'split_B'), ('B', 'split_A')):
            s = e[sk]; n = s['n_pred_clean']; mc = int(round(s['precision'] * n))
            tr = e['train'][peer]
            rows.append({'unit_id': u, 'noise_percent': noise, 'raw_epoch': e['epoch'], 'completed_cotrain_epochs': e['epoch'] + 1, 'peer': peer,
                         'split_from_peer': sk[-1], 'main_n': n, 'main_observed_clean': mc, 'main_observed_mismatch': n - mc,
                         'noisy_n': N - n, 'noisy_observed_clean': tot_clean[noise] - mc, 'noisy_observed_mismatch': (N - tot_clean[noise]) - (n - mc),
                         'precision': s['precision'], 'recall': s['recall'], 'dev_rsum': e['dev']['rsum'],
                         **{k: tr.get(k) for k in ('steps', 'r30_main_raw_mean', 'r30_main_scaled_mean', 'r30_preclip_norm_mean', 'r30_clip_rate',
                                                   'r30_gnorm_main_mean', 'r30_gnorm_intra_mean', 'gnorm_enc_clean_mean', 'gnorm_enc_noisy_mean',
                                                   'gnorm_refiner_mean', 'i2t_n', 't2i_n', 'i2t_no_proto', 't2i_no_proto')}})
keys = list(rows[0])
with open(f'{HERE}/out/LOGGED_SPLIT_COUNTS.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, keys, restval='NA'); w.writeheader(); w.writerows(rows)
print(len(rows), 'rows;', 'total observed-clean', tot_clean)
