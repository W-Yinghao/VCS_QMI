#!/usr/bin/env python3
"""§3 add-on: FIFO queue size actually SAVED in each R29 full state (state_epoch10 / state_epoch20 / state_last; raw epoch
from the state's own 'epoch'). Read-only (mmap, CPU). Epochs without a saved state are not reconstructed.
Usage: export_queue_r29.py RUNS_ROOT OUT.json"""
import glob, json, os, sys
import torch

runs, out = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(runs, '*', '*', 'state_*.pt'))):
    cell, arm = p.split('/')[-3], p.split('/')[-2]
    st = torch.load(p, map_location='cpu', mmap=True, weights_only=False)
    rec = {'unit': f'{cell}__{arm}', 'file': os.path.basename(p), 'raw_epoch': int(st['epoch']), 'completed_cotrain_epochs': int(st['epoch']) + 1}
    for peer in ('A', 'B'):
        s = st['model_' + peer]
        if 'queue' not in s:
            rec[peer] = 'NA'; continue
        q = s['queue']; n = int(q['img'].shape[0]); filled = n if q['is_full'] else int(q['ptr'])
        ids = s.get('r27_queue_ids')
        rec[peer] = {'capacity': n, 'ptr': int(q['ptr']), 'is_full': bool(q['is_full']), 'filled': filled,
                     'id_bearing_slots': int((ids['cap'][:filled] >= 0).sum()) if ids is not None else 'NA',
                     'distinct_captions_in_queue': int(torch.unique(ids['cap'][:filled][ids['cap'][:filled] >= 0]).numel()) if ids is not None else 'NA',
                     'write_epoch_min_max': [int(ids['wepoch'][:filled].min()), int(ids['wepoch'][:filled].max())] if ids is not None and filled else 'NA'}
    rows.append(rec); del st
json.dump({'what': 'queue size saved in R29 full states (only epochs whose full state was kept)', 'rows': rows}, open(out, 'w'), indent=1)
print(len(rows))
