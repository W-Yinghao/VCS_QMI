"""Per-run record of 1-sample labeled-batch events (R25 NATIVE_FIDELITY §3) for every R26 run. Writes LT2_EVENTS.json.
Peer A trains on peer B's clean split (split_B) and vice versa; split size % 128 == 1 -> last labeled batch has 1 sample.
For each run: epochs with such an event, skip_labeled_lt2 counts (0 in runs started before the fix), whether the run
started with the fixed code (r26_co_train.py sha in RUN_*.json), and a NaN/collapse flag (DEV rsum < 50 after epoch 1)."""
import json, glob, os, hashlib

O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027/runs'
D = '/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027'
REV = json.load(open(f'{D}/RUN_LOCK.json'))['code_revisions']
FIXED = REV[0]['sha_changes']['r26_co_train.py']['after']
DATA_FIXED = REV[1]['sha_changes']['data.py']['after']
out = {}
for ep_path in sorted(glob.glob(f'{O}/*/*/EPOCHS.json')):
    run_dir = os.path.dirname(ep_path); cell, arm = run_dir.split('/')[-2:]
    if cell == 'smoketest':
        continue
    h = json.load(open(ep_path)); co = h.get('cotrain', [])
    events = [(e['epoch'], p) for e in co for p, sp in (('A', 'split_B'), ('B', 'split_A')) if e[sp]['n_pred_clean'] % 128 == 1]
    skipped = sum(e['train'][p].get('skip_labeled_lt2', 0) for e in co for p in ('A', 'B') if p in e.get('train', {}))
    runs = sorted(glob.glob(f'{run_dir}/RUN_*.json'))
    code = [json.load(open(r))['env']['code_sha256'].get('r26_co_train.py') for r in runs]
    dcode = [json.load(open(r))['env']['code_sha256'].get('data.py') for r in runs]
    out[f'{cell}/{arm}'] = dict(epochs_done=len(co), lt2_events=events, skip_labeled_lt2=skipped,
                                attempts=[os.path.basename(r) for r in runs], fixed_code=[c == FIXED for c in code], data_py_fixed=[c == DATA_FIXED for c in dcode],
                                collapsed=any(e['epoch'] > 1 and e['dev']['rsum'] < 50 for e in co),
                                done=os.path.exists(f'{run_dir}/DONE.json'))
json.dump(out, open(f'{D}/LT2_EVENTS.json', 'w'), indent=1)
for k, v in out.items():
    print(k, v['epochs_done'], 'events', v['lt2_events'], 'skips', v['skip_labeled_lt2'], 'lt2fix', v['fixed_code'], 'datafix', v['data_py_fixed'], 'collapsed', v['collapsed'])
