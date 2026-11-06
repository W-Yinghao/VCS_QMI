#!/usr/bin/env python3
"""Retrying CPU driver (no --dependency): submits remaining groups, then per-seed transfers when both
groups of the seed are done, then runs the summarizer. Failed jobs are recorded, never blindly resubmitted."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import H, OUT, DATASETS, SEEDS, dump, load, check_frozen
check_frozen()
ST = H / 'DRIVER_STATE.json'
st = load(ST) if ST.exists() else {'jobs': {}, 'failed': {}, 'pre_submitted': {}}
st['jobs'].update({k: v for k, v in json.loads(sys.argv[1]).items()} if len(sys.argv) > 1 else {})
MEM = {'Flickr30K': '32G', 'COCO': '64G'}
def done(u):
    if u.startswith('T_'):
        return (OUT / 'receipts' / 'transfer' / f'd{u[2:]}' / 'TRANSFER_DONE.json').exists()
    ds, s = u.split(':'); return (OUT / 'receipts' / ds / f'd{s}' / 'GROUP_DONE.json').exists()
def queued(j):
    r = subprocess.run(['squeue', '-h', '-j', str(j), '-o', '%T'], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ''
def submit(u):
    if u.startswith('T_'):
        cmd = ['sbatch', '--parsable', f'--job-name=cdr1_t_{u[2:]}', str(H / 'sbatch/transfer.sbatch'), u[2:]]
    else:
        ds, s = u.split(':')
        cmd = ['sbatch', '--parsable', f'--job-name=cdr1_g_{ds}_{s}', f'--mem={MEM[ds]}', str(H / 'sbatch/group.sbatch'), ds, s]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.strip().split(';')[0]
    print(time.strftime('%H:%M:%S'), 'submit retry later', u, r.stderr.strip()[:200], flush=True); return None
units = [f'{ds}:{s}' for s in SEEDS for ds in DATASETS] + [f'T_{s}' for s in SEEDS]
while True:
    for u in units:
        if done(u) or u in st['failed']: continue
        j = st['jobs'].get(u)
        if j and not queued(j):
            time.sleep(20)
            if not done(u):
                st['failed'][u] = {'job': j, 'note': 'left queue without DONE marker; not resubmitted (inspect log)'}
                print('FAILED', u, j, flush=True)
            continue
        if j: continue
        if u.startswith('T_'):
            s = u[2:]
            if not all(done(f'{ds}:{s}') for ds in DATASETS):
                if any(f'{ds}:{s}' in st['failed'] for ds in DATASETS): st['failed'][u] = {'note': 'blocked by failed group'}
                continue
        jid = submit(u)
        if jid: st['jobs'][u] = jid; print(time.strftime('%H:%M:%S'), 'SUBMITTED', u, jid, flush=True)
    dump(ST, st)
    if all(done(u) or u in st['failed'] for u in units): break
    time.sleep(120)
r = subprocess.run(['/home/infres/yinwang/CS_QMI/env_nr1/bin/python', '-I', '-B', str(H / 'site/summarize_cdr1.py')], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
print('CDR1_DRIVER_DONE', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'failed=', list(st['failed']), 'summarizer_rc=', r.returncode, flush=True)
