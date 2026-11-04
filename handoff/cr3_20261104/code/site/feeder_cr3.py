"""CR3 student feeder (skips units whose frozen plan is REUSE) (CPU Slurm job). No --dependency; a unit that leaves the queue without
UNIT_COMPLETE.json is marked FAILED and is NOT resubmitted. STOP: touch STOP_FEEDER."""
import csv, json, subprocess, time
from pathlib import Path
TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr3_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104')
STATE = TREE/'FEEDER_STATE.json'; MAX_CR2, MAX_USER = 12, 27
plan = json.load(open(TREE/'PROTOCOL_FROZEN_CR3.json'))['unit_plan']
rows = [r for r in csv.DictReader(open(TREE/'package'/'configs'/'MATRIX.csv')) if plan[r['unit_id']] != 'REUSE']
done = lambda r: (OUT/'students'/r['condition']/r['unit_id']/'UNIT_COMPLETE.json').exists()
st = json.loads(STATE.read_text()) if STATE.exists() else {}
while not (TREE/'STOP_FEEDER').exists():
    try:
        q = dict(l.split(None, 1) for l in subprocess.run(['squeue', '-u', 'yinwang', '-h', '-o', '%i %j'], capture_output=True, text=True, timeout=60, check=True).stdout.splitlines() if l.strip())
    except Exception as e:
        print('squeue failed', e, flush=True); time.sleep(60); continue
    for u, s in st.items():
        if s['status'] == 'SUBMITTED' and s['job'] not in q:
            r = next(x for x in rows if x['unit_id'] == u); s['status'] = 'DONE' if done(r) else 'FAILED'; print(u, s['status'], s['job'], flush=True)
    running = sum(s['status'] == 'SUBMITTED' for s in st.values())
    for r in rows:
        u = r['unit_id']
        if u in st or running >= MAX_CR2 or len(q) >= MAX_USER: continue
        name = 'cr3_' + u.replace('CR3_', '').replace('_s1001', '').replace('CC152K_RECOVERED_20261104', 'CC')
        p = subprocess.run(['sbatch', '--parsable', f'--job-name={name}', str(TREE/'sbatch'/'student.sbatch'), u], capture_output=True, text=True, cwd=TREE/'sbatch')
        if p.returncode: print('submit failed', u, p.stderr.strip(), flush=True); break
        j = p.stdout.strip().split(';')[0]; t = time.strftime('%FT%TZ', time.gmtime())
        st[u] = dict(status='SUBMITTED', job=j, submit_utc=t); running += 1; q[j] = name
        with open(TREE/'SUBMITTED_CR3.tsv', 'a') as f: f.write(f'{j}\t{name}\tSTUDENT\tGPU\t{t}\tfeeder {u}\n')
        print('submitted', u, j, flush=True)
    STATE.write_text(json.dumps(st, indent=1))
    if len(st) == len(rows) and all(s['status'] != 'SUBMITTED' for s in st.values()):
        print('ALL_UNITS_FINISHED', {u: s['status'] for u, s in st.items() if s['status'] != 'DONE'}, flush=True); break
    time.sleep(60)
