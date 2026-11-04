"""CR1 DAG feeder (CPU Slurm job). Submits a stage only after its prerequisites' output files exist.
No --dependency. A job that leaves the queue without its outputs is marked FAILED and NOT resubmitted.
STOP: touch handoff/cr1_20261104/STOP_FEEDER.   State: FEEDER_STATE.json.
"""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path

TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104'); SB = TREE/'sbatch'
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
CONDS = ('Flickr00', 'Flickr60', 'CC152K_RECOVERED_20261104'); OBJS = ('J', 'BCE', 'NCE')
ARMS = ('OBS', 'BASE_ROW', 'BASE_CAP', 'J_ROW', 'J_CAP', 'BCE_CAP', 'NCE_CAP')
MODES = {'BASE': ('ROW', 'CAP'), 'J': ('ROW', 'CAP'), 'BCE': ('CAP',), 'NCE': ('CAP',)}
MAX_CR1, MAX_USER = 10, 27
STATE = TREE/'FEEDER_STATE.json'


def nodes():
    N = {}
    for c in CONDS:
        for o in OBJS:
            N[f'EST:{c}:{o}'] = dict(deps=[], done=[OUT/c/'experts'/o/f'E{e}'/'DONE.json' for e in ('01', '02', '03', '12', '13', '23')],
                                    cmd=['est.sbatch', c, o], name=f'cr1_est_{c[:9]}_{o}')
        for s in ('BASE',) + OBJS:
            N[f'SCORE:{c}:{s}'] = dict(deps=[] if s == 'BASE' else [f'EST:{c}:{s}'], done=[OUT/c/'scores'/f'{s}.npz.json'],
                                      cmd=['score.sbatch', c, s], name=f'cr1_score_{c[:9]}_{s}')
            N[f'MATCH:{c}:{s}'] = dict(deps=[f'SCORE:{c}:{s}'], done=[OUT/c/'relations'/f'{s}_{m}.npz.json' for m in MODES[s]],
                                      cmd=['match.sbatch', c, s, *MODES[s]], name=f'cr1_match_{c[:9]}_{s}')
        for arm in ARMS:
            dep = [] if arm == 'OBS' else [f"MATCH:{c}:{arm.split('_')[0]}"]
            N[f'STU:{c}:{arm}'] = dict(deps=dep, done=[OUT/c/'students'/arm/'UNIT_COMPLETE.json'],
                                      cmd=['student.sbatch', c, arm], name=f'cr1_stu_{c[:9]}_{arm}')
        N[f'DIAG:{c}'] = dict(deps=[f'MATCH:{c}:{s}' for s in MODES], done=[OUT/c/'GRAPH_DIAG.json'],
                              cmd=['cpu_py.sbatch', f'$CR1_PY -I -B site/graph_diag.py --condition {c} --work $CR1_OUT --banks $CR1_OUT/banks --out $CR1_OUT/{c}/GRAPH_DIAG.json'],
                              name=f'cr1_diag_{c[:9]}', extra=['--cpus-per-task=4', '--mem=32G', '--time=01:00:00'])
    return N


def queue():
    r = subprocess.run(['squeue', '-u', 'yinwang', '-h', '-o', '%i %j'], capture_output=True, text=True, timeout=60)
    if r.returncode: raise RuntimeError(r.stderr)
    rows = [l.split(None, 1) for l in r.stdout.splitlines() if l.strip()]
    return {x[0]: (x[1] if len(x) > 1 else '') for x in rows}


def main():
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    N = nodes()
    while not (TREE/'STOP_FEEDER').exists():
        try: q = queue()
        except Exception as e: print('squeue failed', e, flush=True); time.sleep(60); continue
        isdone = lambda k: all(Path(p).exists() for p in N[k]['done'])
        for k, s in st.items():
            if s['status'] == 'SUBMITTED' and s['job'] not in q:
                s['status'] = 'DONE' if isdone(k) else 'FAILED'; s['end_utc'] = time.strftime('%FT%TZ', time.gmtime())
                print(k, s['status'], s['job'], flush=True)
        running = sum(1 for s in st.values() if s['status'] == 'SUBMITTED'); new = 0
        for k, n in N.items():
            if k in st or running >= MAX_CR1 or len(q) >= MAX_USER: continue
            if isdone(k): st[k] = dict(status='DONE', job=None, note='outputs already present'); continue
            if not all(d in st and st[d]['status'] == 'DONE' for d in n['deps']): continue
            cmd = ['sbatch', '--parsable', f"--job-name={n['name']}", *n.get('extra', []), str(SB/n['cmd'][0]), *n['cmd'][1:]]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=SB)
            if r.returncode: print('submit failed', k, r.stderr.strip(), flush=True); break
            j = r.stdout.strip().split(';')[0]; st[k] = dict(status='SUBMITTED', job=j, submit_utc=time.strftime('%FT%TZ', time.gmtime()))
            with open(TREE/'SUBMITTED_CR1.tsv', 'a') as f: f.write(f"{j}\t{n['name']}\t{k}\t{n['cmd'][0]}\t{st[k]['submit_utc']}\tfeeder\n")
            running += 1; new += 1; q[j] = n['name']; print('submitted', k, j, flush=True)
        STATE.write_text(json.dumps(st, indent=1))
        if len(st) == len(N) and all(s['status'] in ('DONE', 'FAILED') for s in st.values()):
            print('ALL_NODES_FINISHED', {k: s['status'] for k, s in st.items() if s['status'] != 'DONE'}, flush=True); break
        if running == 0 and new == 0 and len(q) < MAX_USER and len(st) < len(N):
            print('BLOCKED (failed prerequisites)', sorted(set(N)-set(st)), flush=True); break
        time.sleep(60)


if __name__ == '__main__': main()
